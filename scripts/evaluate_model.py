import matplotlib.pyplot as plt
import numpy as np
import torch
from prefect.logging import get_logger
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from torch import nn
from torchvision import models

from utils.utils import (
    DEVICE,
    plot_to_tensorboard,
)

LOGGER = get_logger()


def _setup_model(config, best_model_filepath):
    model = getattr(models, config["model_architecture"])()
    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    model.fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 1),
        nn.Sigmoid(),
    )
    model.load_state_dict(torch.load(best_model_filepath, weights_only=True))
    model.to(DEVICE)
    model.eval()
    return model


def run(config, test_data_loader, tb_writer, best_model_filepath):
    model = _setup_model(config, best_model_filepath)

    targets, predictions = [], []
    with torch.no_grad():
        for inputs, targets in test_data_loader:
            inputs, targets = inputs.to(DEVICE), targets.float().to(DEVICE).view(-1, 1)
            predictions = model(inputs).round().cpu().numpy()
            targets = targets.cpu().numpy()

    cm = confusion_matrix(targets, predictions)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(ax=ax)
    plot_to_tensorboard(tb_writer, fig, "Confusion Matrix", global_step=0)

    fpr, tpr, roc_thresholds = roc_curve(targets, [p[0] for p in predictions])
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})"
    )
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax.set_xlim(xmin=0.0, xmax=1.0)
    ax.set_ylim(ymin=0.0, ymax=1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic")
    ax.legend(loc="lower right")
    plot_to_tensorboard(tb_writer, fig, "ROC Curve", global_step=0)

    precision, recall, precision_recall_thresholds = precision_recall_curve(
        targets, [p[0] for p in predictions]
    )
    pr_auc = average_precision_score(targets, [p[0] for p in predictions])
    fig, ax = plt.subplots()
    ax.plot(
        recall,
        precision,
        color="blue",
        lw=2,
        label=f"Precision-Recall curve (area = {pr_auc:.2f})",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    plot_to_tensorboard(tb_writer, fig, "Precision-Recall Curve", global_step=0)

    f1_scores = 2 * recall * precision / (recall + precision)

    # "optimal": maximizes f1_score
    optimal_idx = np.nanargmax(f1_scores)
    optimal_f1_score = f1_scores[optimal_idx]
    optimal_precision = precision[optimal_idx]
    optimal_recall = recall[optimal_idx]

    metrics_table = "| Metric     | Value |\n|------------|-------|\n"
    metrics_table += f"| Precision  | {optimal_precision:.2f} |\n"
    metrics_table += f"| Recall     | {optimal_recall:.2f} |\n"
    metrics_table += f"| F1 Score   | {optimal_f1_score:.2f} |\n"

    tb_writer.add_text("Optimal Metrics", metrics_table, global_step=0)
