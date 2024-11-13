import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from prefect.logging import get_logger
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import models
from zoneinfo import ZoneInfo

import training_logs
from utils.utils import (
    DEVICE,
    get_nested_attr,
    model_mode_context,
)

TRAINING_LOGS_DIR = Path(training_logs.__file__).parent
LOGGER = get_logger()


def get_distribution_in_batch(targets_in_batch):
    num_zeros, num_ones = 0, 0
    for target in targets_in_batch:
        if target == 0:
            num_zeros += 1
        else:
            num_ones += 1
    LOGGER.debug(f"num_damaged_images_in_batch={num_ones}")
    LOGGER.debug(f"num_whole_images_in_batch={num_zeros}")


def eval_batches_in_epoch(
    model, optimizer, criterion, data_loader, is_training_set: bool = True
):
    epoch_losses, epoch_accuracies = [], []

    with model_mode_context(model, is_training_set):
        for batch_idx, (images, targets) in enumerate(data_loader):
            images, targets = images.to(DEVICE), targets.float().to(DEVICE)

            LOGGER.debug(f"batch #{batch_idx}")
            if LOGGER.isEnabledFor(logging.DEBUG):
                get_distribution_in_batch(targets)

            if is_training_set:
                optimizer.zero_grad()
            outputs = model(images)
            curr_loss = criterion(outputs, targets)
            epoch_losses.append(curr_loss)
            curr_accuracy = (outputs.round() == targets).float().mean()
            epoch_accuracies.append(curr_accuracy.item())

            if is_training_set:
                curr_loss.backward()
                optimizer.step()
    loss_for_epoch = torch.tensor(epoch_losses).mean().item()
    accuracy_for_epoch = np.mean(epoch_accuracies)

    return loss_for_epoch, accuracy_for_epoch


def run(config, train_data_loader, val_data_loader):
    TRAINING_START_TIME = datetime.now(tz=ZoneInfo("UTC"))

    torch.cuda.empty_cache()
    tb_run_dir = Path(TRAINING_LOGS_DIR) / config["run_name"]
    model_checkpoints_dir = tb_run_dir / "model_checkpoints"
    model_checkpoints_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(str(tb_run_dir))

    model_arch = getattr(models, config["model_architecture"])
    model_weights = get_nested_attr(models, config["pretrained_weights"])

    model = model_arch(weights=model_weights)
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False
    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    model.fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 1),
        nn.Sigmoid(),
    )
    model.to(DEVICE)

    criterion = getattr(nn, config["loss_criterion"])()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        betas=config["betas"],
        eps=config["eps"],
        weight_decay=config["weight_decay"],
    )
    scheduler = StepLR(
        optimizer,
        step_size=config["scheduler_step_size"],
        gamma=config["scheduler_gamma"],
    )

    best_val_loss = float("inf")

    for epoch in range(config["num_epochs"]):
        LOGGER.info(f"epoch #{epoch}")

        train_loss_for_epoch, train_accuracy_for_epoch = eval_batches_in_epoch(
            model, optimizer, criterion, train_data_loader
        )
        val_loss_for_epoch, val_accuracy_for_epoch = eval_batches_in_epoch(
            model, optimizer, criterion, val_data_loader, is_training_set=False
        )

        LOGGER.info(f"train loss for epoch {epoch}: {train_loss_for_epoch:.2f}")
        LOGGER.info(f"train accuracy for epoch {epoch}: {train_accuracy_for_epoch:.2f}")
        LOGGER.info(f"validation loss for epoch {epoch}: {val_loss_for_epoch:.2f}")
        LOGGER.info(
            f"validation accuracy for epoch {epoch}: {val_accuracy_for_epoch:.2f}"
        )

        tb_writer.add_scalars(
            "Loss",
            {"Training": train_loss_for_epoch, "Validation": val_loss_for_epoch},
            epoch,
        )

        tb_writer.add_scalars(
            "Accuracy",
            {
                "Training": train_accuracy_for_epoch,
                "Validation": val_accuracy_for_epoch,
            },
            epoch,
        )

        scheduler.step()

        if epoch % 5 == 0:
            torch.save(
                model.state_dict(),
                str(
                    model_checkpoints_dir
                    / f"model_start_time_{TRAINING_START_TIME}_epoch_{epoch}.pth"
                ),
            )

        if val_loss_for_epoch < best_val_loss:
            best_model_filepath = str(
                model_checkpoints_dir
                / f"model_start_time_{TRAINING_START_TIME}_best_model.pth"
            )

            torch.save(
                model.state_dict(),
                best_model_filepath,
            )

    torch.save(
        model.state_dict(),
        str(
            model_checkpoints_dir
            / f"model_start_time_{TRAINING_START_TIME}_final_model.pth"
        ),
    )

    return tb_writer, best_model_filepath
