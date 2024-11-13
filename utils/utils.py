import io
import logging
import random
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_nested_attr(obj, attr):
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def setup_logger(log_level="info"):
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s|%(levelname)s|%(pathname)s:%(lineno)s|%(funcName)s> %(message)s",
    )
    logger = logging.getLogger()
    return logger


def set_rng_seeds(seed=1337):
    """Sets seeds for random number generators used by torch (cpu and gpu), numpy, and random libraries"""
    logging.debug(f"RNG {seed=}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_to_tensorboard(writer, figure, tag, global_step):
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    pil_image = Image.open(buf)
    pil_to_tensor = transforms.ToTensor()
    tensor_image = pil_to_tensor(pil_image)
    writer.add_image(tag, tensor_image, global_step=global_step)
    plt.close(figure)


@contextmanager
def model_mode_context(model, is_training_set: bool):
    """
    Sets model.eval() and returns `torch.no_grad()` context manager if `is_training_set` is False.
    Otherwise, sets model.train() and returns a dummy context manager.

    """
    if is_training_set:
        model.train()
        yield

    else:
        model.eval()
        with torch.no_grad():
            yield
