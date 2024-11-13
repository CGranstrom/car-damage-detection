from prefect.logging import get_logger
from torch.utils.data import DataLoader

from data.dataset import ImageDataset
from transforms.transforms import train_transform, val_transform

LOGGER = get_logger()


def get_dataloader(
    data_dir, transform, mode, batch_size=None, shuffle=False, validation_split=None
):
    dataset_args = {"data_dir": data_dir, "transform": transform, "mode": mode}
    if validation_split is not None:
        dataset_args["validation_split"] = validation_split

    dataset = ImageDataset(**dataset_args)

    batch_size = len(dataset) if batch_size is None else batch_size

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


def run(config):
    train_loader = get_dataloader(
        data_dir=config["train_val_data_dir"],
        transform=train_transform,
        mode="train",
        batch_size=config["batch_size"],
        shuffle=True,
        validation_split=config["validation_split"],
    )

    val_loader = get_dataloader(
        data_dir=config["train_val_data_dir"],
        transform=val_transform,
        mode="validation",
        batch_size=config["batch_size"],
        validation_split=config["validation_split"],
    )

    test_loader = get_dataloader(
        data_dir=config["test_data_dir"],
        transform=val_transform,
        mode="test",
    )

    return train_loader, val_loader, test_loader
