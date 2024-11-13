from pathlib import Path

import yaml
from prefect import flow, task
from prefect.logging import get_logger

import scripts.evaluate_model as evaluate_model
import scripts.process_data as process_data
import scripts.train_model as train_model
from data import datasets
from utils.utils import set_rng_seeds

BASE_DATA_DIR = Path(datasets.__file__).parent


def _convert_relative_paths_to_absolute_paths(config):
    config["data_preprocessing"]["train_val_data_dir"] = (
        BASE_DATA_DIR / config["data_preprocessing"]["train_val_data_dir"]
    )
    config["global"]["test_data_dir"] = (
        BASE_DATA_DIR / config["global"]["test_data_dir"]
    )


def load_config(config_file="config.yml"):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    _convert_relative_paths_to_absolute_paths(config)

    global_config_params = config.get("global", {})
    for section in config:
        if section != "global":
            config[section] = {**global_config_params, **config.get(section, {})}

    return config


@task(persist_result=False)
def data_processing(config):
    return process_data.run(config["data_preprocessing"])


@task(persist_result=False)
def model_training(config, train_data_loader, val_data_loader):
    return train_model.run(config["training"], train_data_loader, val_data_loader)


@task(persist_result=False)
def model_evaluation(config, test_data_loader, tb_writer, best_model_filepath):
    evaluate_model.run(
        config["evaluation"], test_data_loader, tb_writer, best_model_filepath
    )


@flow
def pipeline(config_file="config.yml"):
    logger = get_logger()
    logger.info("Starting pipeline")

    config = load_config(config_file)
    set_rng_seeds(config["global"]["rng_seed"])

    train_data_loader, val_data_loader, test_data_loader = data_processing(config)

    tb_writer, best_model_filepath = model_training(
        config, train_data_loader, val_data_loader
    )
    model_evaluation(config, test_data_loader, tb_writer, best_model_filepath)


if __name__ == "__main__":
    pipeline()
