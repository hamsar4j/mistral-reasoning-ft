import os
from datasets import load_dataset
import logging


def load_training_dataset(dataset_path: str):
    if os.path.exists(dataset_path):
        return load_dataset(dataset_path)
    else:
        # from hf hub
        return load_dataset(dataset_path)["train"]


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
