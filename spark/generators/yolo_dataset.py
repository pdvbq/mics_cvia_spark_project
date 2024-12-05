from typing import Optional
from spark.settings import Settings
from spark.converters.labels import default_to_yolo_format
import csv
import os
import yaml
import logging

logger = logger = logging.getLogger(__name__)


def __generate_yolo_cfg(filename: str, dataset_dir: str, settings: Settings) -> bool:
    """
    Generates a YOLO configuration file in YAML format.

    Args:
        filename (str): The name of the YAML file to create (without extension).
        dataset_dir (str): The directory where the dataset is located.
        settings (Settings): A settings object containing dataset configuration, including class mappings.

    Returns:
        bool: True if the configuration file is successfully created, False otherwise.
    """
    config = {
        "path": dataset_dir,
        "train": "train",
        "val": "val",
        "test": "test",
        "names": {},
    }

    for name, idx in settings.dataset_cfg.class_map.items():
        config["names"][idx] = name

    if not os.path.isdir(dataset_dir):
        logger.error(f"Dataset director {dataset_dir} does not exist.")
        return False

    with open(os.path.join(dataset_dir, f"{filename}.yaml"), "w") as f:
        yaml.dump(config, f)

    return True


def __process_split(dataset_dir: str, split: str, settings: Settings) -> Optional[list]:
    """
    Processes a dataset split and extracts labels in YOLO format.

    Args:
        dataset_dir (str): The directory containing the dataset.
        split (str): The split to process (e.g., 'train', 'val').
        settings (Settings): A settings object containing dataset configuration, including image dimensions and class mappings.

    Returns:
        Optional[list]: A list of label information in YOLO format, or None if the split file does not exist.
    """
    labels = []

    file = os.path.join(dataset_dir, f"{split}.csv")

    if not os.path.isfile(file):
        logger.error(f"Split information file {file} does not exist.")
        return None

    with open(file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_name = row["filename"]
            class_name = row["class"]
            bbox = default_to_yolo_format(
                settings.dataset_cfg.image_width,
                settings.dataset_cfg.image_height,
                *eval(row["bbox"]),
            )
            labels.append([img_name, settings.dataset_cfg.class_map[class_name], *bbox])
    return labels


def __write_labels(dataset_dir: str, split: str, labels: list):
    """
    Writes label files for a specific dataset split.

    Args:
        dataset_dir (str): The directory containing the dataset.
        split (str): The split for which to write labels (e.g., 'train', 'val').
        labels (list): A list of label data in YOLO format.
    """

    labels_dir = os.path.join(dataset_dir, "labels")
    split_dir = os.path.join(labels_dir, split)

    if not os.path.isdir(labels_dir):
        logger.info(f"Creating labels dir at {labels_dir}")
        os.mkdir(labels_dir)

    if not os.path.isdir(split_dir):
        logger.info(f"Creating split dir at {split_dir}")
        os.mkdir(split_dir)

    for labels in labels:
        filename = f"{labels[0].split('.')[0]}.txt"
        with open(
            os.path.join(split_dir, filename),
            "w",
        ) as f:
            f.write(f"{labels[1]} {labels[2]} {labels[3]} {labels[4]} {labels[5]}")


def generate_yolo_dataset(
    dataset_name: str, dataset_dir: str, settings: Settings
) -> bool:
    if not __generate_yolo_cfg(dataset_name, dataset_dir, settings):
        logger.error("Failed to generate YOLO configuration. Aborting")
        return False

    train_labels = __process_split(dataset_dir, "train", settings)
    val_labels = __process_split(dataset_dir, "val", settings)

    if train_labels is None:
        logger.error("Failed to process train labels")
        return False
    __write_labels(dataset_dir, "train", train_labels)

    if val_labels is None:
        logger.error("Failed to process validation labels")
        return False
    __write_labels(dataset_dir, "val", val_labels)

    return True
