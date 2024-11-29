from typing import Set
from torch.utils.data import DataLoader
from spark.utils import SparkDataset
from spark.settings import Settings
from ultralytics import YOLO
import os
from spark.utils.dataset_converter import convert_to_yolo_format
import spark.tools as tools
import logging
import typer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(fetch: bool = False):
    if fetch:
        tools.download_dataset(Settings.stream1_url, "stream1", Settings.data_root_dir)


if __name__ == "__main__":
    typer.run(main)
