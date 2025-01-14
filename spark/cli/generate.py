import typer
import matplotlib.pyplot as plt
import logging
import os
from spark.settings import settings
from spark.generators.yolo_dataset import generate_yolo_dataset
from spark.utils.yolo_metrics import generate_yolo_metrics
from scipy.ndimage import gaussian_filter1d
import pandas as pd


app = typer.Typer()
logger = logger = logging.getLogger(__name__)


@app.command()
def all():
    generate_yolo_dataset(
        "stream1", os.path.join(settings.download_cfg.data_dir, "stream1"), settings
    )
    generate_yolo_dataset(
        "stream2", os.path.join(settings.download_cfg.data_dir, "stream2"), settings
    )


@app.command()
def stream1():
    generate_yolo_dataset(
        "stream1", os.path.join(settings.download_cfg.data_dir, "stream1"), settings
    )


@app.command()
def stream2():
    generate_yolo_dataset(
        "stream2", os.path.join(settings.download_cfg.data_dir, "stream2"), settings
    )

@app.command()
def train_metrics(results_path: str):
    generate_yolo_metrics(results_path)
