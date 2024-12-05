import typer
import os
from spark import settings
from spark.generators.yolo_dataset import generate_yolo_dataset
from spark.settings import Settings


app = typer.Typer()


@app.command()
def all():
    generate_yolo_dataset(
        "stream1", os.path.join(Settings.fetch_cfg.data_dir, "stream1"), Settings()
    )
    generate_yolo_dataset(
        "stream2", os.path.join(Settings.fetch_cfg.data_dir, "stream2"), Settings()
    )


@app.command()
def stream1():
    generate_yolo_dataset(
        "stream1", os.path.join(Settings.fetch_cfg.data_dir, "stream1"), Settings()
    )


@app.command()
def stream2():
    generate_yolo_dataset(
        "stream2", os.path.join(Settings.fetch_cfg.data_dir, "stream2"), Settings()
    )
