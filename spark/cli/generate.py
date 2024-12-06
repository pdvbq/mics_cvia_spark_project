import typer
import os
from spark.settings import settings
from spark.generators.yolo_dataset import generate_yolo_dataset


app = typer.Typer()


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
