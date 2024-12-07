import typer
from spark.pipelines.yolo import YoloPipeline
from spark.settings import settings

app = typer.Typer()


@app.command()
def yolo(instruction: str):
    config = settings.pipeline_cfg.frameworks["yolo"]
    pipeline = YoloPipeline(config["model_input"])

    if instruction == "train":
        pipeline.train(**config)
    elif instruction == "val":
        pipeline.validate(**config)
    elif instruction == "test":
        pipeline.test(**config)
