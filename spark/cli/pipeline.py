import typer
from spark.pipelines.yolo import YoloPipeline
from spark.pipelines.maskrcnn import MaskRCNNPipeline
from spark.pipelines.rtdetr import RTDETRPipeline
from spark.pipelines.yolort import YoloRTPipeline
from spark.settings import settings

app = typer.Typer()


@app.command()
def yolo(instruction: str):
    config = settings.pipeline_cfg.yolo
    pipeline = YoloPipeline(config["model_input"])

    if instruction == "train":
        pipeline.train(**config)
    elif instruction == "val":
        pipeline.validate(**config)
    elif instruction == "test":
        pipeline.test(**config)

@app.command()
def maskrcnn(instruction: str):
    config = settings.pipeline_cfg.maskrcnn
    pipeline = MaskRCNNPipeline(config)
    if instruction == "train":
        pipeline.train(**config)
    elif instruction == "val":
        pipeline.validate(**config)
    elif instruction == "test":
        pipeline.test(**config)

@app.command()
def rtdetr(instruction: str):
    config = settings.pipeline_cfg.rtdetr
    pipeline = RTDETRPipeline(config["model_input"])

    if instruction == "train":
        pipeline.train(**config)
    elif instruction == "val":
        pipeline.validate(**config)
    elif instruction == "test":
        pipeline.test(**config)


@app.command()
def yolort(instruction: str):
    config = settings.pipeline_cfg.yolort
    pipeline = YoloRTPipeline(config["yolo_model_input"], config["rt_model_input"])

    if instruction == "test":
        pipeline.test(**config)
