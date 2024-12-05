import typer
from spark.pipelines.yolo import YoloPipeline

app = typer.Typer()


@app.command()
def yolo(instruction: str):
    # TODO: We should create a pipeline settings for yolo and use them here
    # For example, the number of epochs.
    pipeline = YoloPipeline("yolo11n.pt")
    if instruction == "train":
        pipeline.train(data="./data/stream1/stream1.yaml")
    elif instruction == "val":
        pipeline.validate()
    elif instruction == "test":
        pipeline.test(data="./data/stream1/stream1.yaml")
