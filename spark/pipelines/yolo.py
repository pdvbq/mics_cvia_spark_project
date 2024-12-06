from coloredlogs import logging
from ultralytics import YOLO
from spark.pipelines.pipeline import Pipeline

logger = logging.getLogger(__name__)


class YoloPipeline(Pipeline):
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def train(self, **kwargs):
        epochs = kwargs["train"]["epochs"]
        data = kwargs["dataset_metadata"]
        save_file = kwargs["train"]["save_file"]

        self.model.train(data=data, epochs=epochs)

        if save_file != "":
            self.model.save(save_file)

    def test(self, **kwargs):
        logger.warning("Test is not yet implemented")

    def validate(self, **kwargs):
        data = kwargs["dataset_metadata"]
        self.model.val(data=data)
