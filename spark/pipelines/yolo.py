from coloredlogs import logging
import cv2
from ultralytics import YOLO
from spark.pipelines.pipeline import Pipeline
import csv

logger = logging.getLogger(__name__)


class YoloPipeline(Pipeline):
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def train(self, **kwargs):
        epochs = kwargs["train"]["epochs"]
        data = kwargs["dataset_metadata"]
        save_file = kwargs["train"].get("save_file", "")
        batch = kwargs["train"]["batch"]

        self.model.train(data=data, epochs=epochs, batch=batch)

        if save_file != "":
            self.model.save(save_file)

    def test(self, **kwargs):
        source = kwargs["test"]["source"]
        results = self.model.predict(source=source, stream=True)
        file = kwargs["test"]["output"]

        # TODO: Currently, this is saving as a yolo format.
        # The advised format is image_name, class, bbox

        for result in results:
            # This saves as class_id bbox (using yolo format)
            result.save_txt(file)

    def validate(self, **kwargs):
        data = kwargs["dataset_metadata"]
        self.model.val(data=data)
