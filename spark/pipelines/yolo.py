from coloredlogs import logging
import os
from ultralytics import YOLO
import csv
from spark.pipelines.pipeline import Pipeline
from spark.converters.labels import yolo_to_default_format
import csv
from rich.progress import track

logger = logging.getLogger(__name__)


class YoloPipeline(Pipeline):
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def train(self, **kwargs):
        epochs = kwargs["train"]["epochs"]
        data = kwargs["dataset_metadata"]
        save_file = kwargs["train"].get("save_file", "")
        batch = kwargs["train"]["batch"]
        optimizer = kwargs["train"]["optimizer"]
        cos_lr = kwargs["train"]["cos_lr"]
        self.model.train(
            data=data, epochs=epochs, batch=batch, cos_lr=cos_lr, optimizer=optimizer
        )

        if save_file != "":
            self.model.save(save_file)

    def test(self, **kwargs):
        source = kwargs["test"]["source"]
        results = self.model.predict(source=source, stream=True, verbose=False)
        file = kwargs["test"]["output"]

        total_imgs = len(os.listdir(source))

        with open(file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "class", "bbox"])
            for result in track(results, description="Predicting...", total=total_imgs):
                filename = os.path.basename(result.path)
                if result.boxes is None:
                    writer.writerow([filename, "", ""])
                    continue
                for box in result.boxes:
                    cls = result.names[box.cls.tolist()[0]]
                    xyxy = yolo_to_default_format(
                        *result.orig_shape, *result.boxes.xywhn.tolist()[0]
                    )
                    writer.writerow([filename, cls, xyxy])

    def validate(self, **kwargs):
        data = kwargs["dataset_metadata"]
        self.model.val(data=data)
