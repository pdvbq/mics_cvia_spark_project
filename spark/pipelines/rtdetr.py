from os.path import isdir
from coloredlogs import logging
import os
from ultralytics import RTDETR
import csv
from spark.pipelines.pipeline import Pipeline
from spark.converters.labels import yolo_to_default_format
import csv
from rich.progress import track

logger = logging.getLogger(__name__)


class RTDETRPipeline(Pipeline):
    def __init__(self, model_path: str):
        self.model = RTDETR(model_path)

    def train(self, **kwargs):
        epochs = kwargs["train"]["epochs"]
        data = kwargs["dataset_metadata"]
        save_file = kwargs["train"].get("save_file", "")
        batch = kwargs["train"]["batch"]
        optimizer = kwargs["train"]["optimizer"]
        cos_lr = kwargs["train"]["cos_lr"]
        lr0 = kwargs["train"]["lr0"]
        lrf = kwargs["train"]["lrf"]

        self.model.train(
            data=data,
            epochs=epochs,
            batch=batch,
            optimizer=optimizer,
            cos_lr=cos_lr,
            lr0=lr0,
            lrf=lrf,
        )

        if save_file != "":
            self.model.save(save_file)

    def test(self, **kwargs):
        source = kwargs["test"]["source"]
        results = self.model.predict(source=source, stream=True, verbose=False)
        file = kwargs["test"]["output"]
        dirname = os.path.dirname(file)

        if not os.path.isdir(dirname):
            logger.info(f"Save directory does not exist Creating it at {dirname}")
            os.makedirs(dirname)

        total_imgs = len(os.listdir(source))

        with open(file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "class", "bbox"])
            for result in track(results, description="Predicting...", total=total_imgs):
                filename = os.path.basename(result.path)
                # INFO: Trick in order to change extension type back to .png
                # remove this in case it's fixed in codalab
                filename = f"{filename.split('.')[0]}.png"
                if result.boxes is None or len(result.boxes) == 0:
                    writer.writerow([filename, "", []])
                    continue

                best_box = [0, 0]
                for idx, box in enumerate(result.boxes):
                    conf = box.conf.tolist()[0]
                    if conf > best_box[0]:
                        best_box[0] = conf
                        best_box[1] = idx

                box = result.boxes[best_box[1]]
                cls = result.names[box.cls.tolist()[0]]
                xyxy = list(
                    yolo_to_default_format(*result.orig_shape, *box.xywhn.tolist()[0])
                )
                writer.writerow([filename, cls, xyxy])

    def validate(self, **kwargs):
        data = kwargs["dataset_metadata"]
        self.model.val(data=data)
