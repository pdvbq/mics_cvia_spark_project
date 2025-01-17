from os.path import isdir
from coloredlogs import logging
import os
from torch import sigmoid
from ultralytics import RTDETR, YOLO
import csv
from spark.pipelines.pipeline import Pipeline
from spark.converters.labels import yolo_to_default_format
from spark.utils.bbox_mean import averaged_bbox
import csv
from rich.progress import track

logger = logging.getLogger(__name__)


class YoloRTPipeline(Pipeline):
    def __init__(self, yolo_model_path: str, rt_model_path: str):
        self.yolo_model = YOLO(yolo_model_path)
        self.rt_model = RTDETR(rt_model_path)

    def test(self, **kwargs):
        source = kwargs["test"]["source"]
        yolo_results = self.yolo_model.predict(
            source=source, stream=True, verbose=False
        )
        rt_results = self.rt_model.predict(source=source, stream=True, verbose=False)
        file = kwargs["test"]["output"]
        dirname = os.path.dirname(file)

        if not os.path.isdir(dirname):
            logger.info(f"Save directory does not exist Creating it at {dirname}")
            os.makedirs(dirname)

        total_imgs = len(os.listdir(source))

        with open(file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "class", "bbox"])
            for yolo_result, rt_result in track(
                zip(yolo_results, rt_results),
                description="Predicting...",
                total=total_imgs,
            ):
                filename = os.path.basename(yolo_result.path)
                # INFO: Trick in order to change extension type back to .png
                # remove this in case it's fixed in codalab
                filename = f"{filename.split('.')[0]}.png"

                if (yolo_result.boxes is None or len(yolo_result.boxes) == 0) and (
                    rt_result.boxes is None or len(rt_result.boxes) == 0
                ):
                    logger.debug("Both models did not detect an object")
                    writer.writerow([filename, "", []])
                    continue

                best_yolo_box = [-1, -1]
                best_rt_box = [-1, -1]
                if rt_result.boxes is not None or len(rt_result.boxes) > 0:
                    for idx, box in enumerate(rt_result.boxes):
                        conf = box.conf.tolist()[0]
                        if conf > best_rt_box[0]:
                            best_rt_box[0] = conf
                            best_rt_box[1] = idx

                if yolo_result.boxes is not None:
                    for idx, box in (
                        enumerate(yolo_result.boxes) or len(yolo_result.boxes) > 0
                    ):
                        conf = box.conf.tolist()[0]
                        if conf > best_yolo_box[0]:
                            best_yolo_box[0] = conf
                            best_yolo_box[1] = idx

                best_box = None
                consider_bbox_mean = False
                if best_yolo_box[1] >= 0 and best_rt_box[1] >= 0:
                    if best_yolo_box[0] > best_rt_box[0]:
                        logger.debug("YOLO has best confidence")
                        best_box = yolo_result.boxes[best_yolo_box[1]]
                    elif best_yolo_box[0] < best_rt_box[0]:
                        logger.debug("RTDETR has best confidence")
                        best_box = rt_result.boxes[best_rt_box[1]]
                    else:
                        consider_bbox_mean = True
                        best_box = yolo_result.boxes[best_yolo_box[1]]

                elif best_yolo_box[1] < 0 and best_rt_box[1] >= 0:
                    logger.debug("YOLO model did not detect any object")
                    best_box = rt_result.boxes[best_rt_box[1]]
                elif best_yolo_box[1] >= 0 and best_rt_box[1] < 0:
                    logger.debug("RTDETR model did not detect any object")
                    best_box = yolo_result.boxes[best_yolo_box[1]]
                else:
                    logger.error("Both models did not detect????")
                    logger.info(f"{rt_result.boxes[0]}")
                    logger.info(f"{yolo_result.boxes[0]}")

                if not consider_bbox_mean:
                    xyxy = list(
                        yolo_to_default_format(
                            *yolo_result.orig_shape, *best_box.xywhn.tolist()[0]
                        )
                    )
                else:
                    logger.debug("Taking the average of both model's bbox")
                    xyxy_yolo = list(
                        yolo_to_default_format(
                            *yolo_result.orig_shape,
                            *yolo_result.boxes[best_yolo_box[1]].xywhn.tolist()[0],
                        )
                    )
                    xyxy_rt = list(
                        yolo_to_default_format(
                            *rt_result.orig_shape,
                            *rt_result.boxes[best_rt_box[1]].xywhn.tolist()[0],
                        )
                    )

                    xyxy = averaged_bbox(xyxy_yolo, xyxy_rt)

                cls = yolo_result.names[best_box.cls.tolist()[0]]
                writer.writerow([filename, cls, xyxy])
