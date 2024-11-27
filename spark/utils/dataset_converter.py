from spark.utils.bbox import convert_bbox_to_yolo
import os
import csv
import yaml
from spark.settings import Settings

IMG_W = 1024
IMG_H = 1024


def convert_to_yolo_format(root_dir: str, class_map: dict[str, int]):
    yolo_cfg = {
        "path": root_dir,
        "train": "train",
        "val": "val",
        "test": "test",
        "names": {},
    }

    # Add class names
    for name, idx in class_map.items():
        yolo_cfg["names"][idx] = name

    def process_split(split: str):
        data = []
        # Retrieve info from .csv
        with open(os.path.join(root_dir, f"{split}.csv"), "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = row["filename"]
                class_name = row["class"]
                bbox = convert_bbox_to_yolo(IMG_W, IMG_H, *eval(row["bbox"]))
                data.append([img_name, class_map[class_name], *bbox])
        return data

    def write_labels(split: str, data: list):
        for data in data:
            if not os.path.isdir(os.path.join(root_dir, "labels")):
                os.mkdir(os.path.join(root_dir, "labels"))

            if not os.path.isdir(os.path.join(root_dir, "labels", split)):
                os.mkdir(os.path.join(root_dir, "labels", split))

            with open(
                os.path.join(root_dir, "labels", split, f"{data[0].split('.')[0]}.txt"),
                "w+",
            ) as f:
                f.write(f"{data[1]} {data[2]} {data[3]} {data[4]} {data[5]}")

    write_labels("train", process_split("train"))
    write_labels("val", process_split("val"))

    with open(os.path.join(root_dir, "yolo.yaml"), "w") as f:
        yaml.dump(yolo_cfg, f)
