from torch.utils.data import DataLoader
from spark.utils import SparkDataset
from spark.settings import Settings
from ultralytics import YOLO
import os
from spark.utils.dataset_converter import convert_to_yolo_format


def main():
    dataset = SparkDataset(
        class_map=Settings.class_map,
        root_dir=os.path.join(Settings.data_root_dir, Settings.stream1_dir),
        split="train",
    )
    tain_loader = DataLoader(dataset)

    # model = YOLO(model="yolo11n.pt")
    # model.train(data="./data/stream1/yolo.yaml")


if __name__ == "__main__":
    main()
    # convert_to_yolo_format("./data/stream1", Settings.class_map)
