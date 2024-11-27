from torch.utils.data import DataLoader
from spark.utils import SparkDataset
from spark.settings import Settings
import os


def main():
    dataset = SparkDataset(
        class_map=Settings.class_map,
        root_dir=os.path.join(Settings.data_root_dir, Settings.stream1_dir),
        split="train",
    )
    tain_loader = DataLoader(dataset)

    for i, sample in enumerate(tain_loader):
        print(
            f"image shape {sample[0].shape} labels shape {sample[1].shape} bounding box shape {sample[2].shape}"
        )
        if i == 3:
            break


if __name__ == "__main__":
    main()
