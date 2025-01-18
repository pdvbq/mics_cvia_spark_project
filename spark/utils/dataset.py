"""Utility class to load Spark dataset taken from template provided by challenge"""

import torch
from ast import literal_eval
import os
from skimage import io
from torch.utils.data import Dataset
import pandas as pd


def process_labels(labels_dir, split):
    path = os.path.join(labels_dir, f"{split}.csv")
    labels = pd.read_csv(path)
    return labels


class SparkDataset(Dataset):
    """SPARK dataset that can be used with DataLoader for PyTorch training."""

    def __init__(
        self, class_map, split="train", root_dir="", transform=None, detection=True
    ):
        if split not in {"train", "validation", "test"}:
            raise ValueError(
                "Invalid split, has to be either 'train', 'validation' or 'test'"
            )

        self.class_map = class_map

        self.detection = detection
        self.split = split if split != "validation" else "val"
        # self.root_dir = os.path.join(root_dir, self.split)
        self.root_dir = os.path.join(root_dir, "images", self.split)

        if split == "test":
            self.labels = pd.DataFrame(
                {"filename": os.listdir(self.root_dir), "class": "", "bbox": ""}
            )
        else:
            self.labels = process_labels(root_dir, split)

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        sat_name = self.labels.iloc[idx]["class"]
        img_name = self.labels.iloc[idx]["filename"]
        image_name = f"{self.root_dir}/{img_name}"

        image = io.imread(image_name)

        if self.transform is not None:
            torch_image = self.transform(image)

        else:
            torch_image = torch.from_numpy(image).permute(2, 1, 0)

        if self.detection:

            bbox = self.labels.iloc[idx]["bbox"]
            bbox = literal_eval(bbox)

            return torch_image, self.class_map[sat_name], torch.tensor(bbox)

        return torch_image, torch.tensor(self.class_map[sat_name])
