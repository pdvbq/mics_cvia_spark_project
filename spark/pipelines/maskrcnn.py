import torch
from torch import nn, optim, split
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2

from spark.pipelines.pipeline import Pipeline
from spark.settings import DetectionDatasetCfg
from spark.utils.dataset import SparkDataset


class MaskRCNNPipeline(Pipeline):
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available else "cpu"
        self.model = maskrcnn_resnet50_fpn_v2(num_classes=11)
        self.optim = optim.AdamW(self.model.parameters(), lr=0.005)
        self.num_epochs = 5
        self.train_ds = SparkDataset(
            DetectionDatasetCfg().class_map, root_dir="data/stream1", split="train"
        )
        self.val_ds = SparkDataset(
            DetectionDatasetCfg().class_map, root_dir="data/stream1", split="validation"
        )
        # self.test_ds = SparkDataset(
        #     DetectionDatasetCfg().class_map, root_dir="data/stream1", split="test"
        # )
        self.train_dl = DataLoader(self.train_ds, batch_size=64, shuffle=True)
        self.val_dl = DataLoader(self.val_ds, batch_size=64, shuffle=True)
        # self.test_dl = DataLoader(self.test_ds, batch_size=64, shuffle=True)
        # self.

    def train(self, **kwargs):
        # TODO:
        # requires train loop logic in model definition
        self.model.train()
        for i in range(self.num_epochs):
            for img_batch, label_batch in self.train_dl:
                self.optim.zero_grad()
                images = list(img_batch.to(self.device))
                labels = list(label_batch.to(self.device))
                loss_dict = self.model(images, labels)
                losses = sum(loss for loss in loss_dict.values())
                losses.backward()
                self.optim.step()

    def val(self, **kwargs):
        self.model.eval()
        with torch.no_grad():
            for img_batch, label_batch in self.val_dl:
                images = list(img_batch.to(self.device))
                labels = list(label_batch.to(self.device))
                loss_dict = self.model(images, labels)
                losses = sum(loss for loss in loss_dict.values())

    def test(self, **kwargs):
        self.model.eval()
        with torch.no_grad():
            for img_batch, label_batch in self.val_dl:
                images = list(img_batch.to(self.device))
                labels = list(label_batch.to(self.device))
                loss_dict = self.model(images, labels)
                losses = sum(loss for loss in loss_dict.values())


if __name__ == "__main__":
    pipeline = MaskRCNNPipeline()
    pipeline.train()
