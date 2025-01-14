import torch
from torch import nn, optim, split
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2

from spark.pipelines.pipeline import Pipeline
from spark.settings import DetectionDatasetCfg
from spark.utils.dataset import SparkDataset
from spark.utils.transforms import ImgTransform
from spark.utils.mask import generate_masks


class MaskRCNNPipeline(Pipeline):
    def _optim_select(self, config: dict):
        """
        Select optimiser based on config
        """
        option_dict = {"SGD": optim.SGD, "Adam": optim.Adam, "AdamW": optim.AdamW}
        assert (
            config["train"]["optimizer"] in option_dict.keys()
        ), f"Invalid optimiser: {config['train']['optimizer']}"
        return option_dict[config["train"]["optimizer"]]

    def __init__(self, config):
        self.config = config
        BATCH_SIZE = config["train"]["batch"]
        
        self.device = "cuda:0" if torch.cuda.is_available else "cpu"
        self.model = maskrcnn_resnet50_fpn_v2(num_classes=11).to(self.device)
        self.optim = self._optim_select(config)(
            self.model.parameters(), lr=config["train"]["lr0"]
        )
        self.num_epochs = config["train"]["epochs"]
        self.train_ds = SparkDataset(
            DetectionDatasetCfg().class_map,
            root_dir="../datasets/stream1",
            split="train",
            transform=ImgTransform(config["input_size"]),
        )
        self.val_ds = SparkDataset(
            DetectionDatasetCfg().class_map,
            root_dir="../datasets/stream1",
            split="validation",
            transform=ImgTransform(config["input_size"]),
        )
        # self.test_ds = SparkDataset(
        #     DetectionDatasetCfg().class_map, root_dir="data/stream1", split="test"
        # )
        self.train_dl = DataLoader(self.train_ds, batch_size=BATCH_SIZE, shuffle=True)
        self.val_dl = DataLoader(self.val_ds, batch_size=BATCH_SIZE, shuffle=True)
        # self.test_dl = DataLoader(self.test_ds, batch_size=64, shuffle=True)

    def train(self, **kwargs):
        self.model.train()
        for i in range(self.num_epochs):
            for img_batch, label_batch, bbox_batch in self.train_dl:
                self.optim.zero_grad()
                images = list(img_batch.to(self.device))
                labels = list(label_batch.to(self.device))
                bboxes = list(bbox_batch.to(self.device))
                targets = [
                    {"labels": lbl, "boxes": bbox.unsqueeze(0), "masks": generate_masks(bbox.unsqueeze(0), self.config["input_size"])}
                    for lbl, bbox in zip(labels, bboxes)
                ]
                loss_dict = self.model(images, targets=targets)
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
