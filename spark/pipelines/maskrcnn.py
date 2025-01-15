import os
from datetime import datetime
from coloredlogs import logging
import torch
from torch import nn, optim, split
from torch.utils.data import DataLoader
from torchvision.models.resnet import ResNet50_Weights
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
    maskrcnn_resnet50_fpn_v2,
)
from tqdm import tqdm


from spark.pipelines.pipeline import Pipeline
from spark.settings import DetectionDatasetCfg
from spark.utils.dataset import SparkDataset
from spark.utils.transforms import ImgTransform
from spark.utils.mask import generate_masks
from spark.metrics import Precision, Recall, FScore, IoU

logger = logging.getLogger(__name__)


def fill_loss_dict(loss_dict: torch.DictType, losses: dict):
    for k in loss_dict.keys():
        if k in losses.keys():
            losses[k].append(loss_dict[k].item())


def get_avg_loss(losses: dict):
    return {k: sum(v) / len(v) for k, v in losses.items()}


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

    def _scheduler_select(self, config: dict):
        """
        Select scheduler based on config
        """
        option_dict = {
            "auto": optim.lr_scheduler.StepLR,
            "cosine": optim.lr_scheduler.CosineAnnealingLR,
        }
        assert (
            config["train"]["scheduler"] in option_dict.keys()
        ), f"Invalid scheduler: {config['train']['scheduler']}"
        return option_dict[config["train"]["scheduler"]]

    def _save_model(self, epoch, path):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optim_state_dict": self.optim.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
            },
            path,
        )

    def _load_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optim.load_state_dict(checkpoint["optim_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return checkpoint["epoch"]

    def _make_output_dir(self):
        os.makedirs(f"output/{self.exp_name}/{self.timestamp}", exist_ok=True)

    @property
    def exp_name(self):
        flags = [
            "maskrcnn",
            self.config["input_size"][0],
            self.config["train"]["epochs"],
            self.config["train"]["batch"],
            self.config["train"]["optimizer"],
            self.config["train"]["lr0"],
            self.config["train"]["scheduler"],
        ]
        flags = [str(f) for f in flags]

        return "_".join(flags)

    def __init__(self, config):
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._make_output_dir()
        BATCH_SIZE = config["train"]["batch"]

        self.device = "cuda:0" if torch.cuda.is_available else "cpu"
        self.model = maskrcnn_resnet50_fpn_v2(
            weights_backbone=ResNet50_Weights.IMAGENET1K_V2, num_classes=12
        ).to(self.device)
        self.model.roi_heads.mask_predictor = None
        self.optim = self._optim_select(config)(
            self.model.parameters(), lr=config["train"]["lr0"]
        )
        self.scheduler = self._scheduler_select(config)(
            self.optim, T_max=config["train"]["epochs"]
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
        self.test_ds = SparkDataset(
            DetectionDatasetCfg().class_map,
            root_dir="../datasets/stream1",
            split="test",
            transform=ImgTransform(config["input_size"]),
        )
        self.train_dl = DataLoader(self.train_ds, batch_size=BATCH_SIZE, shuffle=True)
        self.val_dl = DataLoader(self.val_ds, batch_size=BATCH_SIZE, shuffle=False)
        self.test_dl = DataLoader(self.test_ds, batch_size=BATCH_SIZE, shuffle=False)

    def train(self, **kwargs):
        self.model.train()
        for i in tqdm(range(self.num_epochs)):
            # expected loss dict: dict_keys(['loss_classifier', 'loss_box_reg', 'loss_mask', 'loss_objectness', 'loss_rpn_box_reg'])
            loss_tracker = {
                "loss_classifier": [],
                "loss_box_reg": [],
                # "loss_mask": [],
                "loss_objectness": [],
                "loss_rpn_box_reg": [],
            }
            cnt = 0
            for img_batch, label_batch, bbox_batch in tqdm(self.train_dl):
                self.optim.zero_grad()
                images = list(img_batch.to(self.device))
                labels = list(label_batch.to(self.device))
                bboxes = list(bbox_batch.to(self.device))
                targets = [
                    {
                        "labels": lbl.unsqueeze(0),
                        "boxes": bbox.unsqueeze(0),
                        "masks": generate_masks(
                            bbox.unsqueeze(0), self.config["input_size"]
                        ),
                    }
                    for lbl, bbox in zip(labels, bboxes)
                ]
                loss_dict = self.model(images, targets=targets)
                ld = loss_dict.copy()
                fill_loss_dict(ld, loss_tracker)
                losses = sum(loss for loss in loss_dict.values())
                losses.backward()
                self.optim.step()
                cnt += 1
            self.scheduler.step()
            loss_tracker = get_avg_loss(loss_tracker)
            logger.info(
                f"Epoch {i+1}/{self.num_epochs}"
                + " | ".join([f"{k}: {v:.4f}" for k, v in loss_tracker.items()])
            )
            # self.val()
            self._save_model(
                i, f"output/{self.exp_name}/{self.timestamp}/model_{i}.pth"
            )

    def val(self, **kwargs):
        self.model.eval()
        with torch.no_grad():
            loss_tracker = {
                "loss_classifier": [],
                "loss_box_reg": [],
                # "loss_mask": [],
                "loss_objectness": [],
                "loss_rpn_box_reg": [],
            }
            for img_batch, label_batch, bbox_batch in self.val_dl:
                images = list(img_batch.to(self.device))
                labels = list(label_batch.to(self.device))
                bboxes = list(bbox_batch.to(self.device))
                targets = [
                    {
                        "labels": lbl.unsqueeze(0),
                        "boxes": bbox.unsqueeze(0),
                        "masks": generate_masks(
                            bbox.unsqueeze(0), self.config["input_size"]
                        ),
                    }
                    for lbl, bbox in zip(labels, bboxes)
                ]
                loss_dict = self.model(images, targets=targets)
                import pdb

                pdb.set_trace()
                ld = loss_dict.copy()
                fill_loss_dict(ld, loss_tracker)
                losses = sum(loss for loss in loss_dict.values())
                break
            loss_tracker = get_avg_loss(loss_tracker)
            logger.info(
                f"Validation"
                + " | ".join([f"{k}: {v:.4f}" for k, v in loss_tracker.items()])
            )

    def test(self, **kwargs):
        self.model.eval()
        with torch.no_grad():
            for img_batch, _, _ in self.val_dl:
                images = list(img_batch.to(self.device))
                # labels = list(label_batch.to(self.device))
                loss_dict = self.model(images)
                losses = sum(loss for loss in loss_dict.values())


if __name__ == "__main__":
    pipeline = MaskRCNNPipeline()
    pipeline.train()
