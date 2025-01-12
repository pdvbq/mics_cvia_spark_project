from dataclasses import dataclass, field
from typing import Dict, Any
import yaml
import os


@dataclass
class DetectionDatasetCfg:
    image_width: int = 1024
    image_height: int = 1024
    class_map: Dict[str, int] = field(
        default_factory=lambda: {
            "proba_2": 0,
            "cheops": 1,
            "debris": 2,
            "double_star": 3,
            "earth_observation_sat_1": 4,
            "lisa_pathfinder": 5,
            "proba_3_csc": 6,
            "proba_3_ocs": 7,
            "smart_1": 8,
            "soho": 9,
            "xmm_newton": 10,
        }
    )


@dataclass
class DownloadCfg:
    data_dir: str = "data"
    streams: Dict[str, str] = field(
        default_factory=lambda: {
            "stream1": "https://uniluxembourg-my.sharepoint.com/:u:/g/personal/0200566850_uni_lu/EbTNBxOOdhxNoZC-mtqnfL8BW705Gold48M6p6kLhaPy9w?download=1",
            "stream2": "",
        }
    )
    stream_dirs: Dict[str, str] = field(
        default_factory=lambda: {"stream1": "stream1", "stream2": "stream2"}
    )


@dataclass
class PipelineCfg:
    yolo: Dict[str, Any] = field(
        default_factory=lambda: {
            "input_size": [1024, 1024],
            "train": {
                "epochs": 100,
                "batch": 0.95,
                "optimizer": "auto",
                "cos_lr": False,
                "lr0": 0.01,
                "lrf": 0.01,
            },
        }
    )
    maskrcnn: Dict[str, Any] = field(
        default_factory=lambda: {
            "input_size": [1024, 1024],
            "train": {
                "epochs": 100,
                "batch": 64,
                "optimizer": "SGD",
                "cos_lr": False,
                "lr0": 0.01,
                "lrf": 0.01,
            },
        }
    )


@dataclass
class Settings:
    download_cfg: DownloadCfg = DownloadCfg()
    dataset_cfg: DetectionDatasetCfg = DetectionDatasetCfg()
    pipeline_cfg: PipelineCfg = PipelineCfg()

    @staticmethod
    def load_from_yaml(file_path: str) -> "Settings":
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                data = yaml.safe_load(file)
                return Settings(
                    download_cfg=DownloadCfg(**data.get("download", {})),
                    dataset_cfg=DetectionDatasetCfg(**data.get("dataset", {})),
                    pipeline_cfg=PipelineCfg(**data.get("pipeline", {})),
                )
        else:
            raise FileNotFoundError(f"Configuration file not found: {file_path}")


settings = Settings().load_from_yaml("./config/settings.yaml")
