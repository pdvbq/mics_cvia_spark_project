from dataclasses import dataclass


@dataclass
class DatasetCfg:
    image_width = 1024
    image_height = 1024
    class_map = {
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


@dataclass
class FetchDatasetCfg:
    data_dir = "data"
    stream1_url = "https://uniluxembourg-my.sharepoint.com/:u:/g/personal/pedro_bastos_uni_lu/EfXBeG7PFypEs5sM3NzMB0MByiEaKFaoD8WKfzQN_6UPog?download=1"
    stream2_url = ""
    stream1_dir = "stream1"
    stream2_dir = "stream2"


@dataclass
class Settings:
    fetch_cfg = FetchDatasetCfg()
    dataset_cfg = DatasetCfg()
