from dataclasses import dataclass


@dataclass
class Settings:
    stream1_url = "TO FILL IN"
    data_root_dir = "data"
    stream1_dir = "stream1"
    stream2_dir = "stream2"
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
