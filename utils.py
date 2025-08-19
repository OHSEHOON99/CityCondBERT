import json
import os

import numpy as np



def id_to_xy(loc_ids, grid_width=200):
    """Convert location IDs to (x, y) coordinates."""
    x = (loc_ids // grid_width) + 1
    y = (loc_ids % grid_width) + 1
    return np.stack([x, y], axis=1)  # shape: (L, 2)


def xy_to_id(x, y, grid_width=200):
    """Convert (x, y) coordinates to location IDs."""
    return (x - 1) * grid_width + (y - 1)


def save_config(config_dict, save_dir, filename='config.json'):
    """
    하이퍼파라미터 설정을 JSON 파일로 저장합니다.
    """
    path = os.path.join(save_dir, filename)
    with open(path, 'w') as f:
        json.dump(config_dict, f, indent=4)
    print(f"[Saved] Config saved at {path}")