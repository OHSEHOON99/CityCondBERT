import os
import json
import copy
import re
import glob
from datetime import datetime

import numpy as np
import wandb
from models.model import CityCondBERT



def id_to_xy(loc_ids, grid_width=200):
    """Convert location IDs to (x, y) coordinates (1-based)."""
    x = (loc_ids // grid_width) + 1
    y = (loc_ids % grid_width) + 1
    return np.stack([x, y], axis=1)


def xy_to_id(x, y, grid_width=200):
    """Convert (x, y) (1-based) to 0-based location IDs."""
    return (x - 1) * grid_width + (y - 1)


def save_config(config_dict, save_dir, filename='config.json'):
    """Save configuration as JSON."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    with open(path, 'w') as f:
        json.dump(config_dict, f, indent=4)
    print(f"[Saved] Config saved at {path}")
    return path


def deep_merge(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_merge(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)
    return dst


def load_json(path):
    if not path:
        return {}
    with open(path, "r") as f:
        return json.load(f)


def args_to_dict(args):
    d = vars(args).copy()
    return {k: v for k, v in d.items() if v is not None}


def _parse_ts(ts_str):
    return datetime.strptime(ts_str, "%Y%m%d_%H%M%S")


def _scan_wandb_by_timestamp_fuzzy(wandb_dir, target_run_name, tolerance_sec=5):
    m = re.search(r"(\d{8}_\d{6})", target_run_name)
    if not m:
        print(f"[W&B] No timestamp found in target_run_name={target_run_name}")
        return None, None
    target_ts = m.group(1)
    target_dt = _parse_ts(target_ts)

    candidates = []
    for pat in (os.path.join(wandb_dir, "run-*"),
                os.path.join(wandb_dir, "wandb", "run-*")):
        for run_root in glob.glob(pat):
            base = os.path.basename(run_root)
            m2 = re.match(r"run-(\d{8}_\d{6})-([a-z0-9]+)", base)
            if not m2:
                continue
            ts_str, run_id = m2.groups()
            try:
                dt = _parse_ts(ts_str)
            except ValueError:
                continue
            diff = abs((dt - target_dt).total_seconds())
            candidates.append((diff, dt, run_id, run_root))

    if not candidates:
        return None, None

    candidates.sort(key=lambda x: (x[0], x[1]))
    best_diff, best_dt, best_id, best_root = candidates[0]

    if best_diff <= tolerance_sec:
        return best_id, best_root
    else:
        print(f"[W&B] Closest run is {best_dt.strftime('%Y%m%d_%H%M%S')} (Δ={int(best_diff)}s) > tolerance({tolerance_sec}s).")
        return None, None


def resume_wandb_by_timestamp(wandb_dir, target_run_name,
                               project, entity=None,
                               tolerance_sec=5):
    if wandb.run is not None:
        return
    run_id, run_root = _scan_wandb_by_timestamp_fuzzy(wandb_dir, target_run_name, tolerance_sec=tolerance_sec)
    if run_id is None:
        print(f"[W&B] No run matched timestamp (±{tolerance_sec}s) for name='{target_run_name}' in {wandb_dir}. Skip resuming.")
        return

    print(f"[W&B] Resuming run id={run_id} (closest ts to {target_run_name}) from {run_root}")
    wandb.init(
        project=project,
        entity=entity,
        id=run_id,
        resume="allow",
        dir=wandb_dir,
        reinit=True,
        name=target_run_name
    )


def build_model_from_config(cfg, device, num_location_ids):
    if "transformer" in cfg and isinstance(cfg["transformer"], dict):
        transformer_cfg = cfg["transformer"]
    else:
        transformer_cfg = {
            "hidden_size":     int(cfg["hidden_size"]),
            "hidden_layers":   int(cfg["hidden_layers"]),
            "attention_heads": int(cfg["attention_heads"]),
            "dropout":         float(cfg["dropout"]),
            "max_seq_length":  int(cfg["max_seq_length"]),
        }

    if "embedding_sizes" in cfg and isinstance(cfg["embedding_sizes"], dict):
        embedding_sizes = cfg["embedding_sizes"]
    else:
        embedding_sizes = {
            "day":      int(cfg["day_embedding_size"]),
            "time":     int(cfg["time_embedding_size"]),
            "dow":      int(cfg["day_of_week_embedding_size"]),
            "weekday":  int(cfg["weekday_embedding_size"]),
            "location": int(cfg["location_embedding_size"]),
        }

    delta_embedding_dims = tuple(cfg.get("delta_embedding_dims", (8, 8, 8)))
    feature_configs = cfg["feature_configs"]
    feature_combine_mode = cfg.get("feature_combine_mode",
                                   cfg.get("embedding_combine_mode", "cat"))

    use_film       = cfg.get("use_film", True)
    apply_film_at  = cfg.get("apply_film_at", "post")
    film_share     = cfg.get("film_share", True)
    num_cities     = int(cfg.get("num_cities", 5))
    city_emb_dim   = int(cfg.get("city_emb_dim", 32))

    use_adapter    = cfg.get("use_adapter", True)
    adapter_layers = int(cfg.get("adapter_layers", 2))
    adapter_r      = int(cfg.get("adapter_r", 16))
    adapter_dropout= float(cfg.get("adapter_dropout", 0.0))

    model = CityCondBERT(
        num_location_ids=num_location_ids,
        transformer_cfg=transformer_cfg,
        embedding_sizes=embedding_sizes,
        delta_embedding_dims=delta_embedding_dims,
        feature_configs=feature_configs,
        embedding_combine_mode=feature_combine_mode,
        num_cities=num_cities,
        city_emb_dim=city_emb_dim,
        use_film=use_film,
        apply_film_at=apply_film_at,
        film_share=film_share,
        use_adapter=use_adapter,
        adapter_layers=adapter_layers,
        adapter_r=adapter_r,
        adapter_dropout=adapter_dropout,
    ).to(device)

    if str(cfg.get("freeze_backbone", "false")).lower() == "true":
        model.freeze_backbone()
    if str(cfg.get("freeze_head", "false")).lower() == "true":
        for p in model.output_projection.parameters():
            p.requires_grad = False

    return model


def safe_ids_to_xy(ids_np, W):
    """Call utils.id_to_xy with or without grid_width depending on signature."""
    try:
        return id_to_xy(ids_np, grid_width=W)
    except TypeError:
        return id_to_xy(ids_np)