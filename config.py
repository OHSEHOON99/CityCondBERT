# config.py
# ---------------------------------------
# Constants
# ---------------------------------------
PAD_VALUE = 0
MASK_TOKEN_VALUE = 40001
IGNORE_INDEX = -100

# ----------------------------
# City id mapping
# ----------------------------
CITY2ID = {"A": 1, "B": 2, "C": 3, "D": 4}
ID2CITY = {v: k for k, v in CITY2ID.items()}

# ---------------------------------------
# Embedding feature config (for EmbeddingLayer)
# ---------------------------------------
def default_feature_configs():
    return {
        "day":      {"modes": ["categorical"],              "combine_mode_inner": "cat"},
        "time":     {"modes": ["categorical", "periodic"],  "combine_mode_inner": "cat"},
        "dow":      {"modes": ["categorical", "periodic"],  "combine_mode_inner": "cat"},
        "weekday":  {"modes": ["categorical"],              "combine_mode_inner": "cat"},
        "location": {"modes": ["categorical"],              "combine_mode_inner": "cat"},
        "delta":    {"modes": ["fourier"],                  "combine_mode_inner": "cat"},
    }

# ---------------------------------------
# Task presets (overrides by task)
# ---------------------------------------
TASK_PRESETS = {
    "pretrain": {
        "mode": "pretrain",
        "city": "ALL",
        "num_epochs": 100,
        "freeze_backbone": False,
        "freeze_head": False,
        "loss_name": "ce",
    },
    "transfer": {
        "mode": "transfer",
        "city": "A",
        "num_epochs": 100,
        "freeze_backbone": True,
        "freeze_head": False,
        "loss_name": "combo",
        "lr": 1e-5,
    },
    "predict": {
        "mode": "predict",
        "num_epochs": 0,
        "batch_size": 8,
        "city": "ALL",
        "mask_only": False
    },
}

# ---------------------------------------
# Global defaults (can be overridden by JSON/CLI)
# ---------------------------------------
def base_defaults(task=None):
    base = {
        # ===== General =====
        "device": 0,
        "seed": 42,
        "city": "ALL",
        "base_path": "/home/sehoon/Desktop/0829_transfer",
        "mode": "train",
        "model_name": None,
        "wandb_api_key": "",
        "wandb_project": "ACM SIGSPATIAL Cup 2025",

        # ===== Data =====
        "split_ratio": (8, 1, 1),
        "batch_size": 8,
        "subsample": False,
        "subsample_number": 1000,
        "mask_days": (60, 74),

        # ===== BERT Backbone =====
        "hidden_size": 768,
        "hidden_layers": 12,
        "attention_heads": 12,
        "dropout": 0.1,
        "max_seq_length": 75 * 48,

        # ===== Embedding sizes =====
        "day_embedding_size": 64,
        "time_embedding_size": 64,
        "day_of_week_embedding_size": 32,
        "weekday_embedding_size": 16,
        "location_embedding_size": 256,
        "delta_embedding_dims": (16, 32, 16),

        # ===== Embedding combine =====
        "feature_combine_mode": "cat",
        "feature_configs": default_feature_configs(),

        # ===== Transfer knobs (CityCondBERT) =====
        "use_film": True,
        "apply_film_at": "post",
        "film_share": False,
        "num_cities": 5,
        "city_emb_dim": 16,

        # Adapter
        "use_adapter": True,
        "adapter_layers": 2,
        "adapter_r": 16,
        "adapter_dropout": 0.0,

        # ===== Optim / Train =====
        "lr": 1e-4,
        "location_embedding_lr": None,
        "lr_transfer": None,
        "lr_head": None,
        "weight_decay": 0.01,
        "num_epochs": 100,
        "use_amp": True,

        "freeze_backbone": False,
        "freeze_head": False,

        # ===== Loss / Metrics =====
        "loss_name": "ce",
        "H": 200, "W": 200,
        "cell_km_x": 0.5, "cell_km_y": 0.5,

        "ddce": {
            "win": 7, "beta": 0.5, "distance_scale": 2.0,
            "ignore_index": None, "reduction": "mean",
        },
        "geobleu": {
            "n_list": [1, 2, 3, 4, 5],
            "win": 7, "beta": 0.5, "distance_scale": 2.0,
            "eps": 0.1, "n_iters": 30, "weights": None,
        },

        "combo": {
            "ce_name": "ce",
            "ce_kwargs": {"ignore_index": None, "reduction": "mean"},
            "geobleu_kwargs": {
                "H": 200, "W": 200, "n_list": [1, 2, 3, 4, 5],
                "win": 9, "beta": 0.5, "cell_km_x": 0.5, "cell_km_y": 0.5,
                "distance_scale": 2.0, "eps": 0.05, "n_iters": 30, "weights": None
            },
            "alpha_init": 1.0,
            "ema_m": 0.99,
            "track_mavg": True,
            "alpha_warmup_epochs": 3,
            "alpha_transition_epochs": 20,
            "alpha_start": 0.9,
            "alpha_end": 0.3,
        },
    }

    if task:
        preset = TASK_PRESETS.get(str(task).lower(), {})
        base = {**base, **preset}
    return base

# ---------------------------------------
# Helpers for assembling final config
# ---------------------------------------
def coerce_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("1", "true", "t", "yes", "y")
    return bool(v)

def deep_merge(dst, src):
    """In-place deep merge."""
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst

def assemble_config(task, file_cfg, cli_cfg, normalize_bools=True):
    """
    Make final config by merging defaults(task) <- file_cfg <- cli_cfg.
    """
    cfg = base_defaults(task=task)
    deep_merge(cfg, file_cfg or {})
    deep_merge(cfg, cli_cfg or {})

    if normalize_bools:
        for k in ["use_film", "film_share", "use_adapter", "freeze_backbone", "freeze_head", "use_amp"]:
            if k in cfg and cfg[k] is not None:
                cfg[k] = coerce_bool(cfg[k])

    if task and cfg.get("mode") is None:
        if task == "pretrain":
            cfg["mode"] = "train"
        elif task == "transfer":
            cfg["mode"] = "transfer"
        elif task == "predict":
            cfg["mode"] = "predict"

    return cfg