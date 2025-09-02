import os
import json
from datetime import datetime

import wandb

from trainer.configs import to_canonical_config, make_loss_kwargs_from_cfg
from trainer.loops import train_model
from trainer.optim import configure_optimizer



# ----------------------------
# Entrypoint from main.py
# ----------------------------
def train(cfg, model, train_loader, val_loader, device, num_classes):
    """
    - num_classes: 40000 (class count for CE)
    """
    # Basic config checks
    has_transformer = ("transformer" in cfg) or all(
        k in cfg for k in ["hidden_size", "hidden_layers", "attention_heads", "dropout", "max_seq_length"]
    )
    if not has_transformer:
        raise ValueError("[cfg] transformer settings required.")

    has_emb_sizes = ("embedding_sizes" in cfg) or all(
        k in cfg for k in ["day_embedding_size", "time_embedding_size",
                           "day_of_week_embedding_size", "weekday_embedding_size",
                           "location_embedding_size"]
    )
    if not has_emb_sizes:
        raise ValueError("[cfg] embedding sizes required.")

    for k in ["city", "lr", "num_epochs", "base_path", "delta_embedding_dims", "feature_configs"]:
        if k not in cfg:
            raise ValueError(f"[cfg] Missing required key: {k}")

    # Canonical config
    num_location_ids = int(num_classes) + 2
    canonical_config = to_canonical_config(cfg, num_location_ids)

    # Loss config
    loss_name = cfg.get("loss_name", "ce").lower()
    loss_kwargs = make_loss_kwargs_from_cfg(cfg)
    canonical_config["loss"] = {"name": loss_name, "kwargs": loss_kwargs}

    # Optimizer
    optimizer = configure_optimizer(
        model,
        base_lr=cfg["lr"],
        location_embedding_lr=cfg.get("location_embedding_lr"),
        lr_transfer=cfg.get("lr_transfer"),
        lr_head=cfg.get("lr_head"),
        weight_decay=cfg.get("weight_decay", 0.01),
    )

    # === Run/Checkpoint naming: add mode/city prefix ===
    mode = str(cfg.get("mode", "train")).lower()
    city_code = str(cfg.get("city", "ALL")).upper()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if mode == "transfer":
        prefix = f"transfer-{city_code}"
    else:
        # treat everything else as pretrain (train/all-cities)
        prefix = "pretrain"

    run_name = f"{prefix}-{model.__class__.__name__}-{ts}"

    # W&B
    if cfg.get("wandb_api_key", ""):
        wandb.login(key=cfg["wandb_api_key"])
    wandb.init(
        project=cfg.get("wandb_project", "ACM SIGSPATIAL Cup 2025"),
        dir=os.path.join(cfg["base_path"], "wandb"),
        name=run_name,
        config={
            **cfg,
            "run_name": run_name,
            "canonical_config": canonical_config,
            "loss_name": loss_name,
            "loss_kwargs": loss_kwargs,
        },
    )

    # Paths
    run_dir = os.path.join(cfg["base_path"], "checkpoints", run_name)
    os.makedirs(os.path.join(run_dir, "results"), exist_ok=True)

    canonical_config_path = os.path.join(run_dir, "config.json")
    with open(canonical_config_path, "w") as f:
        json.dump(canonical_config, f, indent=4)
    print(f"[Config] Saved canonical config → {canonical_config_path}")

    run = wandb.run
    run_meta = {
        "wandb": {
            "id": run.id,
            "project": run.project,
            "entity": getattr(run, "entity", None),
            "name": run.name,
            "url": run.url,
        },
        "created_at": datetime.now().isoformat(),
    }
    with open(os.path.join(run_dir, "run_meta.json"), "w") as f:
        json.dump(run_meta, f, indent=2)

    # Train
    best_model_path, _ = train_model(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=cfg["num_epochs"],
        device=device,
        run_dir=run_dir,
        cfg=cfg,
        loss_name=loss_name,
        loss_kwargs=loss_kwargs,
    )
    return best_model_path, canonical_config_path