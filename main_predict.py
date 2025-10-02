# Standard library
import os
import copy

# Third-party
import torch

# Local modules
from config import base_defaults, ID2CITY
from data_loader import load_and_split_data
from parser import get_predict_parser, coerce_bool
from predict import predict, predict_masked_uid
from utils import (
    load_json,
    deep_merge,
    args_to_dict,
    build_model_from_config,
    resume_wandb_by_timestamp,
)

# PyTorch/SDPA
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True



def main():
    args = get_predict_parser().parse_args()

    # Merge config (task preset: "predict")
    defaults = base_defaults("predict")
    file_cfg = load_json(args.config) if getattr(args, "config", None) else {}
    cli_cfg = args_to_dict(args)
    cfg = deep_merge(deep_merge(copy.deepcopy(defaults), file_cfg), cli_cfg)

    for k in ["use_film", "film_share", "use_adapter", "freeze_backbone", "freeze_head", "use_amp", "mask_only"]:
        if k in cfg and cfg[k] is not None:
            cfg[k] = coerce_bool(cfg[k])

    # Device
    if cfg["device"] != -1 and not torch.cuda.is_available():
        print("[WARN] CUDA not available. Falling back to CPU.")
        cfg["device"] = -1
    device = torch.device("cpu") if cfg["device"] == -1 else torch.device(f'cuda:{cfg["device"]}')

    # Paths
    base_path = cfg["base_path"]
    data_path = os.path.join(base_path, "data")
    ckpt_dir = os.path.join(base_path, "checkpoints", cfg["model_name"])
    model_path = os.path.join(ckpt_dir, "bert_best.pth")
    can_cfg_path = os.path.join(ckpt_dir, "config.json")

    assert os.path.isfile(model_path), f"Model not found: {model_path}"
    assert os.path.isfile(can_cfg_path), f"Canonical config not found: {can_cfg_path}"

    # Model
    H, W = 200, 200
    num_classes = H * W
    num_location_ids = num_classes + 2

    can_cfg = load_json(can_cfg_path)
    can_cfg["city"] = cfg["city"]

    model = build_model_from_config(can_cfg, device, num_location_ids)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"[Mask] Loaded model from {model_path}")

    # DataLoader: transfer mode for a single city (get test + mask loaders)
    _, _, test_loader, mask_loader = load_and_split_data(
        mobility_data_path=data_path,
        batch_size=cfg.get("batch_size", defaults["batch_size"]),
        split_ratio=cfg["split_ratio"],
        use_subsample=cfg["subsample"],
        subsample_number=cfg["subsample_number"],
        random_seed=cfg["seed"],
        mask_days=cfg["mask_days"],
        mode="predict",
        city=cfg["city"],
        W=W,
    )

    # Output dir
    output_dir = args.output_dir or os.path.join(ckpt_dir, "results")
    os.makedirs(output_dir, exist_ok=True)

    # 1) If mask_only is False, also evaluate on the test set and resume W&B
    if not bool(cfg.get("mask_only", False)):
        if test_loader is None:
            print("[Predict] No test_loader available. Skipping test evaluation.")
            return

        # Resume W&B run by timestamp inferred from run directory name
        wandb_dir = os.path.join(base_path, "wandb")
        target_run_name = cfg["model_name"]
        try:
            resume_wandb_by_timestamp(
                wandb_dir=wandb_dir,
                target_run_name=target_run_name,
                project=cfg.get("wandb_project", "ACM SIGSPATIAL Cup 2025"),
                entity=None,
                tolerance_sec=5,
            )
            print(f"[W&B] Resumed run for name≈{target_run_name}")
        except Exception as e:
            print(f"[W&B] Resume failed: {e}")

        geobleu, dtw, acc = predict(
            model=model,
            test_loader=test_loader,
            device=device,
            output_dir=output_dir,
            W=W,
            save_csv=True,
        )

        print(f"[Predict/Test] GEO-BLEU: {geobleu:.4f}, DTW: {dtw:.2f}, Accuracy: {acc*100:.2f}%")

    # 2) Predict masked segments (x==999) -> results/mask/{team}_cityX_humob25.csv
    if mask_loader is None:
        print("[Mask] No mask_loader produced (no x==999). Skipping masked prediction.")
    else:
        predict_masked_uid(
            model=model,
            mask_loader=mask_loader,
            device=device,
            city=cfg["city"],
            output_dir=output_dir,
            W=W,
            team_name="SCSI"
        )
        print(f"[Mask] Saved masked predictions")

if __name__ == "__main__":
    main()