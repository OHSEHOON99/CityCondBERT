# Standard library
import os
import copy

# Third-party
import torch

# Local
from config import base_defaults, ID2CITY
from data_loader import load_and_split_data
from parser import get_pretrain_parser, coerce_bool
from predict import predict
from train import train
from utils import load_json, deep_merge, args_to_dict, build_model_from_config

# PyTorch/SDPA
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True



def main():
    args = get_pretrain_parser().parse_args()

    # 1) Merge config: defaults <- file <- cli
    defaults = base_defaults("pretrain")
    file_cfg = load_json(args.config) if getattr(args, "config", None) else {}
    cli_cfg = args_to_dict(args)
    final_cfg = deep_merge(deep_merge(copy.deepcopy(defaults), file_cfg), cli_cfg)

    # Normalize booleans if given as strings
    for k in ["use_film", "film_share", "use_adapter", "freeze_backbone", "freeze_head", "use_amp"]:
        if k in final_cfg and final_cfg[k] is not None:
            final_cfg[k] = coerce_bool(final_cfg[k])

    # 2) Device (CUDA guard)
    if final_cfg["device"] != -1 and not torch.cuda.is_available():
        print("[WARN] CUDA not available. Falling back to CPU.")
        final_cfg["device"] = -1
    device = torch.device("cpu") if final_cfg["device"] == -1 else torch.device(f'cuda:{final_cfg["device"]}')

    # 3) Paths / constants
    data_path = os.path.join(final_cfg["base_path"], "data")
    H, W = 200, 200
    num_classes = H * W
    num_location_ids = num_classes + 2  # PAD + [1..num_classes] + MASK

    # 4) DataLoader (load all cities; mask loader not used in pretrain)
    train_loader, val_loader, test_loader, _ = load_and_split_data(
        mobility_data_path=data_path,
        batch_size=final_cfg["batch_size"],
        split_ratio=final_cfg["split_ratio"],
        use_subsample=final_cfg["subsample"],
        subsample_number=final_cfg["subsample_number"],
        random_seed=final_cfg["seed"],
        mask_days=final_cfg["mask_days"],
        mode="pretrain",
        city=final_cfg["city"],
    )

    # 5) Train
    model = build_model_from_config(final_cfg, device, num_location_ids)
    best_model_path, canonical_config_path = train(
        cfg=final_cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_classes=num_classes,
    )
    print("[Pretrain] Best model:", best_model_path)

    # 6) Reload best for evaluation
    can_cfg = load_json(canonical_config_path)
    model = build_model_from_config(can_cfg, device, num_location_ids)
    state = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    # 7) Evaluate (pretrain: city=ALL)
    output_dir = os.path.join(os.path.dirname(best_model_path), "results")
    os.makedirs(output_dir, exist_ok=True)

    geobleu, dtw, acc = predict(
        model=model,
        test_loader=test_loader,
        device=device,
        output_dir=output_dir,
        W=W,
        save_csv=True,   # per-city CSVs under results/test & results/metric + summary.txt
    )

    print(f"[Pretrain/Test] GEO-BLEU: {geobleu:.4f}, DTW: {dtw:.2f}, Accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    main()