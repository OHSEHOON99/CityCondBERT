# Standard library
import os
import copy

# Third-party
import torch

# Local
from config import base_defaults, ID2CITY
from data_loader import load_and_split_data
from parser import get_transfer_parser, coerce_bool
from predict import predict, predict_masked_uid
from train import train
from utils import load_json, deep_merge, args_to_dict, build_model_from_config

# PyTorch/SDPA
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True



def main():
    args = get_transfer_parser().parse_args()

    # 1) Merge config: defaults <- file <- cli
    defaults = base_defaults("transfer")
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
    device = torch.device("cpu") if final_cfg["device"] == -1 else torch.device(f"cuda:{final_cfg['device']}")

    # 3) Paths / constants
    base_path = final_cfg["base_path"]
    data_path = os.path.join(base_path, "data")
    ckpt_dir = os.path.join(base_path, "checkpoints", final_cfg["model_name"])
    pretrained_model_path = os.path.join(ckpt_dir, "bert_best.pth")
    canonical_config_path = os.path.join(ckpt_dir, "config.json")

    assert os.path.isfile(pretrained_model_path), f"Pretrained model not found: {pretrained_model_path}"
    assert os.path.isfile(canonical_config_path), f"Canonical config not found: {canonical_config_path}"

    H, W = 200, 200
    num_classes = H * W
    num_location_ids = num_classes + 2  # PAD(0), 1..V, MASK(V+1)

    # 4) DataLoader (single target city + mask_loader for x==999)
    train_loader, val_loader, test_loader, mask_loader = load_and_split_data(
        mobility_data_path=data_path,
        batch_size=final_cfg["batch_size"],
        split_ratio=final_cfg["split_ratio"],
        use_subsample=final_cfg["subsample"],
        subsample_number=final_cfg["subsample_number"],
        random_seed=final_cfg["seed"],
        mask_days=final_cfg["mask_days"],
        mode="transfer",
        city=final_cfg["city"],
        W=W,
    )

    # 5) Build model from canonical config and load pretrained weights
    can_cfg = load_json(canonical_config_path)
    can_cfg["city"] = final_cfg.get("city", can_cfg.get("city", "A"))

    model = build_model_from_config(can_cfg, device, num_location_ids)
    state = torch.load(pretrained_model_path, map_location=device)
    model.load_state_dict(state, strict=False)
    print(f"[Transfer] Loaded pretrained weights from {pretrained_model_path}")

    # Optional freezing
    if str(final_cfg.get("freeze_backbone", "false")).lower() == "true":
        model.freeze_backbone()
    if str(final_cfg.get("freeze_head", "false")).lower() == "true":
        for p in model.output_projection.parameters():
            p.requires_grad = False

    # 6) Transfer training (canonical cfg as base, overridden by final_cfg)
    xfer_cfg = deep_merge(copy.deepcopy(can_cfg), final_cfg)
    best_model_path, new_canonical_config_path = train(
        cfg=xfer_cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_classes=num_classes,
    )
    print("[Transfer] Best model:", best_model_path)
    print("[Transfer] Canonical config:", new_canonical_config_path)

    # 7) Reload best for evaluation
    output_dir = os.path.join(os.path.dirname(best_model_path), "results")
    os.makedirs(output_dir, exist_ok=True)

    best_can_cfg = load_json(new_canonical_config_path)
    eval_model = build_model_from_config(best_can_cfg, device, num_location_ids)
    best_state = torch.load(best_model_path, map_location=device)
    eval_model.load_state_dict(best_state, strict=False)
    eval_model.eval()
    print(f"[Load] Best model loaded from {best_model_path}")

    # 새 predict 시그니처에 맞게 호출 (save_name/id2city_map/save_city_csv 제거)
    geobleu, dtw, acc = predict(
        model=eval_model,
        test_loader=test_loader,
        device=device,
        output_dir=output_dir,
        W=W,
        save_csv=True,  # 결과를 results/test & results/metric에 저장 + summary.txt 생성
    )
    print(f"[Transfer/Test] GEO-BLEU: {geobleu:.4f}, DTW: {dtw:.2f}, Accuracy: {acc*100:.2f}%")

    # 8) x==999 masked-UID prediction using mask_loader
    if mask_loader is not None:
        city_code = str(final_cfg.get("city", "A")).upper()  # 전이 단계: 설정에서 city 사용
        predict_masked_uid(
            model=eval_model,
            mask_loader=mask_loader,
            device=device,
            city=city_code,            # "ALL"이면 A/B/C/D로 분리 저장
            output_dir=output_dir,
            W=W,
            team_name="SCSI",          # 항상 저장하도록 구현됨 (save_csv 인자 제거)
        )
        print(f"[Transfer/Mask] Saved masked predictions for city {city_code}")


if __name__ == "__main__":
    main()