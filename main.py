# main.py
import argparse, os, json, copy
import torch
import pandas as pd

from train import train
from model import MobilityBERT
from data_loader import load_and_split_data
from config import base_defaults
from predict import *


# -------------------------------
# 1) Helpers
# -------------------------------
def parse_tuple(s):
    try:
        return tuple(int(x.strip()) for x in s.strip('()').split(','))
    except:
        raise argparse.ArgumentTypeError("Expected format: '(32,64,128)'")

def deep_merge(dst, src):
    # dict 재귀적 병합: src가 dst를 덮어씀
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
    # 필요 시 내부 전용 키 제거 가능
    return {k: v for k, v in d.items() if v is not None}


# -------------------------------
# 2) argparse
# -------------------------------
def parse_args():
    p = argparse.ArgumentParser(description='Train/predict MobilityBERT with canonical config.')

    # 입력 config 파일 (선택): 실행 전에 미리 작성한 실험 설정
    p.add_argument('--config', type=str, default=None, help='Input config.json path (optional)')

    # General
    p.add_argument('--device', type=int, default=None)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--city', type=str, default=None)
    p.add_argument('--base_path', type=str, default=None)
    p.add_argument('--mode', type=str, choices=['train', 'predict'], default=None)
    p.add_argument('--model_name', type=str, default=None)
    p.add_argument('--wandb_api_key', type=str, default=None)

    # Data
    p.add_argument('--input_seq_length', type=int, default=None)
    p.add_argument('--predict_seq_length', type=int, default=None)
    p.add_argument('--look_back_len', type=int, default=None)
    p.add_argument('--split_ratio', type=parse_tuple, default=None)
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--subsample', action='store_true')
    p.add_argument('--subsample_number', type=int, default=None)

    # Model
    p.add_argument('--hidden_size', type=int, default=None)
    p.add_argument('--hidden_layers', type=int, default=None)
    p.add_argument('--attention_heads', type=int, default=None)
    p.add_argument('--dropout', type=float, default=None)
    p.add_argument('--max_seq_length', type=int, default=None)

    # Embedding sizes
    p.add_argument('--day_embedding_size', type=int, default=None)
    p.add_argument('--time_embedding_size', type=int, default=None)
    p.add_argument('--day_of_week_embedding_size', type=int, default=None)
    p.add_argument('--weekday_embedding_size', type=int, default=None)
    p.add_argument('--location_embedding_size', type=int, default=None)
    p.add_argument('--delta_embedding_dims', type=parse_tuple, default=None)

    # EmbeddingLayer 상위 결합
    p.add_argument('--feature_combine_mode', type=str, default=None)  # cat|sum|mlp

    # Train
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--location_embedding_lr', type=float, default=None)
    p.add_argument('--num_epochs', type=int, default=None)

    return p.parse_args()

# -------------------------------
# 3) model builder (dict 기반)
# -------------------------------
def build_model_from_config(cfg, device, num_locations):
    # canonical 우선, legacy 호환
    transformer_cfg = cfg.get("transformer", {
        "hidden_size":     cfg["hidden_size"],
        "hidden_layers":   cfg["hidden_layers"],
        "attention_heads": cfg["attention_heads"],
        "dropout":         cfg["dropout"],
        "max_seq_length":  cfg["max_seq_length"],
    })

    embedding_sizes = cfg.get("embedding_sizes", {
        "day":      cfg["day_embedding_size"],
        "time":     cfg["time_embedding_size"],
        "dow":      cfg["day_of_week_embedding_size"],
        "weekday":  cfg["weekday_embedding_size"],
        "location": cfg["location_embedding_size"],
    })

    delta_embedding_dims = tuple(cfg["delta_embedding_dims"])

    # ✅ 기본은 base_defaults에 있으므로 여기선 REQUIRED
    feature_configs = cfg["feature_configs"]

    # ✅ 키 통일 (읽기만 호환)
    feature_combine_mode = cfg.get("feature_combine_mode",
                            cfg.get("embedding_combine_mode", "cat"))

    model = MobilityBERT(
        num_location_ids=num_locations,
        transformer_cfg=transformer_cfg,
        embedding_sizes=embedding_sizes,
        delta_embedding_dims=delta_embedding_dims,
        feature_configs=feature_configs,
        embedding_combine_mode=feature_combine_mode,
    ).to(device)

    return model


# -------------------------------
# 4) main
# -------------------------------
def main():
    args = parse_args()

    # 4-1) 최종 설정 병합: defaults -> (옵션) 파일 -> (옵션) CLI
    defaults = base_defaults()
    file_cfg = load_json(args.config)
    cli_cfg = args_to_dict(args)
    final_cfg = deep_merge(deep_merge(copy.deepcopy(defaults), file_cfg), cli_cfg)

    # 4-2) 디바이스
    device = torch.device('cpu') if final_cfg["device"] == -1 else torch.device(f'cuda:{final_cfg["device"]}')

    # 4-3) 데이터 경로
    data_csv_path = os.path.join(final_cfg["base_path"], f"data/city_{final_cfg['city']}_challengedata.csv")
    num_locations = 200 * 200

    # 4-4) DataLoader
    train_loader, val_loader, test_loader = load_and_split_data(
        mobility_data_path=data_csv_path,
        input_seq_length=final_cfg["input_seq_length"],
        predict_seq_length=final_cfg["predict_seq_length"],
        sliding_step=final_cfg["look_back_len"],
        batch_size=final_cfg["batch_size"],
        split_ratio=final_cfg["split_ratio"],
        use_subsample=final_cfg["subsample"],
        subsample_number=final_cfg["subsample_number"],
        random_seed=final_cfg["seed"]
    )

    # ===== Branch: train or predict =====
    if final_cfg["mode"] == "train":
        # 모델 생성 (훈련 전용; train()이 받는 model)
        model = build_model_from_config(final_cfg, device, num_locations)

        # 학습 + 캐논 config 저장
        best_model_path, canonical_config_path = train(
            cfg=final_cfg,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_locations=num_locations,
        )
        print("[Train] Best model:", best_model_path)
        print("[Train] Canonical config:", canonical_config_path)

        # 추후 공통 predict 블록에서 사용할 경로 지정
        model_path = best_model_path
        output_dir = os.path.join(os.path.dirname(best_model_path), "results")

    elif final_cfg["mode"] == "predict":
        # 기존 체크포인트 디렉토리에서 불러오기
        model_dir = os.path.join(final_cfg["base_path"], "checkpoints", final_cfg["model_name"])
        model_path = os.path.join(model_dir, "bert_best.pth")
        canonical_config_path = os.path.join(model_dir, "config.json")
        output_dir = os.path.join(model_dir, "results")
    else:
        raise ValueError(f"Invalid mode: {final_cfg['mode']}")

    # 캐논 config로 모델 구성
    can_cfg = load_json(canonical_config_path)
    model = build_model_from_config(can_cfg, device, num_locations)

    # 가중치 로드 (누락되어 있던 부분)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"[Load] Best model loaded from {model_path}")

    os.makedirs(output_dir, exist_ok=True)
    save_name = f"city_{can_cfg.get('city','A')}_results"

    geobleu, dtw, acc = predict(
        model=model,
        test_loader=test_loader,
        device=device,
        output_dir=output_dir,
        save_name=save_name
    )
    print(f"[Test] GEO-BLEU: {geobleu:.4f}, DTW: {dtw:.2f}, Accuracy: {acc * 100:.2f}%")

    print("[INFO] Predicting masked UID...")
    df_all = pd.read_csv(data_csv_path)
    masked_uids = df_all[df_all['x'] == 999]['uid'].unique()
    df_masked_only = df_all[df_all['uid'].isin(masked_uids)]

    masked_output_path = os.path.join(output_dir, f"city_{can_cfg.get('city','A')}_masked_output.csv")
    predict_masked_uid(
        model=model,
        df_all=df_masked_only,
        config=can_cfg,              # ← can_cfg로 교체
        device=device,
        city=can_cfg.get("city", "A"),
        output_path=masked_output_path
    )

if __name__ == "__main__":
    main()