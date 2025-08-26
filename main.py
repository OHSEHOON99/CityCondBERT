# main.py
import argparse, os, json, copy, glob, re
import torch
import pandas as pd
from datetime import datetime

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


def _parse_ts(ts_str: str) -> datetime:
    # 로컬시간 기준으로 기록된 폴더명이 보통이지만, 서로 비교만 하면 되므로 naive datetime 사용
    return datetime.strptime(ts_str, "%Y%m%d_%H%M%S")


def _scan_wandb_by_timestamp_fuzzy(wandb_dir: str, target_run_name: str, tolerance_sec: int = 5):
    """
    wandb_dir 이하 run-*/ 폴더명을 훑어 target_run_name에 들어있는 타임스탬프와
    가장 가까운 run을 찾는다. 기본 허용 오차는 ±5초.
    반환: (run_id, run_root) or (None, None)
    """
    m = re.search(r"(\d{8}_\d{6})", target_run_name)
    if not m:
        print(f"[W&B] No timestamp found in target_run_name={target_run_name}")
        return None, None
    target_ts = m.group(1)
    target_dt = _parse_ts(target_ts)

    # 두 종류 경로 모두 커버: <wandb_dir>/run-* 와 <wandb_dir>/wandb/run-*
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

    # diff(초)가 가장 작은 후보 선택
    candidates.sort(key=lambda x: (x[0], x[1]))  # (diff, dt)로 tie-break
    best_diff, best_dt, best_id, best_root = candidates[0]

    if best_diff <= tolerance_sec:
        return best_id, best_root
    else:
        # 허용 오차를 넘으면 매칭 실패 처리
        print(f"[W&B] Closest run is {best_dt.strftime('%Y%m%d_%H%M%S')} (Δ={int(best_diff)}s) > tolerance({tolerance_sec}s).")
        return None, None


def _resume_wandb_by_timestamp(wandb_dir: str, target_run_name: str,
                               project: str, entity: str | None = None,
                               tolerance_sec: int = 5):
    """
    폴더명(run-<ts>-<id>)에서 target_run_name의 ts와 가장 가까운 run을 찾아 resume.
    tolerance_sec 내면 채택.
    """
    if wandb.run is not None:
        return  # 이미 열린 run 사용

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
    p.add_argument('--loss_name', type=str,
                choices=['ce', 'ddce', 'geobleu'],
                default=None,
                help='Loss to use: ce | ddce | geobleu')
    
    return p.parse_args()

# -------------------------------
# 3) model builder (dict 기반)
# -------------------------------
def build_model_from_config(cfg, device, num_locations):
    # ----- 1) Transformer 설정 (canonical 우선, legacy 호환) -----
    if "transformer" in cfg and isinstance(cfg["transformer"], dict):
        transformer_cfg = cfg["transformer"]
    else:
        # legacy: 평탄화 키들을 요구
        required = ["hidden_size", "hidden_layers", "attention_heads", "dropout", "max_seq_length"]
        missing = [k for k in required if k not in cfg]
        if missing:
            raise KeyError(f"Missing transformer keys (legacy format expected): {missing}")
        transformer_cfg = {
            "hidden_size":     int(cfg["hidden_size"]),
            "hidden_layers":   int(cfg["hidden_layers"]),
            "attention_heads": int(cfg["attention_heads"]),
            "dropout":         float(cfg["dropout"]),
            "max_seq_length":  int(cfg["max_seq_length"]),
        }

    # ----- 2) Embedding sizes (canonical 우선, legacy 호환) -----
    if "embedding_sizes" in cfg and isinstance(cfg["embedding_sizes"], dict):
        embedding_sizes = cfg["embedding_sizes"]
    else:
        required = ["day_embedding_size", "time_embedding_size", "day_of_week_embedding_size",
                    "weekday_embedding_size", "location_embedding_size"]
        missing = [k for k in required if k not in cfg]
        if missing:
            raise KeyError(f"Missing embedding size keys (legacy format expected): {missing}")
        embedding_sizes = {
            "day":      int(cfg["day_embedding_size"]),
            "time":     int(cfg["time_embedding_size"]),
            "dow":      int(cfg["day_of_week_embedding_size"]),
            "weekday":  int(cfg["weekday_embedding_size"]),
            "location": int(cfg["location_embedding_size"]),
        }

    # ----- 3) Delta embedding dims -----
    delta_embedding_dims = tuple(cfg.get("delta_embedding_dims", (8, 8, 8)))

    # ----- 4) Feature configs (canonical 요구) -----
    if "feature_configs" not in cfg:
        raise KeyError("`feature_configs` is required in the canonical config.")
    feature_configs = cfg["feature_configs"]

    # ----- 5) Combine mode key 호환 -----
    feature_combine_mode = cfg.get("feature_combine_mode",
                                   cfg.get("embedding_combine_mode", "cat"))

    # ----- 6) 모델 생성 -----
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

    # === predict 직전 ===
    wandb_dir = os.path.join(final_cfg["base_path"], "wandb", "wandb")
    target_run_name = final_cfg["model_name"]  # 예: "MobilityBERT-20250819_155854"

    _resume_wandb_by_timestamp(
        wandb_dir=wandb_dir,
        target_run_name=target_run_name,
        project="ACM SIGSPATIAL Cup 2025",
        entity=None,  # 팀/조직 계정 사용 중이면 지정
    )

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