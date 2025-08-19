import os
from datetime import datetime

import numpy as np

import torch
from torch import nn
from transformers import get_linear_schedule_with_warmup

from tqdm import tqdm
from joblib import Parallel, delayed

import wandb
import geobleu

from utils import *
from visualization_feature import *



def _to_canonical_config(cfg, num_locations):
    """cfg(legacy/canonical 혼재)를 단일 canonical dict로 변환."""
    # transformer
    transformer_cfg = cfg.get("transformer", {
        "hidden_size":     cfg["hidden_size"],
        "hidden_layers":   cfg["hidden_layers"],
        "attention_heads": cfg["attention_heads"],
        "dropout":         cfg["dropout"],
        "max_seq_length":  cfg["max_seq_length"],
    })

    # embedding sizes
    embedding_sizes = cfg.get("embedding_sizes", {
        "day":      cfg["day_embedding_size"],
        "time":     cfg["time_embedding_size"],
        "dow":      cfg["day_of_week_embedding_size"],
        "weekday":  cfg["weekday_embedding_size"],
        "location": cfg["location_embedding_size"],
    })

    # feature configs (필수)
    feature_configs = cfg["feature_configs"]

    # 상위 결합 키 통일
    feature_combine_mode = cfg.get("feature_combine_mode",
                            cfg.get("embedding_combine_mode", "cat"))

    delta_embedding_dims = tuple(cfg["delta_embedding_dims"])

    canonical_config = {
        "schema_version": "1.0.0",
        "city": cfg["city"],
        "num_location_ids": num_locations,
        "transformer": transformer_cfg,
        "embedding_sizes": embedding_sizes,
        "delta_embedding_dims": list(delta_embedding_dims),  # JSON 저장용
        "feature_configs": feature_configs,
        "feature_combine_mode": feature_combine_mode,
    }
    return canonical_config


def log_epoch_metrics(
    epoch, train_metrics, val_metrics, log_path
):
    """
    학습 및 검증 지표를 로그 파일에 저장합니다.

    Args:
        epoch (int): 현재 에폭 (0-based index)
        train_metrics (dict): {'loss': ..., 'geobleu': ..., 'dtw': ..., 'acc': ...}
        val_metrics (dict): {'loss': ..., 'geobleu': ..., 'dtw': ..., 'acc': ...}
        log_path (str): 로그 파일 경로
    """
    with open(log_path, 'a') as f:
        f.write(
            f"[Epoch {epoch+1:02d}]\n"
            f"Train     => Loss: {train_metrics['loss']:.4f}, "
            f"GEO-BLEU: {train_metrics['geobleu']:.4f}, "
            f"DTW: {train_metrics['dtw']:.2f}, "
            f"Acc: {train_metrics['acc']*100:.2f}%\n"
            f"Validation=> Loss: {val_metrics['loss']:.4f}, "
            f"GEO-BLEU: {val_metrics['geobleu']:.4f}, "
            f"DTW: {val_metrics['dtw']:.2f}, "
            f"Acc: {val_metrics['acc']*100:.2f}%\n"
            f"{'-'*80}\n"
        )


def configure_optimizer(model, base_lr, location_embedding_lr):
    """
    - 기본 파라미터는 base_lr
    - EmbeddingLayer 내부 location 블록(embedding.feature_blocks.location.*)은 location_embedding_lr
    - location 블록에 categorical/periodic/fourier가 켜져 있으면 전부 포함됨
    """
    base_params, location_params = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "embedding.feature_blocks.location" in name:
            location_params.append(param)
        else:
            base_params.append(param)

    if location_embedding_lr is None:
        location_embedding_lr = base_lr

    optimizer = torch.optim.AdamW(
        [
            {"params": base_params},
            {"params": location_params, "lr": location_embedding_lr},
        ],
        lr=base_lr,
        weight_decay=0.01,
        foreach=False,
    )

    # 간단한 확인 로그 (필요 없으면 제거 가능)
    print(f"[Optimizer] Base LR: {base_lr}, Location Embedding LR: {location_embedding_lr}")
    print(f"[Optimizer] #base params: {sum(p.numel() for p in base_params):,}, "
          f"#location params: {sum(p.numel() for p in location_params):,}")

    return optimizer


def _forward_pass_with_metrics(
    model,
    input_seq_feature,
    historical_locations,
    predict_seq_feature,
    future_locations,
    device,
    metric_eval=False
):
    # Move to device
    input_seq_feature, historical_locations, predict_seq_feature, future_locations = [
        b.to(device) for b in [input_seq_feature, historical_locations, predict_seq_feature, future_locations]
    ]

    # Forward pass
    logits = model(input_seq_feature, historical_locations, predict_seq_feature)
    preds = torch.argmax(logits, dim=-1)

    # Accuracy
    pred_locs = preds.cpu().numpy()
    true_locs = future_locations.cpu().numpy()
    avg_accuracy = np.mean(pred_locs == true_locs)

    if not metric_eval:
        return logits, 0.0, 0.0, avg_accuracy, future_locations

    # 시간 정보 추출
    day_seq = predict_seq_feature[:, :, 0].cpu().numpy()
    time_seq = predict_seq_feature[:, :, 1].cpu().numpy()

    pred_coords = id_to_xy(pred_locs).astype(np.float64)  # (B, 2, T)
    true_coords = id_to_xy(true_locs).astype(np.float64)

    # Metric 계산 함수
    def compute_metrics(i, pred_coords, true_coords, day_seq, time_seq):
        pred_seq = [(int(day_seq[i][t]), int(time_seq[i][t]), x, y)
                    for t, (x, y) in enumerate(pred_coords[i].T)]
        true_seq = [(int(day_seq[i][t]), int(time_seq[i][t]), x, y)
                    for t, (x, y) in enumerate(true_coords[i].T)]

        geobleu_score = geobleu.calc_geobleu_single(pred_seq, true_seq)
        dtw_distance = geobleu.calc_dtw_single(pred_seq, true_seq)
        return geobleu_score, dtw_distance

    results = Parallel(n_jobs=-1)(
        delayed(compute_metrics)(i, pred_coords, true_coords, day_seq, time_seq)
        for i in range(len(pred_coords))
    )

    geobleu_scores, dtw_distances = zip(*results)
    avg_geobleu = np.mean(geobleu_scores)
    avg_dtw = np.mean(dtw_distances)

    return logits, avg_geobleu, avg_dtw, avg_accuracy, future_locations
    

def train_step(model, optimizer, criterion, input_seq_feature, historical_locations, predict_seq_feature, future_locations, _, device):
    model.train()
    optimizer.zero_grad()

    # Return future_locations after moving to device to avoid
    # device mismatch in loss computation (CPU vs CUDA)
    logits, geobleu, dtw, accuracy, future_locations = _forward_pass_with_metrics(
        model, input_seq_feature, historical_locations, predict_seq_feature, future_locations, device
    )

    loss = criterion(logits.view(-1, logits.size(-1)), future_locations.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item(), geobleu, dtw, accuracy


def val_step(model, criterion, input_seq_feature, historical_locations, predict_seq_feature, future_locations, _, device):
    model.eval()

    # Return future_locations after moving to device to avoid
    # device mismatch in loss computation (CPU vs CUDA)
    with torch.no_grad():
        logits, geobleu, dtw, accuracy, future_locations = _forward_pass_with_metrics(
            model, input_seq_feature, historical_locations, predict_seq_feature, future_locations, device, True
        )

        loss = criterion(logits.view(-1, logits.size(-1)), future_locations.view(-1))
        return loss.item(), geobleu, dtw, accuracy


def train_model(model, optimizer, train_loader, val_loader, num_epochs, device, run_dir):
    criterion = nn.CrossEntropyLoss()

    total_steps = max(1, num_epochs * len(train_loader))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps
    )

    best_val_loss = float("inf")
    best_model_path = None
    no_improve_count = 0
    patience = 5
    log_file_path = os.path.join(run_dir, 'train_log.txt')
    os.makedirs(run_dir, exist_ok=True)

    wandb.watch(model, log="all", log_freq=100)

    for epoch in range(num_epochs):
        # === Train ===
        total_train_loss = total_train_geobleu = total_train_dtw = total_train_acc = 0.0
        num_train_batches = len(train_loader)

        train_prog = tqdm(enumerate(train_loader), total=num_train_batches,
                          desc=f"[Train] Epoch {epoch+1}/{num_epochs}", ncols=100)

        for _, batch in train_prog:
            loss, geobleu, dtw, acc = train_step(model, optimizer, criterion, *batch, device=device)
            total_train_loss += loss
            total_train_geobleu += geobleu
            total_train_dtw += dtw
            total_train_acc += acc
            scheduler.step()
            train_prog.set_postfix(loss=loss)

        avg_train_loss  = total_train_loss  / max(1, num_train_batches)
        avg_train_geobl = total_train_geobleu/ max(1, num_train_batches)
        avg_train_dtw   = total_train_dtw   / max(1, num_train_batches)
        avg_train_acc   = total_train_acc   / max(1, num_train_batches)

        print(f"[Train] Loss: {avg_train_loss:.4f}, GEO-BLEU: {avg_train_geobl:.4f}, "
              f"DTW: {avg_train_dtw:.2f}, Acc: {avg_train_acc*100:.2f}%")

        # === Val ===
        total_val_loss = total_val_geobleu = total_val_dtw = total_val_acc = 0.0
        num_val_batches = len(val_loader)

        val_prog = tqdm(enumerate(val_loader), total=num_val_batches,
                        desc=f"[Validation] Epoch {epoch+1}/{num_epochs}", ncols=100)

        for _, batch in val_prog:
            loss, geobleu, dtw, acc = val_step(model, criterion, *batch, device=device)
            total_val_loss += loss
            total_val_geobleu += geobleu
            total_val_dtw += dtw
            total_val_acc += acc
            val_prog.set_postfix(loss=loss)

        avg_val_loss  = total_val_loss  / max(1, num_val_batches)
        avg_val_geobl = total_val_geobleu/ max(1, num_val_batches)
        avg_val_dtw   = total_val_dtw   / max(1, num_val_batches)
        avg_val_acc   = total_val_acc   / max(1, num_val_batches)

        print(f"[Validation] Loss: {avg_val_loss:.4f}, GEO-BLEU: {avg_val_geobl:.4f}, "
              f"DTW: {avg_val_dtw:.2f}, Acc: {avg_val_acc*100:.2f}%")

        # === Log ===
        log_epoch_metrics(
            epoch=epoch,
            train_metrics={'loss': avg_train_loss, 'geobleu': avg_train_geobl,
                           'dtw': avg_train_dtw, 'acc': avg_train_acc},
            val_metrics={'loss': avg_val_loss, 'geobleu': avg_val_geobl,
                         'dtw': avg_val_dtw, 'acc': avg_val_acc},
            log_path=log_file_path
        )
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": avg_train_loss, "train/geobleu": avg_train_geobl,
            "train/dtw": avg_train_dtw,   "train/acc": avg_train_acc,
            "val/loss": avg_val_loss,     "val/geobleu": avg_val_geobl,
            "val/dtw": avg_val_dtw,       "val/acc": avg_val_acc,
        })

        for fname in ["day", "time", "location"]:
            emb_np, ids_np = compute_feature_codebook(model, fname, device)
            log_codebook_tsne(fname, emb_np, ids_np, epoch)
            log_codebook_heatmap(fname, emb_np, epoch)

        # delta: 연속값 → 그리드 범위/스텝 조정 가능
        emb_np, ids_np = compute_feature_codebook(model, "delta", device, delta_max=2000, delta_steps=256)
        log_codebook_tsne("delta", emb_np, ids_np, epoch)

        # === Save best + early stop ===
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve_count = 0
            best_model_path = os.path.join(run_dir, 'bert_best.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"[Saved] Best model saved at {best_model_path}")
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            print(f"[EarlyStopping] Stopped early at epoch {epoch+1} due to no improvement.")
            break

    # Final save
    final_model_path = os.path.join(run_dir, 'bert_final.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"[Saved] Final model saved at {final_model_path}")

    return best_model_path, final_model_path


def train(cfg, model, train_loader, val_loader, device, num_locations):
    """
    main.py에서:
      best_model_path, canonical_config_path = train(cfg=final_cfg, model=model, ...)
    """
    # ----------------------------
    # 1) 필수 필드 검증(유연하게)
    # ----------------------------
    # transformer: dict 또는 legacy 5키 중 하나
    has_transformer = ("transformer" in cfg) or all(
        k in cfg for k in ["hidden_size", "hidden_layers", "attention_heads", "dropout", "max_seq_length"]
    )
    if not has_transformer:
        raise ValueError("[cfg] transformer 설정이 없습니다. "
                         "transformer 블록 또는 legacy 키(hidden_size, hidden_layers, attention_heads, dropout, max_seq_length) 중 하나를 제공하세요.")

    # embedding_sizes: dict 또는 legacy 5키 중 하나
    has_emb_sizes = ("embedding_sizes" in cfg) or all(
        k in cfg for k in ["day_embedding_size", "time_embedding_size",
                           "day_of_week_embedding_size", "weekday_embedding_size",
                           "location_embedding_size"]
    )
    if not has_emb_sizes:
        raise ValueError("[cfg] embedding sizes 설정이 없습니다. "
                         "embedding_sizes 블록 또는 legacy 키(day/time/dow/weekday/location_embedding_size) 중 하나를 제공하세요.")

    # 공통 필수
    for k in ["city", "lr", "num_epochs", "base_path", "delta_embedding_dims", "feature_configs"]:
        if k not in cfg:
            raise ValueError(f"[cfg] Missing required key: {k}")

    # ----------------------------
    # 2) Canonical config 구성(중복 제거: 헬퍼 사용)
    # ----------------------------
    canonical_config = _to_canonical_config(cfg, num_locations)

    # ----------------------------
    # 3) Optimizer / wandb
    # ----------------------------
    optimizer = configure_optimizer(model, cfg["lr"], cfg.get("location_embedding_lr"))

    run_name = f"{model.__class__.__name__}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if cfg.get("wandb_api_key", ""):
        wandb.login(key=cfg["wandb_api_key"])
    wandb.init(
        project="ACM SIGSPATIAL Cup 2025",
        dir=os.path.join(cfg["base_path"], 'wandb'),
        name=run_name,
        config={**cfg, "run_name": run_name, "canonical_config": canonical_config}
    )

    # ----------------------------
    # 4) 경로 / config 저장
    # ----------------------------
    run_dir = os.path.join(cfg["base_path"], 'checkpoints', run_name)
    os.makedirs(os.path.join(run_dir, "results"), exist_ok=True)

    canonical_config_path = os.path.join(run_dir, "config.json")
    with open(canonical_config_path, "w") as f:
        json.dump(canonical_config, f, indent=4)
    print(f"[Config] Saved canonical config → {canonical_config_path}")

    # ----------------------------
    # 5) 학습
    # ----------------------------
    best_model_path, _ = train_model(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=cfg["num_epochs"],
        device=device,
        run_dir=run_dir
    )
    return best_model_path, canonical_config_path