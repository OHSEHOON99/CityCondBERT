import os
from datetime import datetime
from collections import defaultdict
import math

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
from losses import build_criterion



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


def _make_loss_kwargs_from_cfg(cfg):
    # 공통 그리드/셀
    H = int(cfg.get("H", 200))
    W = int(cfg.get("W", 200))
    cell_km_x = float(cfg.get("cell_km_x", 0.5))
    cell_km_y = float(cfg.get("cell_km_y", 0.5))

    loss_name = cfg.get("loss_name", "ce").lower()

    if loss_name == "ce":
        return {}

    elif loss_name == "ddce":
        d = cfg.get("ddce", {})
        return {
            "H": H, "W": W,
            "win": int(d.get("win", 7)),
            "beta": float(d.get("beta", 0.5)),
            "cell_km_x": cell_km_x, "cell_km_y": cell_km_y,
            "distance_scale": float(d.get("distance_scale", 2.0)),
            "ignore_index": d.get("ignore_index", None),
            "reduction": d.get("reduction", "mean"),
        }

    elif loss_name == "geobleu":
        g = cfg.get("geobleu", {})
        return {
            "H": H, "W": W,
            "n_list": tuple(g.get("n_list", [1, 2, 3, 4, 5])),
            "win": int(g.get("win", 7)),
            "beta": float(g.get("beta", 0.5)),
            "cell_km_x": cell_km_x, "cell_km_y": cell_km_y,
            "distance_scale": float(g.get("distance_scale", 2.0)),
            "eps": float(g.get("eps", 0.1)),
            "n_iters": int(g.get("n_iters", 30)),
            "weights": g.get("weights", None),
        }

    elif loss_name == "combo":
        # combo 설정 읽기
        c = cfg.get("combo", {})
        ce_name = c.get("ce_name", "ce").lower()

        # ce_kwargs 구성
        if ce_name == "ddce":
            # ddce를 CE로 쓸 경우 grid/셀크기 포함 필요
            base = cfg.get("ddce", {})
            ck = c.get("ce_kwargs", {})
            ce_kwargs = {
                "H": H, "W": W,
                "win": int(ck.get("win", base.get("win", 7))),
                "beta": float(ck.get("beta", base.get("beta", 0.5))),
                "cell_km_x": cell_km_x, "cell_km_y": cell_km_y,
                "distance_scale": float(ck.get("distance_scale", base.get("distance_scale", 2.0))),
                "ignore_index": ck.get("ignore_index", base.get("ignore_index", None)),
                "reduction": ck.get("reduction", base.get("reduction", "mean")),
            }
        else:
            # 일반 CE
            ce_kwargs = c.get("ce_kwargs", {})

        # geobleu_kwargs 구성 (필수 H/W/셀 크기 ensure)
        gk = c.get("geobleu_kwargs", {})
        geobleu_kwargs = {
            "H": int(gk.get("H", H)),
            "W": int(gk.get("W", W)),
            "n_list": tuple(gk.get("n_list", cfg.get("geobleu", {}).get("n_list", [1,2,3,4,5]))),
            "win": int(gk.get("win", cfg.get("geobleu", {}).get("win", 7))),
            "beta": float(gk.get("beta", cfg.get("geobleu", {}).get("beta", 0.5))),
            "cell_km_x": float(gk.get("cell_km_x", cell_km_x)),
            "cell_km_y": float(gk.get("cell_km_y", cell_km_y)),
            "distance_scale": float(gk.get("distance_scale", cfg.get("geobleu", {}).get("distance_scale", 2.0))),
            "eps": float(gk.get("eps", cfg.get("geobleu", {}).get("eps", 0.1))),
            "n_iters": int(gk.get("n_iters", cfg.get("geobleu", {}).get("n_iters", 30))),
            "weights": gk.get("weights", cfg.get("geobleu", {}).get("weights", None)),
        }

        return {
            "ce_name": ce_name,
            "ce_kwargs": ce_kwargs,
            "geobleu_kwargs": geobleu_kwargs,
            "alpha_init": float(c.get("alpha_init", 1.0)),
            "ema_m": float(c.get("ema_m", 0.99)),
            "track_mavg": bool(c.get("track_mavg", True)),
            # 선택: α가 매우 높을 땐 GeoBLEU 계산 스킵하여 속도↑
            "skip_geobleu_when_alpha_ge": float(c.get("skip_geobleu_when_alpha_ge", 0.999)),
        }

    else:
        raise ValueError(f"[cfg] Unknown loss_name: {loss_name}")


def _alpha_sched(epoch: int, combo_cfg: dict) -> float:
    """
    cfg['combo']에서 스케줄 파라미터 읽어 α를 반환.
    - warmup 동안 α=1.0 (CE only)
    - transition 동안 선형으로 a_start -> a_end
    """
    e_warm = int(combo_cfg.get("alpha_warmup_epochs", 5))
    e_trans = int(combo_cfg.get("alpha_transition_epochs", 3))
    a_start = float(combo_cfg.get("alpha_start", 0.9))
    a_end   = float(combo_cfg.get("alpha_end", 0.3))

    if epoch < e_warm:
        return 1.0
    r = min(1.0, (epoch - e_warm) / max(1, e_trans))  # 0→1
    return a_start + (a_end - a_start) * r            # linear



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
    

def train_step(model, optimizer, criterion, input_seq_feature, historical_locations,
               predict_seq_feature, future_locations, _, device):
    # ▶ 학습 모드 전환 (모델 + 손실함수 둘 다)
    model.train()
    if hasattr(criterion, "train"):
        criterion.train()

    optimizer.zero_grad(set_to_none=True)

    logits, geobleu, dtw, accuracy, future_locations = _forward_pass_with_metrics(
        model, input_seq_feature, historical_locations, predict_seq_feature, future_locations, device
    )

    # CrossEntropyLoss면 (B,T,V)/(B,T) → (B*T,V)/(B*T,)로 평탄화
    if isinstance(criterion, nn.CrossEntropyLoss):
        V = logits.size(-1)
        loss = criterion(logits.reshape(-1, V), future_locations.reshape(-1).long())
        aux = {}
    else:
        out = criterion(logits, future_locations)
        loss, aux = out if isinstance(out, tuple) else (out, {})

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item(), geobleu, dtw, accuracy, aux


@torch.no_grad()
def val_step(model, criterion, input_seq_feature, historical_locations,
             predict_seq_feature, future_locations, _, device):
    # ▶ 평가 모드 전환 (모델 + 손실함수 둘 다)
    model.eval()
    if hasattr(criterion, "eval"):
        criterion.eval()

    logits, geobleu, dtw, accuracy, future_locations = _forward_pass_with_metrics(
        model, input_seq_feature, historical_locations, predict_seq_feature, future_locations, device, metric_eval=True
    )

    if isinstance(criterion, nn.CrossEntropyLoss):
        V = logits.size(-1)
        loss = criterion(logits.reshape(-1, V), future_locations.reshape(-1).long())
        aux = {}
    else:
        out = criterion(logits, future_locations)
        loss, aux = out if isinstance(out, tuple) else (out, {})

    return loss.item(), geobleu, dtw, accuracy, aux


def train_model(model, optimizer, train_loader, val_loader, num_epochs, device, run_dir, cfg,
                loss_name="ce", loss_kwargs=None):
    if loss_kwargs is None:
        loss_kwargs = {}

    criterion = build_criterion(loss_name, **loss_kwargs)

    total_steps = max(1, num_epochs * len(train_loader))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps
    )

    best_val_geobleu = float("-inf")
    best_model_path = None
    no_improve_count = 0
    patience = 5
    log_file_path = os.path.join(run_dir, 'train_log.txt')
    os.makedirs(run_dir, exist_ok=True)

    wandb.watch(model, log="all", log_freq=100)

    for epoch in range(num_epochs):
        # ✅ combo면 epoch마다 α 업데이트 (cfg['combo'] 기반)
        if loss_name == "combo":
            combo_cfg = cfg.get("combo", {})
            alpha = _alpha_sched(epoch, combo_cfg)
            if hasattr(criterion, "set_alpha"):
                criterion.set_alpha(alpha)

        # === Train ===
        total_train_loss = total_train_geobleu = total_train_dtw = total_train_acc = 0.0
        num_train_batches = len(train_loader)
        aux_train_sum = defaultdict(float)  # <-- aux 집계
        aux_train_cnt = 0

        train_prog = tqdm(enumerate(train_loader), total=num_train_batches,
                          desc=f"[Train] Epoch {epoch+1}/{num_epochs}", ncols=100)

        for _, batch in train_prog:
            loss, geobleu, dtw, acc, aux = train_step(model, optimizer, criterion, *batch, device=device)
            total_train_loss += loss
            total_train_geobleu += geobleu
            total_train_dtw += dtw
            total_train_acc += acc

            # aux 집계
            if isinstance(aux, dict):
                for k, v in aux.items():
                    aux_train_sum[f"train/{k}"] += float(v)
                aux_train_cnt += 1

            scheduler.step()
            train_prog.set_postfix(loss=loss)

        avg_train_loss  = total_train_loss  / max(1, num_train_batches)
        avg_train_geobl = total_train_geobleu/ max(1, num_train_batches)
        avg_train_dtw   = total_train_dtw   / max(1, num_train_batches)
        avg_train_acc   = total_train_acc   / max(1, num_train_batches)
        avg_train_aux   = {k: v / max(1, aux_train_cnt) for k, v in aux_train_sum.items()}

        print(f"[Train] Loss: {avg_train_loss:.4f}, GEO-BLEU: {avg_train_geobl:.4f}, "
              f"DTW: {avg_train_dtw:.2f}, Acc: {avg_train_acc*100:.2f}%")

        # === Val ===
        total_val_loss = total_val_geobleu = total_val_dtw = total_val_acc = 0.0
        num_val_batches = len(val_loader)
        aux_val_sum = defaultdict(float)
        aux_val_cnt = 0

        val_prog = tqdm(enumerate(val_loader), total=num_val_batches,
                        desc=f"[Validation] Epoch {epoch+1}/{num_epochs}", ncols=100)

        for _, batch in val_prog:
            loss, geobleu, dtw, acc, aux = val_step(model, criterion, *batch, device=device)
            total_val_loss += loss
            total_val_geobleu += geobleu
            total_val_dtw += dtw
            total_val_acc += acc
            if isinstance(aux, dict):
                for k, v in aux.items():
                    aux_val_sum[f"val/{k}"] += float(v)
                aux_val_cnt += 1
            val_prog.set_postfix(loss=loss, acc=f"{acc:.3f}")

        avg_val_loss  = total_val_loss  / max(1, num_val_batches)
        avg_val_geobl = total_val_geobleu/ max(1, num_val_batches)
        avg_val_dtw   = total_val_dtw   / max(1, num_val_batches)
        avg_val_acc   = total_val_acc   / max(1, num_val_batches)
        avg_val_aux   = {k: v / max(1, aux_val_cnt) for k, v in aux_val_sum.items()}

        print(f"[Validation] Loss: {avg_val_loss:.4f}, GEO-BLEU: {avg_val_geobl:.4f}, "
              f"DTW: {avg_val_dtw:.2f}, Acc: {avg_val_acc*100:.2f}%")

        # === Log ===
        base_train_log = {
            "epoch": epoch + 1,
            "train/loss": avg_train_loss,
            "train/geobleu": avg_train_geobl,
            "train/dtw": avg_train_dtw,
            "train/acc": avg_train_acc,
        }
        base_val_log = {
            "val/loss": avg_val_loss,
            "val/geobleu": avg_val_geobl,
            "val/dtw": avg_val_dtw,
            "val/acc": avg_val_acc,
        }

        # combo라면 α도 로깅
        if loss_name == "combo" and hasattr(criterion, "alpha"):
            base_train_log["train/alpha"] = criterion.alpha
            base_val_log["val/alpha"] = criterion.alpha

        wandb.log({**base_train_log, **base_val_log, **avg_train_aux, **avg_val_aux})

        log_epoch_metrics(
            epoch=epoch,
            train_metrics={'loss': avg_train_loss, 'geobleu': avg_train_geobl,
                           'dtw': avg_train_dtw, 'acc': avg_train_acc, **avg_train_aux},
            val_metrics={'loss': avg_val_loss, 'geobleu': avg_val_geobl,
                         'dtw': avg_val_dtw, 'acc': avg_val_acc, **avg_val_aux},
            log_path=log_file_path
        )

        # === Feature codebook 시각화 ===
        for fname in ["day", "time", "location"]:
            emb_np, ids_np = compute_feature_codebook(model, fname, device)
            if emb_np is None:
                continue
            log_codebook_tsne(fname, emb_np, ids_np, epoch)
            log_codebook_heatmap(fname, emb_np, epoch)

        emb_np, ids_np = compute_feature_codebook(model, "delta", device, delta_max=2000, delta_steps=256)
        if emb_np is not None:
            log_codebook_tsne("delta", emb_np, ids_np, epoch)

        metric = avg_val_geobl if math.isfinite(avg_val_geobl) else float("-inf")
        if metric > best_val_geobleu:
            best_val_geobleu = metric
            no_improve_count = 0
            best_model_path = os.path.join(run_dir, 'bert_best.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"[Saved] Best model (by GEO-BLEU) saved at {best_model_path} — val/geobleu={best_val_geobleu:.6f}")
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            print(f"[EarlyStopping] Stopped early at epoch {epoch+1} — no val/GEO-BLEU improvement for {patience} evals.")
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
    has_transformer = ("transformer" in cfg) or all(
        k in cfg for k in ["hidden_size", "hidden_layers", "attention_heads", "dropout", "max_seq_length"]
    )
    if not has_transformer:
        raise ValueError("[cfg] transformer 설정이 없습니다. "
                         "transformer 블록 또는 legacy 키(hidden_size, hidden_layers, attention_heads, dropout, max_seq_length) 중 하나를 제공하세요.")

    has_emb_sizes = ("embedding_sizes" in cfg) or all(
        k in cfg for k in ["day_embedding_size", "time_embedding_size",
                           "day_of_week_embedding_size", "weekday_embedding_size",
                           "location_embedding_size"]
    )
    if not has_emb_sizes:
        raise ValueError("[cfg] embedding sizes 설정이 없습니다. "
                         "embedding_sizes 블록 또는 legacy 키(day/time/dow/weekday/location_embedding_size) 중 하나를 제공하세요.")

    for k in ["city", "lr", "num_epochs", "base_path", "delta_embedding_dims", "feature_configs"]:
        if k not in cfg:
            raise ValueError(f"[cfg] Missing required key: {k}")

    # ----------------------------
    # 2) Canonical config 구성
    # ----------------------------
    canonical_config = _to_canonical_config(cfg, num_locations)

    # ✅ 손실 이름/하이퍼 생성 & canonical에 기록
    loss_name = cfg.get("loss_name", "ce").lower()
    loss_kwargs = _make_loss_kwargs_from_cfg(cfg)
    canonical_config["loss"] = {"name": loss_name, "kwargs": loss_kwargs}

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
        # wandb.config에 loss 설정도 함께 기록
        config={**cfg, "run_name": run_name, "canonical_config": canonical_config,
                "loss_name": loss_name, "loss_kwargs": loss_kwargs}
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

    run = wandb.run  # 현재 run 핸들
    run_meta = {
        "wandb": {
            "id": run.id,
            "project": run.project,
            "entity": getattr(run, "entity", None),
            "name": run.name,
            "url": run.url,
        },
        "created_at": datetime.now().isoformat()
    }
    run_meta_path = os.path.join(run_dir, "run_meta.json")
    with open(run_meta_path, "w") as f:
        json.dump(run_meta, f, indent=2)

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
        run_dir=run_dir,
        cfg=cfg,  
        loss_name=loss_name,
        loss_kwargs=loss_kwargs,
    )
    return best_model_path, canonical_config_path