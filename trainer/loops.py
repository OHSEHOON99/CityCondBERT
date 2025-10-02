import os
import math
from collections import defaultdict

import numpy as np
import torch
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from joblib import Parallel, delayed
import wandb
import geobleu
from torch.cuda.amp import GradScaler as CudaGradScaler

from utils import id_to_xy
from visualization_feature import (
    compute_feature_codebook,
    log_codebook_tsne,
    log_codebook_heatmap,
)
from losses.builders import build_criterion
from .configs import alpha_sched



def log_epoch_metrics(epoch, train_metrics, val_metrics, log_path):
    """
    Log training and validation metrics to a text file.
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


# ----------------------------------------------------------------
# Forward pass (masked logits only) + accuracy + (optional) metrics
# ----------------------------------------------------------------
def _forward_pass_with_metrics(
    model,
    input_seq_feature,
    location_labels,
    attention_mask,
    city_ids,
    device,
    metric_eval: bool = False
):
    # Move tensors to device
    input_seq_feature = input_seq_feature.to(device, non_blocking=True)
    location_labels = location_labels.to(device, non_blocking=True)
    attention_mask = attention_mask.to(device, non_blocking=True)
    city_ids = city_ids.to(device, non_blocking=True)

    # Forward pass (masked logits only)
    logits_masked, mask_pos = model(
        input_seq_feature,
        attention_mask,
        labels=location_labels,
        city_ids=city_ids,
    )

    # Accuracy on masked positions
    if mask_pos.any().item():
        pred_masked = torch.argmax(logits_masked, dim=-1).detach()
        targets_masked = location_labels[mask_pos]
        avg_accuracy = (pred_masked == targets_masked).float().mean().item()
    else:
        split_art = {
            "targets_masked": None,
            "counts": [],
            "p_list": [],
            "y_list": [],
            "buckets": {},
            "pred_masked": None,
            "pred_list": [],
        }
        return logits_masked, 0.0, 0.0, 0.0, mask_pos, split_art

    # Split by sequence length
    counts = mask_pos.sum(dim=1).tolist()
    nz = [c for c in counts if c > 0]

    p_list = list(torch.split(logits_masked, nz, dim=0))
    y_list = list(torch.split(targets_masked, nz, dim=0))
    pred_list = list(torch.split(pred_masked, nz, dim=0))

    buckets = defaultdict(list)
    for p_i, y_i in zip(p_list, y_list):
        buckets[p_i.size(0)].append((p_i, y_i))
    for T, items in list(buckets.items()):
        P = torch.stack([p for p, _ in items], dim=0)
        Y = torch.stack([y for _, y in items], dim=0)
        buckets[T] = (P, Y)

    split_art = {
        "targets_masked": targets_masked,
        "counts": counts,
        "p_list": p_list,
        "y_list": y_list,
        "buckets": buckets,
        "pred_masked": pred_list[0] if len(pred_list) > 0 else None,
        "pred_list": pred_list,
    }

    # Optional heavy metrics
    geobleu_val = 0.0
    dtw_val = 0.0
    if metric_eval:
        day_masked = input_seq_feature[:, :, 0][mask_pos]
        time_masked = input_seq_feature[:, :, 1][mask_pos]
        day_list = list(torch.split(day_masked, nz, dim=0))
        time_list = list(torch.split(time_masked, nz, dim=0))

        d_np_list = [(day_list[i].detach().cpu().numpy().astype(np.int32) - 1).clip(min=0) for i in range(len(p_list))]
        t_np_list = [(time_list[i].detach().cpu().numpy().astype(np.int32) - 1).clip(min=0) for i in range(len(p_list))]
        p_np_list = [split_art["pred_list"][i].detach().cpu().numpy().astype(np.int64, copy=False) for i in range(len(p_list))]
        y_np_list = [y_list[i].detach().cpu().numpy().astype(np.int64, copy=False) for i in range(len(p_list))]

        def compute_metrics_np(d_i, t_i, p_i, y_i):
            if p_i.size == 0:
                return None
            p_xy = id_to_xy(p_i)
            t_xy = id_to_xy(y_i)
            generated = list(zip(d_i, t_i, p_xy[:, 0], p_xy[:, 1]))
            reference = list(zip(d_i, t_i, t_xy[:, 0], t_xy[:, 1]))
            return geobleu.calc_geobleu_single(generated, reference), geobleu.calc_dtw_single(generated, reference)

        results = Parallel(n_jobs=os.cpu_count() // 2)(
            delayed(compute_metrics_np)(d_np_list[i], t_np_list[i], p_np_list[i], y_np_list[i])
            for i in range(len(p_np_list))
        )
        results = [r for r in results if r is not None]
        if results:
            geos, dtws = zip(*results)
            geobleu_val = float(np.mean(geos))
            dtw_val = float(np.mean(dtws))

    return logits_masked, geobleu_val, dtw_val, avg_accuracy, mask_pos, split_art



# ----------------------------------------------------------------
# Train / Validation Step (with AMP)
# ----------------------------------------------------------------
def train_step(model, optimizer, criterion,
               input_seq_feature, location_labels, attention_mask, city_ids,
               device, scaler, amp_dtype, metric_eval: bool = False):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    use_amp = scaler.is_enabled() if hasattr(scaler, "is_enabled") else True
    autocast_ctx = torch.amp.autocast(
        device_type=device.type, dtype=amp_dtype,
        enabled=(device.type == "cuda" and use_amp)
    )
    with autocast_ctx:
        logits_masked, geobl, dtw, acc, mask_pos, split_art = _forward_pass_with_metrics(
            model, input_seq_feature, location_labels, attention_mask, city_ids, device, metric_eval=metric_eval
        )
        if not mask_pos.any().item():
            return 0.0, 0.0, 0.0, 0.0, {}

        targets_masked = split_art["targets_masked"]
        buckets = split_art["buckets"]

        if hasattr(criterion, "forward_pre_split"):
            out = criterion.forward_pre_split(buckets)
        elif getattr(criterion, "expects_masked_inputs", False):
            out = criterion(logits_masked, targets_masked, mask_pos=mask_pos)
        else:
            out = criterion(logits_masked, targets_masked)

        if isinstance(out, tuple):
            loss, aux = out[0], (out[1] if len(out) > 1 and isinstance(out[1], dict) else {})
        else:
            loss, aux = out, {}

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()

    return float(loss.item()), float(geobl), float(dtw), float(acc), aux


@torch.no_grad()
def val_step(model, criterion,
             input_seq_feature, location_labels, attention_mask, city_ids,
             device, amp_dtype, metric_eval: bool = True):
    model.eval()
    autocast_ctx = torch.amp.autocast(device_type=device.type, dtype=amp_dtype,
                                      enabled=(device.type == "cuda"))
    with autocast_ctx:
        logits_masked, geobl, dtw, acc, mask_pos, split_art = _forward_pass_with_metrics(
            model, input_seq_feature, location_labels, attention_mask, city_ids, device, metric_eval=metric_eval
        )
        if not mask_pos.any().item():
            return 0.0, 0.0, 0.0, float(acc), {}

        targets_masked = split_art["targets_masked"]
        buckets = split_art["buckets"]

        if hasattr(criterion, "forward_pre_split"):
            out = criterion.forward_pre_split(buckets)
        elif getattr(criterion, "expects_masked_inputs", False):
            out = criterion(logits_masked, targets_masked, mask_pos=mask_pos)
        else:
            out = criterion(logits_masked, targets_masked)

        if isinstance(out, tuple):
            loss, aux = out[0], (out[1] if len(out) > 1 and isinstance(out[1], dict) else {})
        else:
            loss, aux = out, {}

    return float(loss.item()), float(geobl), float(dtw), float(acc), aux



# ----------------------------------------------------------------
# Epoch loop
# ----------------------------------------------------------------
def train_model(model, optimizer, train_loader, val_loader, num_epochs, device, run_dir, cfg,
                loss_name="ce", loss_kwargs=None):
    if loss_kwargs is None:
        loss_kwargs = {}

    # Build loss
    criterion = build_criterion(loss_name, **loss_kwargs)

    # ---- AMP (always float16) ----
    use_amp = (device.type == "cuda") and bool(cfg.get("use_amp", True))
    amp_dtype = torch.float16
    scaler_enabled = use_amp

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        # Recommended path for torch >= 2.1
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:
        # Fallback for older versions
        scaler = CudaGradScaler(enabled=scaler_enabled)

    print(f"[AMP] enabled={use_amp}, dtype={amp_dtype}, grad_scaler_enabled={scaler_enabled}")

    # ---- Scheduler ----
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
    os.makedirs(run_dir, exist_ok=True)

    wandb.watch(model, log="all", log_freq=100)

    for epoch in range(num_epochs):
        # Update α per epoch if using combo loss
        if loss_name == "combo":
            combo_cfg = cfg.get("combo", {})
            alpha = alpha_sched(epoch, combo_cfg)
            if hasattr(criterion, "set_alpha"):
                criterion.set_alpha(alpha)

        # === Training phase ===
        total_train_loss = total_train_geobleu = total_train_dtw = total_train_acc = 0.0
        num_train_batches = len(train_loader)
        aux_train_sum = defaultdict(float)
        aux_train_cnt = 0

        train_prog = tqdm(enumerate(train_loader), total=num_train_batches,
                          desc=f"[Train] Epoch {epoch+1}/{num_epochs}", ncols=100)

        for _, batch in train_prog:
            # collate returns: feats, labels_mlm, attention_mask, uids, city_ids
            feats, labels, attn_mask, _, city_ids = batch
            out = train_step(
                model, optimizer, criterion,
                feats, labels, attn_mask, city_ids,
                device, scaler, amp_dtype
            )

            if isinstance(out, tuple) and len(out) >= 4:
                loss, geobleu, dtw, acc = out[:4]
                aux = out[4] if len(out) > 4 else {}
            else:
                loss, geobleu, dtw, acc, aux = out, 0.0, 0.0, 0.0, {}

            total_train_loss += float(loss)
            total_train_geobleu += float(geobleu)
            total_train_dtw += float(dtw)
            total_train_acc += float(acc)

            if isinstance(aux, dict):
                for k, v in aux.items():
                    aux_train_sum[f"train/{k}"] += float(v)
                aux_train_cnt += 1

            scheduler.step()
            train_prog.set_postfix(loss=f"{float(loss):.4f}")

        avg_train_loss = total_train_loss / max(1, num_train_batches)
        avg_train_geobl = total_train_geobleu / max(1, num_train_batches)
        avg_train_dtw = total_train_dtw / max(1, num_train_batches)
        avg_train_acc = total_train_acc / max(1, num_train_batches)
        avg_train_aux = {k: v / max(1, aux_train_cnt) for k, v in aux_train_sum.items()}

        print(f"[Train] Loss: {avg_train_loss:.4f}, GEO-BLEU: {avg_train_geobl:.4f}, "
              f"DTW: {avg_train_dtw:.2f}, Acc: {avg_train_acc*100:.2f}%")

        # === Validation phase ===
        total_val_loss = total_val_geobleu = total_val_dtw = total_val_acc = 0.0
        num_val_batches = len(val_loader)
        aux_val_sum = defaultdict(float)
        aux_val_cnt = 0

        val_prog = tqdm(enumerate(val_loader), total=num_val_batches,
                        desc=f"[Validation] Epoch {epoch+1}/{num_epochs}", ncols=100)

        for _, batch in val_prog:
            feats, labels, attn_mask, uids, city_ids = batch
            out = val_step(
                model, criterion,
                feats, labels, attn_mask, city_ids,
                device, amp_dtype, metric_eval=True
            )

            if isinstance(out, tuple) and len(out) >= 4:
                loss, geobleu, dtw, acc = out[:4]
                aux = out[4] if len(out) > 4 else {}
            else:
                loss, geobleu, dtw, acc, aux = out, 0.0, 0.0, 0.0, {}

            total_val_loss += float(loss)
            total_val_geobleu += float(geobleu)
            total_val_dtw += float(dtw)
            total_val_acc += float(acc)

            if isinstance(aux, dict):
                for k, v in aux.items():
                    aux_val_sum[f"val/{k}"] += float(v)
                aux_val_cnt += 1

            val_prog.set_postfix(loss=f"{float(loss):.4f}", acc=f"{float(acc):.3f}")

        avg_val_loss = total_val_loss / max(1, num_val_batches)
        avg_val_geobl = total_val_geobleu / max(1, num_val_batches)
        avg_val_dtw = total_val_dtw / max(1, num_val_batches)
        avg_val_acc = total_val_acc / max(1, num_val_batches)
        avg_val_aux = {k: v / max(1, aux_val_cnt) for k, v in aux_val_sum.items()}

        print(f"[Validation] Loss: {avg_val_loss:.4f}, GEO-BLEU: {avg_val_geobl:.4f}, "
              f"DTW: {avg_val_dtw:.2f}, Acc: {avg_val_acc*100:.2f}%")

        # === Logging ===
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

        if loss_name == "combo" and hasattr(criterion, "alpha"):
            base_train_log["train/alpha"] = criterion.alpha
            base_val_log["val/alpha"] = criterion.alpha

        try:
            wandb.log({**base_train_log, **base_val_log, **avg_train_aux, **avg_val_aux})
        except Exception:
            pass

        log_epoch_metrics(
            epoch=epoch,
            train_metrics={'loss': avg_train_loss, 'geobleu': avg_train_geobl,
                           'dtw': avg_train_dtw, 'acc': avg_train_acc, **avg_train_aux},
            val_metrics={'loss': avg_val_loss, 'geobleu': avg_val_geobl,
                         'dtw': avg_val_dtw, 'acc': avg_val_acc, **avg_val_aux},
            log_path=os.path.join(run_dir, 'train_log.txt')
        )

        # === Feature codebook visualization (optional) ===
        for fname in ["day", "time", "location"]:
            emb_np, ids_np = compute_feature_codebook(model, fname, device)
            if emb_np is None:
                continue
            log_codebook_tsne(fname, emb_np, ids_np, epoch)
            log_codebook_heatmap(fname, emb_np, epoch)

        emb_np, ids_np = compute_feature_codebook(model, "delta", device, delta_max=2000, delta_steps=256)
        if emb_np is not None:
            log_codebook_tsne("delta", emb_np, ids_np, epoch)

        # === Early stopping based on GEO-BLEU ===
        metric = avg_val_geobl if math.isfinite(avg_val_geobl) else float("-inf")
        if metric > best_val_geobleu:
            best_val_geobleu = metric
            no_improve_count = 0
            best_model_path = os.path.join(run_dir, 'bert_best.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"[Saved] Best model (by GEO-BLEU) → {best_model_path} — val/geobleu={best_val_geobleu:.6f}")
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            print(f"[EarlyStopping] No val/GEO-BLEU improvement for {patience} evals. Stop at epoch {epoch+1}.")
            break

    # Final save
    final_model_path = os.path.join(run_dir, 'bert_final.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"[Saved] Final model → {final_model_path}")

    return best_model_path, final_model_path