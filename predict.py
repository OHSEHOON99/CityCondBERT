import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

from geobleu import calc_geobleu_single, calc_dtw_single
from utils import safe_ids_to_xy
from config import ID2CITY



def predict(
    model,
    test_loader,
    device,
    output_dir="./results",
    W=200,
    save_csv=True
):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    # --- 새 경로 (요구사항) ---
    pred_dir   = os.path.join(output_dir, "test")
    metric_dir = os.path.join(output_dir, "metric")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(metric_dir, exist_ok=True)

    # 버퍼: 도시별 토큰 / UID 묶음
    tok_buf_per_city = defaultdict(list)
    uid_rows = []

    # 전역 micro 정확도
    total_tokens = 0
    total_correct = 0

    # 도시별 micro 정확도 누적
    city_tokens  = defaultdict(int)
    city_correct = defaultdict(int)

    # --- ID -> 도시 문자열 태그 ---
    def city_tag(cid: int) -> str:
        try:
            return str(ID2CITY.get(int(cid), str(cid)))
        except Exception:
            return str(cid)

    def compute_metrics_np(d_i, t_i, p_i, y_i):
        if p_i.size == 0:
            return None
        pred_xy = safe_ids_to_xy(p_i, W)
        true_xy = safe_ids_to_xy(y_i, W)
        pred_seq = list(zip(d_i, t_i, pred_xy[:, 0], pred_xy[:, 1]))
        true_seq = list(zip(d_i, t_i, true_xy[:, 0], true_xy[:, 1]))
        g = calc_geobleu_single(pred_seq, true_seq)
        d = calc_dtw_single(pred_seq, true_seq)
        acc = float((p_i == y_i).mean())
        return g, d, acc

    with torch.inference_mode():
        for batch in tqdm(test_loader, desc="[Predict-MLM]", ncols=100):
            if len(batch) == 5:
                feats, labels, attn_mask, uids, city_ids = batch
            else:
                feats, labels, attn_mask, uids = batch
                city_ids = torch.ones_like(uids, dtype=torch.long)

            feats     = feats.to(device, non_blocking=True)
            labels    = labels.to(device, non_blocking=True)
            attn_mask = attn_mask.to(device, non_blocking=True)
            uids      = uids.to(device, non_blocking=True)
            city_ids  = city_ids.to(device, non_blocking=True)

            logits_masked, mask_pos = model(feats, attn_mask, labels=labels, city_ids=city_ids)
            if not mask_pos.any().item():
                continue

            # --- 배치 평탄화 ---
            b_idx, t_idx = torch.where(mask_pos)
            true_all = labels[b_idx, t_idx].detach().cpu().numpy()
            pred_all = torch.argmax(logits_masked, dim=-1).detach().cpu().numpy()

            d_all = (feats[b_idx, t_idx, 0] - 1).detach().cpu().numpy().astype(np.int32)
            t_all = (feats[b_idx, t_idx, 1] - 1).detach().cpu().numpy().astype(np.int32)
            uid_all  = uids[b_idx].detach().cpu().numpy()
            city_all = city_ids[b_idx].detach().cpu().numpy()

            # --- 전역 micro ---
            total_tokens  += true_all.size
            batch_correct = int((pred_all == true_all).sum())
            total_correct += batch_correct

            # --- 도시별 micro 누적 ---
            if true_all.size:
                sort_c = np.argsort(city_all, kind="mergesort")
                c_sorted = city_all[sort_c]
                eq_sorted = (pred_all[sort_c] == true_all[sort_c]).astype(np.int32)
                uniq_c, start_c, counts_c = np.unique(c_sorted, return_index=True, return_counts=True)
                for cid, s, c in zip(uniq_c, start_c, counts_c):
                    city_tokens[int(cid)]  += int(c)
                    city_correct[int(cid)] += int(eq_sorted[s:s+c].sum())

            # --- UID 그룹핑 ---
            sort_idx = np.argsort(uid_all, kind="mergesort")
            uid_s = uid_all[sort_idx]
            d_s   = d_all[sort_idx]
            t_s   = t_all[sort_idx]
            p_s   = pred_all[sort_idx]
            y_s   = true_all[sort_idx]
            c_s   = city_all[sort_idx]

            uniq_uid, start_idx, counts = np.unique(uid_s, return_index=True, return_counts=True)
            city_per_uid = c_s[start_idx]

            results = Parallel(n_jobs=8)(
                delayed(compute_metrics_np)(d_s[s:s+c], t_s[s:s+c], p_s[s:s+c], y_s[s:s+c])
                for s, c in zip(start_idx, counts)
            )

            for u, cty, c, res in zip(uniq_uid.astype(np.int64), city_per_uid.astype(np.int64), counts, results):
                if res is None:
                    continue
                g, d, acc = res
                row = {
                    "uid": int(u),
                    "city_id": int(cty),
                    "seq_len": int(c),
                    "geobleu": float(g),
                    "dtw": float(d),
                    "acc": float(acc),
                    "city": city_tag(int(cty)),
                }
                uid_rows.append(row)

            # --- 토큰 CSV 버퍼에 저장 ---
            if save_csv:
                pred_xy = safe_ids_to_xy(pred_all, W).astype(np.int32, copy=False)
                true_xy = safe_ids_to_xy(true_all, W).astype(np.int32, copy=False)
                df_tok = pd.DataFrame({
                    "uid": uid_all.astype(np.int64, copy=False),
                    "city_id": city_all.astype(np.int32, copy=False),
                    "d": d_all,
                    "t": t_all,
                    "true_x": true_xy[:, 0],
                    "true_y": true_xy[:, 1],
                    "predict_x": pred_xy[:, 0],
                    "predict_y": pred_xy[:, 1],
                })
                for cid, gdf in df_tok.groupby("city_id", sort=False):
                    tag = city_tag(int(cid))
                    tok_buf_per_city[tag].append(gdf)

    # --- UID 메트릭: 도시별 파일 저장 ---
    df_uid = pd.DataFrame(uid_rows) if uid_rows else pd.DataFrame(
        columns=["uid","city_id","seq_len","geobleu","dtw","acc","city"]
    )
    if save_csv and not df_uid.empty:
        for tag, gdf in df_uid.groupby("city", sort=False):
            gdf.drop(columns=["city"], errors="ignore").to_csv(
                os.path.join(metric_dir, f"city_{tag}.csv"), index=False
            )

    # --- 토큰 CSV: 도시별 저장 ---
    if save_csv:
        for tag, buflist in tok_buf_per_city.items():
            df_all = pd.concat(buflist, ignore_index=True) if buflist else pd.DataFrame()
            df_all.to_csv(os.path.join(pred_dir, f"city_{tag}.csv"), index=False)

    # --- summary.txt ---
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        avg_geobleu = float(df_uid["geobleu"].mean()) if not df_uid.empty else 0.0
        avg_dtw     = float(df_uid["dtw"].mean())     if not df_uid.empty else 0.0
        avg_acc     = float(total_correct / max(1, total_tokens))

        f.write("[Global]\n")
        f.write(f"tokens={total_tokens}, acc_micro={avg_acc:.6f}, geobleu_macro_uid={avg_geobleu:.6f}, dtw_macro_uid={avg_dtw:.6f}\n\n")

        if not df_uid.empty or city_tokens:
            f.write("[Per City]\n")
            seen_cids = set(int(c) for c in df_uid["city_id"].unique()) if not df_uid.empty else set()
            seen_cids |= set(city_tokens.keys())
            for cid in sorted(seen_cids):
                tag = city_tag(int(cid))
                n_tok = city_tokens.get(int(cid), 0)
                n_cor = city_correct.get(int(cid), 0)
                acc_micro = (n_cor / n_tok) if n_tok > 0 else 0.0

                if not df_uid.empty:
                    sub = df_uid[df_uid["city_id"] == cid]
                    geobleu_macro = float(sub["geobleu"].mean()) if not sub.empty else 0.0
                    dtw_macro     = float(sub["dtw"].mean())     if not sub.empty else 0.0
                    acc_macro     = float(sub["acc"].mean())     if not sub.empty else 0.0
                    n_uids        = int(sub["uid"].nunique())
                    n_tokens_uid  = int(sub["seq_len"].sum())
                else:
                    geobleu_macro = dtw_macro = acc_macro = 0.0
                    n_uids = n_tokens_uid = 0

                f.write(
                    f"city={tag} | tokens={n_tok}, acc_micro={acc_micro:.6f}, "
                    f"uids={n_uids}, tokens_uid_sum={n_tokens_uid}, "
                    f"geobleu_macro_uid={geobleu_macro:.6f}, dtw_macro_uid={dtw_macro:.6f}, acc_macro_uid={acc_macro:.6f}\n"
                )
    # ===== WandB logging (per-city + global) =====
    try:
        import wandb
        # global
        wandb.log({
            "test/geobleu": float(df_uid["geobleu"].mean()) if not df_uid.empty else 0.0,
            "test/dtw": float(df_uid["dtw"].mean()) if not df_uid.empty else 0.0,
            "test/acc": float(total_correct / max(1, total_tokens))
        })

        # per-city
        if not df_uid.empty or city_tokens:
            seen_cids = set(int(c) for c in df_uid["city_id"].unique()) if not df_uid.empty else set()
            seen_cids |= set(city_tokens.keys())
            for cid in sorted(seen_cids):
                tag = city_tag(int(cid))
                n_tok = city_tokens.get(int(cid), 0)
                n_cor = city_correct.get(int(cid), 0)
                acc_micro = (n_cor / n_tok) if n_tok > 0 else 0.0

                if not df_uid.empty:
                    sub = df_uid[df_uid["city_id"] == cid]
                    geobleu_macro = float(sub["geobleu"].mean()) if not sub.empty else 0.0
                    dtw_macro     = float(sub["dtw"].mean())     if not sub.empty else 0.0
                    acc_macro     = float(sub["acc"].mean())     if not sub.empty else 0.0
                    n_uids        = int(sub["uid"].nunique())
                else:
                    geobleu_macro = dtw_macro = acc_macro = 0.0
                    n_uids = 0

                wandb.log({
                    f"{tag}/geobleu": geobleu_macro,
                    f"{tag}/dtw": dtw_macro,
                    f"{tag}/acc": acc_macro
                })
    except Exception:
        pass

    return avg_geobleu, avg_dtw, avg_acc


# ---------------------------
# Predict masked UID segments (submission-style) -> save under results/mask
# ---------------------------
@torch.no_grad()
def predict_masked_uid(
    model,
    mask_loader,
    device,
    city="A",                 # "A" | "B" | "C" | "D" | "ALL"
    output_dir="./results",
    W=200,
    team_name="SCSI"
):
    """
    Save files under: {output_dir}/mask
    - city == "ALL": write {team}_cityA_humob25.csv, ..., {team}_cityD_humob25.csv
    - else          : write {team}_city{city}_humob25.csv
    """

    model.eval()
    mask_dir = os.path.join(output_dir, "mask")
    os.makedirs(mask_dir, exist_ok=True)

    if mask_loader is None:
        print("[predict_masked_uid] mask_loader is None. Nothing to do.")
        return pd.DataFrame(columns=["uid", "d", "t", "x", "y"])

    city_upper = str(city).upper()

    # fallback id->letter (dataset이 1~4면 이 매핑 사용)
    fallback_id2city = {1: "A", 2: "B", 3: "C", 4: "D"}

    # city == ALL 이면 도시별 버킷으로 누적
    per_city_rows = defaultdict(list)
    single_city_rows = []  # 단일 도시 모드에서만 사용

    pbar = tqdm(mask_loader, desc=f"[Predict-999|city={city_upper}]", ncols=100)
    for batch in pbar:
        if len(batch) == 5:
            feats_pad, labels_mlm, attention_mask, uids, city_ids = batch
        else:
            feats_pad, labels_mlm, attention_mask, uids = batch
            city_ids = torch.full((uids.shape[0],), 1, dtype=torch.long)

        feats_pad = feats_pad.to(device, non_blocking=True)
        labels_mlm = labels_mlm.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        uids = uids.to(device, non_blocking=True)
        city_ids = city_ids.to(device, non_blocking=True)

        logits_masked, mask_pos = model(
            feats_pad, attention_mask, labels=labels_mlm, city_ids=city_ids
        )  # logits_masked: (N_mask,V), mask_pos: (B,L)

        if not mask_pos.any().item():
            continue

        B, L = mask_pos.shape
        pred_ids_all = torch.argmax(logits_masked, dim=-1).long()  # (N_mask,)

        counts = mask_pos.sum(dim=1).tolist()
        offsets = np.cumsum([0] + counts)

        for b in range(B):
            n = counts[b]
            if n == 0:
                continue

            s, e = offsets[b], offsets[b + 1]
            pos_b = mask_pos[b].nonzero(as_tuple=True)[0]  # (n,)

            d_b = (feats_pad[b, pos_b, 0] - 1).detach().cpu().numpy().astype(np.int64)
            t_b = (feats_pad[b, pos_b, 1] - 1).detach().cpu().numpy().astype(np.int64)
            uid_b = int(uids[b].item())
            pred_ids_b = pred_ids_all[s:e].detach().cpu().numpy()
            pred_xy_b = safe_ids_to_xy(pred_ids_b, W).astype(np.int64)

            # 이 샘플의 city tag 결정
            cid_b = int(city_ids[b].item())
            tag_b = fallback_id2city.get(cid_b, str(cid_b))  # ex) 1->"A"

            # rows 작성
            for (dd, tt), (px, py) in zip(zip(d_b, t_b), pred_xy_b):
                row = {"uid": uid_b, "d": int(dd), "t": int(tt), "x": int(px), "y": int(py)}
                if city_upper == "ALL":
                    per_city_rows[tag_b].append(row)
                else:
                    single_city_rows.append(row)

    # === 저장 ===
    if city_upper == "ALL":
        dfs = {}
        for tag, rows in sorted(per_city_rows.items()):
            df = pd.DataFrame(rows, columns=["uid", "d", "t", "x", "y"])
            if not df.empty:
                df = df.sort_values(["uid", "d", "t"]).reset_index(drop=True)
            dfs[tag] = df
            fname = f"{team_name}_city{tag}_humob25.csv"
            fpath = os.path.join(mask_dir, fname)
            df.to_csv(fpath, index=False)
    else:
        df_pred = pd.DataFrame(single_city_rows, columns=["uid", "d", "t", "x", "y"])
        if not df_pred.empty:
            df_pred = df_pred.sort_values(["uid", "d", "t"]).reset_index(drop=True)
        fname = f"{team_name}_city{city_upper}_humob25.csv"
        fpath = os.path.join(mask_dir, fname)
        df_pred.to_csv(fpath, index=False)
