import os

import geobleu
import numpy as np
import pandas as pd
import torch

from joblib import Parallel, delayed
from tqdm import tqdm

from utils import *



def evaluate_sequence(pred_xy, true_xy, day_seq, time_seq):
    """
    GEO-BLEU, DTW, Accuracy 계산 (day_seq, time_seq는 필수)
    """
    pred_seq = [(d, t, *p) for d, t, p in zip(day_seq, time_seq, pred_xy)]
    true_seq = [(d, t, *t_) for d, t, t_ in zip(day_seq, time_seq, true_xy)]

    geobleu_score = geobleu.calc_geobleu_single(pred_seq, true_seq)
    dtw_dist = geobleu.calc_dtw_single(pred_seq, true_seq)
    acc = np.mean([p == t for p, t in zip(pred_xy, true_xy)])

    return geobleu_score, dtw_dist, acc


def process_trajectory(trajectory_id, uid, day_seq, time_seq, pred_locs, true_locs):
    pred_xy = id_to_xy(pred_locs)
    true_xy = id_to_xy(true_locs)

    geobleu_score, dtw_dist, accuracy = evaluate_sequence(pred_xy, true_xy, day_seq, time_seq)

    rows = []
    for j in range(len(true_locs)):
        d = int(day_seq[j])
        t = int(time_seq[j])
        true_x, true_y = true_xy[j]
        pred_x, pred_y = pred_xy[j]

        rows.append({
            "trajectory_id": trajectory_id,
            "uid": uid,
            "d": d,
            "t": t,
            "true_x": true_x,
            "true_y": true_y,
            "predict_x": pred_x,
            "predict_y": pred_y,
            "seq_geobleu": geobleu_score,
            "seq_dtw": dtw_dist,
            "seq_accuracy": accuracy
        })

    return rows, geobleu_score, dtw_dist, accuracy


def predict(model, test_loader, device, output_dir="./results", save_name="test"):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    input_rows = []
    output_rows = []

    total_geobleu = 0.0
    total_dtw = 0.0
    total_accuracy = 0.0
    total_tokens = 0

    trajectory_id_counter = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="[Predict]", ncols=100):
            input_seq_feature, historical_locations, predict_seq_feature, future_locations = batch[:4]
            uid_seq = batch[4]

            input_seq_feature = input_seq_feature.to(device)
            historical_locations = historical_locations.to(device)
            predict_seq_feature = predict_seq_feature.to(device)
            future_locations = future_locations.to(device)

            logits = model(input_seq_feature, historical_locations, predict_seq_feature)
            preds = torch.argmax(logits, dim=-1)

            B, _ = preds.shape
            uid_seq = uid_seq.cpu().numpy()
            pred_locs = preds.cpu().numpy()
            true_locs = future_locations.cpu().numpy()
            day_seq = predict_seq_feature[:, :, 0].cpu().numpy()
            time_seq = predict_seq_feature[:, :, 1].cpu().numpy()

            results = Parallel(n_jobs=os.cpu_count()//2)(
                delayed(process_trajectory)(
                    trajectory_id_counter + i,
                    int(uid_seq[i]),
                    day_seq[i],
                    time_seq[i],
                    pred_locs[i],
                    true_locs[i]
                ) for i in range(B)
            )

            for i, (rows, geobleu_score, dtw_dist, accuracy) in enumerate(results):
                output_rows.extend(rows)
                total_geobleu += geobleu_score
                total_dtw += dtw_dist
                num_tokens = len(true_locs[i])
                total_accuracy += accuracy * num_tokens
                total_tokens += num_tokens

            # input row 기록은 병렬 처리할 필요 없음 (정답 좌표 기록용)
            for i in range(B):
                uid = int(uid_seq[i])
                hist_day_seq = input_seq_feature[i, :, 1].cpu().numpy()
                hist_time_seq = input_seq_feature[i, :, 2].cpu().numpy()
                hist_loc_ids = historical_locations[i].cpu().numpy()
                hist_xy = id_to_xy(hist_loc_ids)

                input_rows.extend([
                    {
                        "trajectory_id": trajectory_id_counter + i,
                        "uid": uid,
                        "d": int(d),
                        "t": int(t),
                        "true_x": int(x),
                        "true_y": int(y)
                    }
                    for d, t, (x, y) in zip(hist_day_seq, hist_time_seq, hist_xy)
                ])

            trajectory_id_counter += B

    # 평균 지표 출력
    avg_geobleu = total_geobleu / trajectory_id_counter
    avg_dtw = total_dtw / trajectory_id_counter
    avg_accuracy = total_accuracy / total_tokens

    for row in output_rows:
        row["avg_geobleu"] = avg_geobleu
        row["avg_dtw"] = avg_dtw
        row["avg_accuracy"] = avg_accuracy

    pd.DataFrame(input_rows).to_csv(f"{output_dir}/{save_name}_input.csv", index=False)
    pd.DataFrame(output_rows).to_csv(f"{output_dir}/{save_name}_output.csv", index=False)

    return avg_geobleu, avg_dtw, avg_accuracy


########################################################################################################################


def compute_dow_is_weekday(day, city='A'):
    if city in ['C', 'D']:
        dow = (day - 60 + 2) % 7 if day >= 60 else day % 7
    else:
        dow = day % 7
    is_weekday = 1 if 2 <= dow <= 6 else 0
    return dow, is_weekday


def generate_sequence(df, grid_width=200):
    seq_x, seq_y = [], []
    df = df.sort_values(['d', 't']).reset_index(drop=True)
    prev_d, prev_t = df.loc[0, 'd'], df.loc[0, 't']
    for _, row in df.iterrows():
        d, t = row['d'], row['t']
        days = d - 1
        prev_days = prev_d - 1
        delta_t = (days * 48 + t) - (prev_days * 48 + prev_t)
        dow = days % 7
        is_weekday = 1 if 2 <= dow <= 6 else 0
        label = xy_to_id(row['x'], row['y'], grid_width) if row['x'] != 999 else 0
        seq_x.append([d, t, dow, is_weekday, delta_t])
        seq_y.append(label)
        prev_d, prev_t = d, t
    return np.array(seq_x), np.array(seq_y)


def pad_predict_sequence(df_masked, predict_seq_len):
    df_masked['is_dummy'] = False
    remain = len(df_masked) % predict_seq_len
    if remain > 0:
        dummy_rows = df_masked.iloc[[-1] * (predict_seq_len - remain)].copy()
        dummy_rows['is_dummy'] = True
        df_masked = pd.concat([df_masked, dummy_rows], ignore_index=True)
    return df_masked


def predict_masked_uid(model, df_all, config, device, city='A', output_path=None):
    input_seq_len = config.get("input_seq_length", 240)
    predict_seq_len = config.get("predict_seq_length", 48)

    grid_width = 200
    predicted_rows = []

    for uid in tqdm(df_all['uid'].unique(), desc="Predicting masked UIDs", unit="uid"):
        df_uid = df_all[df_all['uid'] == uid].copy().sort_values(['d', 't'])
        df_input = df_uid[df_uid['d'] <= 60]
        df_masked = df_uid[(df_uid['d'] > 60) & (df_uid['x'] == 999)].copy()
        if df_masked.empty:
            continue

        df_masked = pad_predict_sequence(df_masked, predict_seq_len)
        input_x_np, input_y_np = generate_sequence(df_input, grid_width)

        input_x_np = input_x_np[-input_seq_len:]
        input_y_np = input_y_np[-input_seq_len:]

        input_seq_feature = torch.tensor(input_x_np, dtype=torch.long, device=device).unsqueeze(0)
        hist_locs = torch.tensor(input_y_np, dtype=torch.long, device=device).unsqueeze(0)

        prev_d, prev_t = df_input.iloc[-1]['d'], df_input.iloc[-1]['t']
        prev_d = prev_d - 1

        for start in range(0, len(df_masked), predict_seq_len):
            chunk = df_masked.iloc[start:start + predict_seq_len]
            predict_feats = []
            for _, row in chunk.iterrows():
                d, t = row['d'], row['t']
                days = d - 1
                prev_days = prev_d - 1
                delta_t = (days * 48 + t) - (prev_days * 48 + prev_t)
                dow, is_weekday = compute_dow_is_weekday(days, city)
                predict_feats.append([days, t, dow, is_weekday, delta_t])
                prev_d, prev_t = d, t

            predict_feat_tensor = torch.tensor([predict_feats], dtype=torch.long, device=device)

            # print("input_seq_feature:", input_seq_feature.min(), input_seq_feature.max()) ############################
            # print("hist_locs:", hist_locs.min(), hist_locs.max()) ########################
            # print("predict_feat_tensor:", predict_feat_tensor.min(), predict_feat_tensor.max()) ########################

            with torch.no_grad():
                # print(f"predict_feat_tensor : {predict_feat_tensor}") ##############
                logits = model(input_seq_feature, hist_locs, predict_feat_tensor)
                pred_labels = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()

            pred_coords = id_to_xy(pred_labels, grid_width=grid_width)
            # print(f"pred_labels : {pred_labels}") ###################
            # print(f"pred_coords : {pred_coords}") ###################

            for idx, (x, y) in enumerate(pred_coords):
                row = chunk.iloc[idx]
                predicted_rows.append({
                    'uid': uid,
                    'd': int(row['d']),  # 복원
                    't': int(row['t']),
                    'predict_x': x,
                    'predict_y': y,
                    'is_dummy': bool(row.get('is_dummy', False))
                })

            new_input_seq = torch.tensor(predict_feats, dtype=torch.long, device=device).unsqueeze(0)
            new_hist = torch.tensor(pred_labels, dtype=torch.long, device=device).unsqueeze(0)

            input_seq_feature = torch.cat([input_seq_feature[:, predict_seq_len:], new_input_seq], dim=1)
            hist_locs = torch.cat([hist_locs[:, predict_seq_len:], new_hist], dim=1)

    df_pred = pd.DataFrame(predicted_rows)

    if output_path:
        df_pred.to_csv(output_path, index=False)
        print(f"Masked uid predictions saved to: {output_path}")