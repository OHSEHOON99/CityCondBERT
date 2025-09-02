# Standard library
import glob
import multiprocessing
import os
import re

# Third-party
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm

# Local
from config import PAD_VALUE, MASK_TOKEN_VALUE, IGNORE_INDEX, CITY2ID, ID2CITY



# ----------------------------
# Collate function
# ----------------------------
def collate_fn(batch):
    """
    batch: list of dicts from MobilitySequenceDataset.__getitem__()
      - input_seq_feature: (T, 6)
      - input_seq_label:   (T,)
      - uid: int
      - city_id: int (1..4)
    Returns:
      feats_pad[B, L, 6], labels_mlm[B, L], attention_mask[B, L], uids[B], city_ids[B]
    """
    feats = [item["input_seq_feature"] for item in batch]
    labs  = [item["input_seq_label"]   for item in batch]
    uids  = [item["uid"]               for item in batch]
    cids  = [item["city_id"]           for item in batch]

    feats_pad  = pad_sequence(feats, batch_first=True, padding_value=PAD_VALUE)     # (B, L, 6)
    labels_pad = pad_sequence(labs,  batch_first=True, padding_value=IGNORE_INDEX)  # (B, L)

    attention_mask = (feats_pad[..., 0] != PAD_VALUE).long()  # use day_idx column
    masked_pos = (feats_pad[..., 5] == MASK_TOKEN_VALUE) & (attention_mask == 1)  # loc_input_idx
    labels_mlm = torch.where(masked_pos, labels_pad, torch.full_like(labels_pad, IGNORE_INDEX))

    return (
        feats_pad,
        labels_mlm,
        attention_mask,
        torch.tensor(uids, dtype=torch.long),
        torch.tensor(cids, dtype=torch.long),
    )


# ----------------------------
# Dataset
# ----------------------------
class MobilitySequenceDataset(Dataset):
    """
    Stores per-UID sequences.
      - feature: (T, 6)  [day_idx, time_idx, dow_idx, weekday_idx, delta, loc_input_idx]
      - label  : (T,)    [cell_label for fixed_days or IGNORE_INDEX/0 for x999]
      - city_id: {1:A,2:B,3:C,4:D}
    """
    def __init__(self, city_ids_per_uid, feats_list, labels_list, uid_list, mask_days=15):
        self.mask_days = mask_days
        self.input_seq_feature = [torch.tensor(seq, dtype=torch.long) for seq in feats_list]
        self.input_seq_label   = [torch.tensor(labs, dtype=torch.long) for labs in labels_list]
        self.uid_seq  = torch.tensor(uid_list, dtype=torch.long)
        self.city_ids = torch.tensor([city_ids_per_uid[int(uid)] for uid in uid_list], dtype=torch.long)

    def __len__(self):
        return len(self.input_seq_feature)

    def __getitem__(self, idx):
        return {
            "input_seq_feature": self.input_seq_feature[idx],
            "input_seq_label":   self.input_seq_label[idx],
            "uid":   int(self.uid_seq[idx]),
            "city_id": int(self.city_ids[idx]),
        }


# ----------------------------
# Sequence builder (parallel)
# ----------------------------
def build_sequences_parallel(
    raw_uid_np_list,
    *,
    city='A',
    mask_policy="fixed_days",   # "fixed_days" | "x999"
    mask_days=(60, 74),
    W=200,
):
    """
    Parallel sequence builder.
      Common:
        - compute delta (Δt), dow/weekday (+ city C/D correction), +1 shift for categorical indices
      mask_policy:
        * "fixed_days": mask days0 in [s,e] (training/eval)
            - feature.loc_input_idx: labels+1, masked range -> MASK_TOKEN_VALUE
            - label_sequence: full cell_label (0..V-1)
        * "x999": mask positions where x==999 (inference)
            - feature.loc_input_idx: known -> cell_id+1, masked -> MASK_TOKEN_VALUE
            - label_sequence: masked -> 0, others -> IGNORE_INDEX
    Input row format:
        - fixed_days: user_array[:, [uid, d, t, x, y, cell_label]]
        - x999     : user_array[:, [uid, d, t, x, y]]
    Returns:
        input_features_list: List[(T,6)]
        input_labels_list  : List[(T,)]
        uid_list           : List[int]
    """
    city = str(city).upper()

    def process_user(user_array):
        if user_array.shape[0] == 0:
            return None
        uid = int(user_array[0, 0])

        d = user_array[:, 1].astype(np.int64)         # 1-based
        t = user_array[:, 2].astype(np.int64)         # 0..47
        x = user_array[:, 3].astype(np.int64)
        y = user_array[:, 4].astype(np.int64)

        days0    = d - 1
        abs_time = days0 * 48 + t
        delta_t  = np.diff(abs_time, prepend=abs_time[0]).astype(np.int64)

        if city in ['C', 'D']:
            dow_raw = np.where(days0 >= 60, (days0 - 60 + 2) % 7, days0 % 7)
        else:
            dow_raw = days0 % 7
        weekday_raw = ((dow_raw >= 2) & (dow_raw <= 6)).astype(np.int64)

        if mask_policy == "fixed_days":
            s, e = mask_days
            mask_idx = np.where((days0 >= s) & (days0 <= e))[0]
            if mask_idx.size == 0:
                return None
        elif mask_policy == "x999":
            mask_idx = np.where(x == 999)[0]
            if mask_idx.size == 0:
                return None
        else:
            raise ValueError(f"Unknown mask_policy: {mask_policy}")

        day_idx     = (days0 + 1).astype(np.int64)
        time_idx    = (t + 1).astype(np.int64)
        dow_idx     = (dow_raw + 1).astype(np.int64)
        weekday_idx = (weekday_raw + 1).astype(np.int64)

        if mask_policy == "fixed_days":
            labels = user_array[:, 5].astype(np.int64)   # 0..V-1
            loc_input_idx = (labels + 1).astype(np.int64)
            loc_input_idx[mask_idx] = MASK_TOKEN_VALUE
            label_sequence = labels
        else:
            loc_input_idx = np.empty_like(x, dtype=np.int64)
            known_mask = (x != 999)
            if known_mask.any():
                cell_id = (x[known_mask] - 1) * W + (y[known_mask] - 1)
                loc_input_idx[known_mask] = cell_id + 1
            loc_input_idx[~known_mask] = MASK_TOKEN_VALUE
            label_sequence = np.full_like(x, IGNORE_INDEX, dtype=np.int64)
            label_sequence[~known_mask] = 0

        feature_sequence = np.column_stack((day_idx, time_idx, dow_idx, weekday_idx, delta_t, loc_input_idx))
        return feature_sequence, label_sequence, uid

    results = Parallel(n_jobs=-1)(
        delayed(process_user)(user_array)
        for user_array in tqdm(raw_uid_np_list, desc=f"[build_sequences:{city}|{mask_policy}]")
    )

    input_features_list, input_labels_list, uid_list = [], [], []
    for res in results:
        if res is None:
            continue
        feats, labs, uid = res
        input_features_list.append(feats)
        input_labels_list.append(labs)
        uid_list.append(uid)

    return input_features_list, input_labels_list, uid_list


# ----------------------------
# Multi-city loader
# ----------------------------
def load_and_split_data(
    mobility_data_path,
    batch_size,
    split_ratio=(8, 1, 1),
    use_subsample=False,
    subsample_number=10000,
    random_seed=42,
    mask_days=(60, 74),
    *,
    mode="pretrain",          # 'pretrain' | 'transfer' | 'predict'
    city=None,                # pretrain: None/'all'/'A'..'D' | transfer: 'A'..'D' | predict: 'all'/'A'..'D'
    W=200                     # grid width for cell_label
):
    """
    Returns:
        (train_loader, val_loader, test_loader, mask_loader)

    Modes:
      - pretrain:
          * city in {None, 'all'} → use all; or a single city like 'A'
          * exclude x==999 rows entirely
          * build train/val/test; mask_loader=None
      - transfer:
          * city must be a single city in {'A','B','C','D'}
          * build train/val/test from non-missing (x!=999)
          * additionally build mask set from x==999 users → mask_loader
      - predict:
          * city='all' or single city
          * build train/val/test from non-missing (x!=999)  ← CHANGED
          * also build mask set from x==999 → mask_loader   ← unchanged
    """
    assert os.path.isdir(mobility_data_path), f"Not a directory: {mobility_data_path}"
    mode = str(mode).lower()
    assert mode in {"pretrain", "transfer", "predict"}, f"mode must be one of pretrain/transfer/predict, got {mode!r}"

    def _wants_city(file_city):
        c = None if city is None else str(city).strip().upper()
        fc = str(file_city).strip().upper()

        if mode == "pretrain":
            # ▶ pretrain에서도 city 인자를 반드시 사용
            #    - city=ALL → 전체 사용
            #    - city in {A,B,C,D} → 해당 도시만
            if c is None:
                raise ValueError("mode='pretrain' requires --city in {'A','B','C','D','all'}")
            if c == "ALL":
                return True
            if c not in {"A", "B", "C", "D"}:
                raise ValueError("city must be one of {'A','B','C','D','all'}")
            return fc == c

        elif mode == "transfer":
            # 기존 정책 유지: transfer는 특정 도시만 허용, ALL 금지
            if c is None or c == "ALL" or c not in {"A", "B", "C", "D"}:
                raise ValueError("mode='transfer' requires city to be one of {'A','B','C','D'} (not 'all')")
            return fc == c

        else:  # predict
            # predict는 ALL 허용 (기존 동작 유지)
            if c in (None, "", "ALL"):
                return True
            if c not in {"A", "B", "C", "D"}:
                raise ValueError("city must be one of {'A','B','C','D','all'} for predict")
            return fc == c

    csv_paths = sorted(glob.glob(os.path.join(mobility_data_path, "city_*_*.csv")))
    assert len(csv_paths) > 0, f"No CSV files found under {mobility_data_path} matching 'city_*_*.csv'"

    all_train_datasets, all_val_datasets, all_test_datasets = [], [], []
    all_mask_datasets = []

    for csv_path in csv_paths:
        data_filename = os.path.basename(csv_path)
        m = re.search(r'city_([A-Z])_', data_filename)
        if m is None:
            print(f"  -> Skip (cannot parse city from filename): {data_filename}")
            continue
        file_city = m.group(1)

        if not _wants_city(file_city):
            continue

        from config import CITY2ID
        city_id = CITY2ID.get(file_city)
        if city_id is None:
            print(f"  -> Skip (unknown city name): {file_city}")
            continue

        mobility_df_all = pd.read_csv(csv_path)
        print(f"[{data_filename}] Loaded. Total records: {len(mobility_df_all)}")

        # A) Build train/val/test from non-missing users
        if mode in {"pretrain", "transfer", "predict"}:
            uids_with_missing_x = mobility_df_all[mobility_df_all['x'] == 999]['uid'].unique()
            mobility_df = mobility_df_all[~mobility_df_all['uid'].isin(uids_with_missing_x)].copy()

            if len(mobility_df) == 0:
                print(f"  -> Skip (empty after filtering) city {file_city} for train/val/test")
            else:
                mobility_df['cell_label'] = (mobility_df['x'] - 1) * W + (mobility_df['y'] - 1)

                all_uids = mobility_df['uid'].unique()
                if use_subsample and len(all_uids) > subsample_number:
                    np.random.seed(random_seed)
                    chosen = np.random.choice(all_uids, subsample_number, replace=False)
                    mobility_df = mobility_df[mobility_df['uid'].isin(chosen)]
                    all_uids = mobility_df['uid'].unique()

                if len(all_uids) < 3:
                    print(f"  -> Skip (too few users to split) city {file_city}, users={len(all_uids)}")
                else:
                    total = sum(split_ratio)
                    train_ratio, val_ratio, test_ratio = [r / total for r in split_ratio]
                    train_uids, rest_uids = train_test_split(
                        all_uids, test_size=(1 - train_ratio), random_state=random_seed
                    )
                    val_uids, test_uids = train_test_split(
                        rest_uids,
                        test_size=(test_ratio / (val_ratio + test_ratio)),
                        random_state=random_seed
                    )

                    train_df = mobility_df[mobility_df['uid'].isin(train_uids)]
                    val_df   = mobility_df[mobility_df['uid'].isin(val_uids)]
                    test_df  = mobility_df[mobility_df['uid'].isin(test_uids)]

                    def _group_uid_trajectories_fixed(df):
                        return [
                            g.sort_values(["d", "t"])[['uid', 'd', 't', 'x', 'y', 'cell_label']].to_numpy(dtype=np.int64)
                            for _, g in df.groupby('uid', sort=False)
                        ]

                    tr_uid_arrays = _group_uid_trajectories_fixed(train_df)
                    va_uid_arrays = _group_uid_trajectories_fixed(val_df)
                    te_uid_arrays = _group_uid_trajectories_fixed(test_df)

                    print(f"  City {file_city}: Train users: {len(train_uids)}, Val: {len(val_uids)}, Test: {len(test_uids)}")

                    tr_feats, tr_labs, tr_uids = build_sequences_parallel(
                        tr_uid_arrays, city=file_city, mask_policy="fixed_days", mask_days=mask_days, W=W
                    )
                    va_feats, va_labs, va_uids = build_sequences_parallel(
                        va_uid_arrays, city=file_city, mask_policy="fixed_days", mask_days=mask_days, W=W
                    )
                    te_feats, te_labs, te_uids = build_sequences_parallel(
                        te_uid_arrays, city=file_city, mask_policy="fixed_days", mask_days=mask_days, W=W
                    )

                    tr_citymap = {int(uid): city_id for uid in tr_uids}
                    va_citymap = {int(uid): city_id for uid in va_uids}
                    te_citymap = {int(uid): city_id for uid in te_uids}

                    if len(tr_feats) > 0:
                        all_train_datasets.append(
                            MobilitySequenceDataset(tr_citymap, tr_feats, tr_labs, tr_uids, mask_days=mask_days)
                        )
                    if len(va_feats) > 0:
                        all_val_datasets.append(
                            MobilitySequenceDataset(va_citymap, va_feats, va_labs, va_uids, mask_days=mask_days)
                        )
                    if len(te_feats) > 0:
                        all_test_datasets.append(
                            MobilitySequenceDataset(te_citymap, te_feats, te_labs, te_uids, mask_days=mask_days)
                        )

        # B) Build mask set (x==999)
        if mode in {"transfer", "predict"}:
            df_mask_only = mobility_df_all[mobility_df_all['x'] == 999].copy()
            if not df_mask_only.empty:
                mask_uids = df_mask_only["uid"].unique()
                df_for_mask = mobility_df_all[mobility_df_all["uid"].isin(mask_uids)].copy()

                def _group_uid_trajectories_mask(df):
                    return [
                        g[['uid', 'd', 't', 'x', 'y']].sort_values(["d", "t"]).to_numpy(dtype=np.int64)
                        for _, g in df.groupby('uid', sort=False)
                    ]

                mask_uid_array_list = _group_uid_trajectories_mask(df_for_mask)

                mk_feats, mk_labs, mk_uids = build_sequences_parallel(
                    mask_uid_array_list, city=file_city, mask_policy="x999", W=W
                )
                if mk_feats:
                    citymap = {int(u): city_id for u in mk_uids}
                    all_mask_datasets.append(
                        MobilitySequenceDataset(citymap, mk_feats, mk_labs, mk_uids, mask_days=mask_days)
                    )
            else:
                print(f"  City {file_city}: no x==999 rows → no mask set contributed.")

    # Merge datasets / handle None
    if mode in {"pretrain", "transfer"} and not (all_train_datasets or all_val_datasets or all_test_datasets):
        if not all_mask_datasets:
            raise AssertionError("No city datasets built (train/val/test/mask). Check input folder and file formats.")
        else:
            print("[WARN] No train/val/test datasets; only mask set is available.")

    # In predict mode it's okay to have only mask sets; but we'll still warn if nothing is built.
    if mode == "predict" and not (all_train_datasets or all_val_datasets or all_test_datasets or all_mask_datasets):
        raise AssertionError("In 'predict' mode, no datasets were built (neither splits nor mask).")

    train_dataset = (all_train_datasets[0] if len(all_train_datasets) == 1
                     else ConcatDataset(all_train_datasets) if all_train_datasets else None)
    val_dataset   = (all_val_datasets[0] if len(all_val_datasets) == 1
                     else ConcatDataset(all_val_datasets) if all_val_datasets else None)
    test_dataset  = (all_test_datasets[0] if len(all_test_datasets) == 1
                     else ConcatDataset(all_test_datasets) if all_test_datasets else None)
    mask_dataset  = (all_mask_datasets[0] if len(all_mask_datasets) == 1
                     else ConcatDataset(all_mask_datasets) if all_mask_datasets else None)

    # DataLoaders
    num_workers = min(8, multiprocessing.cpu_count())
    common_kwargs = dict(
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        prefetch_factor=6
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        drop_last=True, **common_kwargs
    ) if train_dataset is not None else None
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        drop_last=False, **common_kwargs
    ) if val_dataset is not None else None
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        drop_last=False, **common_kwargs
    ) if test_dataset is not None else None

    mask_loader = None
    if mode in {"transfer", "predict"} and mask_dataset is not None:
        mask_loader = DataLoader(
            mask_dataset, batch_size=batch_size, shuffle=False,
            drop_last=False, **common_kwargs
        )

    return train_loader, val_loader, test_loader, mask_loader