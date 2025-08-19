import multiprocessing
import os
import re

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from joblib import Parallel, delayed
from tqdm import tqdm



class YJDataset(Dataset):
    """
    YJMob100K 데이터를 UID 단위로 분할된 NumPy 배열 리스트로 받아,
    (입력 시퀀스, 출력 라벨, 예측 시퀀스, 예측 라벨) 형태로 구성된 시퀀스 데이터셋을 생성
    """

    def __init__(self, city, raw_uid_np_list, input_seq_len, predict_seq_len, sliding_step=24):
        """
        Args:
            raw_uid_np_list (List[np.ndarray]): UID별로 분할된 NumPy 배열 리스트
            input_seq_len (int): 입력 시퀀스 길이
            predict_seq_len (int): 예측 시퀀스 길이
            sliding_step (int): 슬라이딩 윈도우 간격
        """
        self.city = city
        self.input_seq_len = input_seq_len
        self.predict_seq_len = predict_seq_len
        self.sliding_step = sliding_step
        self.raw_uid_np_list = raw_uid_np_list
        

        # NumPy 시퀀스 생성
        input_x_np, input_y_np, pred_x_np, pred_y_np, uid_seq_np = build_sequences_parallel(
            raw_uid_np_list, input_seq_len, predict_seq_len, sliding_step, city
        )

        # Tensor로 변환
        self.input_seq_feature = torch.from_numpy(input_x_np).long()
        self.input_seq_label = torch.from_numpy(input_y_np).long()
        self.predict_seq_feature = torch.from_numpy(pred_x_np).long()
        self.predict_seq_label = torch.from_numpy(pred_y_np).long()
        self.uid_seq = torch.from_numpy(uid_seq_np).long()

    def __len__(self):
        return len(self.input_seq_feature)

    def __getitem__(self, idx):
        return (
            self.input_seq_feature[idx],
            self.input_seq_label[idx],
            self.predict_seq_feature[idx],
            self.predict_seq_label[idx],
            int(self.uid_seq[idx])
        )


def build_sequences_parallel(raw_uid_np_list, input_seq_len, predict_seq_len,
                             sliding_step=24, city='A'):
    """
    UID별로 미리 분할된 raw_uid_np_list를 기반으로 병렬로 시퀀스 데이터를 생성
    """

    def generate_uid_sequences(user_array):
        if user_array.shape[0] == 0:
            return None

        uid = int(user_array[0, 0])

        # 날짜(day), 시간대(slot), 셀 레이블을 추출
        days = user_array[:, 1] - 1  # day=1부터 시작하므로, 0부터 시작하도록 조정
        times = user_array[:, 2]
        labels = user_array[:, 5]

        # 절대 시간 계산: 각 레코드의 절대 시점 (ex. day*48 + timeslot)
        absolute_time = days * 48 + times
        delta_t = np.diff(absolute_time, prepend=absolute_time[0])  # 첫 값은 항상 0이 됨

        # 요일 계산 방식 분기
        if city in ['C', 'D']:
            day_of_week = np.where(
                days >= 60,
                (days - 60 + 2) % 7,
                days % 7
            )
        else:
            day_of_week = days % 7

        is_weekday = ((day_of_week >= 2) & (day_of_week <= 6)).astype(int)

        feature_sequence = np.column_stack((days, times, day_of_week, is_weekday, delta_t))
        label_sequence = labels

        total_len = len(feature_sequence)
        required_len = input_seq_len + predict_seq_len
        if total_len < required_len:
            return None

        input_features_list, input_labels_list = [], []
        target_features_list, target_labels_list = [], []
        uid_list = []

        for i in range(0, total_len - required_len + 1, sliding_step):
            input_features_list.append(feature_sequence[i: i + input_seq_len])
            input_labels_list.append(label_sequence[i: i + input_seq_len])
            target_features_list.append(feature_sequence[i + input_seq_len: i + required_len])
            target_labels_list.append(label_sequence[i + input_seq_len: i + required_len])
            uid_list.append(uid)

        return input_features_list, input_labels_list, target_features_list, target_labels_list, uid_list

    results = Parallel(n_jobs=-1)(
        delayed(generate_uid_sequences)(user_array)
        for user_array in tqdm(raw_uid_np_list, desc="[build_sequences_parallel] Generating sequences")
    )

    all_input_features, all_input_labels = [], []
    all_target_features, all_target_labels = [], []
    all_uids = []

    for res in results:
        if res is None:
            continue
        in_feats, in_labs, tgt_feats, tgt_labs, uids = res
        all_input_features.extend(in_feats)
        all_input_labels.extend(in_labs)
        all_target_features.extend(tgt_feats)
        all_target_labels.extend(tgt_labs)
        all_uids.extend(uids)

    all_input_features = np.array(all_input_features, dtype=np.int64)
    all_input_labels = np.array(all_input_labels, dtype=np.int64)
    all_target_features = np.array(all_target_features, dtype=np.int64)
    all_target_labels = np.array(all_target_labels, dtype=np.int64)
    all_uids = np.array(all_uids, dtype=np.int64)

    print(f"[build_sequences_parallel] Generated {len(all_input_features)} sequences.")

    return all_input_features, all_input_labels, all_target_features, all_target_labels, all_uids


def load_and_split_data(
    mobility_data_path,
    input_seq_length,
    predict_seq_length,
    sliding_step,
    batch_size,
    split_ratio=(8, 1, 1),      # train/val/test 비율 실험 가능하도록 추가.
    use_subsample=False,
    subsample_number=10000,
    random_seed=42
):
    """
    Human Mobility Data를 불러와 전처리하고, UID 기준으로 주어진 비율만큼
    train/val/test로 나눈 뒤 PyTorch DataLoader 형태로 반환.
    """

    # 1. CSV 로드
    mobility_df = pd.read_csv(mobility_data_path)
    print(f"Original data loaded. Total records: {len(mobility_df)}")

    data_filename = os.path.basename(mobility_data_path)
    city = re.search(r'city_([A-Z])_', data_filename).group(1)

    # 2. 사전 필터링
    uids_with_missing_x = mobility_df[mobility_df['x'] == 999]['uid'].unique()
    mobility_df = mobility_df[~mobility_df['uid'].isin(uids_with_missing_x)].copy()

    # 3. sparse cell label → dense label 매핑
    mobility_df['cell_label'] = (mobility_df['x'] - 1) * 200 + (mobility_df['y'] - 1)

    # 4. Subsample
    all_uids = mobility_df['uid'].unique()
    if use_subsample and len(all_uids) > subsample_number:
        np.random.seed(random_seed)
        all_uids = np.random.choice(all_uids, subsample_number, replace=False)
        mobility_df = mobility_df[mobility_df['uid'].isin(all_uids)]

    # 5. UID 기준 분할
    all_uids = mobility_df['uid'].unique()
    total_ratio = sum(split_ratio)
    train_ratio = split_ratio[0] / total_ratio
    val_ratio = split_ratio[1] / total_ratio
    test_ratio = split_ratio[2] / total_ratio

    train_uid_list, temp_uid_list = train_test_split(
        all_uids, test_size=(1 - train_ratio), random_state=random_seed
    )
    val_uid_list, test_uid_list = train_test_split(
        temp_uid_list, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=random_seed
    )

    # 6. UID별로 데이터 분할
    train_split_df = mobility_df[mobility_df['uid'].isin(train_uid_list)]
    val_split_df = mobility_df[mobility_df['uid'].isin(val_uid_list)]
    test_split_df = mobility_df[mobility_df['uid'].isin(test_uid_list)]

    # 7. UID별 그룹핑 후 NumPy 변환
    def group_uid_trajectories(grouped_df):
        return [
            g[['uid', 'd', 't', 'x', 'y', 'cell_label']].to_numpy(dtype=np.int64)
            for _, g in grouped_df.groupby('uid', sort=False)
        ]

    train_uid_array_list = group_uid_trajectories(train_split_df)
    val_uid_array_list = group_uid_trajectories(val_split_df)
    test_uid_array_list = group_uid_trajectories(test_split_df)

    print(f"Train users: {len(train_uid_list)}, Val: {len(val_uid_list)}, Test: {len(test_uid_list)}")

    # 8. 각각 YJDataset 생성
    train_dataset = YJDataset(city, train_uid_array_list, input_seq_length, predict_seq_length, sliding_step)
    val_dataset = YJDataset(city, val_uid_array_list, input_seq_length, predict_seq_length, sliding_step)
    test_dataset = YJDataset(city, test_uid_array_list, input_seq_length, predict_seq_length, sliding_step)

    # 9. DataLoader 생성
    num_workers = min(8, multiprocessing.cpu_count())

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        prefetch_factor=6,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        prefetch_factor=6,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        prefetch_factor=6,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader