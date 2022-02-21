from pathlib import Path
import wget
import pickle
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy as c


def get_project_root() -> Path:
    return Path(__file__).parent.parent


PROJECT_ROOT = get_project_root()


def _exist_dataset_on_disk(dataset):
    file = f'{PROJECT_ROOT}/dataset/METR-LA.csv' if dataset == 'la' else f'{PROJECT_ROOT}/dataset/PEMS-BAY.csv'
    return os.path.isfile(file)


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([c(module) for _ in range(N)])


def mean_by_day(df, null_value=0.0):
    train_df = df.copy()
    train_df = train_df.replace(null_value, np.nan)
    day_df_mean = dict()
    for day, group_df in train_df.groupby(train_df.index.dayofweek):
        day_df_mean[day] = group_df.groupby(
            group_df.index.strftime("%H:%M")).mean()
    return day_df_mean


def split_data(df, rule=[0.7, 0.1, 0.2]):
    assert np.isclose(np.sum(
        rule), 1.0), f"sum of split rule should be 1 (currently sum={np.sum(rule):.2f})"

    num_samples = df.shape[0]
    num_test = round(num_samples * rule[-1])
    num_train = round(num_samples * rule[0])
    num_val = num_samples - num_test - num_train

    train_df = df.iloc[:num_train].copy()
    valid_df = df.iloc[num_train: num_train + num_val].copy()
    test_df = df.iloc[-num_test:].copy()

    return train_df, valid_df, test_df


def get_traffic_data(dataset, null_value=0.0):
    if dataset == 'la':
        fn, adj_name = 'METR-LA.csv', 'adj_mx_METR-LA.pkl'
    elif dataset == 'bay':
        fn, adj_name = 'PEMS-BAY.csv', 'adj_mx_PEMS-BAY.pkl'
    else:
        raise ValueError("dataset name should be either 'bay' or 'la")
    data_url = f'https://zenodo.org/record/5724362/files/{fn}'
    sup_url = f'https://zenodo.org/record/5724362/files/{adj_name}'
    if not _exist_dataset_on_disk(dataset):
        wget.download(data_url, out=f'{PROJECT_ROOT}/dataset')
        wget.download(sup_url, out=f'{PROJECT_ROOT}/dataset')
    df = pd.read_csv(f'{PROJECT_ROOT}/dataset/{fn}', index_col=0)
    df.index = pd.DatetimeIndex(df.index)
    dt = pd.Timedelta(df.index.to_series().diff().mode().values[0])
    df = df.asfreq(freq=dt, fill_value=null_value)
    df = df.replace(0.0, null_value)
    with open(f'{PROJECT_ROOT}/dataset/{adj_name}', 'rb') as f:
        _, _, adj = pickle.load(f, encoding='latin1')
    return df, adj


def convert_timestamp_to_feature(timestamp):
    hour, minute = timestamp.hour, timestamp.minute
    feature = (hour * 60 + minute) / (24 * 60)
    return pd.DataFrame(feature, index=timestamp)


def apply_mask(data, mask):
    data[mask, ...] = 0
    return data


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def transform_adj(A, adj_type='weight'):
    if adj_type == 'weight':
        adj = A.clone()
    elif adj_type == 'connectivity_only':
        adj = A.clone()
        adj[adj > 0] = 1
    elif adj_type == 'identity':
        adj = torch.eye(A.size(0))
        return adj
    elif adj_type == 'random':
        adj = A.clone()
        random = torch.rand_like(adj)
        random[adj == 0] = 0
        adj = random
    return adj


def masked_metric(agg_fn, error_fn, pred, target, null_value=0.0, dim=0):
    mask = ~torch.isnan(target) if np.isnan(
        null_value) else target != null_value
    mask = mask.float()
    mask /= torch.mean(mask, dim=dim, keepdim=True)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    score = error_fn(pred, target)
    score = score * mask
    score = torch.where(torch.isnan(score), torch.zeros_like(score), score)

    return agg_fn(score)


def masked_MAE(pred, target, null_value=0.0, dim=(0, 1, 2)):
    mae = masked_metric(agg_fn=lambda e: torch.mean(e, dim=dim),
                        error_fn=lambda p, t: torch.absolute(p - t),
                        pred=pred, target=target, null_value=null_value, dim=dim)
    return mae


def masked_MSE(pred, target, null_value=0.0, dim=(0, 1, 2)):
    mse = masked_metric(agg_fn=lambda e: torch.mean(e, dim=dim),
                        error_fn=lambda p, t: (p - t) ** 2,
                        pred=pred, target=target, null_value=null_value, dim=dim)
    return mse


def masked_RMSE(pred, target, null_value=0.0, dim=(0, 1, 2)):
    rmse = masked_metric(agg_fn=lambda e: torch.sqrt(torch.mean(e, dim=dim)),
                         error_fn=lambda p, t: (p - t)**2,
                         pred=pred, target=target, null_value=null_value, dim=dim)
    return rmse


def masked_MAPE(pred, target, null_value=0.0, dim=(0, 1, 2)):
    mape = masked_metric(agg_fn=lambda e: torch.mean(torch.absolute(e) * 100, dim=dim),
                         error_fn=lambda p, t: ((p - t) / (t)),
                         pred=pred, target=target, null_value=null_value, dim=dim)
    return mape
