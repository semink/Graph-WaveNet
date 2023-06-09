import torch
import os
from pytorch_lightning import LightningDataModule
import numpy as np
from libs import utils
import pandas as pd
from torch.utils.data import DataLoader, Subset


def encode_features(df, scaler):
    # df must has time index, and sensors as columns
    t = df.index
    num_sensors = df.shape[1]
    additional_features = []
    time_ind = (
        t.values - t.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
    theta = time_ind * 2 * np.pi
    cos = np.tile(np.cos(theta)[np.newaxis, :], (num_sensors, 1))
    sine = np.tile(np.sin(theta)[np.newaxis, :], (num_sensors, 1))
    additional_features.append(cos)
    additional_features.append(sine)

    dow = np.tile((t.day_of_week > 4).astype(float).T, (num_sensors, 1))
    additional_features.append(dow)
    x = torch.stack([torch.tensor(feature) for feature in (
        scaler.transform(df.T.values), *additional_features)], dim=0).float()
    return x


class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, X, scaler, t_features, seq_len=1, horizon=1,
                 between_times=(('00:00', '23:55'),)):
        self.X = X
        self.seq_len = seq_len
        self.horizon = horizon
        self.scaler = scaler
        self.t_features = t_features
        self.num_sensors = X.shape[1]
        subset = pd.concat([self.t_features.between_time(*bt)
                            for bt in between_times])
        subset_idx = [self.t_features.index.get_loc(t) for t in subset.index]
        self.subset_idx = list(filter(lambda x: (
            x+self.horizon <= len(self.t_features)-1) & (x+1 - self.seq_len >= 0), subset_idx))

    def __len__(self):
        return self.subset_idx.__len__()

    def get_data(self, idx):
        tidx = self.subset_idx[idx]
        x_time_index = slice(tidx-self.seq_len+1, tidx+1, 1)
        y_time_index = slice(tidx + 1, tidx + self.horizon+1, 1)
        traffic_X = self.scaler.transform(self.X[x_time_index, :].T)

        additional_features = []
        additional_features.append(np.tile(
            self.t_features[x_time_index].T.values, (self.num_sensors, 1)))

        traffic_Y = self.X[y_time_index, :]

        input = torch.stack([torch.tensor(feature) for feature in (
            traffic_X, *additional_features)], dim=0).float()
        target = torch.tensor(traffic_Y).float()
        return (input, target)

    def __getitem__(self, index):
        X, Y = self.get_data(index)
        return X, Y


class DataModule(LightningDataModule):
    def __init__(self, dataset: str = "bay", batch_size: int = 32,
                 seq_len=24, horizon=12,
                 test_on_time=(('00:00', '23:55'), )):
        super(DataModule, self).__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.horizon = horizon
        self.train_df, self.valid_df, self.test_df = None, None, None
        self.scaler = None
        self.dt = None
        self.adj = None
        self.train_t, self.valid_t, self.test_t = None, None, None
        self.test_on_time = test_on_time

    def get_input_dim(self):
        dim = 2
        return dim

    def prepare_data(self, custom_dataset=None):
        if custom_dataset is None:
            self.df, self.adj = utils.get_traffic_data(self.dataset)
            self.t_features = utils.convert_timestamp_to_feature(self.df.index)
        else:
            self.df, self.t_features, self.adj = custom_dataset
        if self.dataset == 'bay':
            self.df = self.df.astype(np.float32)

        # set scaler
        self._set_scaler()

        # set dataset
        self._set_dataset()

    def get_scaler(self):
        return self.scaler

    def get_num_nodes(self):
        return self.df.shape[1]

    def get_adj(self):
        return torch.tensor(self.adj).float()

    def _set_scaler(self):
        train_df, _, _ = utils.split_data(df=self.df)
        mean = np.nanmean(train_df.replace(0.0, np.nan).values)
        std = np.nanstd(train_df.replace(0.0, np.nan).values)
        self.scaler = utils.StandardScaler(mean, std)

    def _set_dataset(self):
        self.ds = TimeSeriesDataset(self.df.values, scaler=self.scaler,
                                    t_features=self.t_features,
                                    seq_len=self.seq_len, horizon=self.horizon)
        ds_len = len(self.ds)
        self.train_ds_idx = range(0, int(ds_len*0.7))
        self.val_ds_idx = range(int(ds_len*0.7), int(ds_len*0.8))
        self.test_ds_idx = range(int(ds_len*0.8), ds_len)

        self.train_df, self.valid_df, self.test_df = utils.split_data(
            df=self.df)
        self.test_ds = TimeSeriesDataset(self.df.values[self.test_ds_idx[0]:],
                                         scaler=self.scaler,
                                         t_features=self.t_features.iloc[self.test_ds_idx[0]:],
                                         seq_len=self.seq_len, horizon=self.horizon,
                                         between_times=self.test_on_time)

    def _get_custom_dataset(self, df, seq_len=12, horizon=12):
        ds = TimeSeriesDataset(df.values, scaler=self.scaler,
                               t_features=utils.convert_timestamp_to_feature(
                                   df.index),
                               seq_len=seq_len, horizon=horizon)
        return ds

    def get_raw_data(self):
        return self.df, self.adj

    def _cache_check(self, fn):
        return os.path.isfile(fn)

    def _pad_with_last_sample(self, data):
        x, y = zip(*data)
        if len(data) < self.batch_size:
            num_padding = self.batch_size - len(data)
            x_pad = [x[-1].clone() for _ in range(num_padding)]
            x = list(x) + x_pad
            y_pad = [y[-1].clone() for _ in range(num_padding)]
            y = list(y) + y_pad
        return torch.stack(x), torch.stack(y)

    def train_dataloader(self, shuffle=True, pad_with_last_sample=True):
        collate_fn = self._pad_with_last_sample if pad_with_last_sample else None
        return DataLoader(Subset(self.ds, self.train_ds_idx), batch_size=self.batch_size,
                          shuffle=shuffle, collate_fn=collate_fn)

    def val_dataloader(self, shuffle=False, pad_with_last_sample=True):
        collate_fn = self._pad_with_last_sample if pad_with_last_sample else None
        return DataLoader(Subset(self.ds, self.val_ds_idx), batch_size=self.batch_size,
                          shuffle=shuffle, collate_fn=collate_fn)

    def test_dataloader(self, shuffle=False, pad_with_last_sample=True):
        collate_fn = self._pad_with_last_sample if pad_with_last_sample else None
        return DataLoader(self.test_ds, batch_size=self.batch_size,
                          shuffle=shuffle,
                          collate_fn=collate_fn)

    def predict_dataloader(self, df, seq_len=12, horizon=12):
        ds = self._get_custom_dataset(df, seq_len=seq_len, horizon=horizon)
        return DataLoader(ds, batch_size=self.batch_size,
                          shuffle=False,
                          collate_fn=None)
