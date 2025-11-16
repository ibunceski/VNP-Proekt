import random

import torch
from torch.utils.data import Dataset
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


def load_stock_data(ticker="TSLA", period="5y", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df


class StockDataset(Dataset):
    def __init__(self, data, seq_len=30, horizon=5):
        self.seq_len = seq_len
        self.horizon = horizon
        self.data = data
        self.length = len(data) - seq_len - horizon

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + self.seq_len: idx + self.seq_len + self.horizon, 3]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def prepare_data(df, seq_len=30, horizon=5, train_split=0.8):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df.values)
    dataset = StockDataset(scaled, seq_len=seq_len, horizon=horizon)
    train_size = int(len(dataset) * train_split)
    train_ds = torch.utils.data.Subset(dataset, range(train_size))
    valid_ds = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))

    if len(valid_ds) > 0:
        rand_idx = random.randint(0, len(valid_ds) - 1)
        test_example = valid_ds[rand_idx]
    else:
        test_example = dataset[-1]

    return train_ds, valid_ds, scaler, test_example
