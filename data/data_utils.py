import torch
from torch.utils.data import Dataset
import pandas as pd
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
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.horizon, 3]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def prepare_data(df, seq_len=30, horizon=5, train_split=0.8):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df.values)
    dataset = StockDataset(scaled, seq_len=seq_len, horizon=horizon)
    train_size = int(len(dataset) * train_split)
    valid_size = len(dataset) - train_size
    train_ds, valid_ds = torch.utils.data.random_split(dataset, [train_size, valid_size])
    return train_ds, valid_ds, scaler
