import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import CSVLogger
import pandas as pd
import os
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

from data.data_utils import load_stock_data, prepare_data
from models.xlstm_model import TimeSeriesPredictor
from models.basic_lstm import BasicLSTM
from training.trainer_lightning import LitTimeSeries
from utils.helpers import inverse_close, plot_loss, plot_forecast


def main():
    ticker = "NVDA"
    df = load_stock_data(ticker, period="10y", interval="1d")
    seq_len, horizon = 20, 5
    train_ds, valid_ds, scaler, test = prepare_data(df, seq_len, horizon)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False)
    valid_loader = DataLoader(valid_ds, batch_size=32, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    close_idx = df.columns.get_loc("Close")

    logger_xlstm = CSVLogger("logs", name="xlstm")
    early_stop_xlstm = L.pytorch.callbacks.EarlyStopping(monitor="val_loss", patience=10)
    trainer_xlstm = L.Trainer(
        max_epochs=30,
        callbacks=[early_stop_xlstm],
        accelerator="gpu" if device == "cuda" else "cpu",
        logger=logger_xlstm,
        default_root_dir="lightning_logs/xlstm",
        gradient_clip_val=0.4,
    )

    xlstm_model = TimeSeriesPredictor(
        df.shape[1], hidden_dim=128,
        num_blocks=2, context_length=seq_len,
        horizon=horizon, dropout=0.3
    )
    lit_xlstm = LitTimeSeries(xlstm_model, lr=1e-3)
    trainer_xlstm.fit(lit_xlstm, train_loader, valid_loader)

    logger_lstm = CSVLogger("logs", name="lstm")
    early_stop_lstm = L.pytorch.callbacks.EarlyStopping(monitor="val_loss", patience=10)
    trainer_lstm = L.Trainer(
        max_epochs=30,
        callbacks=[early_stop_lstm],
        accelerator="gpu" if device == "cuda" else "cpu",
        logger=logger_lstm,
        default_root_dir="lightning_logs/lstm",
    )

    lstm_model = BasicLSTM(
        df.shape[1], hidden_dim=128,
        num_layers=2, horizon=horizon, dropout=0.3
    )
    lit_lstm = LitTimeSeries(lstm_model, lr=1e-3)
    trainer_lstm.fit(lit_lstm, train_loader, valid_loader)

    lit_xlstm.eval()
    lit_lstm.eval()
    xb, yb = test
    with torch.no_grad():
        x_pred = xlstm_model(xb.unsqueeze(0).to(device)).cpu().numpy()[0]
        l_pred = lstm_model(xb.unsqueeze(0).to(device)).cpu().numpy()[0]

    true_vals = inverse_close(yb.numpy(), scaler, close_idx)
    x_vals = inverse_close(x_pred, scaler, close_idx)
    l_vals = inverse_close(l_pred, scaler, close_idx)

    # Metrics
    mse_x = mean_squared_error(true_vals, x_vals)
    mae_x = mean_absolute_error(true_vals, x_vals)
    rmse_x = np.sqrt(mse_x)

    mse_l = mean_squared_error(true_vals, l_vals)
    mae_l = mean_absolute_error(true_vals, l_vals)
    rmse_l = np.sqrt(mse_l)

    print("True:", true_vals)
    print("xLSTM:", x_vals, f"(MSE={mse_x:.4f}, MAE={mae_x:.4f}, RMSE={rmse_x:.4f})")
    print("LSTM:", l_vals, f"(MSE={mse_l:.4f}, MAE={mae_l:.4f}, RMSE={rmse_l:.4f})")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results_df = pd.DataFrame({
        "timestamp": [timestamp] * horizon,
        "step": list(range(horizon)),
        "true": true_vals,
        "xLSTM": x_vals,
        "LSTM": l_vals,
        "mse_xLSTM": [mse_x] * horizon,
        "mae_xLSTM": [mae_x] * horizon,
        "rmse_xLSTM": [rmse_x] * horizon,
        "mse_LSTM": [mse_l] * horizon,
        "mae_LSTM": [mae_l] * horizon,
        "rmse_LSTM": [rmse_l] * horizon
    })

    results_file = "results.csv"
    if os.path.exists(results_file):
        results_df.to_csv(results_file, mode="a", header=False, index=False)
    else:
        results_df.to_csv(results_file, index=False)

    print(f"Results + metrics saved to {results_file}")

    metrics = {'mse_x': mse_x, 'mae_x': mae_x, 'rmse_x': rmse_x, 'mse_l': mse_l, 'mae_l': mae_l, 'rmse_l': rmse_l}
    plot_forecast(horizon, true_vals, x_vals, l_vals, ticker, metrics)

    plot_loss(logger_xlstm.log_dir, "xLSTM")
    plot_loss(logger_lstm.log_dir, "LSTM")


if __name__ == "__main__":
    main()
