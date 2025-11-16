import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def inverse_close(scaled_values, scaler, close_idx):
    scaled_values = np.array(scaled_values).reshape(-1)

    if isinstance(close_idx, (np.ndarray, list)):
        close_idx = int(close_idx[0])
    else:
        close_idx = int(close_idx)

    dummy = np.zeros((len(scaled_values), scaler.n_features_in_))
    dummy[:, close_idx] = scaled_values
    return scaler.inverse_transform(dummy)[:, close_idx]


def evaluate_preds(y_true, y_pred, label):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{label} → MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    return mse, rmse, mae


def plot_loss(log_dir, model_name):
    """Plot train vs val loss from Lightning CSV logs."""
    metrics_path = os.path.join(log_dir, "metrics.csv")
    if not os.path.exists(metrics_path):
        print(f"No metrics.csv found in {log_dir}")
        return

    df = pd.read_csv(metrics_path)

    train_loss = df[df["train_loss"].notna()][["epoch", "train_loss"]]
    val_loss = df[df["val_loss"].notna()][["epoch", "val_loss"]]

    plt.figure(figsize=(8, 5))
    if not train_loss.empty:
        plt.plot(train_loss["epoch"], train_loss["train_loss"], label="Train Loss")
    if not val_loss.empty:
        plt.plot(val_loss["epoch"], val_loss["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title(f"Training vs Validation Loss - {model_name}")
    plt.legend()
    plt.tight_layout()

    os.makedirs(f"plot/{datetime.now().strftime('%Y%m%d_%H%M')}/loss_plots", exist_ok=True)
    loss_plot_filename = f"plot/{datetime.now().strftime('%Y%m%d_%H%M')}/loss_plots/{model_name}_loss.png"
    plt.savefig(loss_plot_filename, bbox_inches="tight")
    plt.close()

    print(f"Loss plot saved to {loss_plot_filename}")


def plot_forecast(horizon, true_vals, x_vals, l_vals, ticker, metrics):
    os.makedirs(f"plot/{datetime.now().strftime('%Y%m%d_%H%M')}", exist_ok=True)
    plot_filename = f"plot/{datetime.now().strftime('%Y%m%d_%H%M')}/forecast.png"

    plt.figure(figsize=(10, 5))
    plt.plot(range(horizon), true_vals, label="True", color="black")
    plt.plot(range(horizon), x_vals, label="xLSTM", linestyle="--")
    plt.plot(range(horizon), l_vals, label="Basic LSTM", linestyle=":")
    plt.legend()
    plt.title("Multi-step Forecast Comparison")

    metrics_text = (
        f"{ticker}\n"
        f"xLSTM - MSE={metrics['mse_x']:.4f}, MAE={metrics['mae_x']:.4f}, RMSE={metrics['rmse_x']:.4f}\n"
        f"LSTM - MSE={metrics['mse_l']:.4f}, MAE={metrics['mae_l']:.4f}, RMSE={metrics['rmse_l']:.4f}"
    )
    plt.figtext(0.5, -0.05, metrics_text, ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(plot_filename, bbox_inches="tight")
    plt.close()

    print(f"Forecast plot saved to {plot_filename}")
