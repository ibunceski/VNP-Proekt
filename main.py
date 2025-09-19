import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from data.data_utils import load_stock_data, prepare_data
from models.xlstm_model import TimeSeriesPredictor
from models.basic_lstm import BasicLSTM
from training.trainer import train_model
from utils.helpers import inverse_close, evaluate_preds

def main():
    df = load_stock_data("NVDA", period="5y", interval="1d")
    seq_len, horizon = 30, 5
    train_ds, valid_ds, scaler = prepare_data(df, seq_len, horizon)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=32, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    close_idx = df.columns.get_loc("Close")

    # Train xLSTM
    xlstm_model = TimeSeriesPredictor(df.shape[1], hidden_dim=128,
                                      num_blocks=3, context_length=seq_len,
                                      horizon=horizon, dropout=0.3)
    xlstm_model = train_model(xlstm_model, train_loader, valid_loader,
                              epochs=40, lr=1e-3, patience=5, device=device)

    # Train Basic LSTM
    lstm_model = BasicLSTM(df.shape[1], hidden_dim=64,
                           num_layers=2, horizon=horizon, dropout=0.2)
    lstm_model = train_model(lstm_model, train_loader, valid_loader,
                             epochs=40, lr=1e-3, patience=5, device=device)

    # Evaluate
    xlstm_model.eval()
    lstm_model.eval()
    xb, yb = valid_ds[0]
    with torch.no_grad():
        x_pred = xlstm_model(xb.unsqueeze(0).to(device)).cpu().numpy()[0]
        l_pred = lstm_model(xb.unsqueeze(0).to(device)).cpu().numpy()[0]

    true_vals = inverse_close(yb.numpy(), scaler, close_idx)
    x_vals = inverse_close(x_pred, scaler, close_idx)
    l_vals = inverse_close(l_pred, scaler, close_idx)

    print("True:", true_vals)
    print("xLSTM:", x_vals)
    print("LSTM:", l_vals)

    evaluate_preds(true_vals, x_vals, "xLSTM")
    evaluate_preds(true_vals, l_vals, "Basic LSTM")

    # Plot
    plt.figure(figsize=(10,5))
    plt.plot(range(horizon), true_vals, label="True", color="black")
    plt.plot(range(horizon), x_vals, label="xLSTM", linestyle="--")
    plt.plot(range(horizon), l_vals, label="Basic LSTM", linestyle=":")
    plt.legend()
    plt.title("Multi-step Forecast Comparison")
    plt.show()

if __name__ == "__main__":
    main()
