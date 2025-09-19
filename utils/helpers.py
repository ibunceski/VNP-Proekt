import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def inverse_close(scaled_values, scaler, close_idx):
    scaled_values = np.array(scaled_values).reshape(-1)  # (N,)

    # if close_idx is an array (e.g., from np.where), extract the first element
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
