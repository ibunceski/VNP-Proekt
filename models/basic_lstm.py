import torch.nn as nn

class BasicLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2,
                 horizon=5, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, horizon)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
