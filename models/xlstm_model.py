import torch
import torch.nn as nn
from xlstm import (
    xLSTMBlockStack, xLSTMBlockStackConfig,
    mLSTMBlockConfig, mLSTMLayerConfig,
    sLSTMBlockConfig, sLSTMLayerConfig,
    FeedForwardConfig,
)

class TimeSeriesPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_blocks=3,
                 context_length=30, horizon=5, dropout=0.2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)

        cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(conv1d_kernel_size=3,
                                       qkv_proj_blocksize=2, num_heads=1)
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="cuda" if torch.cuda.is_available() else "vanilla",
                    num_heads=4,
                    conv1d_kernel_size=3,
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(proj_factor=1.5, act_fn="gelu"),
            ),
            context_length=context_length,
            num_blocks=num_blocks,
            embedding_dim=hidden_dim,
            slstm_at=[1],
        )

        self.xlstm_stack = xLSTMBlockStack(cfg)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, horizon)

    def forward(self, x):
        x = self.embedding(x)
        x = self.xlstm_stack(x)
        x = self.norm(x)
        x = self.dropout(x)
        return self.output(x[:, -1, :])
