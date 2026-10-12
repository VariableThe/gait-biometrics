"""DeepConvLSTM (paper B3 — PRIMARY classification baseline).

A Conv1D stack extracts local per-window features along time, an LSTM stack
models the temporal sequence of those features, and a classifier maps the
final hidden state to activity logits.
"""

from __future__ import annotations

import torch
from torch import nn


class DeepConvLSTM(nn.Module):
    """Conv1D feature extractor -> LSTM sequence model -> linear classifier.

    Args:
        n_channels: number of sensor channels in the input (N, C, T).
        n_classes: number of activity classes.
        hidden_dim: width of the conv output channels and the LSTM hidden state.
        n_lstm_layers: number of stacked LSTM layers.
        dropout: dropout probability (between LSTM layers and before the head).
    """

    def __init__(self, n_channels: int, n_classes: int, hidden_dim: int = 128,
                 n_lstm_layers: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        # T=128 -> 16 time steps after the three pools; each step is a
        # hidden_dim-dimensional feature vector.
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_lstm_layers,
            batch_first=True,
            dropout=dropout if n_lstm_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)  # (N, hidden_dim, T/8)
        h = h.permute(0, 2, 1)  # (N, T/8, hidden_dim) — time-major for the LSTM
        out, _ = self.lstm(h)
        return self.classifier(out[:, -1, :])
