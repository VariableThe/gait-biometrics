"""Multi-task CNN+LSTM (paper B2 — METIER, Chen et al., 2020).

A shared Conv1D+LSTM trunk extracts robust temporal features from the sensor
window; two linear heads predict the activity class and the subject (user)
identity from the same representation.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MultiTaskNet(nn.Module):
    """Shared CNN+LSTM trunk with activity and subject heads.

    Args:
        n_channels: number of sensor channels in the input (N, C, T).
        n_activities: number of activity classes.
        n_subjects: number of subject classes (user identities).
        hidden_dim: width of the conv output channels and the LSTM hidden state.
        dropout: dropout probability between LSTM layers.
    """

    def __init__(self, n_channels: int, n_activities: int, n_subjects: int,
                 hidden_dim: int = 128, dropout: float = 0.3) -> None:
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
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.activity_head = nn.Linear(hidden_dim, n_activities)
        self.subject_head = nn.Linear(hidden_dim, n_subjects)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(x)  # (N, hidden_dim, T/8)
        h = h.permute(0, 2, 1)  # (N, T/8, hidden_dim)
        out, _ = self.lstm(h)
        feat = out[:, -1, :]
        return self.activity_head(feat), self.subject_head(feat)


def multitask_loss(act_logits: torch.Tensor, subj_logits: torch.Tensor,
                   act_y: torch.Tensor, subj_y: torch.Tensor,
                   alpha: float = 0.5) -> torch.Tensor:
    """Weighted sum of the activity and subject cross-entropy losses.

    `alpha` weights the activity loss; `1 - alpha` weights the subject loss.
    """
    act_loss = F.cross_entropy(act_logits, act_y)
    subj_loss = F.cross_entropy(subj_logits, subj_y)
    return alpha * act_loss + (1.0 - alpha) * subj_loss
