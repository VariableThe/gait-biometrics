"""Multi-sensor fusion CNN (paper B1 — Dehzangi et al., 2017).

Each sensor channel is processed by its own 1D convolutional branch and the
per-branch features are fused by concatenation, summation, or a learned
attention weighting before the activity classifier.
"""

from __future__ import annotations

import torch
from torch import nn

_BRANCH_FEAT_DIM = 64


class _SensorBranch(nn.Module):
    """1D CNN over a single sensor channel: (N, 1, T) -> (N, _BRANCH_FEAT_DIM)."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, _BRANCH_FEAT_DIM, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(1)


class MultiSensorFusionCNN(nn.Module):
    """Per-sensor 1D CNN branches with configurable feature fusion.

    Args:
        n_channels: number of sensor channels (one branch per channel).
        n_classes: number of activity classes.
        fusion: "concat" (concatenate branch features), "sum" (element-wise
            sum), or "attention" (softmax-weighted sum with learned scores).
        dropout: dropout probability before the classifier.
    """

    def __init__(self, n_channels: int, n_classes: int, fusion: str = "concat", dropout: float = 0.3) -> None:
        super().__init__()
        if fusion not in ("concat", "sum", "attention"):
            raise ValueError(f"fusion must be 'concat', 'sum' or 'attention', got {fusion!r}")
        self.fusion = fusion
        self.branches = nn.ModuleList(_SensorBranch() for _ in range(n_channels))

        if fusion == "concat":
            fused_dim = _BRANCH_FEAT_DIM * n_channels
        else:  # "sum" and "attention" both produce one feature vector
            fused_dim = _BRANCH_FEAT_DIM

        if fusion == "attention":
            # Shared scorer: each branch's feature vector gets a logit, then a
            # softmax over branches yields the fusion weights.
            self.attention = nn.Linear(_BRANCH_FEAT_DIM, 1)

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T) -> one (N, 1, T) slice per branch.
        feats = [branch(x[:, i : i + 1, :]) for i, branch in enumerate(self.branches)]
        # feats[i]: (N, _BRANCH_FEAT_DIM)

        if self.fusion == "concat":
            fused = torch.cat(feats, dim=1)
        elif self.fusion == "sum":
            fused = torch.stack(feats, dim=0).sum(dim=0)
        else:  # "attention"
            scores = torch.cat([self.attention(f) for f in feats], dim=1)  # (N, C)
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (N, C, 1)
            stacked = torch.stack(feats, dim=1)  # (N, C, feat_dim)
            fused = (weights * stacked).sum(dim=1)

        return self.classifier(fused)
