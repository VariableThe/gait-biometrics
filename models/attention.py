"""SE-DeepConvNet with Squeeze-and-Excitation channel attention (paper B5).

Wearable-sensor behavioral user authentication using a hybrid deep learning
approach with an SE mechanism (Computers, MDPI, 2024). This module provides
the SE block, a depthwise-separable 1D convolution, and `SEConvNet` — a
deep convolutional classifier whose conv blocks are each followed by an SE
recalibration stage.

`SEConvNet` is parameter-bounded: with `n_channels=9, n_classes=6` and the
defaults (`use_depthwise=True, dropout=0.3, reduction=16`) it has
~146k parameters, comfortably under the 158676 cap (1.2 x
`MultiSensorFusionCNN`'s 132230).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block: channel-wise recalibration.

    Global-average-pools over time, passes the channel descriptor through
    a bottleneck MLP (reduction), and rescales the input channels by the
    resulting sigmoid gates.

    Args:
        channels: number of input/output channels.
        reduction: bottleneck reduction ratio.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T)
        s = x.mean(dim=2)  # (N, C) — squeeze over time
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))  # (N, C)
        return x * s.unsqueeze(2)  # (N, C, T) — excite


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise-separable 1D convolution: depthwise + pointwise.

    Factorizes a standard Conv1d into a per-channel depthwise filter
    followed by a 1x1 pointwise projection, cutting parameters and FLOPs
    while preserving the receptive field.

    Args:
        in_ch: number of input channels.
        out_ch: number of output channels.
        kernel_size: spatial extent of the depthwise filter.
        stride: stride of the depthwise convolution.
        padding: zero-padding of the depthwise convolution.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 stride: int = 1, padding: int = 0) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_ch, in_ch, kernel_size, stride=stride, padding=padding,
            groups=in_ch,
        )
        self.pointwise = nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class SEConvNet(nn.Module):
    """Deep convolutional network with SE-recalibrated blocks.

    Each block is a (depthwise-separable) Conv1d -> BatchNorm -> ReLU ->
    MaxPool -> SEBlock. Three pools reduce T=128 to 16; the flattened
    features feed a linear classifier.

    Args:
        n_channels: number of sensor channels in the input (N, C, T).
        n_classes: number of activity classes.
        use_depthwise: if True, use depthwise-separable convolutions;
            otherwise standard Conv1d.
        dropout: dropout probability before the classifier.
        reduction: SE bottleneck reduction ratio.
    """

    def __init__(self, n_channels: int, n_classes: int, use_depthwise: bool = True,
                 dropout: float = 0.3, reduction: int = 16) -> None:
        super().__init__()
        self.use_depthwise = use_depthwise

        def make_conv(in_ch: int, out_ch: int, kernel_size: int, padding: int) -> nn.Module:
            if use_depthwise:
                return DepthwiseSeparableConv1d(in_ch, out_ch, kernel_size, padding=padding)
            return nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding)

        # Block 1: 9 -> 32, T=128 -> 64
        self.conv1 = make_conv(n_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.se1 = SEBlock(32, reduction)

        # Block 2: 32 -> 64, T=64 -> 32
        self.conv2 = make_conv(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.se2 = SEBlock(64, reduction)

        # Block 3: 64 -> 128, T=32 -> 16
        self.conv3 = make_conv(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.se3 = SEBlock(128, reduction)

        # 128 channels * 16 time steps = 2048 flattened features
        self.classifier = nn.Sequential(
            nn.Linear(128 * 16, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.se1(h)
        h = F.max_pool1d(h, 2)

        # Block 2
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.se2(h)
        h = F.max_pool1d(h, 2)

        # Block 3
        h = F.relu(self.bn3(self.conv3(h)))
        h = self.se3(h)
        h = F.max_pool1d(h, 2)

        # Classifier
        return self.classifier(h.flatten(1))
