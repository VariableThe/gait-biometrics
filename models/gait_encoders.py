"""Opt-in, compact inertial gait encoders; adaptations, not paper reproductions.

Normalization is fitted explicitly on training windows, never in ``forward``.
See docs/gait_encoder_methods.md for feature definitions and protocol limits.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from models.siamese import EmbeddingNet


def _vector_groups(n_channels: int, groups=None) -> tuple[tuple[int, int, int], ...]:
    if groups is None:
        groups = (
            [range(i, i + 3) for i in range(0, n_channels, 3)]
            if n_channels in (3, 9)
            else []
        )
    result = tuple(tuple(int(i) for i in group) for group in groups)
    flat = [i for group in result for i in group]
    if any(len(group) != 3 for group in result) or len(set(flat)) != len(flat):
        raise ValueError(
            "vector_groups must contain disjoint three-axis channel groups"
        )
    if any(i < 0 or i >= n_channels for i in flat):
        raise ValueError("vector_groups channel index out of range")
    return result


class _SE(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(channels, max(4, channels // 8)),
            nn.SiLU(),
            nn.Linear(max(4, channels // 8), channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x.mean(-1)).unsqueeze(-1)


class _MultiscaleBlock(nn.Module):
    def __init__(self, width: int, dropout: float, long_kernel: int) -> None:
        super().__init__()
        branch = max(4, width // 4)
        self.bottleneck = nn.Conv1d(width, branch, 1, bias=False)
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(branch, branch, k, padding=k // 2, bias=False)
                for k in (3, 7, 15, long_kernel)
            ]
        )
        self.mix = nn.Sequential(
            nn.Conv1d(4 * branch, width, 1, bias=False),
            nn.GroupNorm(1, width),
            nn.SiLU(),
            _SE(width),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.bottleneck(x)
        return F.silu(
            x + self.mix(torch.cat([branch(h) for branch in self.branches], dim=1))
        )


class _TemporalBlock(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(
                width, width, 5, padding=2 * dilation, dilation=dilation, bias=False
            ),
            nn.GroupNorm(1, width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(1, width),
            _SE(width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x + self.net(x))


class GaitEncoder(nn.Module):
    """Factory result: raw (N,C,T) windows to L2-normalized (N,D) embeddings.

    ``fit_normalization`` must see only the post-validation-split training X.
    It works in bounded chunks and stores only tensor buffers in checkpoints.
    A fresh encoder has identity input normalization until explicitly fitted.
    Calling ``fit_normalization`` in eval mode is rejected to catch leakage.
    """

    def __init__(self, n_channels: int, model_args: Mapping) -> None:
        super().__init__()
        args = dict(model_args)
        self.n_channels = int(n_channels)
        self.embed_dim = int(args.get("embed_dim", 64))
        self.architecture = str(args.get("architecture", "legacy"))
        width = int(args.get("width", 32))
        dropout = float(args.get("dropout", 0.1))
        if self.n_channels < 1 or self.embed_dim < 1 or width < 4:
            raise ValueError("n_channels/embed_dim must be positive and width >= 4")
        if self.architecture not in {"legacy", "multiscale", "temporal"}:
            raise ValueError(f"unknown gait architecture: {self.architecture}")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        self.normalization = bool(
            args.get("normalization", self.architecture != "legacy")
        )
        self.stats_fusion = bool(args.get("stats_fusion", False))
        self.stats_include_mean = bool(args.get("stats_include_mean", False))
        self.norm_eps = float(args.get("norm_eps", 1e-5))
        if not math.isfinite(self.norm_eps) or self.norm_eps <= 0:
            raise ValueError("norm_eps must be finite and positive")
        self.vector_groups = _vector_groups(self.n_channels, args.get("vector_groups"))
        self.add_magnitudes = bool(
            args.get("add_magnitudes", self.architecture != "legacy")
        )
        if self.architecture == "legacy" and (self.stats_fusion or self.add_magnitudes):
            raise ValueError(
                "legacy preserves EmbeddingNet: disable stats_fusion/add_magnitudes"
            )
        signals = self.n_channels + (
            len(self.vector_groups) if self.add_magnitudes else 0
        )
        self.signal_channels = signals
        self.register_buffer("channel_mean", torch.zeros(1, self.n_channels, 1))
        self.register_buffer("channel_std", torch.ones(1, self.n_channels, 1))
        self.register_buffer(
            "magnitude_mean", torch.zeros(1, signals - self.n_channels, 1)
        )
        self.register_buffer(
            "magnitude_std", torch.ones(1, signals - self.n_channels, 1)
        )
        self.register_buffer("normalization_fitted", torch.tensor(False))
        pairs = torch.triu_indices(signals, signals, offset=1)
        self.register_buffer("correlation_indices", pairs)
        self.stats_dim = (
            signals * (9 + int(self.stats_include_mean)) + pairs.shape[1]
            if self.stats_fusion
            else 0
        )
        self.register_buffer("stats_mean", torch.zeros(1, self.stats_dim))
        self.register_buffer("stats_std", torch.ones(1, self.stats_dim))
        if self.architecture == "legacy":
            self.encoder = EmbeddingNet(self.n_channels, self.embed_dim, dropout)
            return
        self.stem = nn.Sequential(
            nn.Conv1d(signals, width, 5, padding=2, bias=False),
            nn.GroupNorm(1, width),
            nn.SiLU(),
        )
        if self.architecture == "multiscale":
            self.encoder = nn.Sequential(
                *[_MultiscaleBlock(width, dropout, 31) for _ in range(3)]
            )
            pooled_dim = width * 2
        else:
            bins = int(args.get("temporal_bins", 4))
            if bins not in (4, 8):
                raise ValueError("temporal_bins must be 4 or 8")
            self.temporal_bins = bins
            self.encoder = nn.Sequential(
                *[_TemporalBlock(width, d, dropout) for d in (1, 2, 4)]
            )
            pooled_dim = width * (bins + 1)
        if self.stats_fusion:
            self.stats_projection = nn.Sequential(
                nn.Linear(self.stats_dim, width),
                nn.LayerNorm(width),
                nn.SiLU(),
                nn.Dropout(dropout),
            )
            pooled_dim += width
        self.projection = nn.Sequential(
            nn.Linear(pooled_dim, 2 * width),
            nn.LayerNorm(2 * width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * width, self.embed_dim),
        )

    def _validate(self, x: torch.Tensor) -> None:
        if (
            x.ndim != 3
            or x.shape[1] != self.n_channels
            or x.shape[0] == 0
            or x.shape[2] == 0
        ):
            raise ValueError(
                f"expected nonempty (N,{self.n_channels},T), got {tuple(x.shape)}"
            )
        if not x.is_floating_point():
            raise ValueError("gait windows must be floating point")

    def _magnitudes(self, x: torch.Tensor) -> torch.Tensor:
        if not self.add_magnitudes or not self.vector_groups:
            return x[:, :0, :]
        return torch.stack(
            [
                torch.linalg.vector_norm(x[:, group, :], dim=1)
                for group in self.vector_groups
            ],
            dim=1,
        )

    def _signals(self, x: torch.Tensor) -> torch.Tensor:
        # Use physical vectors before axis standardization.
        magnitudes = self._magnitudes(x)
        if self.normalization:
            x = (x - self.channel_mean) / self.channel_std
            magnitudes = (magnitudes - self.magnitude_mean) / self.magnitude_std
        return torch.cat((x, magnitudes), dim=1) if self.add_magnitudes else x

    def _statistics(self, x: torch.Tensor) -> torch.Tensor:
        # Float32 FFT also supports half inputs at arbitrary window lengths.
        x = x.float()
        mean = x.mean(-1)
        centered = x - mean.unsqueeze(-1)
        variance = centered.square().mean(-1)
        std = variance.clamp_min(self.norm_eps**2).sqrt()
        rms = x.square().mean(-1).clamp_min(self.norm_eps**2).sqrt()
        features = [std.log1p(), rms.log1p()]
        if self.stats_include_mean:
            features.append(mean)
        power = torch.fft.rfft(centered, dim=-1, norm="ortho").abs().square()
        frequencies = torch.arange(power.shape[-1], device=x.device) / x.shape[-1]
        total = power.sum(-1).clamp_min(self.norm_eps)
        for lo, hi in ((0, 0.0625), (0.0625, 0.125), (0.125, 0.25), (0.25, 0.5)):
            mask = (frequencies > lo) & (frequencies <= hi)
            features.append((power * mask).sum(-1) / total)
        for fraction in (1 / 16, 1 / 8, 1 / 4):
            lag = max(1, round(x.shape[-1] * fraction))
            corr = (
                (centered[..., lag:] * centered[..., :-lag]).mean(-1)
                if lag < x.shape[-1]
                else variance * 0
            )
            features.append((corr / variance.clamp_min(self.norm_eps**2)).clamp(-1, 1))
        normalized = centered / std.unsqueeze(-1)
        covariance = normalized @ normalized.transpose(1, 2) / x.shape[-1]
        features.append(
            covariance[
                :, self.correlation_indices[0], self.correlation_indices[1]
            ].clamp(-1, 1)
        )
        return torch.cat(features, dim=1)

    @torch.no_grad()
    def fit_normalization(self, X, *, chunk_size: int = 256) -> GaitEncoder:
        """Fit channel and optional feature moments on TRAIN X only; return self.

        Population moments use float64 bounded chunks for numerical stability.
        No dataset is retained. Call after the subject-heldout validation split.
        ``normalization=False`` leaves channel scaling as identity; statistics,
        when requested, are still standardized using training moments.
        """
        if not self.training:
            raise RuntimeError(
                "fit_normalization is train-only; call before eval on training X"
            )
        X = torch.as_tensor(X)
        self._validate(X)
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        device = self.channel_mean.device

        # Merge chunk means/M2 rather than E[x²]-E[x]² to preserve small
        # variation even when a sensor has a large DC offset.
        def collect(transform, reduction, shape):
            count = 0
            mean = torch.zeros(shape, dtype=torch.float64, device=device)
            m2 = torch.zeros_like(mean)
            for chunk in X.split(chunk_size):
                chunk = chunk.to(device=device, dtype=torch.float32)
                if not torch.isfinite(chunk).all():
                    raise ValueError("normalization training windows must be finite")
                values = transform(chunk).double()
                batch_count = math.prod(values.shape[d] for d in reduction)
                batch_mean = values.mean(reduction, keepdim=True)
                batch_m2 = ((values - batch_mean) ** 2).sum(reduction, keepdim=True)
                delta = batch_mean - mean
                total_count = count + batch_count
                m2 += batch_m2 + delta.square() * count * batch_count / total_count
                mean += delta * batch_count / total_count
                count = total_count
            return mean.float(), (m2 / count).sqrt().clamp_min(self.norm_eps).float()

        if self.normalization:
            mean, std = collect(lambda x: x, (0, 2), self.channel_mean.shape)
            self.channel_mean.copy_(mean)
            self.channel_std.copy_(std)
            if self.add_magnitudes and self.vector_groups:
                mean, std = collect(self._magnitudes, (0, 2), self.magnitude_mean.shape)
                self.magnitude_mean.copy_(mean)
                self.magnitude_std.copy_(std)
        if self.stats_fusion:
            mean, std = collect(
                lambda x: self._statistics(self._signals(x)),
                (0,),
                self.stats_mean.shape,
            )
            self.stats_mean.copy_(mean)
            self.stats_std.copy_(std)
        self.normalization_fitted.fill_(True)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate(x)
        if self.architecture == "legacy":
            return self.encoder(self._signals(x))
        signals = self._signals(x)
        h = self.encoder(self.stem(signals))
        std = h.var(dim=-1, unbiased=False).clamp_min(self.norm_eps**2).sqrt()
        if self.architecture == "multiscale":
            pooled = torch.cat((h.mean(-1), std), dim=1)
        else:
            pooled = torch.cat(
                (F.adaptive_avg_pool1d(h, self.temporal_bins).flatten(1), std), dim=1
            )
        if self.stats_fusion:
            stats = (self._statistics(signals) - self.stats_mean) / self.stats_std
            # Unseen constant/near-constant features cannot explode a projection.
            stats = self.stats_projection(stats.clamp(-10, 10).to(h.dtype))
            pooled = torch.cat((pooled, stats), dim=1)
        return F.normalize(self.projection(pooled), p=2, dim=1)


def build_gait_encoder(n_channels: int, model_args: dict | None = None) -> nn.Module:
    """Build an opt-in encoder without changing the legacy EmbeddingNet class."""
    return GaitEncoder(n_channels, model_args or {})
