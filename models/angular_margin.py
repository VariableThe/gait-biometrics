"""ArcFace identity-classification head adapted to inertial gait embeddings.

Research basis: Deng et al., https://arxiv.org/abs/1801.07698v3, Eq. 3.
This head returns logits for cross entropy, rather than computing a loss.
"""

from __future__ import annotations

import math
from numbers import Integral, Real

import torch
from torch import nn
from torch.nn import functional as F


class AngularMarginClassifier(nn.Module):
    """Normalized, bias-free identity head with a target-only angular margin.

    ``embeddings`` must be finite floating-point ``(B, in_features)`` values.
    Optional ``labels`` are integer ``(B,)`` training-class indices on the
    same device. Omitting labels returns ordinary scaled cosine logits,
    independently of train/eval mode. Passing labels applies the standard
    monotonic ArcFace extension, then scales all logits. All head arithmetic
    and returned logits use float32, including inside an autocast context.

    A zero vector normalizes to zero. At cosine +/-1, the square-root term
    uses a finite zero endpoint subgradient while preserving its exact value.
    Save ``state_dict()`` plus constructor arguments, not the module object.
    """

    def __init__(
        self,
        in_features: int,
        n_classes: int,
        scale: float = 30.0,
        margin: float = 0.3,
    ) -> None:
        super().__init__()
        for name, value in (("in_features", in_features), ("n_classes", n_classes)):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be a positive integer")
            if value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (("scale", scale), ("margin", margin)):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a finite real number")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if scale <= 0:
            raise ValueError("scale must be positive")
        if not 0 <= margin < math.pi / 2:
            raise ValueError("margin must be in [0, pi/2) radians")

        self.in_features = int(in_features)
        self.n_classes = int(n_classes)
        self.scale = float(scale)
        self.margin = float(margin)
        self._cos_margin = math.cos(self.margin)
        self._sin_margin = math.sin(self.margin)
        self._threshold = math.cos(math.pi - self.margin)
        self._fallback_offset = math.sin(math.pi - self.margin) * self.margin
        self.weight = nn.Parameter(torch.empty(self.n_classes, self.in_features))
        nn.init.xavier_uniform_(self.weight)

    @staticmethod
    def _normalize(values: torch.Tensor) -> torch.Tensor:
        # A direct fp32 L2 norm can overflow for finite rows such as [1e20, 0].
        # Rescale large rows first; rows near zero retain the original epsilon
        # behavior. The detached scale has no effect on the true derivative:
        # unit normalization is invariant to this positive scalar per row.
        row_scale = values.detach().abs().amax(dim=1, keepdim=True).clamp_min(1.0)
        return F.normalize(values / row_scale, p=2, dim=1, eps=1e-12)

    def forward(
        self, embeddings: torch.Tensor, labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        if not isinstance(embeddings, torch.Tensor):
            raise TypeError("embeddings must be a floating-point tensor")
        if not embeddings.is_floating_point():
            raise TypeError("embeddings must be floating point")
        if (
            embeddings.layout != torch.strided
            or embeddings.ndim != 2
            or embeddings.shape[0] == 0
            or embeddings.shape[1] != self.in_features
        ):
            raise ValueError(
                f"embeddings must have nonempty shape (B, {self.in_features})"
            )
        if embeddings.device != self.weight.device:
            raise ValueError("embeddings and classifier weight must share a device")

        indices = None
        if labels is not None:
            if not isinstance(labels, torch.Tensor):
                raise TypeError("labels must be an integer tensor")
            if labels.dtype not in (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            ):
                raise TypeError("labels must have an integer dtype (not bool)")
            if labels.layout != torch.strided or labels.shape != (len(embeddings),):
                raise ValueError("labels must have shape (B,), one per embedding")
            if labels.device != embeddings.device:
                raise ValueError("labels and embeddings must share a device")
            indices = labels.long().unsqueeze(1)
            if ((indices < 0) | (indices >= self.n_classes)).any():
                raise ValueError(f"labels must be in [0, {self.n_classes})")

        # Disabling autocast is necessary: .float() alone does not prevent
        # autocast from reducing precision again inside the matrix multiply.
        with torch.autocast(device_type=embeddings.device.type, enabled=False):
            values = embeddings.float()
            if not torch.isfinite(values).all():
                raise ValueError(
                    "embeddings must be finite and representable in float32"
                )
            cosine = F.linear(
                self._normalize(values),
                self._normalize(self.weight.float()),
            ).clamp(-1.0, 1.0)
            if indices is None or self.margin == 0:
                return cosine * self.scale

            target = cosine.gather(1, indices)
            sine_squared = (1.0 - target.square()).clamp_min(0.0)
            # sqrt(0) has an infinite derivative, even in an unselected where
            # branch. Floor its input before sqrt, then restore exact zeros.
            # For fp32 cosine values strictly inside [-1,1], 1-cos^2 >= eps.
            sine = sine_squared.clamp_min(torch.finfo(torch.float32).eps).sqrt()
            sine = torch.where(sine_squared > 0, sine, torch.zeros_like(sine))
            angular_target = target * self._cos_margin - sine * self._sin_margin
            target_with_margin = torch.where(
                target > self._threshold,
                angular_target,
                target - self._fallback_offset,
            )
            # Out-of-place scatter changes only the labeled class and retains
            # gradients through both normalized embeddings and class weights.
            return cosine.scatter(1, indices, target_with_margin) * self.scale
