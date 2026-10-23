"""Opt-in serving transforms; restore ordinary checkpoints before applying them."""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from models.gait_encoders import GaitEncoder, _MultiscaleBlock


class _PackedMultiscaleBlock(nn.Module):
    def __init__(self, source: _MultiscaleBlock) -> None:
        super().__init__()
        branches = source.branches
        if len(branches) != 4 or any(
            not isinstance(branch, nn.Conv1d)
            or branch.kernel_size != (kernel,)
            or branch.stride != (1,)
            or branch.dilation != (1,)
            or branch.groups != 1
            or branch.padding != (kernel // 2,)
            or branch.padding_mode != "zeros"
            or branch.bias is not None
            or branch.in_channels != branches[0].in_channels
            or branch.out_channels != branches[0].out_channels
            for branch, kernel in zip(branches, (3, 7, 15, 31))
        ):
            raise ValueError("packing requires the standard 3/7/15/31 Conv1d branches")
        weights = torch.cat(
            [
                F.pad(branch.weight, ((31 - branch.kernel_size[0]) // 2,) * 2)
                for branch in branches
            ],
            dim=0,
        )
        self.bottleneck = source.bottleneck
        # Reuse the copied longest branch's convolution configuration. Only its
        # output width and weight change; no random initialization is needed.
        self.packed = branches[-1]
        self.packed.out_channels = weights.shape[0]
        self.packed.weight = nn.Parameter(weights, requires_grad=False)
        self.mix = source.mix

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x + self.mix(self.packed(self.bottleneck(x))))


class _FrozenPackedEncoder(nn.Module):
    def __init__(self, model: GaitEncoder) -> None:
        super().__init__()
        self.model = model
        self.requires_grad_(False)
        self.train(False)

    def train(self, mode: bool = True) -> _FrozenPackedEncoder:
        if mode:
            raise RuntimeError(
                "packed encoders are inference-only; training is unsupported"
            )
        return super().train(False)

    def fit_normalization(self, *args, **kwargs):
        raise RuntimeError(
            "packed encoders are inference-only; normalization is frozen"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


@torch.no_grad()
def pack_multiscale_branches(model: GaitEncoder) -> nn.Module:
    """Return a deep-copied, frozen eval module with one Conv1d per branch bank.

    Accepts only a multiscale ``GaitEncoder`` with every submodule in eval mode;
    other architectures and training modules raise clear errors. The caller's
    weights, buffers, trainability, and mode remain unchanged. Channel ordering,
    bottlenecks, mixing, and saved normalization are preserved. The four kernels
    are centered in length-31 zero-padded weights and stacked on the output axis.

    This trades extra parameters and zero multiplies for fewer dispatches. Speed
    depends on device and batch size; validate outputs on the serving backend.
    The returned module rejects training and normalization fitting. It is a
    serving artifact, not a new resumable/weights-only checkpoint format: load
    the original checkpoint normally, call ``eval()``, then apply this helper.
    """
    if not isinstance(model, GaitEncoder) or model.architecture != "multiscale":
        raise ValueError("packing requires a multiscale GaitEncoder")
    if any(module.training for module in model.modules()):
        raise ValueError("packing is eval-only; call model.eval() first")
    packed = deepcopy(model)
    blocks = list(packed.encoder.children())
    if not blocks or any(not isinstance(block, _MultiscaleBlock) for block in blocks):
        raise ValueError("packing requires standard multiscale encoder blocks")
    packed.encoder = nn.Sequential(*[_PackedMultiscaleBlock(block) for block in blocks])
    return _FrozenPackedEncoder(packed)
