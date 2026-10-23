"""Opt-in training augmentations on individual inertial windows only.

All defaults are identity. The caller must invoke this only for training.
One proper SO(3) rotation is shared by all triads and timesteps of a window.
"""

from __future__ import annotations

import math

import torch


def _groups(channels, value):
    if value is None:
        value = (
            [range(i, i + 3) for i in range(0, channels, 3)]
            if channels in (3, 9)
            else []
        )
    groups = tuple(tuple(int(i) for i in group) for group in value)
    indices = [i for group in groups for i in group]
    if any(len(group) != 3 for group in groups) or len(set(indices)) != len(indices):
        raise ValueError("vector_groups must contain disjoint three-axis groups")
    if any(i < 0 or i >= channels for i in indices):
        raise ValueError("vector_groups channel index out of range")
    return groups


def augment_gait(x: torch.Tensor, config: dict | None, generator=None) -> torch.Tensor:
    """Augment (N,C,T) or (C,T), preserving rows and input storage.

    Keys: jitter_std=0, scale_std=0 (positive lognormal, one per vector),
    roll_fraction=0 (max circular shift / T), rotation_degrees=0 (max angle),
    vector_groups=None. For C=3/9 contiguous triads are the known defaults.
    Rotation uses the same matrix for every related vector in each window;
    scaling uses a single factor per triad, never separate axis factors.
    Supply a seeded torch.Generator for repeatability (CPU generator works
    with CUDA tensors too). No interpolation, row mixing, or test-time hook.
    """
    config = dict(config or {})
    allowed = {
        "jitter_std",
        "scale_std",
        "roll_fraction",
        "rotation_degrees",
        "vector_groups",
    }
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"unknown gait augmentation settings: {sorted(unknown)}")
    if x.ndim not in (2, 3) or not x.is_floating_point() or min(x.shape) == 0:
        raise ValueError("expected a nonempty floating (C,T) or (N,C,T) tensor")
    original_ndim = x.ndim
    result = x.unsqueeze(0).clone() if original_ndim == 2 else x.clone()
    n, channels, length = result.shape
    groups = _groups(channels, config.get("vector_groups"))
    values = {key: float(config.get(key, 0)) for key in allowed - {"vector_groups"}}
    if any(not math.isfinite(v) or v < 0 for v in values.values()):
        raise ValueError("augmentation strengths must be finite and nonnegative")
    if values["roll_fraction"] > 0.5 or values["rotation_degrees"] > 180:
        raise ValueError("roll_fraction <= .5 and rotation_degrees <= 180 required")

    def sample(shape, *, normal=False):
        random_device = generator.device if generator is not None else result.device
        function = torch.randn if normal else torch.rand
        return function(
            shape, device=random_device, generator=generator, dtype=torch.float32
        ).to(device=result.device, dtype=result.dtype)

    if values["rotation_degrees"]:
        if not groups:
            raise ValueError(
                "rotation requires explicit vector_groups for unknown channel layouts"
            )
        axis = sample((n, 3), normal=True)
        axis = axis / axis.norm(dim=1, keepdim=True).clamp_min(1e-8)
        angle = (sample((n,)) * 2 - 1) * math.radians(values["rotation_degrees"])
        a, b, c = axis.unbind(1)
        zero = torch.zeros_like(a)
        skew = torch.stack((zero, -c, b, c, zero, -a, -b, a, zero), dim=1).reshape(
            n, 3, 3
        )
        identity = torch.eye(3, device=result.device, dtype=result.dtype).expand(
            n, -1, -1
        )
        rotation = (
            identity
            + angle.sin()[:, None, None] * skew
            + (1 - angle.cos())[:, None, None] * (skew @ skew)
        )
        for group in groups:
            result[:, group, :] = rotation @ result[:, group, :]
    if values["scale_std"]:
        assigned = {i for group in groups for i in group}
        scale_groups = list(groups) + [
            (i,) for i in range(channels) if i not in assigned
        ]
        for group in scale_groups:
            scale = (sample((n, 1, 1), normal=True) * values["scale_std"]).exp()
            result[:, group, :] = result[:, group, :] * scale
    if values["roll_fraction"]:
        maximum = int(length * values["roll_fraction"])
        shifts = (sample((n,)) * (2 * maximum + 1)).floor().long() - maximum
        indices = (
            torch.arange(length, device=result.device)[None, :] - shifts[:, None]
        ) % length
        result = result.gather(2, indices[:, None, :].expand(-1, channels, -1))
    if values["jitter_std"]:
        result = result + sample(result.shape, normal=True) * values["jitter_std"]
    return result.squeeze(0) if original_ndim == 2 else result
