"""Shared training utilities: seeding, metric tracking, checkpointing."""

from __future__ import annotations

import random
from collections import defaultdict

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Seed random, numpy and torch; fix torch to 4 threads."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)


class MetricTracker:
    """Accumulate named scalar metrics and expose running averages."""

    def __init__(self) -> None:
        self._sums: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)
        self._history: dict[str, list[float]] = defaultdict(list)

    def update(self, name: str, value: float) -> None:
        self._sums[name] += float(value)
        self._counts[name] += 1
        self._history[name].append(float(value))

    def avg(self, name: str) -> float:
        if self._counts[name] == 0:
            raise KeyError(f"no values recorded for metric {name!r}")
        return self._sums[name] / self._counts[name]

    @property
    def history(self) -> dict:
        return dict(self._history)


def save_checkpoint(path: str, model, optimizer, epoch: int, metrics: dict) -> None:
    """Save model/optimizer state plus epoch and metrics to `path`."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": dict(metrics),
        },
        path,
    )


def load_checkpoint(path: str) -> dict:
    """Load the model state_dict from `path` written by `save_checkpoint`.

    Accepts either the rich checkpoint dict (``{"epoch", "model_state_dict",
    "optimizer_state_dict", "metrics"}``) or a bare ``state_dict``. Always
    loads with ``weights_only=True``: unpickling a checkpoint must never be able
    to execute arbitrary code, so only tensors, containers and plain scalars are
    accepted.

    Raises:
        ValueError: if the file does not contain a state_dict-shaped dict.
    """
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        state = obj["model_state_dict"]
    elif isinstance(obj, dict):
        state = obj
    else:
        raise ValueError(
            f"unrecognized checkpoint format in {path!r}: expected a state_dict "
            f"or a dict with 'model_state_dict', got {type(obj).__name__}"
        )
    if not isinstance(state, dict):
        raise ValueError(
            f"'model_state_dict' in {path!r} is {type(state).__name__}, not a dict"
        )
    return state
