"""Identity-balanced batches and differentiable metric-learning objectives.

Adapted from Hermans et al. (arXiv:1703.07737) and Khosla et al.
(arXiv:2004.11362); these sensor-window adaptations are not reproductions.
"""

from __future__ import annotations

import math
import random

import torch
from torch.nn import functional as F


class IdentityBalancedBatchSampler:
    """P distinct nonsingleton identities, K windows each, deterministic by epoch.

    Every eligible identity is visited before beginning another shuffled cycle.
    The last batch of a cycle is filled without repeating identities within a
    batch. Rows are unique when a class has >=K rows; otherwise every row is
    used before replacement. Singleton identities are reported, never silently
    promoted to a duplicated-row positive. Pass sampled row IDs to the losses
    to exclude copies of the same original window from positive pairs.
    """

    def __init__(
        self,
        labels,
        identities_per_batch: int = 8,
        windows_per_identity: int = 4,
        steps_per_epoch: int | None = None,
        seed: int = 42,
    ):
        labels = torch.as_tensor(labels)
        if (
            labels.ndim != 1
            or not len(labels)
            or labels.dtype
            not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
        ):
            raise ValueError("labels must be a nonempty 1D integer array")
        if identities_per_batch < 2 or windows_per_identity < 2:
            raise ValueError("identity-balanced sampling requires P >= 2 and K >= 2")
        self.groups: dict[int, list[int]] = {}
        for row, label in enumerate(labels.tolist()):
            self.groups.setdefault(int(label), []).append(row)
        self.identities = sorted(s for s, rows in self.groups.items() if len(rows) >= 2)
        self.singleton_ids = sorted(
            s for s, rows in self.groups.items() if len(rows) < 2
        )
        if len(self.identities) < 2:
            raise ValueError(
                "need at least two identities with at least two distinct windows each"
            )
        self.requested_p = int(identities_per_batch)
        self.p = min(self.requested_p, len(self.identities))
        self.k = int(windows_per_identity)
        self.steps = (
            math.ceil(len(labels) / (self.p * self.k))
            if steps_per_epoch is None
            else int(steps_per_epoch)
        )
        if self.steps <= 0:
            raise ValueError("steps_per_epoch must be positive")
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.steps

    @property
    def context(self) -> dict:
        return {
            "requested_identities_per_batch": self.requested_p,
            "identities_per_batch": self.p,
            "windows_per_identity": self.k,
            "steps_per_epoch": self.steps,
            "eligible_subject_ids": self.identities,
            "excluded_singleton_subject_ids": self.singleton_ids,
            "excluded_singleton_windows": len(self.singleton_ids),
            "eligible_identity_count": len(self.identities),
            "negative_identity_coverage": "nonsingleton_optimizer_train_identities_only",
            "identity_coverage_per_epoch_guaranteed": self.steps * self.p
            >= len(self.identities),
            "row_replacement": "only_when_identity_has_fewer_than_K_rows",
        }

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        remaining: list[int] = []
        for _ in range(self.steps):
            selected = []
            while len(selected) < self.p:
                if not remaining:
                    remaining = self.identities.copy()
                    rng.shuffle(remaining)
                # Keep repeated boundary identities for the following batch.
                position = next(
                    (i for i, s in enumerate(remaining) if s not in selected), None
                )
                if position is None:
                    choices = [s for s in self.identities if s not in selected]
                    selected.extend(rng.sample(choices, self.p - len(selected)))
                    break
                selected.append(remaining.pop(position))
            batch = []
            for subject in selected:
                rows = self.groups[subject]
                draw = rng.sample(rows, min(self.k, len(rows)))
                draw.extend(rng.choices(rows, k=self.k - len(draw)))
                batch.extend(draw)
            yield batch


def _masks(embeddings, labels, sample_ids=None):
    if embeddings.ndim != 2 or embeddings.shape[1] < 1:
        raise ValueError("embeddings must have shape (N, D), D >= 1")
    labels = torch.as_tensor(labels, device=embeddings.device)
    if labels.shape != (len(embeddings),):
        raise ValueError("labels must have one entry per embedding")
    different_row = ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    if sample_ids is not None:
        sample_ids = torch.as_tensor(sample_ids, device=labels.device)
        if sample_ids.shape != labels.shape:
            raise ValueError("sample_ids must have one entry per embedding")
        different_row &= sample_ids[:, None] != sample_ids[None, :]
    same_label = labels[:, None] == labels[None, :]
    positive = same_label & different_row
    negative = ~same_label & different_row
    return positive, negative


def batch_hard_triplet_loss(
    embeddings: torch.Tensor,
    labels,
    margin: float = 0.2,
    *,
    softplus: bool = False,
    sample_ids=None,
) -> torch.Tensor:
    """Mean hardest-positive/nearest-negative Euclidean triplet loss.

    Invalid anchors are skipped. With no valid anchors the differentiable
    zero has finite gradients. The direct cdist kernel handles coincident and
    zero embeddings without a singular hand-written square-root gradient.
    Softplus applies to d_positive - d_negative (the margin is unused).
    """
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("margin must be finite and nonnegative")
    positive, negative = _masks(embeddings, labels, sample_ids)
    valid = positive.any(1) & negative.any(1)
    if not valid.any():
        return embeddings.sum() * 0.0
    distances = torch.cdist(
        embeddings, embeddings, p=2, compute_mode="donot_use_mm_for_euclid_dist"
    )
    hardest_positive = distances.masked_fill(~positive, -torch.inf).max(1).values[valid]
    hardest_negative = distances.masked_fill(~negative, torch.inf).min(1).values[valid]
    difference = hardest_positive - hardest_negative
    return (
        F.softplus(difference).mean()
        if softplus
        else F.relu(difference + margin).mean()
    )


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels,
    temperature: float = 0.1,
    *,
    sample_ids=None,
) -> torch.Tensor:
    """SupCon outside-log positive average; excludes self and duplicated rows.

    A single sensor window supplies each view; P×K supplies same-identity
    positives. Embeddings are normalized here, including in direct API calls.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    positive, negative = _masks(embeddings, labels, sample_ids)
    valid = positive.any(1) & negative.any(1)
    if not valid.any():
        return embeddings.sum() * 0.0
    z = F.normalize(embeddings, p=2, dim=1)
    logits = (z @ z.T) / temperature
    allowed = positive | negative
    # Only valid rows reach logsumexp, avoiding all-masked invalid rows.
    logits = logits[valid]
    positives = positive[valid]
    log_denominator = torch.logsumexp(
        logits.masked_fill(~allowed[valid], -torch.inf), dim=1
    )
    positive_mean = torch.where(positives, logits, 0.0).sum(1) / positives.sum(1)
    return (log_denominator - positive_mean).mean()


# Short alias for callers familiar with the paper's name.
supcon_loss = supervised_contrastive_loss
