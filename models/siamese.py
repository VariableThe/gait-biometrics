"""Siamese denoising autoencoders for gait verification (paper B4).

Sheng & Li, 2020, Neurocomputing — Siamese network over joint-trajectory
reconstructions with a denoising-autoencoder pretext task. This module
provides the embedding network, the siamese wrapper, triplet/contrastive
losses, and a window-level triplet sampler that forms genuine (same-subject)
and impostor (different-subject) pairs from a `WindowData` instance.

The DAE pretext head is included for completeness but is OFF the default
forward path (`EmbeddingNet.forward` never invokes it); a separate training
script can wire it in for the reconstruction objective.
"""

from __future__ import annotations

import random

import torch
from torch import nn
from torch.nn import functional as F


class EmbeddingNet(nn.Module):
    """Convolutional embedding network: (N, C, T) -> (N, embed_dim), L2-normalized.

    Args:
        n_channels: number of sensor channels in the input (N, C, T).
        embed_dim: dimensionality of the output embedding.
        dropout: dropout probability before the projection head.
    """

    def __init__(self, n_channels: int, embed_dim: int = 128, dropout: float = 0.3) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.embed_dim = embed_dim

        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.projection = nn.Sequential(
            nn.Linear(128, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

        # Optional DAE pretext head — OFF the default forward path.
        # A separate training script can call `forward_dae` for the
        # reconstruction objective.
        self.dae_decoder = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_channels * 128),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed a batch of windows: (N, C, T) -> (N, embed_dim), L2-normalized."""
        h = self.features(x).flatten(1)  # (N, 128)
        z = self.projection(h)  # (N, embed_dim)
        return F.normalize(z, p=2, dim=1)

    def forward_dae(self, x: torch.Tensor, noise_std: float = 0.1) -> torch.Tensor:
        """Denoising-autoencoder reconstruction: (N, C, T) -> (N, C * T).

        Adds Gaussian noise to the input, embeds the noisy window, and
        decodes a reconstruction of the clean input. Not called by
        `forward`; provided for the paper's pretext objective.
        """
        if noise_std > 0:
            noisy = x + torch.randn_like(x) * noise_std
        else:
            noisy = x
        z = self.forward(noisy)  # (N, embed_dim)
        return self.dae_decoder(z)  # (N, C * T)


class SiameseNet(nn.Module):
    """Siamese wrapper around an `EmbeddingNet`.

    The same embedding network processes both inputs (weight sharing), and
    the two embeddings are returned for a distance-based loss.
    """

    def __init__(self, embedding: EmbeddingNet) -> None:
        super().__init__()
        self.embedding = embedding

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed two batches: (N, C, T) x2 -> (N, embed_dim) x2."""
        return self.embedding(x1), self.embedding(x2)


def triplet_loss(a: torch.Tensor, p: torch.Tensor, n: torch.Tensor,
                 margin: float = 1.0) -> torch.Tensor:
    """Triplet loss: max(0, d(a, p) - d(a, n) + margin).

    Args:
        a: anchor embeddings (N, D).
        p: positive embeddings (N, D) — same subject as anchor.
        n: negative embeddings (N, D) — different subject from anchor.
        margin: minimum desired separation between positive and negative distances.
    """
    d_pos = (a - p).pow(2).sum(dim=1)  # squared L2
    d_neg = (a - n).pow(2).sum(dim=1)
    return F.relu(d_pos - d_neg + margin).mean()


def contrastive_loss(z1: torch.Tensor, z2: torch.Tensor,
                     label: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """Contrastive loss: y * d^2 + (1 - y) * max(0, margin - d)^2.

    Args:
        z1: embeddings of the first batch (N, D).
        z2: embeddings of the second batch (N, D).
        label: 1.0 for genuine (same-subject) pairs, 0.0 for impostor pairs (N,).
        margin: minimum separation for impostor pairs.
    """
    dist = (z1 - z2).pow(2).sum(dim=1)  # squared L2
    positive = label * dist
    negative = (1.0 - label) * F.relu(margin - dist.sqrt()).pow(2)
    return (positive + negative).mean()


class TripletWindowSampler:
    """Samples (anchor, positive, negative) window triplets from `WindowData`.

    Positives share the anchor's subject id; negatives do not. Requires
    subject labels — raises `ValueError` when `data.y_subject is None`
    (WISDM cannot form genuine/impostor pairs).

    Args:
        data: a `WindowData` instance with `y_subject` labels.
        n_triplets: number of triplets to generate per epoch.
        seed: random seed for reproducibility.
    """

    def __init__(self, data, n_triplets: int, seed: int = 42) -> None:
        if getattr(data, "y_subject", None) is None:
            raise ValueError(
                "TripletWindowSampler requires subject labels (y_subject); "
                "the provided data has y_subject=None, so genuine/impostor "
                "pairs cannot be formed."
            )
        self.data = data
        self.n_triplets = int(n_triplets)
        self.seed = int(seed)

        # Build subject -> row-index mapping.
        subjects = data.y_subject.cpu().numpy()
        self.subject_to_indices: dict[int, list[int]] = {}
        for idx, s in enumerate(subjects):
            self.subject_to_indices.setdefault(int(s), []).append(idx)
        self.subjects = sorted(self.subject_to_indices.keys())

    def __iter__(self):
        rng = random.Random(self.seed)
        X = self.data.X
        for _ in range(self.n_triplets):
            # Pick a subject that has at least 2 windows (anchor + positive).
            valid_subjects = [s for s in self.subjects
                              if len(self.subject_to_indices[s]) >= 2]
            if not valid_subjects:
                raise ValueError(
                    "no subject has at least 2 windows; cannot form triplets"
                )
            anchor_subject = rng.choice(valid_subjects)
            anchor_idx, pos_idx = rng.sample(self.subject_to_indices[anchor_subject], 2)

            # Pick a negative from a different subject.
            neg_subjects = [s for s in self.subjects if s != anchor_subject]
            neg_subject = rng.choice(neg_subjects)
            neg_idx = rng.choice(self.subject_to_indices[neg_subject])

            yield X[anchor_idx], X[pos_idx], X[neg_idx]

    def __len__(self) -> int:
        return self.n_triplets
