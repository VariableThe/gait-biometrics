"""Siamese embedding training (triplet / contrastive) for gait verification.

CLI:
    python -m engine.train_siamese --config configs/siamese.yaml --epochs 2 \
        --subset 2000 --loss triplet --out-dir runs/smoke_verification/siamese

Reads a YAML config; CLI flags override `epochs`, `subset`, `out_dir`, and
`loss` (CLI `--loss` wins over config `loss_type`). Trains an `EmbeddingNet`
wrapped in `SiameseNet` with Adam, writing `<out_dir>/best.pt` (via
`save_checkpoint`) and `<out_dir>/metrics.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import random

import torch
import yaml
from torch.utils.data import default_collate

from data.adapter import load_windows
from engine.train import _git_commit, _stratified_subset
from models.common import save_checkpoint, seed_everything
from models.siamese import (
    EmbeddingNet,
    SiameseNet,
    TripletWindowSampler,
    contrastive_loss,
    triplet_loss,
)


def _batched(iterable, batch_size: int):
    """Yield lists of up to `batch_size` items from `iterable`."""
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == batch_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _contrastive_pairs(data, n_pairs: int, seed: int):
    """Yield (x1, x2, label) pairs: 1.0 = genuine (same subject), 0.0 = impostor."""
    rng = random.Random(seed)
    subjects = data.y_subject.cpu().numpy()
    subject_to_indices: dict[int, list[int]] = {}
    for idx, s in enumerate(subjects):
        subject_to_indices.setdefault(int(s), []).append(idx)
    all_subjects = sorted(subject_to_indices)
    if len(all_subjects) < 2:
        raise ValueError(
            "contrastive training requires at least 2 subjects; "
            f"found {len(all_subjects)}"
        )
    valid = [s for s in all_subjects if len(subject_to_indices[s]) >= 2]
    X = data.X
    for _ in range(n_pairs):
        if valid and rng.random() < 0.5:
            s = rng.choice(valid)
            i, j = rng.sample(subject_to_indices[s], 2)
            yield X[i], X[j], 1.0
        else:
            s1, s2 = rng.sample(all_subjects, 2)
            i = rng.choice(subject_to_indices[s1])
            j = rng.choice(subject_to_indices[s2])
            yield X[i], X[j], 0.0


def train_siamese(config: dict, epochs: int, subset: int | None,
                  loss_type: str, out_dir: str) -> dict:
    """Train the siamese embedding net per `config`; returns the metrics dict."""
    if loss_type not in ("triplet", "contrastive"):
        raise ValueError(
            f"unsupported loss_type {loss_type!r}; use 'triplet' or 'contrastive'"
        )
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    torch.set_num_threads(int(config.get("torch_threads", 4)))

    source = config["source"]
    data = load_windows(
        source,
        "train",
        window_size=int(config.get("window_size", 128)),
        stride=int(config.get("stride", 64)),
    )
    if data.y_subject is None:
        raise ValueError(
            f"siamese training requires subject labels, but source {source!r} "
            "has y_subject=None; genuine/impostor pairs cannot be formed. "
            "Use source 'har'."
        )
    if subset is not None:
        data = _stratified_subset(data, int(subset), seed)

    model_args = dict(config.get("model_args", {}))
    embed_dim = int(model_args.get("embed_dim", 128))
    dropout = float(model_args.get("dropout", 0.3))
    margin = float(model_args.get("margin", 1.0))
    n_channels = int(data.X.shape[1])

    model = SiameseNet(EmbeddingNet(n_channels, embed_dim=embed_dim, dropout=dropout))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.get("lr", 1e-3)),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )

    batch_size = int(config.get("batch_size", 64))
    n_samples = int(subset) if subset is not None else len(data.X)

    per_epoch: list[dict] = []
    best_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        n_seen = 0
        # seed + epoch - 1: deterministic per epoch, different across epochs.
        if loss_type == "triplet":
            sampler = TripletWindowSampler(data, n_triplets=n_samples, seed=seed + epoch - 1)
            pair_iter = _batched(sampler, batch_size)
            for chunk in pair_iter:
                a, p, neg = default_collate(chunk)
                optimizer.zero_grad()
                za = model.embedding(a)
                zp = model.embedding(p)
                zn = model.embedding(neg)
                loss = triplet_loss(za, zp, zn, margin)
                loss.backward()
                optimizer.step()
                bs = a.size(0)
                loss_sum += loss.item() * bs
                n_seen += bs
        else:  # contrastive
            sampler = _contrastive_pairs(data, n_samples, seed + epoch - 1)
            for chunk in _batched(sampler, batch_size):
                x1, x2, label = default_collate(chunk)
                optimizer.zero_grad()
                z1, z2 = model(x1, x2)
                loss = contrastive_loss(z1, z2, label, margin)
                loss.backward()
                optimizer.step()
                bs = x1.size(0)
                loss_sum += loss.item() * bs
                n_seen += bs

        epoch_loss = loss_sum / max(n_seen, 1)
        per_epoch.append({"epoch": epoch, "loss": epoch_loss})
        print(f"epoch {epoch}/{epochs} loss={epoch_loss:.4f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch
            os.makedirs(out_dir, exist_ok=True)
            save_checkpoint(
                os.path.join(out_dir, "best.pt"),
                model,
                optimizer,
                epoch,
                {"loss": epoch_loss},
            )

    metrics = {
        "model": "siamese",
        "loss_type": loss_type,
        "embed_dim": embed_dim,
        "source": source,
        "epochs": epochs,
        "subset": subset,
        "per_epoch": per_epoch,
        "final_loss": per_epoch[-1]["loss"],
        "best_epoch": best_epoch,
        "n_params": sum(p.numel() for p in model.parameters()),
        "margin": margin,
        "seed": seed,
        "git_commit": _git_commit(),
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Siamese embedding training (triplet/contrastive)"
    )
    parser.add_argument("--config", required=True, help="path to the YAML config")
    parser.add_argument("--epochs", type=int, default=None, help="override config epochs")
    parser.add_argument("--subset", type=int, default=None, help="override config subset")
    parser.add_argument(
        "--loss", choices=["triplet", "contrastive"], default=None,
        help="override config loss_type",
    )
    parser.add_argument("--out-dir", default=None, help="override config out_dir")
    args = parser.parse_args(argv)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    epochs = args.epochs if args.epochs is not None else int(config["epochs"])
    subset = args.subset if args.subset is not None else config.get("subset")
    loss_type = args.loss if args.loss is not None else str(config.get("loss_type", "triplet"))
    out_dir = args.out_dir if args.out_dir is not None else config["out_dir"]

    train_siamese(config, epochs, subset, loss_type, out_dir)


if __name__ == "__main__":
    main()
