"""Siamese embedding training (triplet / contrastive) for gait verification.

CLI:
    python -m engine.train_siamese --config configs/siamese.yaml --epochs 2 \
        --subset 2000 --loss triplet --out-dir runs/smoke_verification/siamese

Reads a YAML config; CLI flags override `epochs`, `subset`, `out_dir`, and
`loss` (CLI `--loss` wins over config `loss_type`). Trains an `EmbeddingNet`
wrapped in `SiameseNet` with Adam, writing `<out_dir>/best.pt` (via
`save_checkpoint`) and `<out_dir>/metrics.json`.

Two optional config keys tune the training protocol; both default to the
previous behaviour:

  - `triplets_per_epoch` (int, default 1): triplet sampler passes per epoch.
    The sampler draws `triplets_per_epoch * n_train_windows` triplets, so the
    optimiser budget scales with this key rather than being capped at one
    dataset pass per epoch. Applies to the triplet loss only.
  - `val_subjects` (int, default 0 = disabled): number of distinct subjects
    carved out of the TRAIN split as a subject-disjoint validation set. When
    > 0, `best.pt` is selected on validation EER (lower is better) computed by
    `engine.evaluate.verification_metrics` instead of the training loss.
"""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
import yaml
from torch.utils.data import default_collate

from data.adapter import WindowData, build_loader, load_windows
from engine.evaluate import verification_metrics
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


def _n_triplets(config: dict, n_samples: int) -> int:
    """Number of triplets to draw per epoch.

    `triplets_per_epoch` (default 1) multiplies one dataset pass, so the
    default reproduces the historical budget exactly.
    """
    return int(config.get("triplets_per_epoch", 1)) * int(n_samples)


def _selection(val_data) -> tuple[str, str]:
    """Checkpoint selection ``(metric_name, per_epoch_key)``.

    With a validation split, selection minimises validation EER; without one
    (`val_subjects` = 0) the historical training-loss selection is preserved
    verbatim. The report name is "train_loss" while the per-epoch record field
    is "loss", so both are returned from this single source.
    """
    if val_data is None:
        return "train_loss", "loss"
    return "val_eer", "val_eer"


def _select_best_epoch(per_epoch: list[dict], metric: str) -> int:
    """Epoch with the lowest `metric` so far (ties resolve to the earliest).

    Both selectable metrics (training loss, validation EER) are error measures,
    so lower is better. This reproduces the historical ``value < best`` rule:
    an epoch is a new best exactly when it is the running minimum, and equal
    values keep the earlier epoch.
    """
    return min(per_epoch, key=lambda record: record[metric])["epoch"]


def _split_validation(data: WindowData, val_subjects: int, seed: int):
    """Carve a subject-disjoint validation split out of `data`.

    Rule: the distinct subject ids present in `data` are sorted ascending,
    shuffled with ``random.Random(seed)``, and the first `val_subjects` become
    the validation subjects. The split is deterministic for a fixed seed and
    no subject appears in both portions. Returns
    ``(train_data, val_data, val_subject_ids)`` as `WindowData` objects so
    `build_loader` keeps working.
    """
    subjects = sorted(int(s) for s in torch.unique(data.y_subject).tolist())
    if not 1 <= val_subjects < len(subjects):
        raise ValueError(
            f"val_subjects must satisfy 1 <= val_subjects < {len(subjects)} "
            f"(the distinct subjects in the training split); got {val_subjects}"
        )
    val_subject_ids = sorted(random.Random(seed).sample(subjects, val_subjects))
    val_set = set(val_subject_ids)

    val_mask = torch.tensor(
        [int(s) in val_set for s in data.y_subject], dtype=torch.bool
    )

    def _take(idx: torch.Tensor) -> WindowData:
        return WindowData(
            X=data.X[idx],
            y_activity=data.y_activity[idx],
            y_subject=data.y_subject[idx],
            activity_names=data.activity_names,
            subject_ids=data.subject_ids,
            sampling_rate=data.sampling_rate,
        )

    train_idx = torch.nonzero(~val_mask, as_tuple=False).flatten()
    val_idx = torch.nonzero(val_mask, as_tuple=False).flatten()
    return _take(train_idx), _take(val_idx), val_subject_ids


def _embed_windows(embedding_net: EmbeddingNet, data: WindowData, batch_size: int,
                   num_workers: int):
    """Embed every window in `data`; returns an (N, embed_dim) numpy array."""
    embedding_net.eval()
    loader = build_loader(data, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers)
    embs = []
    with torch.no_grad():
        for batch in loader:
            embs.append(embedding_net(batch[0]).numpy())
    return np.concatenate(embs)


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

    # Optional subject-disjoint validation carve (default 0 = disabled).
    val_subjects = int(config.get("val_subjects", 0))
    val_data = None
    n_train_subjects_total = n_val_subjects = n_val_windows = 0
    if val_subjects > 0:
        n_train_subjects_total = int(torch.unique(data.y_subject).numel())
        data, val_data, val_subject_ids = _split_validation(data, val_subjects, seed)
        n_val_subjects = len(val_subject_ids)
        n_val_windows = int(len(val_data.X))

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
    best_epoch = 0
    selection_metric, selection_key = _selection(val_data)

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        n_seen = 0
        # seed + epoch - 1: deterministic per epoch, different across epochs.
        if loss_type == "triplet":
            sampler = TripletWindowSampler(
                data, n_triplets=_n_triplets(config, n_samples), seed=seed + epoch - 1
            )
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
        record = {"epoch": epoch, "loss": epoch_loss}
        val_metrics = None
        if val_data is not None:
            val_metrics = verification_metrics(
                _embed_windows(
                    model.embedding, val_data, batch_size,
                    int(config.get("num_workers", 0)),
                ),
                val_data.y_subject.numpy(),
                n_pairs=10000,
                seed=seed,
            )
            record["val_eer"] = val_metrics["eer"]
            record["val_roc_auc"] = val_metrics["roc_auc"]
        per_epoch.append(record)

        if val_metrics is None:
            print(f"epoch {epoch}/{epochs} loss={epoch_loss:.4f}")
        else:
            print(
                f"epoch {epoch}/{epochs} loss={epoch_loss:.4f} "
                f"val_eer={val_metrics['eer']:.4f} "
                f"val_roc_auc={val_metrics['roc_auc']:.4f}"
            )

        if _select_best_epoch(per_epoch, selection_key) == epoch:
            best_epoch = epoch
            os.makedirs(out_dir, exist_ok=True)
            checkpoint_metrics = {
                "loss": epoch_loss,
                # Lowest mean validation EER when a subject-disjoint
                # validation split exists, else the mean training pair/triplet
                # loss (the default `val_subjects: 0`).
                "selection_metric": selection_metric,
            }
            if val_metrics is not None:
                checkpoint_metrics["val_eer"] = val_metrics["eer"]
                checkpoint_metrics["val_roc_auc"] = val_metrics["roc_auc"]
            save_checkpoint(
                os.path.join(out_dir, "best.pt"),
                model,
                optimizer,
                epoch,
                checkpoint_metrics,
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
        # Selected on the lowest validation EER when `val_subjects` > 0, else
        # on the mean training loss (the default, no validation split).
        "selection_metric": selection_metric,
        "n_params": sum(p.numel() for p in model.parameters()),
        "margin": margin,
        "seed": seed,
        "git_commit": _git_commit(),
    }
    if val_data is not None:
        n_train_windows = int(len(data.X))
        metrics["n_val_subjects"] = n_val_subjects
        metrics["n_val_windows"] = n_val_windows
        metrics["val_subject_fraction"] = n_val_subjects / n_train_subjects_total
        metrics["val_window_fraction"] = n_val_windows / (
            n_train_windows + n_val_windows
        )
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
