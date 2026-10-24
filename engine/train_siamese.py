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
import inspect
import json
import math
import os
import random
import time

import numpy as np
import torch
import yaml

from data.adapter import WindowData, load_windows
from engine.evaluate import _verification_pairs, verification_metrics
from engine.train import (
    _filter_activities,
    _finite_setting,
    _git_commit,
    _integer_setting,
    _make_loader,
    _protocol,
    _runtime,
    _take,
    _validate_windows,
)
from engine.train import (
    _split_validation as _subject_split,
)
from models.common import save_checkpoint
from models.siamese import (
    EmbeddingNet,
    SiameseNet,
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
    multiplier = _integer_setting(
        config.get("triplets_per_epoch", 1), "triplets_per_epoch", 1
    )
    return multiplier * _integer_setting(n_samples, "n_samples", 1)


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
    return _subject_split(data, val_subjects, seed)


class _IdentitySampler:
    """Cache identity groups once; draw negatives without per-triplet scans."""

    def __init__(self, data: WindowData):
        if data.y_subject is None:
            raise ValueError("verification requires real subject labels")
        self.groups = {}
        for index, subject in enumerate(data.y_subject.tolist()):
            self.groups.setdefault(subject, []).append(index)
        self.subjects = sorted(self.groups)
        self.valid = [s for s in self.subjects if len(self.groups[s]) >= 2]
        if len(self.subjects) < 2:
            raise ValueError("verification requires at least 2 subjects in each split")
        if not self.valid:
            raise ValueError(
                "verification requires a subject with at least 2 windows for genuine pairs"
            )
        self.positions = {s: i for i, s in enumerate(self.subjects)}

    def indices(self, count: int, seed: int, triplets: bool):
        rng = random.Random(seed)
        for _ in range(count):
            if triplets or rng.random() < 0.5:
                subject = rng.choice(self.valid)
                anchor, positive = rng.sample(self.groups[subject], 2)
                if not triplets:
                    yield anchor, positive, 1
                    continue
                # Uniformly select one of S-1 alternatives in O(1).
                neg_position = rng.randrange(len(self.subjects) - 1)
                neg_position += neg_position >= self.positions[subject]
                negative = rng.choice(self.groups[self.subjects[neg_position]])
                yield anchor, positive, negative
            else:
                s1, s2 = rng.sample(self.subjects, 2)
                yield rng.choice(self.groups[s1]), rng.choice(self.groups[s2]), 0


def _identity_subset(data: WindowData, n: int, seed: int) -> WindowData:
    """Preserve every identity and at least two windows where available."""
    n = _integer_setting(n, "subset", 1)
    sampler = _IdentitySampler(data)
    if n >= len(data.X):
        return data
    rng = random.Random(seed)
    required = [
        i
        for s in sampler.subjects
        for i in rng.sample(sampler.groups[s], min(2, len(sampler.groups[s])))
    ]
    if n < len(required):
        raise ValueError(
            f"subset={n} cannot preserve identity coverage; need at least {len(required)} windows"
        )
    excluded = set(required)
    extra = rng.sample(
        [i for i in range(len(data.X)) if i not in excluded], n - len(required)
    )
    return _take(data, torch.tensor(sorted(required + extra)))


def _embed_windows(
    embedding_net: EmbeddingNet,
    data: WindowData,
    batch_size: int,
    num_workers: int,
    config: dict | None = None,
):
    """Embed every window in `data`; returns an (N, embed_dim) numpy array."""
    embedding_net.eval()
    device = next(embedding_net.parameters()).device
    loader = _make_loader(
        data,
        {**(config or {}), "batch_size": batch_size, "num_workers": num_workers},
        False,
        device,
    )
    embs = []
    with torch.inference_mode():
        for batch in loader:
            embs.append(
                embedding_net(batch[0].to(device, non_blocking=True)).cpu().numpy()
            )
    return np.concatenate(embs)


def train_siamese(
    config: dict, epochs: int, subset: int | None, loss_type: str, out_dir: str
) -> dict:
    """Train the siamese embedding net per `config`; returns the metrics dict."""
    if loss_type not in ("triplet", "contrastive"):
        raise ValueError(
            f"unsupported loss_type {loss_type!r}; use 'triplet' or 'contrastive'"
        )
    # Static settings must fail before data loading or allocating a model.
    _n_triplets(config, 1)  # Validate even when using contrastive loss.
    val_pairs = _integer_setting(config.get("val_pairs", 10000), "val_pairs", 2)
    val_subjects = _integer_setting(config.get("val_subjects", 0), "val_subjects")
    if val_subjects == 1:
        raise ValueError("verification val_subjects must be 0 (legacy) or at least 2")
    pairing = config.get("verification_pairing", "unrestricted")
    if pairing not in ("same_activity", "unrestricted"):
        raise ValueError("verification_pairing must be same_activity or unrestricted")
    model_args = dict(config.get("model_args", {}))
    embed_dim = _integer_setting(
        model_args.get("embed_dim", 128), "model_args.embed_dim", 1
    )
    dropout = _finite_setting(model_args.get("dropout", 0.3), "model_args.dropout")
    if not 0 <= dropout <= 1:
        raise ValueError("model_args.dropout must be between 0 and 1")
    margin = _finite_setting(model_args.get("margin", 1.0), "model_args.margin")
    if margin < 0:
        raise ValueError("model_args.margin must be nonnegative")
    started = time.monotonic()
    device = _runtime(config, epochs, subset)
    seed = int(config.get("seed", 42))

    source = config["source"]
    data = load_windows(
        source,
        "train",
        window_size=int(config.get("window_size", 128)),
        stride=int(config.get("stride", 64)),
        data_root=config.get("data_root", "datasets"),
    )
    _validate_windows(data, config)
    data = _filter_activities(data, config.get("activity_ids"))
    if data.y_subject is None:
        raise ValueError(
            f"siamese training requires subject labels, but source {source!r} "
            "has y_subject=None; genuine/impostor pairs cannot be formed. "
            "Use an identity-preserving source such as 'har' or 'wisdm_v1'."
        )

    # Optional subject-disjoint validation carve (default 0 = disabled).
    val_data = None
    n_train_subjects_total = n_val_subjects = n_val_windows = 0
    if val_subjects > 0:
        n_train_subjects_total = int(torch.unique(data.y_subject).numel())
        data, val_data, val_subject_ids = _split_validation(data, val_subjects, seed)
        n_val_subjects = len(val_subject_ids)
        n_val_windows = len(val_data.X)
        _IdentitySampler(val_data)  # Fail before training if EER is undefined.
        # Use the evaluator's exact eligibility rule with only two probe pairs.
        # Identity coverage alone cannot establish same-activity feasibility.
        _verification_pairs(
            val_data.y_subject,
            2,
            seed,
            activity_ids=val_data.y_activity if pairing == "same_activity" else None,
        )
    if subset is not None:
        data = _identity_subset(data, int(subset), seed)
    sampler = _IdentitySampler(data)
    protocol = _protocol(data, val_data, config)
    protocol["verification_pairing"] = pairing
    protocol.update(n_train_windows=len(data.X), n_val_windows=n_val_windows)
    protocol_enabled = val_data is not None or config.get("activity_ids") is not None

    n_channels = int(data.X.shape[1])

    model = SiameseNet(
        EmbeddingNet(n_channels, embed_dim=embed_dim, dropout=dropout)
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.get("lr", 1e-3)),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )

    batch_size = int(config.get("batch_size", 64))
    n_samples = len(data.X)

    per_epoch: list[dict] = []
    best_epoch = 0
    selection_metric, selection_key = _selection(val_data)

    for epoch in range(1, epochs + 1):
        epoch_started = time.monotonic()
        model.train()
        loss_sum = 0.0
        n_seen = 0
        # seed + epoch - 1: deterministic per epoch, different across epochs.
        if loss_type == "triplet":
            pair_iter = _batched(
                sampler.indices(_n_triplets(config, n_samples), seed + epoch - 1, True),
                batch_size,
            )
            for chunk in pair_iter:
                # Gather only this batch; no full dataset or pair cache on GPU.
                a, p, neg = data.X[torch.tensor(chunk).T].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
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
            for chunk in _batched(
                sampler.indices(n_samples, seed + epoch - 1, False), batch_size
            ):
                indices = torch.tensor(chunk).T
                x1, x2 = data.X[indices[:2]].to(device, non_blocking=True)
                label = indices[2].to(device, dtype=torch.float32)
                optimizer.zero_grad(set_to_none=True)
                z1, z2 = model(x1, x2)
                loss = contrastive_loss(z1, z2, label, margin)
                loss.backward()
                optimizer.step()
                bs = x1.size(0)
                loss_sum += loss.item() * bs
                n_seen += bs

        epoch_loss = loss_sum / max(n_seen, 1)
        if not math.isfinite(epoch_loss):
            raise ValueError("nonfinite Siamese training loss")
        record = {"epoch": epoch, "loss": epoch_loss}
        val_metrics = None
        if val_data is not None:
            pair_options = {}
            if (
                pairing == "same_activity"
                and "activity_ids" in inspect.signature(verification_metrics).parameters
            ):
                pair_options["activity_ids"] = val_data.y_activity.cpu().numpy()
            val_metrics = verification_metrics(
                _embed_windows(
                    model.embedding,
                    val_data,
                    batch_size,
                    int(config.get("num_workers", 0)),
                    config,
                ),
                val_data.y_subject.numpy(),
                n_pairs=val_pairs,
                seed=seed,
                **pair_options,
            )
            record["val_eer"] = val_metrics["eer"]
            record["val_roc_auc"] = val_metrics["roc_auc"]
            if not math.isfinite(record["val_eer"]):
                raise ValueError("nonfinite validation EER")
        if protocol_enabled:
            record["elapsed_seconds"] = time.monotonic() - epoch_started
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
            if protocol_enabled:
                checkpoint_metrics["protocol"] = protocol
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
        "protocol": protocol,
        "device": str(device),
        "elapsed_seconds": time.monotonic() - started,
        "n_train_windows": len(data.X),
        "n_train_subjects": len(sampler.subjects),
    }
    if val_data is not None:
        n_train_windows = len(data.X)
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
    parser.add_argument(
        "--epochs", type=int, default=None, help="override config epochs"
    )
    parser.add_argument(
        "--subset", type=int, default=None, help="override config subset"
    )
    parser.add_argument(
        "--loss",
        choices=["triplet", "contrastive"],
        default=None,
        help="override config loss_type",
    )
    parser.add_argument("--out-dir", default=None, help="override config out_dir")
    parser.add_argument("--device", default=None, help="cpu, cuda[:index], or auto")
    args = parser.parse_args(argv)

    with open(args.config) as f:
        config = yaml.safe_load(f)
    if args.device is not None:
        config["device"] = args.device

    epochs = args.epochs if args.epochs is not None else config["epochs"]
    subset = args.subset if args.subset is not None else config.get("subset")
    loss_type = (
        args.loss if args.loss is not None else str(config.get("loss_type", "triplet"))
    )
    out_dir = args.out_dir if args.out_dir is not None else config["out_dir"]

    train_siamese(config, epochs, subset, loss_type, out_dir)


if __name__ == "__main__":
    main()
