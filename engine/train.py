"""Activity training with subject-disjoint validation and bounded device batches.

Only the source's original train split is loaded here. The original test split
belongs exclusively to final evaluation; checkpoint selection uses activity
cross entropy on held-out training subjects, including for multitask models.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import time

import numpy as np
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from data.adapter import WindowData, load_windows
from models.common import save_checkpoint, seed_everything
from models.registry import build_model


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _take(data: WindowData, idx) -> WindowData:
    subjects = data.y_subject[idx] if data.y_subject is not None else None
    return WindowData(
        X=data.X[idx],
        y_activity=data.y_activity[idx],
        y_subject=subjects,
        activity_names=list(data.activity_names),
        subject_ids=sorted(int(s) for s in subjects.unique().tolist())
        if subjects is not None
        else [],
        sampling_rate=data.sampling_rate,
    )


def _stratified_subset(data: WindowData, n: int, seed: int) -> WindowData:
    """Deterministic activity-stratified subset; never use this to split subjects."""
    if n <= 0:
        raise ValueError("subset must be positive")
    idx = np.arange(len(data.X))
    if n < len(idx):
        idx, _ = train_test_split(
            idx, train_size=n, stratify=data.y_activity.cpu().numpy(), random_state=seed
        )
        idx = np.sort(idx)
    return _take(data, idx)


def _split_validation(data: WindowData, val_subjects: int, seed: int):
    """Partition original identities deterministically before any subsampling."""
    if data.y_subject is None:
        raise ValueError(
            "subject-disjoint validation requires real y_subject labels; use an identity-preserving source"
        )
    subjects = sorted(int(s) for s in data.y_subject.unique().tolist())
    if not 1 <= val_subjects < len(subjects):
        raise ValueError(
            f"val_subjects must satisfy 1 <= val_subjects < {len(subjects)}; got {val_subjects}"
        )
    val_ids = sorted(random.Random(seed).sample(subjects, val_subjects))
    mask = torch.isin(data.y_subject, torch.tensor(val_ids, dtype=data.y_subject.dtype))
    return _take(data, ~mask), _take(data, mask), val_ids


def _chronological_validation(data: WindowData, config: dict):
    """Explicit fallback for activity CSVs without identities; purge overlap."""
    fraction = float(config.get("val_fraction", 0.2))
    if not 0 < fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    stride = int(config.get("stride", 64))
    if stride <= 0:
        raise ValueError("stride must be positive")
    purge = math.ceil(data.X.shape[2] / stride)
    boundary = int(len(data.X) * (1 - fraction))
    if boundary <= purge or boundary >= len(data.X):
        raise ValueError(
            "chronological validation needs enough windows for a nonempty split and overlap purge"
        )
    return (
        _take(data, slice(None, boundary - purge)),
        _take(data, slice(boundary, None)),
        purge,
    )


def _validate_windows(data: WindowData, config: dict) -> None:
    if data.X.ndim != 3 or any(s <= 0 for s in data.X.shape):
        raise ValueError("X must be nonempty (N, C, T) windows")
    if data.X.dtype != torch.float32:
        raise ValueError("X must use float32 sensor values")
    n = len(data.X)
    if data.y_activity.shape != (n,) or data.y_activity.dtype != torch.long:
        raise ValueError("y_activity must be int64 with one label per window")
    if (
        not data.activity_names
        or data.y_activity.min() < 0
        or data.y_activity.max() >= len(data.activity_names)
    ):
        raise ValueError("activity labels must index activity_names")
    if data.y_subject is not None and (
        data.y_subject.shape != (n,)
        or data.y_subject.dtype != torch.long
        or data.y_subject.min() < 0
    ):
        raise ValueError(
            "y_subject must be nonnegative int64 with one real identity per window"
        )
    # Bound the temporary finite-value mask even for large datasets.
    for chunk in data.X.split(1024):
        if not torch.isfinite(chunk).all():
            raise ValueError("sensor windows contain nonfinite values")
    expected = config.get("model_args", {})
    for name, actual in (
        ("n_channels", data.X.shape[1]),
        ("n_classes", len(data.activity_names)),
    ):
        if name in expected and int(expected[name]) != actual:
            raise ValueError(
                f"model_args.{name}={expected[name]} disagrees with data {actual}"
            )
    if "window_size" in config and int(config["window_size"]) != data.X.shape[2]:
        raise ValueError("configured window_size disagrees with loaded data")


def _filter_activities(data: WindowData, activity_ids) -> WindowData:
    if activity_ids is None:
        return data
    ids = [int(a) for a in activity_ids]
    if not ids or any(a < 0 or a >= len(data.activity_names) for a in ids):
        raise ValueError("activity_ids must contain valid zero-indexed activity labels")
    selected = _take(data, torch.isin(data.y_activity, torch.tensor(ids)))
    if not len(selected.X):
        raise ValueError("activity_ids selected no windows")
    return selected


def _device(config: dict) -> torch.device:
    requested = str(config.get("device", "cpu"))
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("device must be cpu, cuda[:index], or auto")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return device


def _runtime(config: dict, epochs: int, subset: int | None) -> torch.device:
    for name, value in (
        ("epochs", epochs),
        ("batch_size", config.get("batch_size", 64)),
        ("torch_threads", config.get("torch_threads", 4)),
    ):
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive")
    if subset is not None and int(subset) <= 0:
        raise ValueError("subset must be positive")
    if int(config.get("num_workers", 0)) < 0:
        raise ValueError("num_workers must be nonnegative")
    if str(config.get("optimizer", "adam")).lower() != "adam":
        raise ValueError("only optimizer 'adam' is supported")
    seed_everything(int(config.get("seed", 42)))
    torch.set_num_threads(int(config.get("torch_threads", 4)))
    return _device(config)


def _make_loader(
    data: WindowData, config: dict, shuffle: bool, device: torch.device
) -> DataLoader:
    """Keep datasets on CPU and copy bounded batches to the selected device."""
    tensors = [data.X, data.y_activity]
    if data.y_subject is not None:
        tensors.append(data.y_subject)
    workers = int(config.get("num_workers", 0))
    options = {}
    if workers:
        options["persistent_workers"] = bool(config.get("persistent_workers", True))
        options["prefetch_factor"] = int(config.get("prefetch_factor", 2))
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=int(config.get("batch_size", 64)),
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=bool(config.get("pin_memory", device.type == "cuda")),
        generator=torch.Generator().manual_seed(int(config.get("seed", 42))),
        **options,
    )


def _protocol(data: WindowData, val_data: WindowData | None, config: dict) -> dict:
    return {
        "source": str(config["source"]),
        "data_root": str(config.get("data_root", "datasets")),
        "train_subject_ids": data.subject_ids,
        "val_subject_ids": val_data.subject_ids if val_data is not None else [],
        "activity_ids": list(config["activity_ids"])
        if config.get("activity_ids") is not None
        else None,
        "n_channels": int(data.X.shape[1]),
        "window_size": int(data.X.shape[2]),
        "stride": int(config.get("stride", 64)),
        "sampling_rate": float(data.sampling_rate),
        "normalization": "none",
        "split": "subject_disjoint" if val_data is not None else "train_only",
        "seed": int(config.get("seed", 42)),
    }


def _balanced_class_weights(y: torch.Tensor, n_classes: int) -> torch.Tensor:
    counts = torch.bincount(y, minlength=n_classes).float()
    return counts.sum() / (n_classes * counts.clamp(min=1.0))


def _activity_validation(model, loader, device: torch.device) -> dict:
    """Score activity only: validation identities are unseen by the subject head."""
    model.eval()
    total_loss = correct = count = 0
    with torch.inference_mode():
        for batch in loader:
            xb, yb = [t.to(device, non_blocking=True) for t in batch[:2]]
            logits = model(xb)
            if isinstance(logits, tuple):
                logits = logits[0]
            total_loss += F.cross_entropy(logits, yb, reduction="sum").item()
            correct += (logits.argmax(1) == yb).sum().item()
            count += len(xb)
    return {
        "val_activity_loss": total_loss / count,
        "val_activity_acc": correct / count,
    }


def train(config: dict, epochs: int, subset: int | None, out_dir: str) -> dict:
    started = time.monotonic()
    device = _runtime(config, epochs, subset)
    seed = int(config.get("seed", 42))
    data = load_windows(
        config["source"],
        "train",
        window_size=int(config.get("window_size", 128)),
        stride=int(config.get("stride", 64)),
        data_root=config.get("data_root", "datasets"),
    )
    _validate_windows(data, config)
    data = _filter_activities(data, config.get("activity_ids"))
    strategy = config.get("validation_strategy", "subject_disjoint")
    purge = 0
    if strategy == "subject_disjoint":
        n_subjects = 0 if data.y_subject is None else data.y_subject.unique().numel()
        n_val = int(config.get("val_subjects", max(1, math.ceil(n_subjects * 0.2))))
        data, val_data, _ = _split_validation(data, n_val, seed)
    elif strategy == "chronological":
        if config["model"] == "multitask":
            raise ValueError(
                "multitask training requires subject-disjoint validation with real identities"
            )
        if data.y_subject is not None:
            raise ValueError(
                "chronological validation is only for activity data without subject labels"
            )
        data, val_data, purge = _chronological_validation(data, config)
    else:
        raise ValueError(
            "validation_strategy must be subject_disjoint or chronological"
        )
    if subset is not None:
        data = _stratified_subset(data, int(subset), seed)
    protocol = _protocol(data, val_data, config)
    protocol.update(
        n_train_windows=len(data.X),
        n_val_windows=len(val_data.X),
        split=strategy,
        purged_windows=purge,
    )
    model_name = config["model"]
    is_multitask = model_name == "multitask"
    if is_multitask:
        # Map only training identities; unseen validation subjects are never
        # given training subject targets or evaluated as known identities.
        original_ids = data.subject_ids
        mapping = {s: i for i, s in enumerate(original_ids)}
        protocol["subject_label_map"] = {str(s): i for s, i in mapping.items()}
        data.y_subject = torch.tensor(
            [mapping[int(s)] for s in data.y_subject], dtype=torch.long
        )
        if int(config.get("model_args", {}).get("n_subjects", 0)) < len(mapping):
            raise ValueError(
                "model_args.n_subjects is smaller than the distinct training subjects"
            )
    n_classes = len(data.activity_names)
    model = build_model(
        model_name, config.get("model_args", {}), data.X.shape[1], n_classes
    ).to(device)
    loader = _make_loader(data, config, True, device)
    val_loader = _make_loader(val_data, config, False, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.get("lr", 1e-3)),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    weighting = config.get("class_weight")
    if weighting not in (None, "balanced", "none", "null"):
        raise ValueError("class_weight must be 'balanced' or null")
    weights = (
        _balanced_class_weights(data.y_activity, n_classes).to(device)
        if weighting == "balanced"
        else None
    )
    per_epoch = []
    best_loss = float("inf")
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        epoch_started = time.monotonic()
        model.train()
        loss_sum = act_sum = subj_sum = act_correct = subj_correct = n_total = 0
        for batch in loader:
            batch = [t.to(device, non_blocking=True) for t in batch]
            xb, yb = batch[:2]
            optimizer.zero_grad(set_to_none=True)
            out = model(xb)
            logits = out[0] if is_multitask else out
            act_loss = F.cross_entropy(logits, yb, weight=weights)
            loss = act_loss
            if is_multitask:
                subject_loss = F.cross_entropy(out[1], batch[2])
                loss = 0.5 * (act_loss + subject_loss)
                subj_sum += subject_loss.item() * len(xb)
                subj_correct += (out[1].argmax(1) == batch[2]).sum().item()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(xb)
            act_sum += act_loss.item() * len(xb)
            act_correct += (logits.argmax(1) == yb).sum().item()
            n_total += len(xb)
        record = {
            "epoch": epoch,
            "train_loss": loss_sum / n_total,
            "train_acc": act_correct / n_total,
            **_activity_validation(model, val_loader, device),
        }
        record["elapsed_seconds"] = time.monotonic() - epoch_started
        if not math.isfinite(record["train_loss"]) or not math.isfinite(
            record["val_activity_loss"]
        ):
            raise ValueError("nonfinite training or validation loss")
        per_epoch.append(record)
        print(
            f"epoch {epoch}/{epochs} train_loss={record['train_loss']:.4f} train_acc={record['train_acc']:.4f} "
            f"val_activity_loss={record['val_activity_loss']:.4f} val_activity_acc={record['val_activity_acc']:.4f}",
            flush=True,
        )
        if record["val_activity_loss"] < best_loss:
            best_loss, best_epoch = record["val_activity_loss"], epoch
            os.makedirs(out_dir, exist_ok=True)
            save_checkpoint(
                os.path.join(out_dir, "best.pt"),
                model,
                optimizer,
                epoch,
                {
                    **record,
                    "selection_metric": "val_activity_loss",
                    "protocol": protocol,
                },
            )
    metrics = {
        "model": model_name,
        "source": config["source"],
        "epochs": epochs,
        "subset": subset,
        "per_epoch": per_epoch,
        "best_epoch": best_epoch,
        "selection_metric": "val_activity_loss",
        "best_val_activity_loss": best_loss,
        "final_train_loss": per_epoch[-1]["train_loss"],
        "final_train_acc": per_epoch[-1]["train_acc"],
        "protocol": protocol,
        "device": str(device),
        "elapsed_seconds": time.monotonic() - started,
        "n_params": sum(p.numel() for p in model.parameters()),
        "git_commit": _git_commit(),
    }
    if is_multitask:
        metrics.update(
            final_activity_loss=act_sum / n_total,
            final_subject_loss=subj_sum / n_total,
            final_subject_acc=subj_correct / n_total,
        )
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--subset", type=int)
    parser.add_argument("--out-dir")
    parser.add_argument("--device", help="cpu, cuda[:index], or auto")
    args = parser.parse_args(argv)
    with open(args.config) as f:
        config = yaml.safe_load(f)
    if args.device is not None:
        config["device"] = args.device
    train(
        config,
        args.epochs if args.epochs is not None else int(config["epochs"]),
        args.subset if args.subset is not None else config.get("subset"),
        args.out_dir if args.out_dir is not None else config["out_dir"],
    )


if __name__ == "__main__":
    main()
