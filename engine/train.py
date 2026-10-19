"""Shared classification training loop.

CLI:
    python -m engine.train --config PATH --epochs INT --subset INT --out-dir PATH

Reads a YAML config, optionally overrides `epochs` / `subset` / `out_dir` from
the CLI, trains the registered model with Adam + CrossEntropyLoss, and writes
`<out_dir>/best.pt` (via `save_checkpoint`) and `<out_dir>/metrics.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess

import numpy as np
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch.nn import functional as F

from data.adapter import WindowData, build_loader, load_windows
from models.common import MetricTracker, save_checkpoint, seed_everything
from models.multitask import multitask_loss
from models.registry import build_model


def _git_commit() -> str | None:
    """Return the current git commit hash, or None if not obtainable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _stratified_subset(data: WindowData, n: int, seed: int) -> WindowData:
    """Deterministic stratified sample of `n` windows (activity-preserving)."""
    idx = np.arange(len(data.X))
    if n is not None and n < len(idx):
        idx, _ = train_test_split(
            idx, train_size=n, stratify=data.y_activity.numpy(), random_state=seed
        )
        idx = np.sort(idx)
    return WindowData(
        X=data.X[idx],
        y_activity=data.y_activity[idx],
        y_subject=data.y_subject[idx] if data.y_subject is not None else None,
        activity_names=data.activity_names,
        subject_ids=data.subject_ids,
        sampling_rate=data.sampling_rate,
    )


def _balanced_class_weights(y: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights: N / (n_classes * count_c)."""
    counts = torch.bincount(y, minlength=n_classes).float()
    return counts.sum() / (n_classes * counts.clamp(min=1.0))


def train(config: dict, epochs: int, subset: int | None, out_dir: str) -> dict:
    """Train per `config`; returns the metrics dict written to metrics.json."""
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    torch.set_num_threads(int(config.get("torch_threads", 4)))

    source = config["source"]
    model_name = config["model"]
    is_multitask = model_name == "multitask"

    data = load_windows(
        source,
        "train",
        window_size=int(config.get("window_size", 128)),
        stride=int(config.get("stride", 64)),
    )
    if is_multitask and data.y_subject is None:
        raise ValueError(
            f"model {model_name!r} trains a subject head, but source {source!r} "
            "has no subject labels (y_subject is None). Use source 'har' or a "
            "model without a subject head."
        )
    if is_multitask:
        # The frozen adapter emits 0-indexed but NON-CONTIGUOUS subject ids
        # (HAR train subjects are raw 1,3,5,...,30 -> 0,2,4,...,29). Remap to
        # contiguous ranks so the subject head size equals the distinct
        # subject count (n_subjects: 21 in configs/multitask.yaml).
        uniq = sorted(int(s) for s in torch.unique(data.y_subject).tolist())
        remap = {s: i for i, s in enumerate(uniq)}
        data = WindowData(
            X=data.X,
            y_activity=data.y_activity,
            y_subject=torch.tensor(
                [remap[int(s)] for s in data.y_subject], dtype=torch.long
            ),
            activity_names=data.activity_names,
            subject_ids=uniq,
            sampling_rate=data.sampling_rate,
        )
        n_subjects = int(config.get("model_args", {}).get("n_subjects", 0))
        if n_subjects < len(uniq):
            raise ValueError(
                f"model_args.n_subjects={n_subjects} is smaller than the "
                f"{len(uniq)} distinct subjects in the loaded {source} data"
            )
    if subset is not None:
        data = _stratified_subset(data, int(subset), seed)

    n_channels = int(data.X.shape[1])
    n_classes = len(data.activity_names)
    model = build_model(model_name, config.get("model_args", {}), n_channels, n_classes)

    loader = build_loader(
        data,
        batch_size=int(config.get("batch_size", 64)),
        shuffle=True,
        num_workers=int(config.get("num_workers", 0)),
    )

    optimizer_name = str(config.get("optimizer", "adam")).lower()
    if optimizer_name != "adam":
        raise ValueError(f"unsupported optimizer {optimizer_name!r}; only 'adam' is supported")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.get("lr", 1e-3)),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )

    class_weight = config.get("class_weight")
    class_weights = None
    if class_weight == "balanced":
        class_weights = _balanced_class_weights(data.y_activity, n_classes)
    elif class_weight not in (None, "none", "null"):
        raise ValueError(
            f"unsupported class_weight {class_weight!r}; use 'balanced' or null"
        )

    per_epoch: list[dict] = []
    best_loss = float("inf")
    best_epoch = 0
    final_activity_loss = final_subject_loss = final_subject_acc = None

    for epoch in range(1, epochs + 1):
        model.train()
        tracker = MetricTracker()
        loss_sum = 0.0
        act_correct = subj_correct = n_total = 0

        for batch in loader:
            # build_loader appends y_subject to every batch whenever the
            # loaded data has subject labels (always the case for HAR).
            if len(batch) == 3:
                xb, yb_act, yb_subj = batch
            else:
                xb, yb_act = batch
                yb_subj = None

            optimizer.zero_grad()
            out = model(xb)

            if is_multitask:
                act_logits, subj_logits = out
                if class_weights is not None:
                    # Balanced weighting applies to the (imbalanced) activity
                    # head; the subject head keeps plain cross-entropy.
                    act_loss = F.cross_entropy(act_logits, yb_act, weight=class_weights)
                    subj_loss = F.cross_entropy(subj_logits, yb_subj)
                    loss = 0.5 * act_loss + 0.5 * subj_loss
                else:
                    loss = multitask_loss(act_logits, subj_logits, yb_act, yb_subj)
                    with torch.no_grad():
                        act_loss = F.cross_entropy(act_logits, yb_act)
                        subj_loss = F.cross_entropy(subj_logits, yb_subj)
                act_correct += (act_logits.argmax(1) == yb_act).sum().item()
                subj_correct += (subj_logits.argmax(1) == yb_subj).sum().item()
                tracker.update("activity_loss", act_loss.item())
                tracker.update("subject_loss", subj_loss.item())
            else:
                loss = F.cross_entropy(out, yb_act, weight=class_weights)
                act_correct += (out.argmax(1) == yb_act).sum().item()

            loss.backward()
            optimizer.step()

            bs = xb.size(0)
            loss_sum += loss.item() * bs
            n_total += bs

        train_loss = loss_sum / n_total
        train_acc = act_correct / n_total
        per_epoch.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc})

        if is_multitask:
            final_activity_loss = tracker.avg("activity_loss")
            final_subject_loss = tracker.avg("subject_loss")
            final_subject_acc = subj_correct / n_total
            print(
                f"epoch {epoch}/{epochs} train_loss={train_loss:.4f} "
                f"train_acc={train_acc:.4f} act_loss={final_activity_loss:.4f} "
                f"subj_loss={final_subject_loss:.4f} subj_acc={final_subject_acc:.4f}"
            )
        else:
            print(f"epoch {epoch}/{epochs} train_loss={train_loss:.4f} train_acc={train_acc:.4f}")

        if train_loss < best_loss:
            best_loss = train_loss
            best_epoch = epoch
            os.makedirs(out_dir, exist_ok=True)
            save_checkpoint(
                os.path.join(out_dir, "best.pt"),
                model,
                optimizer,
                epoch,
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    # There is no validation split (and no data protocol yet), so
                    # the checkpoint is chosen on training loss only. Recorded
                    # explicitly so the checkpoint's provenance is unambiguous.
                    "selection_metric": "train_loss",
                },
            )

    metrics = {
        "model": model_name,
        "source": source,
        "epochs": epochs,
        "subset": subset,
        "per_epoch": per_epoch,
        "best_epoch": best_epoch,
        # best.pt is selected on training loss, NOT on validation performance:
        # no validation split exists yet (the held-out test split is evaluated
        # separately by engine/evaluate.py and is never used for selection).
        "selection_metric": "train_loss",
        "final_train_loss": per_epoch[-1]["train_loss"],
        "final_train_acc": per_epoch[-1]["train_acc"],
        "n_params": sum(p.numel() for p in model.parameters()),
        "git_commit": _git_commit(),
    }
    if is_multitask:
        metrics["final_activity_loss"] = final_activity_loss
        metrics["final_subject_loss"] = final_subject_loss
        metrics["final_subject_acc"] = final_subject_acc

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Shared classification training loop")
    parser.add_argument("--config", required=True, help="path to the YAML config")
    parser.add_argument("--epochs", type=int, default=None, help="override config epochs")
    parser.add_argument("--subset", type=int, default=None, help="override config subset")
    parser.add_argument("--out-dir", default=None, help="override config out_dir")
    args = parser.parse_args(argv)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    epochs = args.epochs if args.epochs is not None else int(config["epochs"])
    subset = args.subset if args.subset is not None else config.get("subset")
    out_dir = args.out_dir if args.out_dir is not None else config["out_dir"]

    train(config, epochs, subset, out_dir)


if __name__ == "__main__":
    main()
