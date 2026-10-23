"""Opt-in identity metric training, selected only on held-out TRAIN subjects.

No test split is loaded by this module. Custom encoder checkpoints deliberately
use a separate loader; the legacy evaluator cannot infer their architecture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path

import torch
import yaml
from data.adapter import load_windows
from models.common import seed_everything
from models.metric_learning import (
    IdentityBalancedBatchSampler,
    batch_hard_triplet_loss,
    supervised_contrastive_loss,
)
from torch import nn
from torch.nn import functional as F

from engine.evaluate import verification_metrics
from engine.train import (
    _device,
    _filter_activities,
    _git_commit,
    _split_validation,
    _validate_windows,
)


def build_gait_encoder(n_channels: int, model_args: dict):
    """Late import keeps loss/sampler tools usable without the optional encoder."""
    from models.gait_encoders import build_gait_encoder as factory

    return factory(n_channels, model_args)


def _resolved_config(config: dict) -> dict:
    # Round-trip restricts checkpoint metadata to plain JSON values, no objects.
    config = json.loads(json.dumps(config, allow_nan=False))
    defaults = {
        "epochs": 40,
        "seed": config.get("seedmodel", 42),
        "split_seed": 42,
        "val_seed": 42,
        "val_subjects": 4,
        "activity_ids": [0],
        "window_size": 128,
        "stride": 64,
        "objective": "batch_hard",
        "identities_per_batch": 8,
        "windows_per_identity": 4,
        "ce_weight": 0.5,
        "label_smoothing": 0.05,
        "margin": 0.2,
        "temperature": 0.1,
        "softplus": False,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "gradient_clip": 1.0,
        "device": "auto",
        "torch_threads": 2,
        "validation_pairs": 10000,
        "eval_batch_size": 256,
        "data_root": "datasets",
        "augment": {},
    }
    resolved = {**defaults, **config}
    if "source" not in resolved:
        raise ValueError("config.source is required")
    if resolved["objective"] not in {"batch_hard", "supcon", "hybrid"}:
        raise ValueError("objective must be batch_hard, supcon, or hybrid")
    for key in ("epochs", "torch_threads", "eval_batch_size", "window_size", "stride"):
        if int(resolved[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    if int(resolved["val_subjects"]) < 2:
        raise ValueError("val_subjects >= 2 is required for validation EER selection")
    if int(resolved["validation_pairs"]) < 2:
        raise ValueError("validation_pairs must be >= 2")
    for key in ("ce_weight", "weight_decay", "margin"):
        if not math.isfinite(float(resolved[key])) or float(resolved[key]) < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    for key in ("lr", "temperature", "gradient_clip"):
        if not math.isfinite(float(resolved[key])) or float(resolved[key]) <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not 0 <= float(resolved["label_smoothing"]) < 1:
        raise ValueError("label_smoothing must be in [0, 1)")
    if "seedmodel" in config and int(config["seedmodel"]) != int(resolved["seed"]):
        raise ValueError("seed and seedmodel disagree")
    resolved["model_args"] = {
        "architecture": "multiscale",
        "embed_dim": 64,
        "width": 32,
        "dropout": 0.1,
        "stats_fusion": False,
        "normalization": True,
        **config.get("model_args", {}),
    }
    if not isinstance(resolved["augment"], dict):
        raise TypeError("augment must be a mapping")
    return resolved


def _normalization_fingerprint(encoder) -> dict:
    """Fingerprint embedded input mean/std buffers, excluding BatchNorm state."""
    buffers = {
        name: value.detach().cpu().contiguous()
        for name, value in encoder.named_buffers()
        if "running_" not in name
        and (
            "mean" in name.lower()
            or "std" in name.lower()
            or name == "normalization_fitted"
        )
    }
    digest = hashlib.sha256()
    entries = {}
    for name, value in sorted(buffers.items()):
        entry = {"shape": list(value.shape), "dtype": str(value.dtype)}
        entries[name] = entry
        digest.update(json.dumps([name, entry], sort_keys=True).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return {"sha256": digest.hexdigest(), "buffers": entries}


def _development_fingerprint(train, val):
    checksum = hashlib.sha256()
    for data in (train, val):
        for values in (data.X, data.y_subject, data.y_activity):
            checksum.update(values.contiguous().numpy().tobytes())
    return checksum.hexdigest()


def _embed(encoder, windows, batch_size, device):
    encoder.eval()
    parts = []
    with torch.inference_mode():
        for batch in windows.split(batch_size):
            parts.append(encoder(batch.to(device)).detach().cpu())
    return torch.cat(parts)


def _collapsed_fraction(z: torch.Tensor) -> float:
    """Fraction of distinct batch positions with distance <= 1e-4."""
    z = z.detach()[:512]
    if len(z) < 2:
        return 0.0
    return float((torch.pdist(z.float()) <= 1e-4).float().mean().item())


def _augment(x, config):
    if not config:
        return x
    from data.gait_transforms import augment_gait

    return augment_gait(x, config)


def load_metric_encoder(checkpoint, device="cpu"):
    """Safely restore raw encoder state and embedded train-only normalization."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("checkpoint_format") != "gait_metric_v1"
    ):
        raise ValueError("expected a gait_metric_v1 checkpoint")
    config = payload.get("config")
    protocol = payload.get("metrics", {}).get("protocol")
    if not isinstance(config, dict) or not isinstance(protocol, dict):
        raise TypeError(
            "metric checkpoint requires config and metrics.protocol mappings"
        )
    # Enforce the same plain-metadata requirement when loading, too.
    json.dumps({"config": config, "protocol": protocol}, allow_nan=False)
    if config.get("model_args") != protocol.get("model_args"):
        raise ValueError("checkpoint config/model_args disagree with protocol")
    encoder = build_gait_encoder(int(protocol["n_channels"]), config["model_args"])
    encoder.load_state_dict(payload["model_state_dict"], strict=True)
    if _normalization_fingerprint(encoder) != protocol.get("normalization_fingerprint"):
        raise ValueError("checkpoint normalization fingerprint mismatch")
    encoder.to(device).eval()
    return encoder, protocol, config


def train_metric(config: dict, out_dir: str) -> dict:
    """Train optimizer identities only; select earliest minimum validation EER."""
    started = time.monotonic()
    config = _resolved_config(config)
    seed = int(config["seed"])
    split_seed, val_seed = int(config["split_seed"]), int(config["val_seed"])
    seed_everything(seed)
    torch.set_num_threads(int(config["torch_threads"]))
    device = _device(config)
    data = load_windows(
        config["source"],
        "train",
        window_size=int(config["window_size"]),
        stride=int(config["stride"]),
        data_root=config["data_root"],
    )
    _validate_windows(data, config)
    expected_channels = {"har": 9, "wisdm_v1": 3}.get(config["source"])
    if expected_channels is not None and data.X.shape[1] != expected_channels:
        raise ValueError(
            f"{config['source']} requires {expected_channels} source channels"
        )
    # Match the walking baseline: select the declared activity cohort before
    # the fixed subject split (some source identities have no walking rows).
    data = _filter_activities(data, config["activity_ids"])
    train_data, val_data, val_ids = _split_validation(
        data, int(config["val_subjects"]), split_seed
    )
    sampler = IdentityBalancedBatchSampler(
        train_data.y_subject,
        int(config["identities_per_batch"]),
        int(config["windows_per_identity"]),
        config.get("steps_per_epoch"),
        seed,
    )
    # Fail before optimization if the validation cohort cannot supply pairs.
    from engine.evaluate import _verification_pairs

    _verification_pairs(
        val_data.y_subject,
        int(config["validation_pairs"]),
        val_seed,
        val_data.y_activity,
    )
    train_ids = sorted(int(s) for s in train_data.y_subject.unique().tolist())
    encoder = build_gait_encoder(int(train_data.X.shape[1]), config["model_args"])
    # The encoder also fits optional statistical-fusion moments when channel
    # normalization is disabled; this method obeys its normalization flag.
    encoder.fit_normalization(train_data.X)
    fingerprint = _normalization_fingerprint(encoder)
    if config["model_args"]["normalization"] and not fingerprint["buffers"]:
        raise ValueError("normalization requires embedded mean/std buffers")
    encoder.to(device)
    ce_weight = float(config["ce_weight"])
    classifier = (
        nn.Linear(int(config["model_args"]["embed_dim"]), len(train_ids)).to(device)
        if ce_weight > 0
        else None
    )
    parameters = list(encoder.parameters())
    if classifier is not None:
        parameters.extend(classifier.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=float(config["lr"]), weight_decay=float(config["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(config["epochs"])
    )
    remapped = torch.searchsorted(torch.tensor(train_ids), train_data.y_subject)
    hardware = {
        "device": str(device),
        "torch_version": str(torch.__version__),
        "torch_threads": torch.get_num_threads(),
        "platform": platform.platform(),
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else platform.processor(),
    }
    protocol = {
        "name": "held_out_subject_within_recording_discrimination",
        "source": config["source"],
        "data_root": config["data_root"],
        "train_subject_ids": train_ids,
        "val_subject_ids": val_ids,
        "classifier_subject_ids": train_ids,
        "classifier_label_mapping": "sorted_train_ids_to_0_based",
        "activity_ids": config["activity_ids"],
        "n_channels": int(data.X.shape[1]),
        "window_size": int(data.X.shape[2]),
        "stride": int(config["stride"]),
        "sampling_rate": float(data.sampling_rate),
        "split": "subject_disjoint",
        "split_seed": split_seed,
        "seed": seed,
        "train_seed": seed,
        "val_seed": val_seed,
        "verification_pairing": "same_activity",
        "validation_pairs": int(config["validation_pairs"]),
        "normalization": "train_only_channel_standardization"
        if config["model_args"]["normalization"]
        else "none",
        "normalization_fingerprint": fingerprint,
        "model_args": config["model_args"],
        "objective": config["objective"],
        "config": config,
        "sampler": sampler.context,
        "n_train_windows": len(train_data.X),
        "n_val_windows": len(val_data.X),
        "selection_metric": "val_eer",
        "selection_ties": "earliest_epoch",
        "test_split_loaded": False,
        "development_data_sha256": _development_fingerprint(train_data, val_data),
    }
    os.makedirs(out_dir, exist_ok=True)
    records, best_eer, best_epoch = [], math.inf, 0
    log_path = Path(out_dir) / "per_epoch.jsonl"
    with log_path.open("w") as log_file:
        for epoch in range(1, int(config["epochs"]) + 1):
            epoch_started = time.monotonic()
            sampler.set_epoch(epoch - 1)
            encoder.train()
            if classifier is not None:
                classifier.train()
            totals = {
                key: 0.0
                for key in (
                    "loss",
                    "batch_hard_loss",
                    "supcon_loss",
                    "ce_loss",
                    "collapsed_fraction",
                    "grad_norm",
                )
            }
            lr = float(optimizer.param_groups[0]["lr"])
            for rows in sampler:
                indices = torch.tensor(rows, dtype=torch.long)
                x = train_data.X[indices].to(device)
                labels = train_data.y_subject[indices].to(device)
                sample_ids = indices.to(device)
                optimizer.zero_grad(set_to_none=True)
                z = encoder(_augment(x, config["augment"]))
                if z.ndim != 2 or z.shape != (
                    len(rows),
                    int(config["model_args"]["embed_dim"]),
                ):
                    raise ValueError(
                        "encoder output must have shape (batch, embed_dim)"
                    )
                hard = z.sum() * 0.0
                supcon = z.sum() * 0.0
                if config["objective"] in {"batch_hard", "hybrid"}:
                    hard = batch_hard_triplet_loss(
                        z,
                        labels,
                        float(config["margin"]),
                        softplus=bool(config["softplus"]),
                        sample_ids=sample_ids,
                    )
                if config["objective"] in {"supcon", "hybrid"}:
                    supcon = supervised_contrastive_loss(
                        z, labels, float(config["temperature"]), sample_ids=sample_ids
                    )
                metric_loss = (
                    (hard + supcon) / 2
                    if config["objective"] == "hybrid"
                    else hard + supcon
                )
                ce = (
                    F.cross_entropy(
                        classifier(z),
                        remapped[indices].to(device),
                        label_smoothing=float(config["label_smoothing"]),
                    )
                    if classifier is not None
                    else z.sum() * 0.0
                )
                loss = metric_loss + ce_weight * ce
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite metric training loss")
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters, float(config["gradient_clip"]), error_if_nonfinite=True
                )
                optimizer.step()
                for key, value in (
                    ("loss", loss),
                    ("batch_hard_loss", hard),
                    ("supcon_loss", supcon),
                    ("ce_loss", ce),
                    ("grad_norm", grad_norm),
                ):
                    totals[key] += float(value.detach().item())
                totals["collapsed_fraction"] += _collapsed_fraction(z)
            z_val = _embed(encoder, val_data.X, int(config["eval_batch_size"]), device)
            validation = verification_metrics(
                z_val,
                val_data.y_subject,
                n_pairs=int(config["validation_pairs"]),
                seed=val_seed,
                activity_ids=val_data.y_activity,
            )
            record = {
                "epoch": epoch,
                **{key: value / len(sampler) for key, value in totals.items()},
                "val_eer": float(validation["eer"]),
                "val_roc_auc": float(validation["roc_auc"]),
                "val_collapsed_fraction": _collapsed_fraction(z_val),
                "lr": lr,
                "elapsed_seconds": time.monotonic() - epoch_started,
            }
            records.append(record)
            log_file.write(json.dumps(record, allow_nan=False) + "\n")
            log_file.flush()
            if record["val_eer"] < best_eer:
                best_eer, best_epoch = record["val_eer"], epoch
                # Save only raw encoder keys: no wrapper/classification prefix.
                torch.save(
                    {
                        "checkpoint_format": "gait_metric_v1",
                        "epoch": epoch,
                        "model_state_dict": {
                            k: v.detach().cpu() for k, v in encoder.state_dict().items()
                        },
                        "classifier_state_dict": {
                            k: v.detach().cpu()
                            for k, v in classifier.state_dict().items()
                        }
                        if classifier is not None
                        else None,
                        "config": config,
                        "train_seed": seed,
                        "split_seed": split_seed,
                        "val_seed": val_seed,
                        "metrics": {
                            **record,
                            "protocol": protocol,
                            "validation": validation,
                        },
                        "hardware": hardware,
                    },
                    Path(out_dir) / "best.pt",
                )
            scheduler.step()
    metrics = {
        "method": "metric_learning",
        "objective": config["objective"],
        "selection_metric": "val_eer",
        "best_epoch": best_epoch,
        "best_val_eer": best_eer,
        "epochs": int(config["epochs"]),
        "per_epoch": records,
        "protocol": protocol,
        "hardware": hardware,
        "seed": seed,
        "split_seed": split_seed,
        "val_seed": val_seed,
        "elapsed_seconds": time.monotonic() - started,
        "git_commit": _git_commit(),
        "checkpoint": str(Path(out_dir) / "best.pt"),
    }
    (Path(out_dir) / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n"
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    with open(args.config) as stream:
        config = yaml.safe_load(stream)
    if args.epochs is not None:
        config["epochs"] = args.epochs
    print(json.dumps(train_metric(config, args.out_dir), indent=2))


if __name__ == "__main__":
    main()
