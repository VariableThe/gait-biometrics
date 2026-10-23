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
from torch import nn
from torch.nn import functional as F

from data.adapter import load_windows
from engine.evaluate import verification_metrics
from engine.train import (
    _device,
    _filter_activities,
    _git_commit,
    _split_validation,
    _take,
    _validate_windows,
)
from models.common import seed_everything
from models.metric_learning import (
    IdentityBalancedBatchSampler,
    batch_hard_triplet_loss,
    supervised_contrastive_loss,
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
        "classifier_type": "linear",
        "angular_scale": 30.0,
        "angular_margin": 0.3,
        "label_smoothing": 0.05,
        "margin": 0.2,
        "temperature": 0.1,
        "softplus": False,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "gradient_clip": 1.0,
        "device": "auto",
        "precision": "float32",
        "epoch_selection": "validation_eer",
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
    if resolved["classifier_type"] not in {"linear", "arcface"}:
        raise ValueError("classifier_type must be linear or arcface")
    if resolved["precision"] not in {"float32", "bfloat16"}:
        raise ValueError("precision must be float32 or bfloat16")
    if resolved["epoch_selection"] not in {"validation_eer", "final"}:
        raise ValueError("epoch_selection must be validation_eer or final")
    explicit = "train_subject_ids" in resolved or "val_subject_ids" in resolved
    if explicit:
        if not all(key in resolved for key in ("train_subject_ids", "val_subject_ids")):
            raise ValueError(
                "train_subject_ids and val_subject_ids are required together"
            )
        for key in ("train_subject_ids", "val_subject_ids"):
            resolved[key] = _subject_ids(resolved[key], key)
        if set(resolved["train_subject_ids"]) & set(resolved["val_subject_ids"]):
            raise ValueError("train_subject_ids and val_subject_ids must not overlap")
    for key in ("epochs", "torch_threads", "eval_batch_size", "window_size", "stride"):
        if int(resolved[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    if not explicit and int(resolved["val_subjects"]) < 2:
        raise ValueError("val_subjects >= 2 is required for validation EER selection")
    if int(resolved["validation_pairs"]) < 2:
        raise ValueError("validation_pairs must be >= 2")
    for key in ("ce_weight", "weight_decay", "margin"):
        if not math.isfinite(float(resolved[key])) or float(resolved[key]) < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    for key in ("lr", "temperature", "gradient_clip"):
        if not math.isfinite(float(resolved[key])) or float(resolved[key]) <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if (
        not math.isfinite(float(resolved["angular_scale"]))
        or float(resolved["angular_scale"]) <= 0
    ):
        raise ValueError("angular_scale must be finite and positive")
    if (
        not math.isfinite(float(resolved["angular_margin"]))
        or not 0 <= float(resolved["angular_margin"]) < math.pi / 2
    ):
        raise ValueError("angular_margin must be finite and in [0, pi/2)")
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


def _subject_ids(values, name):
    """Reject lossy ID coercions; explicit cohorts describe exact real identities."""
    if not isinstance(values, list) or any(type(s) is not int or s < 0 for s in values):
        raise ValueError(f"{name} must be a list of nonnegative integer subject IDs")
    if len(values) < 2:
        raise ValueError(f"{name} requires at least two subjects")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must contain unique subject IDs")
    return sorted(values)


def _split_cohorts(data, config):
    """Use only post-activity-filter TRAIN rows; unused subjects stay excluded."""
    if "train_subject_ids" not in config:
        return _split_validation(
            data, int(config["val_subjects"]), int(config["split_seed"])
        )
    if data.y_subject is None:
        raise ValueError("explicit subject cohorts require real y_subject labels")
    available = set(data.y_subject.unique().tolist())
    cohorts = []
    for key in ("train_subject_ids", "val_subject_ids"):
        ids = config[key]
        missing = sorted(set(ids) - available)
        if missing:
            raise ValueError(f"{key} unavailable after activity filtering: {missing}")
        mask = torch.isin(data.y_subject, data.y_subject.new_tensor(ids))
        selected = _take(data, mask)
        if selected.subject_ids != ids:
            raise ValueError(f"{key} exact cohort membership mismatch")
        cohorts.append(selected)
    return cohorts[0], cohorts[1], list(config["val_subject_ids"])


def _validate_precision(config, device):
    if config["precision"] == "bfloat16":
        if device.type != "cuda":
            raise ValueError("bfloat16 precision requires CUDA")
        # Check the requested GPU, which need not be the process default device.
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise ValueError(
                    "bfloat16 precision requires a BF16-supported CUDA device"
                )


def _build_classifier(config, n_classes):
    if float(config["ce_weight"]) == 0:
        return None
    dimension = int(config["model_args"]["embed_dim"])
    if config["classifier_type"] == "arcface":
        from models.angular_margin import AngularMarginClassifier

        return AngularMarginClassifier(
            dimension,
            n_classes,
            scale=float(config["angular_scale"]),
            margin=float(config["angular_margin"]),
        )
    return nn.Linear(dimension, n_classes)


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
    # Verification always uses FP32, including inside an outer autocast context.
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        for batch in windows.split(batch_size):
            parts.append(
                encoder(batch.to(device=device, dtype=torch.float32)).float().cpu()
            )
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


def _training_losses(z, labels, sample_ids, classifier, classifier_labels, config):
    """Distance, angular logits and all reductions stay FP32 under encoder AMP."""
    with torch.autocast(device_type=z.device.type, enabled=False):
        z = z.float()
        hard, supcon = z.sum() * 0.0, z.sum() * 0.0
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
                z,
                labels,
                float(config["temperature"]),
                sample_ids=sample_ids,
            )
        metric_loss = (
            (hard + supcon) / 2 if config["objective"] == "hybrid" else hard + supcon
        )
        ce = z.sum() * 0.0
        if classifier is not None:
            # Only optimizer-training IDs reach CE; unseen validation IDs never do.
            logits = (
                classifier(z, classifier_labels)
                if config["classifier_type"] == "arcface"
                else classifier(z)
            )
            ce = F.cross_entropy(
                logits.float(),
                classifier_labels,
                label_smoothing=float(config["label_smoothing"]),
            )
        loss = metric_loss + float(config["ce_weight"]) * ce
    return hard, supcon, ce, loss


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
    """Train exact optimizer identities; select validation EER or a fixed final epoch."""
    started = time.monotonic()
    config = _resolved_config(config)
    seed = int(config["seed"])
    split_seed, val_seed = int(config["split_seed"]), int(config["val_seed"])
    seed_everything(seed)
    torch.set_num_threads(int(config["torch_threads"]))
    device = _device(config)
    _validate_precision(config, device)
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
    train_data, val_data, val_ids = _split_cohorts(data, config)
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
    classifier = _build_classifier(config, len(train_ids))
    if classifier is not None:
        classifier.to(device)
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
    selection_metric = (
        "val_eer" if config["epoch_selection"] == "validation_eer" else "final_epoch"
    )
    classifier_config = {
        "classifier_type": config["classifier_type"],
        "angular_scale": float(config["angular_scale"]),
        "angular_margin": float(config["angular_margin"]),
        "enabled": classifier is not None,
        "n_classes": len(train_ids),
    }
    protocol = {
        "name": "held_out_subject_within_recording_discrimination",
        "source": config["source"],
        "data_root": config["data_root"],
        "train_subject_ids": train_ids,
        "val_subject_ids": val_ids,
        "classifier_subject_ids": train_ids,
        "classifier_label_mapping": "sorted_train_ids_to_0_based",
        "classifier_config": classifier_config,
        "cohort_assignment": "explicit"
        if "train_subject_ids" in config
        else "seeded_split",
        "unused_subject_ids": sorted(
            set(data.subject_ids) - set(train_ids) - set(val_ids)
        ),
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
        "epoch_selection": config["epoch_selection"],
        "selection_metric": selection_metric,
        "selection_ties": "earliest_epoch"
        if config["epoch_selection"] == "validation_eer"
        else "not_applicable",
        "validation_used_for_selection": config["epoch_selection"] == "validation_eer",
        "precision": config["precision"],
        "evaluation_precision": "float32",
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
                x = _augment(x, config["augment"])
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=config["precision"] == "bfloat16",
                ):
                    z = encoder(x)
                if z.ndim != 2 or z.shape != (
                    len(rows),
                    int(config["model_args"]["embed_dim"]),
                ):
                    raise ValueError(
                        "encoder output must have shape (batch, embed_dim)"
                    )
                hard, supcon, ce, loss = _training_losses(
                    z,
                    labels,
                    sample_ids,
                    classifier,
                    remapped[indices].to(device),
                    config,
                )
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
            selected = (
                record["val_eer"] < best_eer
                if config["epoch_selection"] == "validation_eer"
                else epoch == int(config["epochs"])
            )
            if selected:
                best_eer, best_epoch = record["val_eer"], epoch
                # Save only raw encoder keys: no wrapper/classification prefix.
                torch.save(
                    {
                        "checkpoint_format": "gait_metric_v1",
                        "epoch": epoch,
                        "completed_epochs": epoch,
                        "epoch_selection": config["epoch_selection"],
                        "precision": config["precision"],
                        "classifier_config": classifier_config,
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
                            "per_epoch": list(records),
                            "protocol": protocol,
                            "validation": validation,
                        },
                        "hardware": hardware,
                    },
                    Path(out_dir) / "best.pt",
                )
            scheduler.step()
    # An early selected checkpoint still records the entire completed run while
    # retaining its selected encoder/classifier state and per-epoch validation.
    protocol["completed_epochs"] = len(records)
    checkpoint_path = Path(out_dir) / "best.pt"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    payload["completed_epochs"] = len(records)
    payload["metrics"]["per_epoch"] = records
    payload["metrics"]["protocol"] = protocol
    torch.save(payload, checkpoint_path)
    metrics = {
        "method": "metric_learning",
        "objective": config["objective"],
        "epoch_selection": config["epoch_selection"],
        "selection_metric": selection_metric,
        "best_epoch": best_epoch,
        "best_val_eer": best_eer,
        "selected_epoch": best_epoch,
        "selected_val_eer": best_eer,
        "observed_min_val_eer": min(record["val_eer"] for record in records),
        "epochs": int(config["epochs"]),
        "completed_epochs": len(records),
        "precision": config["precision"],
        "classifier_config": classifier_config,
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
