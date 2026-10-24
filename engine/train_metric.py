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
import random
import tempfile
import time
from pathlib import Path

import numpy as np
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
    # Never silently truncate an optimization budget or sampling dimension.
    for key in (
        "epochs",
        "torch_threads",
        "eval_batch_size",
        "window_size",
        "stride",
        "identities_per_batch",
        "windows_per_identity",
        "validation_pairs",
        "val_subjects",
        "seed",
        "split_seed",
        "val_seed",
    ):
        minimum = 0 if key in {"seed", "split_seed", "val_seed", "val_subjects"} else 1
        if type(resolved[key]) is not int or resolved[key] < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    if resolved.get("steps_per_epoch") is not None and (
        type(resolved["steps_per_epoch"]) is not int or resolved["steps_per_epoch"] <= 0
    ):
        raise ValueError("steps_per_epoch must be a positive integer")
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


def _json_digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _window_fingerprint(data):
    """Hash TRAIN tensor roles, shapes, dtypes, order, values and interpretation."""
    checksum = hashlib.sha256()
    metadata = {
        "activity_names": data.activity_names,
        "subject_ids": data.subject_ids,
        "sampling_rate": float(data.sampling_rate),
    }
    checksum.update(json.dumps(metadata, sort_keys=True, allow_nan=False).encode())
    for name in ("X", "y_subject", "y_activity"):
        value = getattr(data, name)
        description = None if value is None else [list(value.shape), str(value.dtype)]
        checksum.update(json.dumps([name, description]).encode())
        if value is not None:
            # memoryview avoids another dataset-sized byte-string allocation.
            checksum.update(
                memoryview(value.detach().cpu().contiguous().numpy()).cast("B")
            )
    return checksum.hexdigest()


def _source_fingerprints():
    root = Path(__file__).resolve().parents[1]
    # Include transitive local model, loader and evaluation dependencies.
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for directory in ("engine", "models", "data")
        for path in sorted((root / directory).rglob("*.py"))
    }


def _cpu_state(value):
    """Detach snapshots, including CPU tensors that otherwise alias live weights."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_state(item) for item in value)
    return value


def _atomic_write(path, write):
    """Replace one artifact only after a complete, flushed same-directory write."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_torch_save(payload, path):
    _atomic_write(path, lambda stream: torch.save(payload, stream))


def _rng_state(device):
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "algorithm": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
    }


def _restore_rng(state, device):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["algorithm"],
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )
    torch.set_rng_state(state["torch_cpu"])
    if device.type == "cuda":
        if len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("resume CUDA RNG device count mismatch")
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    elif state["torch_cuda"]:
        raise ValueError("cannot restore CUDA training RNG on CPU")


def _sampler_state(sampler, completed_epochs):
    # __iter__ owns a fresh local Random(seed + epoch). At an epoch boundary
    # there are no pending draws; persist the exact next iterator's RNG state.
    return {
        "algorithm": "python_random_seed_plus_epoch_v1",
        "seed": sampler.seed,
        "next_epoch": completed_epochs,
        "rng_state": random.Random(sampler.seed + completed_epochs).getstate(),
        "context": sampler.context,
    }


def _resume_provenance(config, out_dir, data_fingerprints, protocol, hardware):
    return {
        "output_dir": str(Path(out_dir).resolve()),
        "data_root": str(Path(config["data_root"]).resolve()),
        "config_sha256": _json_digest(config),
        "sources": _source_fingerprints(),
        "data": data_fingerprints,
        "cohorts_and_protocol": protocol,
        "runtime": {
            **hardware,
            "python_version": platform.python_version(),
            "numpy_version": str(np.__version__),
            "cuda_device_count": torch.cuda.device_count()
            if hardware["device"].startswith("cuda")
            else 0,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
    }


def _load_resume(out_dir, config):
    path = Path(out_dir) / "last.pt"
    if not path.is_file():
        raise ValueError(
            "resume requires last.pt from --resumable; best.pt is inference-only"
        )
    state = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(state, dict)
        or state.get("checkpoint_format") != "gait_metric_resume_v1"
    ):
        raise ValueError(
            "expected a gait_metric_resume_v1 last.pt; legacy best.pt cannot resume"
        )
    if state.get("config") != config:
        raise ValueError(
            "resume config mismatch (including the original total epochs budget)"
        )
    provenance = state.get("provenance", {})
    if provenance.get("output_dir") != str(Path(out_dir).resolve()):
        raise ValueError("resume output directory provenance mismatch")
    if provenance.get("config_sha256") != _json_digest(config):
        raise ValueError("resume config fingerprint mismatch")
    if provenance.get("sources") != _source_fingerprints():
        raise ValueError(
            "resume source fingerprint mismatch; restore the original source checkout"
        )
    records, completed = state.get("history"), state.get("completed_epochs")
    if (
        type(completed) is not int
        or not 0 <= completed <= config["epochs"]
        or not isinstance(records, list)
        or len(records) != completed
        or any(
            not isinstance(row, dict) or row.get("epoch") != i
            for i, row in enumerate(records, 1)
        )
    ):
        raise ValueError("resume completed epoch/history mismatch")
    json.dumps(records, allow_nan=False)
    selected = None
    if records and config["epoch_selection"] == "validation_eer":
        selected = min(records, key=lambda row: row["val_eer"])
    elif completed == config["epochs"]:
        selected = records[-1]
    best = state.get("best_checkpoint")
    if (
        state.get("best_epoch") != (selected["epoch"] if selected else 0)
        or state.get("best_val_eer") != (selected["val_eer"] if selected else None)
        or (selected is None and best is not None)
        or (
            selected is not None
            and (
                not isinstance(best, dict)
                or best.get("checkpoint_format") != "gait_metric_v1"
                or best.get("epoch") != selected["epoch"]
                or best.get("config") != config
                or best.get("metrics", {}).get("val_eer") != selected["val_eer"]
            )
        )
    ):
        raise ValueError("resume best selection/history mismatch")
    if state.get("scaler_state_dict") is not None:
        raise ValueError("this trainer uses float32/BF16 without a gradient scaler")
    return state


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


def train_metric(
    config: dict, out_dir: str, *, resumable: bool = False, resume: bool = False
) -> dict:
    """Train metric identities, optionally committing epoch-boundary last.pt state.

    ``resume=True`` implies ``resumable=True`` and keeps the original total epoch
    budget. Only a matching last.pt in its original output directory is accepted.
    """
    started = time.monotonic()
    config = _resolved_config(config)
    resumable = resumable or resume
    resume_state = _load_resume(out_dir, config) if resume else None
    managed = ("last.pt", "best.pt", "per_epoch.jsonl", "metrics.json")
    if not resume and (
        (resumable and any((Path(out_dir) / name).exists() for name in managed))
        or (Path(out_dir) / "last.pt").exists()
    ):
        raise ValueError(
            "training artifacts already exist; use --resume or a fresh output directory"
        )
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
    data_fingerprints = {"loaded_train": _window_fingerprint(data)} if resumable else {}
    expected_channels = {"har": 9, "wisdm_v1": 3}.get(config["source"])
    if expected_channels is not None and data.X.shape[1] != expected_channels:
        raise ValueError(
            f"{config['source']} requires {expected_channels} source channels"
        )
    # Match the walking baseline: select the declared activity cohort before
    # the fixed subject split (some source identities have no walking rows).
    data = _filter_activities(data, config["activity_ids"])
    train_data, val_data, val_ids = _split_cohorts(data, config)
    if resumable:
        data_fingerprints.update(
            optimizer_train=_window_fingerprint(train_data),
            validation=_window_fingerprint(val_data),
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
    records, best_eer, best_epoch = [], math.inf, 0
    best_payload = None
    completed_epochs, previous_elapsed = 0, 0.0
    provenance = (
        _resume_provenance(config, out_dir, data_fingerprints, protocol, hardware)
        if resumable
        else None
    )
    if resume_state is not None:
        for key, value in provenance.items():
            if resume_state["provenance"].get(key) != value:
                raise ValueError(f"resume {key} provenance mismatch")
        completed_epochs = resume_state["completed_epochs"]
        if resume_state.get("sampler_state") != _sampler_state(
            sampler, completed_epochs
        ):
            raise ValueError("resume sampler RNG state mismatch")
        encoder.load_state_dict(resume_state["model_state_dict"], strict=True)
        if classifier is None:
            if resume_state["classifier_state_dict"] is not None:
                raise ValueError("resume classifier state mismatch")
        else:
            classifier.load_state_dict(
                resume_state["classifier_state_dict"], strict=True
            )
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        scheduler.load_state_dict(resume_state["scheduler_state_dict"])
        if (
            scheduler.last_epoch != completed_epochs
            or scheduler.T_max != config["epochs"]
        ):
            raise ValueError("resume scheduler/epoch budget mismatch")
        records = resume_state["history"]
        best_epoch = resume_state["best_epoch"]
        best_eer = resume_state["best_val_eer"] if best_epoch else math.inf
        best_payload = resume_state["best_checkpoint"]
        previous_elapsed = float(resume_state["elapsed_seconds"])
        _restore_rng(resume_state["rng_state"], device)

    os.makedirs(out_dir, exist_ok=True)
    log_path = Path(out_dir) / "per_epoch.jsonl"

    def save_boundary(completed):
        _atomic_torch_save(
            {
                "checkpoint_format": "gait_metric_resume_v1",
                "config": config,
                "provenance": provenance,
                "completed_epochs": completed,
                "history": list(records),
                "best_epoch": best_epoch,
                "best_val_eer": best_eer if best_epoch else None,
                "best_checkpoint": best_payload,
                "model_state_dict": _cpu_state(encoder.state_dict()),
                "classifier_state_dict": _cpu_state(classifier.state_dict())
                if classifier is not None
                else None,
                "optimizer_state_dict": _cpu_state(optimizer.state_dict()),
                "scheduler_state_dict": scheduler.state_dict(),
                # BF16 has no GradScaler; explicitly record the absence of one.
                "scaler_state_dict": None,
                "rng_state": _rng_state(device),
                "sampler_state": _sampler_state(sampler, completed),
                "elapsed_seconds": previous_elapsed + time.monotonic() - started,
            },
            Path(out_dir) / "last.pt",
        )

    if resume:
        # last.pt is authoritative if interruption left a stale best or a
        # partial/ahead log. Rebuild derived artifacts without duplicate rows.
        encoded = "".join(
            json.dumps(row, allow_nan=False) + "\n" for row in records
        ).encode()
        _atomic_write(log_path, lambda stream: stream.write(encoded))
        if best_payload is not None:
            _atomic_torch_save(best_payload, Path(out_dir) / "best.pt")
        else:
            (Path(out_dir) / "best.pt").unlink(missing_ok=True)
        (Path(out_dir) / "metrics.json").unlink(missing_ok=True)
    elif resumable:
        save_boundary(0)
    with log_path.open("a" if resume else "w") as log_file:
        for epoch in range(completed_epochs + 1, int(config["epochs"]) + 1):
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
                best_payload = {
                    "checkpoint_format": "gait_metric_v1",
                    "epoch": epoch,
                    "completed_epochs": epoch,
                    "epoch_selection": config["epoch_selection"],
                    "precision": config["precision"],
                    "classifier_config": classifier_config,
                    "model_state_dict": {
                        k: v.detach().cpu().clone()
                        for k, v in encoder.state_dict().items()
                    },
                    "classifier_state_dict": {
                        k: v.detach().cpu().clone()
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
                }
                if not resumable:
                    torch.save(best_payload, Path(out_dir) / "best.pt")
            scheduler.step()
            if resumable:
                # Commit optimizer/scheduler/RNG and the selected snapshot in
                # one file before updating the inference-only best.pt copy.
                save_boundary(epoch)
                if selected:
                    _atomic_torch_save(best_payload, Path(out_dir) / "best.pt")
    # An early selected checkpoint still records the entire completed run while
    # retaining its selected encoder/classifier state and per-epoch validation.
    protocol["completed_epochs"] = len(records)
    checkpoint_path = Path(out_dir) / "best.pt"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    payload["completed_epochs"] = len(records)
    payload["metrics"]["per_epoch"] = records
    payload["metrics"]["protocol"] = protocol
    if resumable:
        _atomic_torch_save(payload, checkpoint_path)
    else:
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
        "elapsed_seconds": previous_elapsed + time.monotonic() - started,
        "git_commit": _git_commit(),
        "checkpoint": str(Path(out_dir) / "best.pt"),
    }
    encoded_metrics = json.dumps(metrics, indent=2, allow_nan=False) + "\n"
    if resumable:
        _atomic_write(
            Path(out_dir) / "metrics.json",
            lambda stream: stream.write(encoded_metrics.encode()),
        )
    else:
        (Path(out_dir) / "metrics.json").write_text(encoded_metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument(
        "--resumable",
        action="store_true",
        help="write epoch-boundary last.pt (fresh output directory)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume matching last.pt in --out-dir at the original total epoch budget",
    )
    args = parser.parse_args()
    with open(args.config) as stream:
        config = yaml.safe_load(stream)
    if args.epochs is not None:
        config["epochs"] = args.epochs
    print(
        json.dumps(
            train_metric(
                config, args.out_dir, resumable=args.resumable, resume=args.resume
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
