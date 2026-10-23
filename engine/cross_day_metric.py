"""In-memory, train-only cross-day metric fitting; no dataset or checkpoint I/O.

This is a sensor-window adaptation of supervised contrastive learning, not a
reproduction. See docs/cross_day_metric.md for the objective and protocol limits.
"""

from __future__ import annotations

import hashlib
import json
import marshal
import math
import random
import sys
import time
from collections.abc import Mapping
from numbers import Integral

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from data.gait_transforms import augment_gait
from models.gait_encoders import build_gait_encoder
from models.metric_learning import supervised_contrastive_loss


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _ids(values, name, size=None, device="cpu"):
    if not isinstance(values, (np.ndarray, torch.Tensor)):
        raise TypeError(f"{name} must be an in-memory NumPy array or Torch tensor")
    if isinstance(values, np.ndarray) and values.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integer IDs")
    values = torch.as_tensor(values)
    if values.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise ValueError(f"{name} must contain integer IDs representable as int64")
    if (
        values.ndim != 1
        or not len(values)
        or (size is not None and len(values) != size)
    ):
        raise ValueError(f"{name} must have one integer ID per row")
    return values.detach().to(device=device, dtype=torch.int64)


def cross_day_pair_masks(subjects, days, sample_ids=None):
    """Same-ID/different-day positives; different-ID negatives; no repeated row.

    Same-day same-ID pairs are excluded entirely, rather than made negatives.
    ``sample_ids`` are original input row indices, retained under replacement.
    """
    subjects = _ids(subjects, "subjects", device=getattr(subjects, "device", "cpu"))
    days = _ids(days, "days", len(subjects), subjects.device)
    distinct = ~torch.eye(len(subjects), dtype=torch.bool, device=subjects.device)
    if sample_ids is not None:
        rows = _ids(sample_ids, "sample_ids", len(subjects), subjects.device)
        distinct &= rows[:, None] != rows[None, :]
    same_subject = subjects[:, None] == subjects[None, :]
    positive = same_subject & (days[:, None] != days[None, :]) & distinct
    negative = ~same_subject & distinct
    return positive, negative


def cross_day_supervised_contrastive_loss(
    embeddings,
    subjects,
    days,
    temperature=0.1,
    *,
    sample_ids=None,
):
    """Outside-log positive average with a cross-day-only positive set."""
    if embeddings.ndim != 2 or min(embeddings.shape) < 1:
        raise ValueError("embeddings must have nonempty shape (N,D)")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    labels = _ids(subjects, "subjects", len(embeddings), embeddings.device)
    day_ids = _ids(days, "days", len(embeddings), embeddings.device)
    positive, negative = cross_day_pair_masks(labels, day_ids, sample_ids)
    valid = positive.any(1) & negative.any(1)
    if not valid.any():
        return embeddings.sum() * 0.0
    z = F.normalize(embeddings.float(), p=2, dim=1)
    logits = (z @ z.T)[valid] / temperature
    positives = positive[valid]
    allowed = (positive | negative)[valid]
    denominator = torch.logsumexp(logits.masked_fill(~allowed, -torch.inf), dim=1)
    positive_mean = torch.where(positives, logits, 0.0).sum(1) / positives.sum(1)
    return (denominator - positive_mean).mean()


class CrossDayBatchSampler:
    """P distinct subjects × K rows, split equally between two distinct days.

    All subjects are visited each epoch. Day pairs are shuffled and cycled;
    rows are unique within each day draw when enough originals are available.
    With only one row on a day, replacement retains the original row index.
    Pair matrices never exceed 32×32 with the fitting defaults (P<=8, K=4).
    """

    def __init__(
        self, subjects, days, *, identities_per_batch=8, windows_per_identity=4, seed=42
    ):
        subjects = _ids(subjects, "subjects")
        days = _ids(days, "days", len(subjects))
        self.requested_p = _integer(identities_per_batch, "identities_per_batch", 2)
        self.k = _integer(windows_per_identity, "windows_per_identity", 2)
        if self.k % 2:
            raise ValueError("windows_per_identity must be even")
        self.seed = _integer(seed, "seed", 0)
        self.groups = {}
        for row, (subject, day) in enumerate(zip(subjects.tolist(), days.tolist())):
            self.groups.setdefault(subject, {}).setdefault(day, []).append(row)
        self.identities = sorted(self.groups)
        if len(self.identities) < 2:
            raise ValueError("need at least two training identities")
        if any(len(groups) < 2 for groups in self.groups.values()):
            raise ValueError("every training identity must have at least two days")
        self.p = min(self.requested_p, len(self.identities))
        self.steps = max(
            math.ceil(len(subjects) / (self.p * self.k)),
            math.ceil(len(self.identities) / self.p),
        )
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = _integer(epoch, "epoch", 0)

    def __len__(self):
        return self.steps

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        remaining = []
        day_orders = {}
        for _ in range(self.steps):
            selected = []
            while len(selected) < self.p:
                if not remaining:
                    remaining = self.identities.copy()
                    rng.shuffle(remaining)
                position = next(
                    (i for i, s in enumerate(remaining) if s not in selected),
                    None,
                )
                if position is None:
                    selected.extend(
                        rng.sample(
                            [s for s in self.identities if s not in selected],
                            self.p - len(selected),
                        )
                    )
                    break
                selected.append(remaining.pop(position))
            batch = []
            for subject in selected:
                order = day_orders.setdefault(subject, [])
                if len(order) < 2:
                    choices = list(self.groups[subject])
                    rng.shuffle(choices)
                    # Keep a remaining day and choose a different companion.
                    if order:
                        choices.remove(order[0])
                    order.extend(choices)
                chosen_days = (order.pop(0), order.pop(0))
                for day in chosen_days:
                    rows = self.groups[subject][day]
                    count = self.k // 2
                    draw = rng.sample(rows, min(count, len(rows)))
                    draw.extend(rng.choices(rows, k=count - len(draw)))
                    batch.extend(draw)
            yield batch

    @property
    def context(self):
        return {
            "requested_identities_per_batch": self.requested_p,
            "identities_per_batch": self.p,
            "windows_per_identity": self.k,
            "days_per_identity_per_batch": 2,
            "windows_per_selected_day": self.k // 2,
            "steps_per_epoch": self.steps,
            "batch_size": self.p * self.k,
            "subject_coverage_per_epoch_guaranteed": True,
            "row_replacement": "only_if_selected_day_has_fewer_than_K_over_2_rows",
            "original_row_ids_preserved": True,
        }


def _array_hash(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _source_record():
    # Code-object fingerprints need no source-file reads. Parent runners should
    # additionally freeze ordinary source-file hashes in their experiment lock.
    functions = []
    modules = [
        __name__,
        "models.gait_encoders",
        "models.metric_learning",
        "models.siamese",
        "data.gait_transforms",
    ]
    for name in modules:
        for value in vars(sys.modules[name]).values():
            if getattr(value, "__module__", None) != name:
                continue
            if hasattr(value, "__code__"):
                functions.append(value)
            elif isinstance(value, type):
                for method in vars(value).values():
                    if isinstance(method, property):
                        method = method.fget
                    elif isinstance(method, (staticmethod, classmethod)):
                        method = method.__func__
                    if hasattr(method, "__code__"):
                        functions.append(method)
    return {
        "module": "engine.cross_day_metric",
        "implementation_version": 1,
        "fingerprint_kind": "sha256_of_python_marshal_code_object",
        "callable_code_sha256": {
            f"{f.__module__}.{f.__qualname__}": hashlib.sha256(
                marshal.dumps(f.__code__)
            ).hexdigest()
            for f in functions
        },
        "references": [
            "https://papers.nips.cc/paper_files/paper/2020/file/d89a66c7c80a29b1bdbab0f2a1a94af8-Paper.pdf",
            "https://arxiv.org/abs/2007.15958",
            "https://doi.org/10.1007/s00521-022-07741-0",
        ],
    }


def fit_cross_day_encoder(
    X,
    subjects,
    days,
    *,
    n_channels,
    model_args,
    seed=42,
    epochs=40,
    device="cpu",
    torch_threads=2,
    objective="cross_day_supcon",
):
    """Fit final-epoch FP32 encoder exclusively on caller-supplied TRAIN rows.

    Returns ``(encoder.eval(), JSON-compatible training_record)``. There is no
    validation, early stopping, dataset access, save, or test-time fitting.
    Normalization sees unaugmented training rows exactly once before AdamW.
    The auxiliary training-ID classifier is discarded on return.
    """
    started = time.perf_counter()
    n_channels = _integer(n_channels, "n_channels")
    seed = _integer(seed, "seed", 0)
    if seed >= 2**32:
        raise ValueError("seed must be below 2**32")
    epochs = _integer(epochs, "epochs")
    torch_threads = _integer(torch_threads, "torch_threads")
    if objective not in {"cross_day_supcon", "supcon"}:
        raise ValueError("objective must be cross_day_supcon or supcon")
    if not isinstance(model_args, Mapping):
        raise TypeError("model_args must be an encoder configuration mapping")
    # JSON round-trip rejects nonfinite values and non-serializable objects.
    args = {
        "architecture": "multiscale",
        "embed_dim": 64,
        "width": 32,
        "dropout": 0.1,
        "stats_fusion": False,
        "normalization": True,
        **json.loads(json.dumps(dict(model_args), allow_nan=False)),
    }
    known = {
        "architecture",
        "embed_dim",
        "width",
        "dropout",
        "normalization",
        "stats_fusion",
        "stats_include_mean",
        "norm_eps",
        "vector_groups",
        "add_magnitudes",
        "temporal_bins",
    }
    if set(args) - known:
        raise ValueError(f"unknown encoder model_args: {sorted(set(args) - known)}")
    for key in ("embed_dim", "width"):
        args[key] = _integer(args[key], key, 4 if key == "width" else 1)
    if "temporal_bins" in args:
        args["temporal_bins"] = _integer(args["temporal_bins"], "temporal_bins")
    architecture = args["architecture"]
    args.setdefault("add_magnitudes", architecture != "legacy")
    args.setdefault("stats_include_mean", False)
    args.setdefault("norm_eps", 1e-5)
    if args.get("vector_groups") is None:
        args["vector_groups"] = (
            [list(range(i, i + 3)) for i in range(0, n_channels, 3)]
            if n_channels in (3, 9)
            else []
        )
    if architecture == "temporal":
        args.setdefault("temporal_bins", 4)
    for group in args["vector_groups"]:
        for index in group:
            _integer(index, "vector_groups channel", 0)
    if not isinstance(X, (np.ndarray, torch.Tensor)):
        raise TypeError("X must be an in-memory NumPy array or Torch tensor")
    windows = torch.as_tensor(X)
    if (
        windows.ndim != 3
        or min(windows.shape) == 0
        or windows.shape[1] != n_channels
        or not windows.is_floating_point()
    ):
        raise ValueError(f"X must be nonempty floating (N,{n_channels},T)")
    if architecture == "legacy" and windows.shape[2] < 4:
        raise ValueError("legacy windows require T >= 4")
    windows = windows.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not torch.isfinite(windows).all():
        raise ValueError("X must contain finite FP32-representable values")
    subjects = _ids(subjects, "subjects", len(windows))
    days = _ids(days, "days", len(windows))
    sampler = CrossDayBatchSampler(subjects, days, seed=seed)
    identities, classifier_labels = torch.unique(
        subjects, sorted=True, return_inverse=True
    )
    target = torch.device(device)
    if target.type not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if target.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if target.type == "cuda" and target.index is None:
        target = torch.device("cuda", torch.cuda.current_device())
    config = {
        "n_channels": n_channels,
        "model_args": args,
        "seed": seed,
        "epochs": epochs,
        "epoch_selection": "fixed_final_epoch",
        "device": str(target),
        "precision": "float32",
        "torch_threads": torch_threads,
        "objective": objective,
        "temperature": 0.1,
        "ce_weight": 0.25,
        "classifier": "linear_training_id_only",
        "label_smoothing": 0.0,
        "optimizer": "AdamW",
        "optimizer_args": {
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "amsgrad": False,
            "foreach": None,
            "maximize": False,
            "capturable": False,
            "differentiable": False,
            "fused": False,
        },
        "lr": 0.001,
        "weight_decay": 0.0001,
        "gradient_clip": 1.0,
        "scheduler": "none",
        "augment": {},
        "sampler": sampler.context,
    }
    previous_threads = torch.get_num_threads()
    cuda_devices = (
        [target.index if target.index is not None else torch.cuda.current_device()]
        if target.type == "cuda"
        else []
    )
    history = []
    try:
        torch.set_num_threads(torch_threads)
        # fork_rng restores CPU and the selected CUDA stream's RNG after fitting.
        with torch.random.fork_rng(devices=cuda_devices):
            torch.random.default_generator.manual_seed(seed)
            if cuda_devices:
                with torch.cuda.device(cuda_devices[0]):
                    torch.cuda.manual_seed(seed)
            encoder = build_gait_encoder(n_channels, args).to(
                device=target, dtype=torch.float32
            )
            encoder.train()
            with torch.autocast(device_type=target.type, enabled=False):
                encoder.fit_normalization(windows)  # exactly once, before optimizer
            classifier = nn.Linear(
                encoder.embed_dim, len(identities), dtype=torch.float32
            ).to(target)
            parameters = list(encoder.parameters()) + list(classifier.parameters())
            optimizer = torch.optim.AdamW(
                parameters,
                lr=config["lr"],
                weight_decay=config["weight_decay"],
                **config["optimizer_args"],
            )
            generator = torch.Generator().manual_seed(seed)
            # Disable ambient autocast: this API promises FP32 on CPU and CUDA.
            with torch.autocast(device_type=target.type, enabled=False):
                for epoch in range(epochs):
                    sampler.set_epoch(epoch)
                    totals = np.zeros(3, dtype=np.float64)
                    for rows in sampler:
                        indices = torch.tensor(rows, dtype=torch.int64)
                        batch = augment_gait(
                            windows[indices].to(target), config["augment"], generator
                        )
                        labels = subjects[indices].to(target)
                        day_ids = days[indices].to(target)
                        original_rows = indices.to(target)
                        optimizer.zero_grad(set_to_none=True)
                        z = encoder(batch).float()
                        metric = (
                            cross_day_supervised_contrastive_loss(
                                z,
                                labels,
                                day_ids,
                                config["temperature"],
                                sample_ids=original_rows,
                            )
                            if objective == "cross_day_supcon"
                            else supervised_contrastive_loss(
                                z,
                                labels,
                                config["temperature"],
                                sample_ids=original_rows,
                            )
                        )
                        ce = F.cross_entropy(
                            classifier(z), classifier_labels[indices].to(target)
                        )
                        loss = metric + config["ce_weight"] * ce
                        if not torch.isfinite(loss):
                            raise FloatingPointError(
                                "nonfinite cross-day training loss"
                            )
                        loss.backward()
                        nn.utils.clip_grad_norm_(
                            parameters, config["gradient_clip"], error_if_nonfinite=True
                        )
                        optimizer.step()
                        totals += [
                            float(loss.detach()),
                            float(metric.detach()),
                            float(ce.detach()),
                        ]
                    means = totals / len(sampler)
                    history.append(
                        {
                            "epoch": epoch + 1,
                            "loss": float(means[0]),
                            "metric_loss": float(means[1]),
                            "ce_loss": float(means[2]),
                            "lr": config["lr"],
                            "steps": len(sampler),
                        }
                    )
            encoder.eval()
            if not all(torch.isfinite(v).all() for v in encoder.state_dict().values()):
                raise FloatingPointError("nonfinite trained encoder state")
            if not all(
                torch.isfinite(v).all() for v in classifier.state_dict().values()
            ):
                raise FloatingPointError("nonfinite auxiliary classifier state")
    finally:
        torch.set_num_threads(previous_threads)
    row_ids = np.arange(len(windows), dtype=np.int64)
    record = {
        "schema_version": 1,
        "config": config,
        "source": _source_record(),
        "environment": {
            "torch": str(torch.__version__),
            "numpy": str(np.__version__),
            "cuda": torch.version.cuda,
        },
        "training_subject_ids": identities.tolist(),
        "training_days_by_subject": {
            str(s): sorted(groups) for s, groups in sampler.groups.items()
        },
        "input_shape": list(windows.shape),
        "input_dtype": str(torch.as_tensor(X).dtype),
        "training_dtype": "torch.float32",
        "training_window_count": len(windows),
        "window_id_scope": "caller_input_row_index_with_subject_and_day; not physical recording IDs",
        "training_window_ids_sha256": _array_hash(
            row_ids, subjects.numpy(), days.numpy()
        ),
        "training_input_sha256": _array_hash(
            windows.numpy(), subjects.numpy(), days.numpy()
        ),
        "normalization_fit_calls": 1,
        "normalization_scope": "all_caller_supplied_training_rows_only",
        "encoder_parameters": sum(p.numel() for p in encoder.parameters()),
        "classifier_parameters": sum(p.numel() for p in classifier.parameters()),
        "history": history,
        "epochs_completed": epochs,
        "selected_epoch": epochs,
        "all_finite": True,
        "wall_seconds": time.perf_counter() - started,
        "selection": "none; caller owns subject-fold selection and every held-out evaluation",
        "determinism": "seeded CPU/CUDA RNG and epoch-local sampler; CUDA kernels may be nondeterministic",
    }
    json.dumps(record, allow_nan=False)
    return encoder, record
