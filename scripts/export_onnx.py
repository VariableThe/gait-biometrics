#!/usr/bin/env python3
"""Edge-compression export entry point (paper B6, Student B track).

Loads a trained classification checkpoint, measures it on a deterministic
stratified slice of the test split, optionally applies *structured* pruning,
exports the pruned FP32 model to ONNX, and (optionally) measures a dynamically
quantized copy separately.

Run from the repository root:
    python -m scripts.export_onnx --config configs/lightweight.yaml \
        --checkpoint runs/smoke_classification/cnn_lstm/best.pt --out model.onnx \
        [--prune 0.3 | --no-prune] [--quantize | --no-quantize] \
        [--eval-subset 1000] [--report report.json]

Pruning/quantization are config-driven with CLI overrides winning:
``configs/lightweight.yaml`` ships ``prune: 0.3`` and ``quantize: true``, which
this script uses as its defaults. ``--prune AMOUNT`` / ``--no-prune`` and
``--quantize`` / ``--no-quantize`` override those values explicitly.

Pruning delegates to ``models.compress.apply_pruning``, so it supports exactly
the topology that implementation implements index-for-index: ``DeepConvLSTM``
(B3, the model ``configs/lightweight.yaml`` selects). ``MultiSensorFusionCNN``
(B1), ``MultiTaskNet`` (B2), ``EmbeddingNet`` (B4), ``SEConvNet`` (B5) and
``GaitViT`` (B7) fail fast with a clear error rather than exporting a broken or
silently-unpruned graph; use ``--no-prune`` (or unstructured pruning) for those.

Notes on fidelity:
  * The exported ONNX file is the *pruned FP32* model. When quantization is
    enabled, a *dynamically quantized copy* is measured (params + serialized
    size) only as a separate estimate; it is NOT the artifact written to
    ``--out`` and that file is never claimed to be quantized.
  * The checkpoint is loaded with ``weights_only=True`` (see
    ``models.common.load_checkpoint``), so an untrusted checkpoint cannot
    execute code during load.
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import os
import sys

import torch
from torch import nn

# Repo root on sys.path so the module runs as `python -m scripts.export_onnx`
# from anywhere.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _fail(msg: str) -> None:
    """Print a clear error to stderr and exit non-zero."""
    print(f"export_onnx: ERROR: {msg}", file=sys.stderr)
    raise SystemExit(2)


def resolve_compression_options(config: dict, prune_cli: float | None,
                                no_prune: bool, quantize_cli: bool | None
                                ) -> tuple[float | None, bool]:
    """Effective ``(prune_amount, quantize)`` from a config, with CLI overrides.

    The config supplies the defaults (``prune`` / ``quantize``) and an explicit
    CLI value wins:

    * ``no_prune=True``            -> pruning disabled regardless of the config.
    * ``prune_cli`` is not None    -> that amount, overriding the config.
    * ``quantize_cli`` is not None -> that flag, overriding the config.
    """
    if no_prune:
        prune: float | None = None
    elif prune_cli is not None:
        prune = float(prune_cli)
    else:
        config_prune = config.get("prune")
        prune = None if config_prune is None else float(config_prune)

    quantize = (
        bool(quantize_cli) if quantize_cli is not None
        else bool(config.get("quantize", False))
    )
    return prune, quantize


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def stratified_subset(y: torch.Tensor, n: int, seed: int) -> torch.Tensor:
    """Deterministic stratified subset of indices into ``y`` (size ``n``)."""
    import numpy as np

    y_np = y.cpu().numpy()
    N = len(y_np)
    if n >= N:
        return torch.arange(N)

    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y_np, return_counts=True)
    quota = np.maximum(1, np.round(counts * n / N).astype(int))
    while quota.sum() > n:
        quota[np.argmax(quota)] -= 1
    while quota.sum() < n:
        quota[np.argmax(counts - quota)] += 1

    picked = []
    for c, q in zip(classes, quota):
        idx_c = np.where(y_np == c)[0]
        take = min(int(q), len(idx_c))
        picked.append(rng.choice(idx_c, size=take, replace=False))
    return torch.from_numpy(np.sort(np.concatenate(picked)))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def accuracy(model: nn.Module, X: torch.Tensor, y: torch.Tensor, batch: int = 256) -> float:
    model.eval()
    correct = 0
    for i in range(0, len(X), batch):
        pred = model(X[i:i + batch]).argmax(dim=1)
        correct += (pred == y[i:i + batch]).sum().item()
    return correct / len(X)


def estimate_serialized_size(model: nn.Module) -> int | None:
    """Byte size of the model's state_dict via torch.save (best effort)."""
    try:
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        return buf.tell()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Edge-compression ONNX export.")
    p.add_argument("--config", required=True, help="YAML config path")
    p.add_argument("--checkpoint", required=True, help="checkpoint .pt path")
    p.add_argument("--prune", type=float, default=None,
                   help="structured pruning amount (fraction to remove); "
                        "overrides the config 'prune' key")
    p.add_argument("--no-prune", action="store_true",
                   help="disable pruning even when the config sets 'prune'")
    p.add_argument("--quantize", action="store_true", default=None,
                   help="measure a dynamically-quantized copy (separate; not "
                        "exported); overrides the config 'quantize' key")
    p.add_argument("--no-quantize", action="store_false", dest="quantize",
                   help="disable quantization even when the config sets "
                        "'quantize'")
    p.add_argument("--out", required=True, help="output ONNX path")
    p.add_argument("--eval-subset", type=int, default=1000,
                   help="stratified test-subset size for accuracy (default 1000)")
    p.add_argument("--report", default=None, help="optional JSON report path")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    import yaml
    from data.adapter import load_windows
    from models.common import load_checkpoint, seed_everything
    from models.registry import build_model
    from models import compress as C

    # Dynamic LSTM quantization needs a quantized engine on CPU builds that
    # only ship qnnpack; set it before any quantization call.
    try:
        torch.backends.quantized.engine = "qnnpack"
    except Exception:
        pass

    if not os.path.isfile(args.config):
        _fail(f"config not found: {args.config}")
    if not os.path.isfile(args.checkpoint):
        _fail(f"checkpoint not found: {args.checkpoint}")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Config values are the defaults; explicit CLI flags win.
    prune, quantize = resolve_compression_options(
        config, args.prune, args.no_prune, args.quantize
    )
    if prune is not None and not 0 <= prune < 1:
        _fail(f"pruning amount must be in [0, 1), got {prune} "
              f"(from --prune or config 'prune')")

    seed = int(config.get("seed", 42))
    seed_everything(seed)                      # also fixes torch to 4 threads
    torch.set_num_threads(int(config.get("torch_threads", 4)))

    model_name = config.get("model")
    model_args = config.get("model_args", {}) or {}

    # --- data / shapes ---
    src = config.get("source", "har")
    data = load_windows(
        src, "test",
        window_size=int(config.get("window_size", 128)),
        stride=int(config.get("stride", 64)),
    )
    X_all, y_all = data.X, data.y_activity
    n_channels = int(X_all.shape[1])
    T = int(X_all.shape[2])
    n_classes = int(model_args["n_classes"])

    idx = stratified_subset(y_all, args.eval_subset, seed)
    X, y = X_all[idx], y_all[idx]
    eval_n = len(X)
    print(f"[data] source={src} test={tuple(X_all.shape)} "
          f"eval_subset={eval_n} channels={n_channels} classes={n_classes}")
    print(f"[options] prune={prune} quantize={bool(quantize)} "
          f"(config={args.config})")

    # --- build + load ---
    try:
        model = build_model(model_name, model_args, n_channels, n_classes)
    except Exception as e:
        _fail(f"could not build model {model_name!r}: {e}")

    try:
        model.load_state_dict(load_checkpoint(args.checkpoint))
    except Exception as e:
        _fail(f"could not load checkpoint {args.checkpoint!r}: {e}")
    model.eval()

    input_shape = (1, n_channels, T)

    # --- BEFORE compression ---
    rep_before = C.flops_report(model, input_shape)
    print(f"[before] params={rep_before['params']} flops={rep_before['flops']}")
    acc_before = accuracy(model, X, y)
    print(f"[before] accuracy={acc_before:.4f}  (n={eval_n})")

    params_before = rep_before["params"]
    flops_before = rep_before["flops"]

    # --- pruning (delegated to the shared, topology-checked pruner) ---
    export_model = model
    params_after = params_before
    flops_after = flops_before
    acc_after = acc_before
    param_reduction = 0.0
    if prune is not None:
        supported, reason = C.structured_prune_support(model)
        if not supported:
            _fail(
                f"structured pruning is not supported for model "
                f"{type(model).__name__!r} ({model_name!r}): {reason}. "
                f"Supported models: {sorted(C.STRUCTURED_PRUNE_SUPPORTED)}; "
                "re-export without the 'prune' option or prune with "
                "'unstructured' mode."
            )
        try:
            export_model = C.apply_pruning(model, prune, "structured")
        except Exception as e:
            _fail(f"structured pruning failed: {e}")
        rep_after = C.flops_report(export_model, input_shape)
        params_after = rep_after["params"]
        flops_after = rep_after["flops"]
        param_reduction = (params_before - params_after) / params_before
        print(f"[prune] amount={prune} params={params_after} "
              f"flops={flops_after} param_reduction={param_reduction:.4f}")
        acc_after = accuracy(export_model, X, y)
        delta = acc_after - acc_before
        print(f"[prune] accuracy={acc_after:.4f}  delta={delta:+.4f}")
        if abs(delta) > 0.05:
            print(f"[prune] NOTE: |accuracy_delta|={abs(delta):.4f} exceeds 0.05 "
                  f"(before={acc_before:.4f}, after={acc_after:.4f}) — reported plainly.")

    # --- export pruned FP32 (unquantized) model ---
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    try:
        # Use opset 18 for this deployment artifact. The shared wrapper selects
        # the legacy exporter explicitly, matching its dynamic_axes contract.
        C.export_onnx(export_model, args.out, input_shape, opset=18)
    except Exception as e:
        _fail(f"ONNX export to {args.out!r} failed: {e}")
    onnx_size = os.path.getsize(args.out)
    print(f"[export] wrote {args.out} ({onnx_size} bytes) — FP32 "
          f"{'pruned ' if prune is not None else ''}model (NOT quantized)")

    # --- optional dynamic quantization (measured separately) ---
    quantized_params = None
    quantized_bytes = None
    quant_module_params = None
    if quantize:
        src_q = copy.deepcopy(export_model)
        try:
            quant = C.dynamic_quantize(src_q)
        except Exception as e:
            _fail(f"dynamic quantization failed: {e}")
        quant.eval()
        quant_module_params = C.count_parameters(quant)
        # Dynamic quantization is a dtype-only transform (float32 -> int8): it
        # preserves every tensor shape, so the weight-ELEMENT count is unchanged
        # from the model it was derived from.
        quantized_params = C.count_parameters(export_model)
        quantized_bytes = estimate_serialized_size(quant)
        sz = f"{quantized_bytes} bytes" if quantized_bytes is not None else "n/a"
        print(f"[quantize] dynamically-quantized copy (int8 Linear/LSTM), "
              f"measured separately — NOT the artifact at --out")
        print(f"[quantize] quantized weight-elements (params)={quantized_params}  "
              f"count_parameters(module)={quant_module_params} "
              f"(packed int8 weights are not enumerated as trainable params)")
        print(f"[quantize] estimated serialized size={sz}  "
              f"(vs {onnx_size} bytes FP32 ONNX at --out)")

    # --- report ---
    if args.report:
        report = {
            "config": args.config,
            "checkpoint": args.checkpoint,
            "model": model_name,
            "eval_subset": eval_n,
            "prune_amount": prune,
            "quantize_measured": bool(quantize),
            "accuracy_before": round(acc_before, 6),
            "accuracy_after_prune": round(acc_after, 6),
            "accuracy_delta": round(acc_after - acc_before, 6),
            "params_before": params_before,
            "params_after_prune": params_after,
            "param_reduction": round(param_reduction, 6),
            "flops_before": flops_before,
            "flops_after_prune": flops_after,
            "quantized_params": quantized_params,
            "out": args.out,
        }
        rep_dir = os.path.dirname(args.report)
        if rep_dir:
            os.makedirs(rep_dir, exist_ok=True)
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[report] wrote {args.report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
