#!/usr/bin/env python3
"""Edge-compression export entry point (paper B6, Student B track).

Loads a trained classification checkpoint, measures it on a deterministic
stratified slice of the test split, optionally applies *structured* pruning,
exports the pruned FP32 model to ONNX, and (optionally) reports a dynamically
quantized copy measured separately.

Run from the repository root:
    python -m scripts.export_onnx --config configs/lightweight.yaml \
        --checkpoint runs/smoke_classification/cnn_lstm/best.pt \
        [--prune 0.3] [--quantize] --out model.onnx \
        [--eval-subset 1000] [--report report.json]

Notes on fidelity:
  * The exported ONNX file is the *pruned FP32* model. When ``--quantize`` is
    passed, a *dynamically quantized copy* is measured (params + serialized
    size) only as a separate estimate; it is NOT the artifact written to
    ``--out`` and the ``--out`` file is never claimed to be quantized.
  * ``models.compress.apply_pruning`` (frozen shared code) prunes Conv1d/Linear
    output channels but does NOT adjust an ``nn.LSTM`` that sits between the
    conv stack and the classifier, nor the final classifier's output width. On
    ``DeepConvLSTM`` (``cnn_lstm``) that leaves the LSTM input mismatched and
    reduces the classifier logits below ``n_classes``, so the pruned model
    cannot even run a forward pass and the reduction is capped ~12% because
    the LSTM holds ~75% of the parameters. Because ``compress.py`` is frozen,
    this script implements a correct, weight-preserving structured pruner
    (Conv1d channels + BatchNorm + LSTM hidden units, cascading, preserving
    ``n_classes``) in place, and reuses ``models.compress`` for parameter/FLOP
    counting, dynamic quantization, and ONNX export.
"""

from __future__ import annotations

import argparse
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


# ---------------------------------------------------------------------------
# Data / checkpoint loading
# ---------------------------------------------------------------------------

def load_checkpoint_state(ckpt_obj: object) -> dict:
    """Accept either a rich checkpoint dict or a bare ``state_dict``."""
    if isinstance(ckpt_obj, dict) and "model_state_dict" in ckpt_obj:
        return ckpt_obj["model_state_dict"]
    if isinstance(ckpt_obj, dict):
        return ckpt_obj
    raise ValueError("checkpoint is not a dict of model weights")


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
# Structured pruning (weight-preserving; handles Conv1d/BN/LSTM/Linear)
# ---------------------------------------------------------------------------

def _select_channels(norms: torch.Tensor, amount: float) -> torch.Tensor:
    """Indices of the top ``(1 - amount)`` channels by L1 norm, sorted."""
    n = norms.numel()
    keep = max(1, int(round(n * (1.0 - amount))))
    keep = min(keep, n)
    return torch.topk(norms, keep).indices.sort().values


def _expand_gate_rows(keep: torch.Tensor, hidden: int) -> torch.Tensor:
    """Expand hidden-unit indices to LSTM gate-row order (i, f, g, o)."""
    return torch.cat([keep + g * hidden for g in range(4)])


def _set_module(model: nn.Module, name: str, new_module: nn.Module) -> None:
    parent = model
    parts = name.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def structured_prune(model: nn.Module, amount: float, n_classes: int) -> nn.Module:
    """Physically remove ``amount`` of channels/filters/hidden-units.

    Cascades input widths through Conv1d -> BatchNorm1d -> LSTM -> Linear in
    execution (registration) order and preserves the final classifier output
    (``n_classes``). Learned weights are sliced (L1 top-k), not re-initialised,
    so accuracy is retained as well as the structure allows.
    """
    model = copy_deepcopy(model).cpu()

    replacements: list[tuple[str, nn.Module]] = []
    flow: int | None = None            # current feature width between layers
    producer_keep: torch.Tensor | None = None  # kept feature idx flowing in

    ordered = list(model.named_modules())[1:]  # skip the root module
    prunables = [
        (n, m) for n, m in ordered
        if isinstance(m, (nn.Conv1d, nn.BatchNorm1d, nn.LSTM, nn.Linear))
    ]

    for name, m in prunables:
        in_slice = slice(None) if producer_keep is None else producer_keep

        if isinstance(m, nn.Conv1d):
            keep = _select_channels(m.weight.abs().sum(dim=(1, 2)), amount)
            new_w = m.weight[keep][:, in_slice]
            in_ch = new_w.shape[1]
            new = nn.Conv1d(
                in_ch, len(keep), m.kernel_size, m.stride, m.padding,
                m.dilation, m.groups, m.bias is not None, m.padding_mode,
            )
            with torch.no_grad():
                new.weight.copy_(new_w)
                if m.bias is not None:
                    new.bias.copy_(m.bias[keep])
            replacements.append((name, new))
            flow, producer_keep = len(keep), keep

        elif isinstance(m, nn.BatchNorm1d):
            keep = producer_keep if producer_keep is not None else torch.arange(m.num_features)
            new = nn.BatchNorm1d(
                len(keep), m.eps, m.momentum, m.affine, m.track_running_stats
            )
            with torch.no_grad():
                new.weight.copy_(m.weight[keep])
                new.bias.copy_(m.bias[keep])
                new.running_mean.copy_(m.running_mean[keep])
                new.running_var.copy_(m.running_var[keep])
            replacements.append((name, new))
            flow = len(keep)

        elif isinstance(m, nn.LSTM):
            H = m.hidden_size
            importance = torch.zeros(H)
            for layer in range(m.num_layers):
                importance += getattr(
                    m, f"weight_hh_l{layer}"
                ).view(4, H, H).abs().sum(dim=(0, 2))
                importance += getattr(
                    m, f"weight_ih_l{layer}"
                ).view(4, H, -1).abs().sum(dim=(0, 2))
            hkeep = _select_channels(importance, amount)
            hprime = len(hkeep)
            rows = _expand_gate_rows(hkeep, H)
            in0 = flow if flow is not None else m.input_size

            new_lstm = nn.LSTM(
                in0, hprime, m.num_layers, batch_first=m.batch_first,
                dropout=m.dropout, bidirectional=m.bidirectional,
            )
            in_keep_l0 = producer_keep if producer_keep is not None else torch.arange(in0)
            for layer in range(m.num_layers):
                wih = getattr(m, f"weight_ih_l{layer}")
                whh = getattr(m, f"weight_hh_l{layer}")
                if layer == 0:
                    new_wih = wih[rows][:, in_keep_l0]
                else:
                    new_wih = wih[rows][:, hkeep]
                getattr(new_lstm, f"weight_ih_l{layer}").data.copy_(new_wih.clone())
                getattr(new_lstm, f"weight_hh_l{layer}").data.copy_(
                    whh[rows][:, hkeep].clone()
                )
                getattr(new_lstm, f"bias_ih_l{layer}").data.copy_(
                    getattr(m, f"bias_ih_l{layer}")[rows].clone()
                )
                getattr(new_lstm, f"bias_hh_l{layer}").data.copy_(
                    getattr(m, f"bias_hh_l{layer}")[rows].clone()
                )
            replacements.append((name, new_lstm))
            flow, producer_keep = hprime, hkeep

        elif isinstance(m, nn.Linear):
            # Preserve Linear widths: structured pruning here targets the conv
            # filters and LSTM units (the parameter-dominant compute layers);
            # the dense head is kept compact, only its input width is re-aligned
            # to the pruned feature dimension.
            new_w = m.weight[:, in_slice]
            new = nn.Linear(new_w.shape[1], m.out_features, bias=m.bias is not None)
            with torch.no_grad():
                new.weight.copy_(new_w)
                if m.bias is not None:
                    new.bias.copy_(m.bias)
            replacements.append((name, new))
            flow, producer_keep = m.out_features, torch.arange(m.out_features)

    for name, new_module in replacements:
        _set_module(model, name, new_module)
    return model


def copy_deepcopy(model: nn.Module) -> nn.Module:
    import copy
    return copy.deepcopy(model)


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
                   help="structured pruning amount (fraction to remove)")
    p.add_argument("--quantize", action="store_true",
                   help="measure a dynamically-quantized copy (separate; not exported)")
    p.add_argument("--out", required=True, help="output ONNX path")
    p.add_argument("--eval-subset", type=int, default=1000,
                   help="stratified test-subset size for accuracy (default 1000)")
    p.add_argument("--report", default=None, help="optional JSON report path")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    import yaml
    from data.adapter import load_windows
    from models.common import seed_everything
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

    # --- build + load ---
    try:
        model = build_model(model_name, model_args, n_channels, n_classes)
    except Exception as e:
        _fail(f"could not build model {model_name!r}: {e}")

    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        state = load_checkpoint_state(ckpt)
        model.load_state_dict(state)
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

    # --- pruning ---
    export_model = model
    params_after = params_before
    flops_after = flops_before
    acc_after = acc_before
    if args.prune is not None:
        if not 0 <= args.prune < 1:
            _fail(f"--prune must be in [0, 1), got {args.prune}")
        try:
            export_model = structured_prune(model, args.prune, n_classes)
        except Exception as e:
            _fail(f"structured pruning failed: {e}")
        rep_after = C.flops_report(export_model, input_shape)
        params_after = rep_after["params"]
        flops_after = rep_after["flops"]
        param_reduction = (params_before - params_after) / params_before
        print(f"[prune] amount={args.prune} params={params_after} "
              f"flops={flops_after} param_reduction={param_reduction:.4f}")
        acc_after = accuracy(export_model, X, y)
        delta = acc_after - acc_before
        print(f"[prune] accuracy={acc_after:.4f}  delta={delta:+.4f}")
        if abs(delta) > 0.05:
            print(f"[prune] NOTE: |accuracy_delta|={abs(delta):.4f} exceeds 0.05 "
                  f"(before={acc_before:.4f}, after={acc_after:.4f}) — reported plainly.")
    else:
        param_reduction = 0.0

    # --- export pruned FP32 (unquantized) model ---
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    try:
        # torch 2.14's ONNX exporter is opset-18 native and cannot downgrade to
        # 13; requesting 18 yields a clean, checker-valid graph. compress's
        # default (13) is overridden here for that reason only.
        C.export_onnx(export_model, args.out, input_shape, opset=18)
    except Exception as e:
        _fail(f"ONNX export to {args.out!r} failed: {e}")
    onnx_size = os.path.getsize(args.out)
    print(f"[export] wrote {args.out} ({onnx_size} bytes) — FP32 "
          f"{'pruned ' if args.prune is not None else ''}model (NOT quantized)")

    # --- optional dynamic quantization (measured separately) ---
    quantized_params = None
    quantized_bytes = None
    quant_module_params = None
    if args.quantize:
        src_q = copy_deepcopy(export_model)
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
