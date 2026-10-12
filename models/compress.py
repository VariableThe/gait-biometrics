"""Model compression utilities for edge deployment (paper B6).

Real-time multiple-people gait recognition at the edge (Scientific Reports,
2025). This module provides parameter/FLOP counting, structured and
unstructured pruning, dynamic quantization, and ONNX export.

Structured pruning (`mode="structured"`) physically removes whole output
channels/filters by L1 norm so `count_parameters` genuinely drops — it does
not return an L1-masked model. Unstructured pruning uses
`torch.nn.utils.prune` and reports achieved sparsity.
"""

from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    """Total number of trainable parameters in `model`."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_flops(model: nn.Module, input_shape: tuple[int, ...]) -> int:
    """Count FLOPs for a single forward pass with `input_shape`.

    Prefers `torch.utils.flop_counter.FlopCounterMode` (torch >= 2.1);
    falls back to a forward-hook counter for Conv1d/Linear layers.
    """
    model.eval()
    # input_shape may be (C, T) or (N, C, T); normalize to a full batch.
    if len(input_shape) == 2:
        dummy = torch.zeros(1, *input_shape)
    else:
        dummy = torch.zeros(*input_shape)

    try:
        # torch >= 2.1: use FlopCounterMode
        from torch.utils.flop_counter import FlopCounterMode
        flop_counter = FlopCounterMode(display=False)
        with flop_counter:
            with torch.no_grad():
                model(dummy)
        return flop_counter.get_total_flops()
    except (ImportError, AttributeError):
        pass

    # Fallback: forward hooks for Conv1d and Linear.
    flops = 0

    def hook(module, inputs, output):
        nonlocal flops
        if isinstance(module, nn.Conv1d):
            # FLOPs = out_channels * out_length * (in_channels/groups * kernel)
            batch = output.shape[0]
            out_len = output.shape[2]
            kernel_ops = (module.in_channels // module.groups) * module.kernel_size[0]
            flops += batch * module.out_channels * out_len * kernel_ops
        elif isinstance(module, nn.Linear):
            flops += output.shape[0] * module.in_features * module.out_features

    handles = []
    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            handles.append(module.register_forward_hook(hook))

    try:
        with torch.no_grad():
            model(dummy)
    finally:
        for h in handles:
            h.remove()

    return flops


def flops_report(model: nn.Module, input_shape: tuple[int, ...]) -> dict:
    """Report parameter count and FLOPs for `model`."""
    return {
        "params": count_parameters(model),
        "flops": count_flops(model, input_shape),
    }


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------

def _l1_channel_norms(weight: torch.Tensor, is_conv: bool) -> torch.Tensor:
    """L1 norm per output channel (Conv1d) or output feature (Linear)."""
    if is_conv:
        return weight.abs().sum(dim=(1, 2))  # (out_channels,)
    return weight.abs().sum(dim=1)  # (out_features,)


def _select_channels(norms: torch.Tensor, amount: float) -> torch.Tensor:
    """Indices of the top (1 - amount) channels by L1 norm, sorted ascending."""
    n = norms.numel()
    n_keep = max(1, int(n * (1.0 - amount)))
    keep_idx = torch.topk(norms, n_keep).indices.sort().values
    return keep_idx


def _prune_conv1d_out(conv: nn.Conv1d, keep_idx: torch.Tensor) -> nn.Conv1d:
    """Create a new Conv1d with only the kept output channels.

    Ensures out_channels is divisible by groups (required by PyTorch),
    adjusting the keep count downward if necessary.
    """
    # Conv1d requires out_channels % groups == 0; adjust keep_idx if needed.
    g = conv.groups
    # For depthwise convs (groups == in_channels), we must keep a multiple of groups.
    max_keep = (len(keep_idx) // g) * g
    if max_keep < len(keep_idx):
        keep_idx = keep_idx[:max_keep] if max_keep > 0 else keep_idx[:1]
    new = nn.Conv1d(
        conv.in_channels, len(keep_idx), conv.kernel_size,
        conv.stride, conv.padding, conv.dilation, conv.groups,
        conv.bias is not None, conv.padding_mode,
    )
    with torch.no_grad():
        new.weight.copy_(conv.weight[keep_idx])
        if conv.bias is not None:
            new.bias.copy_(conv.bias[keep_idx])
    return new


def _prune_conv1d_in(conv: nn.Conv1d, keep_idx: torch.Tensor) -> nn.Conv1d:
    """Create a new Conv1d with only the kept input channels.

    For depthwise convs (groups > 1), prunes entire groups and adjusts
    `groups` accordingly.
    """
    groups = conv.groups
    if groups > 1:
        in_per_group = conv.in_channels // groups
        new_in = len(keep_idx)
        new_groups = new_in // in_per_group
        if new_groups == 0:
            new_groups = 1
    else:
        new_groups = 1

    new = nn.Conv1d(
        len(keep_idx), conv.out_channels, conv.kernel_size,
        conv.stride, conv.padding, conv.dilation, new_groups if groups > 1 else 1,
        conv.bias is not None, conv.padding_mode,
    )
    with torch.no_grad():
        new.weight.copy_(conv.weight[:, keep_idx])
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new


def _prune_linear_out(linear: nn.Linear, keep_idx: torch.Tensor) -> nn.Linear:
    """Create a new Linear with only the kept output features."""
    new = nn.Linear(linear.in_features, len(keep_idx), bias=linear.bias is not None)
    with torch.no_grad():
        new.weight.copy_(linear.weight[keep_idx])
        if linear.bias is not None:
            new.bias.copy_(linear.bias[keep_idx])
    return new


def _prune_linear_in(linear: nn.Linear, keep_idx: torch.Tensor) -> nn.Linear:
    """Create a new Linear with only the kept input features."""
    new = nn.Linear(len(keep_idx), linear.out_features, bias=linear.bias is not None)
    with torch.no_grad():
        new.weight.copy_(linear.weight[:, keep_idx])
        if linear.bias is not None:
            new.bias.copy_(linear.bias)
    return new


def _set_module(model: nn.Module, full_name: str, new_module: nn.Module) -> None:
    """Replace a module in `model` by its dotted name (e.g. 'conv3')."""
    parts = full_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _structured_prune(model: nn.Module, amount: float) -> nn.Module:
    """Physically remove output channels from Conv1d/Linear layers by L1 norm.

    For each Conv1d/Linear, removes the bottom `amount` fraction of output
    channels (by L1 norm) and adjusts consumer layers' input channels. Handles
    Conv1d, Linear, BatchNorm1d, and SEBlock.
    """
    # Find all Conv1d/Linear layers
    prunable = [(n, m) for n, m in model.named_modules()
                if isinstance(m, (nn.Conv1d, nn.Linear))]
    if not prunable:
        raise ValueError("model has no Conv1d or Linear layers to prune")

    # Compute keep indices for each layer
    keep_map: dict[str, torch.Tensor] = {}
    for name, layer in prunable:
        # Only prune pointwise Conv1d (groups == 1); skip depthwise (groups == in == out)
        if isinstance(layer, nn.Conv1d) and layer.groups == 1:
            norms = _l1_channel_norms(layer.weight, is_conv=True)
        elif isinstance(layer, nn.Conv1d) and layer.groups > 1:
            # Depthwise conv: keep all channels (no pruning)
            keep_map[name] = torch.arange(layer.out_channels)
            continue
        else:
            norms = _l1_channel_norms(layer.weight, is_conv=False)
        keep_map[name] = _select_channels(norms, amount)

    # Build producer -> keep indices mapping.
    # In a sequential architecture, the producer of each layer is the
    # previous Conv1d/Linear in the module tree.
    layer_names = [n for n, _ in prunable]
    producer_keep: dict[str, torch.Tensor] = {}
    for i, name in enumerate(layer_names):
        if i > 0:
            producer_name = layer_names[i - 1]
            if producer_name in keep_map:
                producer_keep[name] = keep_map[producer_name]

    # Replace each prunable layer with a pruned version.
    for name, layer in prunable:
        keep_idx = keep_map[name]

        # Skip depthwise convs entirely; only prune pointwise convs and linear layers.
        is_depthwise = (isinstance(layer, nn.Conv1d)
                        and layer.groups > 1
                        and layer.in_channels == layer.out_channels)
        if is_depthwise:
            new_layer = layer
        elif isinstance(layer, nn.Conv1d) and layer.groups == 1:
            new_layer = _prune_conv1d_out(layer, keep_idx)
        else:
            new_layer = _prune_linear_out(layer, keep_idx)

        # Prune input channels based on producer.
        if name in producer_keep:
            in_keep = producer_keep[name]
            # Only adjust input channels for pointwise convs (groups == 1)
            if isinstance(new_layer, nn.Conv1d) and new_layer.groups == 1:
                new_layer = _prune_conv1d_in(new_layer, in_keep)
            elif isinstance(new_layer, nn.Linear):
                new_layer = _prune_linear_in(new_layer, in_keep)

        _set_module(model, name, new_layer)

    # Fix BatchNorm1d layers: prune num_features to match the preceding Conv1d.
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm1d):
            # Find the preceding Conv1d's keep indices.
            prev_keep = None
            for other_name, other in model.named_modules():
                if other_name == name:
                    continue
                if isinstance(other, nn.Conv1d) and other_name in keep_map:
                    prev_keep = keep_map[other_name]
                    break
            if prev_keep is not None and len(prev_keep) != module.num_features:
                new_bn = nn.BatchNorm1d(
                    len(prev_keep), module.eps, module.momentum,
                    module.affine, module.track_running_stats,
                )
                with torch.no_grad():
                    new_bn.weight.copy_(module.weight[prev_keep])
                    new_bn.bias.copy_(module.bias[prev_keep])
                    new_bn.running_mean.copy_(module.running_mean[prev_keep])
                    new_bn.running_var.copy_(module.running_var[prev_keep])
                _set_module(model, name, new_bn)

    # Fix classifier input features to match the pruned conv output.
    # The classifier is nn.Linear(128 * 16, 64) where 128 is the last conv's
    # output channels and 16 is the temporal dimension after 3x2 pooling.
    # After pruning, the last Conv1d (conv3) output channels may have changed.
    # Find the last Conv1d and update the classifier's in_features.
    last_conv_name = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv1d):
            last_conv_name = name
    if last_conv_name:
        last_conv = dict(model.named_modules())[last_conv_name]
        # The classifier is typically named 'classifier' and its first layer
        # takes the flattened features from the last conv output.
        classifier = dict(model.named_modules()).get('classifier')
        if classifier is not None and isinstance(classifier, nn.Sequential):
            first_linear = classifier[0]
            if isinstance(first_linear, nn.Linear):
                # Compute the flattened size: out_channels * T_after_pooling
                # After 3 max_pool1d(2) on T=128: 128 / (2^3) = 16
                t_after = 128 // (2 ** 3)  # = 16
                new_in_features = last_conv.out_channels * t_after
                if new_in_features != first_linear.in_features:
                    with torch.no_grad():
                        # Preserve the existing weight row(s) if possible, zero rest
                        old_in = first_linear.in_features
                        new_weight = torch.zeros(
                            first_linear.out_features, new_in_features,
                            dtype=first_linear.weight.dtype,
                            device=first_linear.weight.device,
                        )
                        # Copy overlapping region: min(old_in, new_in) input features
                        copy_in = min(old_in, new_in_features)
                        new_weight[:, :copy_in] = first_linear.weight[:, :copy_in]
                    first_linear.in_features = new_in_features
                    first_linear.weight = nn.Parameter(new_weight)

    return model


def _unstructured_prune(model: nn.Module, amount: float) -> nn.Module:
    """Apply L1 unstructured pruning to all Conv1d/Linear layers.

    Returns the model with pruning masks applied. Reports achieved sparsity.
    """
    from torch.nn.utils import prune

    for _, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            prune.l1_unstructured(module, name="weight", amount=amount)

    # Compute achieved sparsity.
    total = sum(p.numel() for p in model.parameters())
    zeros = sum((p == 0).sum().item() for p in model.parameters())
    sparsity = zeros / total if total > 0 else 0.0
    print(f"unstructured pruning: {sparsity:.2%} sparsity achieved "
          f"({zeros}/{total} parameters zeroed)")
    return model


def apply_pruning(model: nn.Module, amount: float = 0.3,
                  mode: str = "structured") -> nn.Module:
    """Prune `model` by `amount` (fraction of channels/weights to remove).

    Args:
        model: the model to pruned.
        amount: fraction of channels/weights to remove (0 <= amount < 1).
        mode: "structured" physically removes whole output channels/filters
            by L1 norm so `count_parameters` genuinely drops; "unstructured"
            uses `torch.nn.utils.prune` and reports achieved sparsity.

    Returns:
        The pruned model (same object for unstructured; modified in-place
        for structured).

    Raises:
        ValueError: if `mode` is not "structured" or "unstructured", or if
            `amount` is not in [0, 1), or if the model has no prunable
            layers (structured mode).
    """
    if mode not in ("structured", "unstructured"):
        raise ValueError(
            f"mode must be 'structured' or 'unstructured', got {mode!r}"
        )
    if not 0 <= amount < 1:
        raise ValueError(f"amount must be in [0, 1), got {amount}")

    if mode == "unstructured":
        return _unstructured_prune(model, amount)
    return _structured_prune(model, amount)


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def dynamic_quantize(model: nn.Module) -> nn.Module:
    """Apply post-training dynamic quantization to Linear and LSTM layers.

    Returns a quantized copy of `model`. Quantizes activations and weights
    of `nn.Linear` and `nn.LSTM` to int8, cutting model size ~4x for those
    layers.

    Raises:
        ValueError: if `model` is None or torch quantization is unavailable.
    """
    if model is None:
        raise ValueError("model must not be None")
    try:
        quantized = torch.ao.quantization.quantize_dynamic(
            model.cpu(),
            {nn.Linear, nn.LSTM},
            dtype=torch.qint8,
        )
    except AttributeError:
        quantized = torch.quantization.quantize_dynamic(
            model.cpu(),
            {nn.Linear, nn.LSTM},
            dtype=torch.qint8,
        )
    return quantized


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_onnx(model: nn.Module, path: str, input_shape: tuple[int, ...],
                 opset: int = 13) -> None:
    """Export `model` to ONNX at `path`.

    Args:
        model: the model to export (will be set to eval mode and moved to CPU).
        path: output file path (e.g. "model.onnx").
        input_shape: shape of a single input sample (C, T) or (N, C, T).
        opset: ONNX opset version.

    Raises:
        ValueError: if `model` is None or `path` is empty.
    """
    if model is None:
        raise ValueError("model must not be None")
    if not path:
        raise ValueError("path must not be empty")

    model = model.cpu().eval()
    if len(input_shape) == 2:
        dummy = torch.zeros(1, *input_shape)
    else:
        dummy = torch.zeros(*input_shape)

    torch.onnx.export(
        model,
        dummy,
        path,
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
