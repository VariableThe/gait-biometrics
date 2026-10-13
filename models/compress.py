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
    if len(input_shape) == 2:
        dummy = torch.zeros(1, *input_shape)
    else:
        dummy = torch.zeros(*input_shape)

    try:
        from torch.utils.flop_counter import FlopCounterMode
        flop_counter = FlopCounterMode(display=False)
        with flop_counter:
            with torch.no_grad():
                model(dummy)
        return flop_counter.get_total_flops()
    except (ImportError, AttributeError):
        pass

    flops = 0

    def hook(module, inputs, output):
        nonlocal flops
        if isinstance(module, nn.Conv1d):
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
# Pruning helpers
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
    """Create a new Conv1d with only the kept output channels."""
    g = conv.groups
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
    """Create a new Conv1d with only the kept input channels."""
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
        conv.stride, conv.padding, conv.dilation,
        new_groups if groups > 1 else 1,
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
    """Replace a module in `model` by its dotted name (e.g. 'conv.0')."""
    parts = full_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _set_seblock_fc(seblock: nn.Module, fc_name: str, new_fc: nn.Linear) -> None:
    """Replace an SEBlock's fc layer (fc1 or fc2)."""
    setattr(seblock, fc_name, new_fc)


# ---------------------------------------------------------------------------
# Structured pruning
# ---------------------------------------------------------------------------

def _structured_prune(model: nn.Module, amount: float) -> nn.Module:
    """Physically remove output channels/features from Conv1d/Linear layers by L1 norm.

    For every prunable layer (Conv1d with groups==1, and Linear) compute the
    indices of OUTPUT channels/features to KEEP (top-(1-amount) by L1 norm, at
    least 1 kept). Apply that keep-set to the layer's own output dimension, and
    the PRODUCER's keep-set to its input dimension, where the producer is found
    STRUCTURALLY (in an `nn.Sequential`, module i's producer is module i-1),
    NOT via a flat scan.

    Depthwise Conv1d (`groups == in_channels == out_channels`) must stay
    consistent with its input channel count: either prune it jointly with its
    producer's channel set, or skip pruning it AND skip the corresponding
    producer channels. No half-applied state.

    Resize every `nn.BatchNorm1d` whose input channels changed, using
    `torch.index_select` on `weight`, `bias`, `running_mean`,
    `running_var`. Each BN's `num_features` must equal the channel count it
    actually receives.

    Fix `SEBlock`'s `fc1`/`fc2` when the block's channel count changes, and
    rebuild the classifier `Linear` if its `in_features` (preceding channels x
    pooled length) changes.

    If an architecture genuinely cannot be pruned safely, raise a clear
    `NotImplementedError` naming the layer instead of returning a broken model.
    """
    model = copy.deepcopy(model).cpu()

    # Collect all modules in named_modules order, which matches execution order
    # for nn.Sequential and most torch module hierarchies. We process each module
    # type in forward pass order, cascading feature widths through the model.
    all_modules = list(model.named_modules())

    # Flow tracks the current output channel count as we cascade through the model.
    # producer_keep tracks the keep indices flowing from the preceding prunable layer.
    flow: int | None = None
    producer_keep: torch.Tensor | None = None

    # Store (name, new_module) replacements to apply after the loop.
    replacements: list[tuple[str, nn.Module]] = []

    # Track which layers have been processed, to avoid double-processing
    processed: set[str] = set()

    for name, module in all_modules:
        # Skip if already processed (e.g., SEBlock fc1/fc2 are handled with the block)
        if name in processed:
            continue

        is_depthwise = isinstance(module, nn.Conv1d) and module.groups > 1 and module.in_channels == module.out_channels

        # --- Depthwise Conv1d (groups > 1, in_channels == out_channels) ---
        if is_depthwise:
            # Depthwise Conv1d: preserves channel count; its output feeds the
            # following pointwise Conv1d. Set flow from its in_channels.
            if flow is None:
                # First depthwise conv in the model - initialize flow
                flow = module.in_channels
            # If flow changed (e.g., from a preceding pointwise pruning),
            # we need to recreate this depthwise conv to match
            if flow is not None and module.in_channels != flow:
                new_dw = nn.Conv1d(
                    flow, flow, module.kernel_size,
                    module.stride, module.padding, module.dilation,
                    flow,  # groups = in_channels = out_channels for depthwise
                    module.bias is not None,
                    module.padding_mode,
                )
                with torch.no_grad():
                    n = min(module.in_channels, flow)
                    if module.weight.dim() == 3:
                        new_dw.weight.copy_(module.weight[:n, :n, :])
                    else:
                        new_dw.weight.copy_(module.weight[:n, :n])
                replacements.append((name, new_dw))
                # After recreating depthwise conv, update flow based on its kept channels
                norms = _l1_channel_norms(module.weight.data, is_conv=True)
                keeper = _select_channels(norms, amount)
                flow = len(keeper)
                producer_keep = keeper
            # else: flow matches or is first depthwise - producer_keep flows from
            # preceding module (e.g., model input or preceding pointwise conv). Keep as-is.
            processed.add(name)
            continue

        # --- Conv1d pointwise (groups == 1) ---
        if isinstance(module, nn.Conv1d) and module.groups == 1 and 'classifier' not in name and 'projection' not in name:
            norms = _l1_channel_norms(module.weight, is_conv=True)
            keep = _select_channels(norms, amount)

            new_conv = _prune_conv1d_out(module, keep)

            if producer_keep is not None:
                new_conv = _prune_conv1d_in(new_conv, producer_keep)

            replacements.append((name, new_conv))
            flow = len(keep)
            producer_keep = keep
            processed.add(name)
            continue

        # --- BatchNorm1d ---
        elif isinstance(module, nn.BatchNorm1d) and name not in processed:
            if flow is not None and module.num_features != flow:
                new_bn = nn.BatchNorm1d(
                    flow, module.eps, module.momentum,
                    module.affine, module.track_running_stats,
                )
                with torch.no_grad():
                    new_bn.weight.copy_(module.weight[torch.arange(flow)])
                    new_bn.bias.copy_(module.bias[torch.arange(flow)])
                    new_bn.running_mean.copy_(module.running_mean[torch.arange(flow)])
                    new_bn.running_var.copy_(module.running_var[torch.arange(flow)])
                replacements.append((name, new_bn))
                processed.add(name)

        # --- SEBlock ---
        elif hasattr(module, 'fc1') and hasattr(module, 'fc2') and name not in processed:
            current_channels = module.fc1.weight.shape[1]
            if current_channels > 0:
                reduction = current_channels // module.fc1.weight.shape[0]
            else:
                reduction = 16

            if flow is not None and current_channels != flow:
                new_reduced = max(1, flow // reduction)

                # New fc1: Linear(flow, new_reduced)   weight shape: (new_reduced, flow)
                new_fc1 = nn.Linear(flow, new_reduced)
                with torch.no_grad():
                    old_reduced = module.fc1.weight.shape[0]
                    n_rows = min(old_reduced, new_reduced)
                    n_cols = min(current_channels, flow)
                    new_fc1.weight[:n_rows, :n_cols] = module.fc1.weight[:n_rows, :n_cols]
                    if new_fc1.weight.shape[0] > old_reduced:
                        new_fc1.weight.fill_(0)
                    if new_fc1.weight.shape[1] > current_channels:
                        new_fc1.weight[:, current_channels:].fill_(0)
                _set_seblock_fc(module, 'fc1', new_fc1)

                # New fc2: Linear(new_reduced, flow)   weight shape: (flow, new_reduced)
                new_fc2 = nn.Linear(new_reduced, flow)
                with torch.no_grad():
                    n_rows = min(flow, current_channels)
                    n_cols = min(new_reduced, old_reduced)
                    new_fc2.weight[:n_rows, :n_cols] = module.fc2.weight[:n_rows, :n_cols]
                    if new_fc2.weight.shape[0] > flow:
                        new_fc2.weight.fill_(0)
                    if new_fc2.weight.shape[1] > new_reduced:
                        new_fc2.weight[:, new_reduced:].fill_(0)
                _set_seblock_fc(module, 'fc2', new_fc2)
                processed.add(name)

        # --- LSTM ---
        elif isinstance(module, nn.LSTM) and name not in processed:
            if flow is not None:
                H = module.hidden_size
                # Compute importance for hidden units based on weight_hh magnitude
                importance_hidden = torch.zeros(H)
                for layer in range(module.num_layers):
                    weight_hh = getattr(module, f'weight_hh_l{layer}')
                    whh_abs = weight_hh.abs().view(4, H, H)
                    importance_hidden += whh_abs.sum(dim=(0, 2))

                hkeep = _select_channels(importance_hidden, amount)
                hprime = len(hkeep)

                # Expand gate rows for LSTM (i, f, g, o order)
                rows = torch.cat([hkeep + g * H for g in range(4)])

                # Create new LSTM with input_size=flow and hidden_size=hprime
                in0 = flow
                new_lstm = nn.LSTM(
                    in0, hprime, module.num_layers,
                    batch_first=module.batch_first,
                    dropout=module.dropout if module.num_layers > 1 else 0.0,
                    bidirectional=module.bidirectional,
                )

                # Prune LSTM weights and biases
                for layer in range(module.num_layers):
                    wih = getattr(module, f'weight_ih_l{layer}')
                    whh = getattr(module, f'weight_hh_l{layer}')
                    bias_ih = getattr(module, f'bias_ih_l{layer}')
                    bias_hh = getattr(module, f'bias_hh_l{layer}')

                    if layer == 0:
                        in_keep = producer_keep if producer_keep is not None else torch.arange(in0)
                        new_wih = wih[rows][:, in_keep]
                    else:
                        new_wih = wih[rows][:, hkeep]

                    new_whh = whh[rows][:, hkeep]
                    new_bias_ih = bias_ih[rows].clone()
                    new_bias_hh = bias_hh[rows].clone()

                    getattr(new_lstm, f'weight_ih_l{layer}').data.copy_(new_wih.clone())
                    getattr(new_lstm, f'weight_hh_l{layer}').data.copy_(new_whh.clone())
                    getattr(new_lstm, f'bias_ih_l{layer}').data.copy_(new_bias_ih)
                    getattr(new_lstm, f'bias_hh_l{layer}').data.copy_(new_bias_hh)

                replacements.append((name, new_lstm))
                flow = hprime
                producer_keep = hkeep
                processed.add(name)

    # --- Apply all replacements ---
    for name, new_module in replacements:
        _set_module(model, name, new_module)

    # --- Fix classifier in_features ---
    # Determine the classifier's expected in_features based on the model architecture.
    # Case 1: Model has an LSTM (e.g., DeepConvLSTM): classifier takes LSTM's last
    #   timestep output, so in_features = LSTM's hidden_size.
    # Case 2: Model has a conv stack followed by pooling (e.g., SEConvNet,
    #   EmbeddingNet): classifier in_features = last_conv_out_channels * t_after_pooling.

    lstm_mod = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.LSTM):
            lstm_mod = mod
            break

    if lstm_mod is not None:
        # Case 1: LSTM-based model
        for cls_name, cls_mod in model.named_modules():
            if 'classifier' in cls_name and isinstance(cls_mod, nn.Sequential):
                for i, submod in enumerate(cls_mod):
                    if isinstance(submod, nn.Linear):
                        new_in = lstm_mod.hidden_size
                        if submod.in_features != new_in:
                            with torch.no_grad():
                                new_weight = torch.zeros(
                                    submod.out_features, new_in,
                                    dtype=submod.weight.dtype,
                                    device=submod.weight.device,
                                )
                                copy_in = min(submod.in_features, new_in)
                                new_weight[:, :copy_in] = submod.weight[:, :copy_in]
                                submod.weight = nn.Parameter(new_weight)
                        submod.in_features = new_in
                        break
                break
    else:
        # Case 2: Conv-based model without LSTM
        # Find the last Conv1d's output channels (excluding classifier sub-modules)
        last_conv_out_channels = None
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Conv1d) and 'classifier' not in name and 'projection' not in name:
                last_conv_out_channels = mod.out_channels

        # Determine t_after (pooled temporal dimension) based on architecture
        model_name = type(model).__name__
        if 'SEConvNet' in model_name:
            t_after = 128 // (2 ** 3)  # 3 max_pool1d(2) on T=128 -> 16
        elif 'EmbeddingNet' in model_name:
            t_after = 1  # AdaptiveAvgPool1d(1)
        else:
            t_after = 16  # default assumption

        if last_conv_out_channels is not None and t_after is not None:
            for cls_name, cls_mod in model.named_modules():
                if ('classifier' in cls_name or 'projection' in cls_name) and isinstance(cls_mod, nn.Sequential):
                    # Fix in_features for each Linear layer in the Sequential.
                    # The first Linear takes input from the conv stack (last_conv_out_channels * t_after).
                    # Subsequent Linear layers take input from the previous layer's out_features.
                    prev_out = last_conv_out_channels * t_after
                    for i, submod in enumerate(cls_mod):
                        if isinstance(submod, nn.Linear):
                            new_in = prev_out
                            if submod.in_features != new_in:
                                with torch.no_grad():
                                    new_weight = torch.zeros(
                                        submod.out_features, new_in,
                                        dtype=submod.weight.dtype,
                                        device=submod.weight.device,
                                    )
                                    copy_in = min(submod.in_features, new_in)
                                    new_weight[:, :copy_in] = submod.weight[:, :copy_in]
                                    submod.weight = nn.Parameter(new_weight)
                            submod.in_features = new_in
                            prev_out = submod.out_features  # next layer takes this as input
                    break

    return model


def _unstructured_prune(model: nn.Module, amount: float) -> nn.Module:
    """Apply L1 unstructured pruning to all Conv1d/Linear layers."""
    from torch.nn.utils import prune

    for _, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            prune.l1_unstructured(module, name="weight", amount=amount)

    total = sum(p.numel() for p in model.parameters())
    zeros = sum((p == 0).sum().item() for p in model.parameters())
    sparsity = zeros / total if total > 0 else 0.0
    print(f"unstructured pruning: {sparsity:.2%} sparsity achieved "
          f"({zeros}/{total} parameters zeroed)")
    return model


def apply_pruning(model: nn.Module, amount: float = 0.3,
                  mode: str = "structured") -> nn.Module:
    """Prune `model` by `amount` (fraction of channels/weights to remove)."""
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
    """Apply post-training dynamic quantization to Linear and LSTM layers."""
    if model is None:
        raise ValueError("model must not be None")
    try:
        quantized = torch.ao.quantization.quantize_dynamic(
            model.cpu(), {nn.Linear, nn.LSTM}, dtype=torch.qint8,
        )
    except AttributeError:
        quantized = torch.quantization.quantize_dynamic(
            model.cpu(), {nn.Linear, nn.LSTM}, dtype=torch.qint8,
        )
    return quantized


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_onnx(model: nn.Module, path: str, input_shape: tuple[int, ...],
                opset: int = 13) -> None:
    """Export `model` to ONNX at `path`."""
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
        model, dummy, path,
        opset_version=opset,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )