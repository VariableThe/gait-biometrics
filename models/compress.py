"""Model compression utilities for edge deployment (paper B6).

Real-time multiple-people gait recognition at the edge (Scientific Reports,
2025). This module provides parameter/FLOP counting, structured and
unstructured pruning, dynamic quantization, and ONNX export.

Structured pruning (`mode="structured"`) physically removes whole output
channels/filters by L1 norm so `count_parameters` genuinely drops — it does
not return an L1-masked model. Unstructured pruning uses
`torch.nn.utils.prune` and reports achieved sparsity.

Structured pruning is topology-specific. It is implemented for exactly one
topology — the ``DeepConvLSTM`` chain (B3, ``models/cnn_lstm.py``, used by
``configs/lightweight.yaml``)::

    Conv1d -> BatchNorm1d -> ... -> nn.LSTM -> Linear head

One channel keep-set is propagated along that chain and every consumer is
sliced by the producer's *exact* indices: conv input columns, BatchNorm
``weight``/``bias``/``running_mean``/``running_var``, LSTM gate rows and
``weight_hh``/``weight_ih`` columns and both biases, and the head Linear's input
columns. See `_structured_prune` for the rules and
``tests/test_compress_deepconvlstm_alignment.py`` for the index-exact and
mask-equivalent regression tests.

Every other Student B topology raises `NotImplementedError` before the model is
copied or mutated (see `structured_prune_support`). ``SEConvNet`` (B5) and
``EmbeddingNet`` (B4) were previously accepted by a generic cascade that sliced
downstream tensors by *prefix* rather than by the producer's channel indices,
which silently attached affine/running statistics and classifier columns to the
wrong channels. Until each gets topology-specific slicing (SE bottleneck,
depthwise reconstruction, projection head) with exact-index tests, they are
fail-closed like ``MultiSensorFusionCNN`` (B1), ``MultiTaskNet`` (B2) and
``GaitViT`` (B7). Use ``mode="unstructured"`` for those.
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


def _conv1d_select(conv: nn.Conv1d, keep_out: torch.Tensor,
                   keep_in: torch.Tensor | None) -> nn.Conv1d:
    """New Conv1d keeping output channels `keep_out` and input channels `keep_in`.

    `keep_out`/`keep_in` are explicit index tensors into the original weight, so
    the retained filters and input columns are exactly the selected ones (never a
    prefix). Only `groups == 1` convolutions exist here (preflight-enforced).
    """
    in_ch = conv.in_channels if keep_in is None else int(keep_in.numel())
    new = nn.Conv1d(
        in_ch, int(keep_out.numel()), conv.kernel_size, conv.stride,
        conv.padding, conv.dilation, conv.groups,
        conv.bias is not None, conv.padding_mode,
    )
    with torch.no_grad():
        weight = conv.weight[keep_out]
        if keep_in is not None:
            weight = weight[:, keep_in]
        new.weight.copy_(weight)
        if conv.bias is not None:
            new.bias.copy_(conv.bias[keep_out])
    return new


def _batchnorm1d_select(bn: nn.BatchNorm1d, keep: torch.Tensor) -> nn.BatchNorm1d:
    """New BatchNorm1d over exactly the channels in `keep`.

    `weight`, `bias`, `running_mean` and `running_var` are all indexed by `keep`,
    so each statistic stays attached to the filter it was learned for. Indexing
    by a prefix (`arange`) instead would silently mis-pair affine values and
    running statistics with the wrong channels.
    """
    new = nn.BatchNorm1d(
        int(keep.numel()), bn.eps, bn.momentum, bn.affine, bn.track_running_stats
    )
    with torch.no_grad():
        if bn.affine:
            new.weight.copy_(bn.weight[keep])
            new.bias.copy_(bn.bias[keep])
        if bn.track_running_stats:
            new.running_mean.copy_(bn.running_mean[keep])
            new.running_var.copy_(bn.running_var[keep])
    return new


def _linear_select_in(linear: nn.Linear, keep: torch.Tensor) -> nn.Linear:
    """New Linear taking exactly the input features in `keep`."""
    new = nn.Linear(int(keep.numel()), linear.out_features,
                    bias=linear.bias is not None)
    with torch.no_grad():
        new.weight.copy_(linear.weight[:, keep])
        if linear.bias is not None:
            new.bias.copy_(linear.bias)
    return new


def _lstm_hidden_importance(lstm: nn.LSTM) -> torch.Tensor:
    """Per-hidden-unit L1 importance: the sum of that unit's outgoing weights."""
    hidden = lstm.hidden_size
    importance = torch.zeros(hidden)
    for layer in range(lstm.num_layers):
        weight_hh = getattr(lstm, f"weight_hh_l{layer}").detach()
        importance += weight_hh.abs().view(4, hidden, hidden).sum(dim=(0, 2))
    return importance


def _lstm_select(lstm: nn.LSTM, hidden_keep: torch.Tensor,
                 input_keep: torch.Tensor | None) -> nn.LSTM:
    """New LSTM over exactly the hidden units in `hidden_keep`.

    Gate rows are expanded from `hidden_keep` in PyTorch's ``i, f, g, o`` weight
    order; `weight_hh` columns and both biases follow the same set. Each layer's
    `weight_ih` columns follow that layer's *input* set: the producer's kept
    feature channels for layer 0, the previous layer's kept hidden units after
    that. Nothing is copied by prefix.
    """
    hidden = lstm.hidden_size
    hidden_prime = int(hidden_keep.numel())
    in_size = lstm.input_size if input_keep is None else int(input_keep.numel())
    new = nn.LSTM(
        in_size, hidden_prime, lstm.num_layers, batch_first=lstm.batch_first,
        dropout=lstm.dropout if lstm.num_layers > 1 else 0.0,
        bidirectional=False,
    )
    rows = torch.cat([hidden_keep + gate * hidden for gate in range(4)])
    with torch.no_grad():
        for layer in range(lstm.num_layers):
            layer_keep = input_keep if layer == 0 else hidden_keep
            weight_ih = getattr(lstm, f"weight_ih_l{layer}")[rows]
            if layer_keep is not None:
                weight_ih = weight_ih[:, layer_keep]
            getattr(new, f"weight_ih_l{layer}").copy_(weight_ih)
            getattr(new, f"weight_hh_l{layer}").copy_(
                getattr(lstm, f"weight_hh_l{layer}")[rows][:, hidden_keep]
            )
            getattr(new, f"bias_ih_l{layer}").copy_(
                getattr(lstm, f"bias_ih_l{layer}")[rows]
            )
            getattr(new, f"bias_hh_l{layer}").copy_(
                getattr(lstm, f"bias_hh_l{layer}")[rows]
            )
    return new


def _set_module(model: nn.Module, full_name: str, new_module: nn.Module) -> None:
    """Replace a module in `model` by its dotted name (e.g. 'conv.0')."""
    parts = full_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


# ---------------------------------------------------------------------------
# Structured-pruning preflight
# ---------------------------------------------------------------------------

#: Model classes whose topology `_structured_prune` implements and that are
#: covered by exact-index / mask-equivalent regression tests
#: (``tests/test_compress_deepconvlstm_alignment.py``). DeepConvLSTM (B3) is the
#: only one whose downstream tensors are sliced index-for-index end to end;
#: everything else fails closed.
STRUCTURED_PRUNE_SUPPORTED: tuple[str, ...] = (
    "DeepConvLSTM",   # B3 — Conv1d/BN/ReLU/pool stack -> LSTM -> Sequential head
)

#: Why each other Student B topology is deliberately rejected. Kept concrete so
#: the error tells the caller what the index propagation cannot cover.
_STRUCTURED_PRUNE_UNSUPPORTED_REASONS: dict[str, str] = {
    "SEConvNet": (
        "the depthwise-separable reconstruction and the SE `fc1`/`fc2` slices "
        "were selected by prefix/independently instead of by the producer's "
        "channel indices, so surviving features could be attached to the wrong "
        "channels; needs topology-specific slicing plus exact-index tests (B5)"
    ),
    "EmbeddingNet": (
        "its conv/BN stack, projection head and pooled input width are not yet "
        "covered by verified index-exact slicing (B4)"
    ),
    "MultiSensorFusionCNN": (
        "one independent Conv1d branch per sensor channel (nn.ModuleList) has no "
        "single feature-width flow to cascade, and the fused width is fixed at "
        "construction (B1)"
    ),
    "MultiTaskNet": (
        "two parallel Linear heads (activity + subject) read the same trunk "
        "feature vector, so a single-pass index propagation cannot re-slice both "
        "heads (B2)"
    ),
    "GaitViT": (
        "the patch-embedding Conv1d output width must equal embed_dim because "
        "the learned CLS token and positional embeddings are added to it; pruning "
        "the projection width breaks that addition (B7)"
    ),
}


def structured_prune_support(model: nn.Module) -> tuple[bool, str]:
    """Whether `mode="structured"` supports `model`'s topology: ``(ok, reason)``.

    Pure inspection — never touches parameters, buffers or module structure, so
    it is safe to call before any mutation.
    """
    name = type(model).__name__
    if name in STRUCTURED_PRUNE_SUPPORTED:
        return True, ""
    reason = _STRUCTURED_PRUNE_UNSUPPORTED_REASONS.get(
        name,
        "its topology is not the DeepConvLSTM chain (Conv1d/BatchNorm1d stack -> "
        "nn.LSTM -> Linear head) that this pruner slices with exact indices",
    )
    return False, reason


def _check_structured_prune_supported(model: nn.Module) -> None:
    """Raise `NotImplementedError` before any mutation for unsupported topologies."""
    ok, reason = structured_prune_support(model)
    if ok:
        return
    raise NotImplementedError(
        f"structured pruning is not implemented for {type(model).__name__}: "
        f"{reason}. Supported topologies: {sorted(STRUCTURED_PRUNE_SUPPORTED)}. "
        "Use mode='unstructured', or add topology-aware, index-exact slicing "
        "together with regression tests before enabling it."
    )


# ---------------------------------------------------------------------------
# Structured pruning
# ---------------------------------------------------------------------------

def _check_prunable_structure(model: nn.Module) -> None:
    """Fail closed unless `model` contains only modules `_structured_prune` slices.

    The pruner carries one channel index set along a linear module order, so
    anything whose channel semantics it cannot follow is rejected rather than
    silently mis-sliced: grouped/depthwise convs, SE bottlenecks, multi-branch or
    multi-head trunks, and LSTMs whose weight layout differs (bidirectional,
    projections, no biases).
    """
    has_lstm = False
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv1d) and module.groups != 1:
            raise NotImplementedError(
                f"structured pruning does not support grouped/depthwise Conv1d "
                f"{name!r} (groups={module.groups})"
            )
        if hasattr(module, "fc1") and hasattr(module, "fc2"):
            raise NotImplementedError(
                f"structured pruning does not support the SE attention block "
                f"{name!r}: its channel indices are not propagated"
            )
        if isinstance(module, nn.BatchNorm1d) and not (
            module.affine and module.track_running_stats
        ):
            raise NotImplementedError(
                f"structured pruning requires affine BatchNorm1d with running "
                f"statistics, but {name!r} has affine={module.affine}, "
                f"track_running_stats={module.track_running_stats}"
            )
        if isinstance(module, nn.LSTM):
            has_lstm = True
            if module.bidirectional or getattr(module, "proj_size", 0):
                raise NotImplementedError(
                    f"structured pruning supports only unidirectional LSTMs "
                    f"without projections, but {name!r} is bidirectional="
                    f"{module.bidirectional} with proj_size="
                    f"{getattr(module, 'proj_size', 0)}"
                )
            if not getattr(module, "bias", True):
                raise NotImplementedError(
                    f"structured pruning requires an LSTM with biases, but "
                    f"{name!r} has bias=False"
                )
    if not has_lstm:
        raise NotImplementedError(
            "structured pruning is implemented for the DeepConvLSTM chain only "
            "(Conv1d/BatchNorm1d feature stack -> nn.LSTM -> Linear head); no "
            "nn.LSTM was found, so the head-input slicing rule does not apply"
        )


def _structured_prune(model: nn.Module, amount: float) -> nn.Module:
    """Physically remove Conv1d channels and LSTM hidden units by L1 norm.

    Supported topology: the DeepConvLSTM chain (B3 / `configs/lightweight.yaml`)
    — a Conv1d -> BatchNorm1d feature stack, one `nn.LSTM` consuming the last
    conv's channels, and a Linear head consuming the LSTM's last hidden state.

    Index propagation: one explicit index set is carried along the chain, and
    every hop asserts that the consumer's declared width matches the pending
    producer before anything is rebuilt.

    * **Conv1d** keeps its top-(1-amount) output channels by L1 norm and takes
      its input columns from the *producer's* kept indices, i.e.
      ``weight[keep_out][:, keep_in]``; the bias follows ``keep_out``.
    * **BatchNorm1d** is rebuilt over the same indices, indexing ``weight``,
      ``bias``, ``running_mean`` and ``running_var`` by ``keep`` — never by a
      prefix — so each statistic stays with the filter it was learned for.
      BatchNorm preserves channel order, so the pending index set is unchanged.
    * **nn.LSTM** keeps hidden units by recurrent-weight L1 importance; gate rows
      are expanded from ``hidden_keep`` in PyTorch's ``i, f, g, o`` weight order,
      and ``weight_hh`` columns, ``bias_ih``/``bias_hh`` rows and every layer's
      ``weight_ih`` columns (layer 0: the conv's kept channels; later layers: the
      previous layer's kept hidden units) are sliced by those indices.
    * **Linear** consumes the LSTM's last hidden state, so its weight takes
      exactly the kept hidden-unit columns; later head layers are left untouched
      because their input is the previous Linear's output.

    An unsupported topology raises `NotImplementedError` from
    `_check_structured_prune_supported` / `_check_prunable_structure` *before*
    the model is copied or mutated, instead of returning a model whose logits are
    computed from mismatched weights.
    """
    # Fail before any work: an unsupported topology must never be deep-copied,
    # mutated, or returned half-pruned.
    _check_structured_prune_supported(model)
    _check_prunable_structure(model)

    model = copy.deepcopy(model).cpu()

    # `pending_keep` is the kept-channel index set (into the *original* width
    # `pending_width`) that the tensor entering the next consumer corresponds to.
    # None means "not produced by a pruned layer" or "already consumed".
    pending_keep: torch.Tensor | None = None
    pending_width: int | None = None

    # Store (name, new_module) replacements to apply after the loop.
    replacements: list[tuple[str, nn.Module]] = []

    def _expect_consumer(module_name: str, width: int) -> None:
        if pending_width != width:
            raise NotImplementedError(
                f"structured pruning cannot follow {module_name!r}: it consumes "
                f"{width} channels but the preceding pruned layer produced "
                f"{pending_width}"
            )

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv1d):
            if pending_keep is not None:
                _expect_consumer(name, module.in_channels)
            keep = _select_channels(
                _l1_channel_norms(module.weight, is_conv=True), amount
            )
            replacements.append((name, _conv1d_select(module, keep, pending_keep)))
            pending_keep, pending_width = keep, module.out_channels

        elif isinstance(module, nn.BatchNorm1d):
            if pending_keep is None:
                continue
            _expect_consumer(name, module.num_features)
            if len(pending_keep) != module.num_features:
                replacements.append(
                    (name, _batchnorm1d_select(module, pending_keep))
                )
            # BatchNorm preserves the channel order, so `pending_*` is unchanged.

        elif isinstance(module, nn.LSTM):
            if pending_keep is not None:
                _expect_consumer(name, module.input_size)
            hidden_keep = _select_channels(
                _lstm_hidden_importance(module), amount
            )
            replacements.append(
                (name, _lstm_select(module, hidden_keep, pending_keep))
            )
            pending_keep, pending_width = hidden_keep, module.hidden_size

        elif isinstance(module, nn.Linear):
            if pending_keep is None:
                # Head-internal layer: its input is the previous Linear's output.
                continue
            _expect_consumer(name, module.in_features)
            replacements.append((name, _linear_select_in(module, pending_keep)))
            pending_keep, pending_width = None, None

    # --- Apply all replacements ---
    for name, new_module in replacements:
        _set_module(model, name, new_module)

    return model


def effective_weight(module: nn.Module) -> torch.Tensor | None:
    """The weight tensor a Conv1d/Linear actually computes with, or None.

    Under a live `torch.nn.utils.prune` reparametrization the effective weight
    is ``weight_orig * weight_mask``; after `torch.nn.utils.prune.remove` it is
    the plain ``weight`` parameter. Reading `weight_orig` alone reports the
    un-pruned tensor and therefore measures the wrong sparsity.
    """
    if hasattr(module, "weight_orig") and hasattr(module, "weight_mask"):
        return module.weight_orig.detach() * module.weight_mask
    weight = getattr(module, "weight", None)
    if isinstance(weight, torch.Tensor):
        return weight.detach()
    return None


def pruned_weight_sparsity(model: nn.Module) -> dict:
    """Actual mask-induced sparsity of the Conv1d/Linear weights.

    Returns ``{"zeros", "total", "sparsity"}`` where ``sparsity`` is the
    fraction of the pruned layers' weight elements that are zero. This is the
    number `_unstructured_prune` reports; summing ``model.parameters()`` would
    instead count ``weight_orig`` and report ~0%.
    """
    zeros = 0
    total = 0
    for module in model.modules():
        if not isinstance(module, (nn.Conv1d, nn.Linear)):
            continue
        weight = effective_weight(module)
        if weight is None:
            continue
        zeros += int((weight == 0).sum().item())
        total += int(weight.numel())
    return {
        "zeros": zeros,
        "total": total,
        "sparsity": (zeros / total) if total else 0.0,
    }


def _unstructured_prune(model: nn.Module, amount: float) -> nn.Module:
    """Apply L1 unstructured pruning to all Conv1d/Linear weights (in place).

    The reparametrization is materialised with `prune.remove` so the zeros are
    real ``weight`` values: the result deep-copies, dynamically quantizes and
    exports like an ordinarily-trained model, and `pruned_weight_sparsity`
    reports the sparsity the mask actually induced. LSTM weights are left
    untouched (their parameters are not named ``weight``).
    """
    from torch.nn.utils import prune

    for _, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")

    report = pruned_weight_sparsity(model)
    print(f"unstructured pruning: {report['sparsity']:.2%} sparsity achieved "
          f"({report['zeros']}/{report['total']} pruned weight elements zeroed)")
    return model


def apply_pruning(model: nn.Module, amount: float = 0.3,
                  mode: str = "structured") -> nn.Module:
    """Prune `model` by `amount` (fraction of channels/weights to remove).

    ``mode="structured"`` raises `NotImplementedError` before touching the model
    when the topology is unsupported (see `structured_prune_support`);
    ``mode="unstructured"`` works on any topology containing Conv1d/Linear.
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