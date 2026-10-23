# Opt-in gait encoders and training transforms

These methods are research-informed adaptations for the existing within-recording
verification protocol. They are not exact reproductions of published systems.
No HAR/WISDM improvement or measured authentication result is established by the
synthetic tests. Real candidate selection belongs on the fixed validation split;
evaluate a locked configuration on test only after that selection.

## Integration contract

```python
from models.gait_encoders import build_gait_encoder
from data.gait_transforms import augment_gait

args = {
    "architecture": "temporal",  # legacy | multiscale | temporal
    "embed_dim": 64,
    "width": 32,
    "dropout": 0.1,
    "normalization": True,
    "stats_fusion": False,
    "temporal_bins": 4,
}
encoder = build_gait_encoder(n_channels=9, model_args=args)
# X_train must already exclude all validation and test subjects/windows.
encoder.fit_normalization(X_train)
z = encoder(X_train[:4])  # (N,C,T) -> (N,64), L2-normalized

# Explicit training call only; identity if config is None or {}.
x_train_augmented = augment_gait(X_train[:4], {
    "jitter_std": 0.005,
    "scale_std": 0.02,
    "roll_fraction": 0.05,
    "rotation_degrees": 5,
})
encoder.eval()
z_validation = encoder(X_validation)  # no augmentation or fitting here
```

The returned `nn.Module` exposes `n_channels`, `embed_dim`, `architecture`,
`signal_channels`, `stats_dim`, and `fit_normalization(X, chunk_size=256)`.
It accepts a floating tensor (or array for fitting) in `(N,C,T)` layout. New
architectures accept varying positive T, including non-power-of-two lengths;
the legacy network requires T >= 4 because its unchanged backbone pools twice.
Width must be >= 4. Temporal bins must be 4 or 8.

Save `encoder.state_dict()` and the constructor arguments together. Reload by
building the same architecture and using `load_state_dict`; tensor buffers are
compatible with `torch.load(..., weights_only=True)`. New wrapper checkpoints
have `encoder.*` keys and are not raw historical `EmbeddingNet` checkpoints.
For an existing raw checkpoint, instantiate the legacy wrapper with matching
embedding size/dropout and load that historical state into `encoder.encoder`.
Existing `models.siamese.EmbeddingNet` and all existing training/evaluation
paths are unchanged.

## Architecture choices and research basis

`legacy` wraps the unchanged `EmbeddingNet` class. Input normalization defaults
to false; `add_magnitudes` and `stats_fusion` must remain false. Its constructor
uses `embed_dim=64` and `dropout=0.1` unless overridden, so a comparison must
explicitly match the historical experiment's constructor arguments as well as
its training budget. Width has no effect on this backbone.

`multiscale` uses a 5-tap stem and three residual blocks. Each block applies a
1x1 bottleneck, parallel kernels of lengths 3, 7, 15, and 31, a 1x1 mixing
projection, GroupNorm, SiLU, an SE channel gate, and dropout. Pooling concatenates
the temporal mean and population standard deviation before an embedding head.
The bottleneck and parallel temporal resolutions are inspired by
[InceptionTime](https://arxiv.org/abs/1909.04939). This is a single small network
with different kernels, normalization, pooling, and objective; it omits that
paper's classification ensemble and max-pooling branch. Channel recalibration
adapts [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507) to 1D.

`temporal` uses three compact residual blocks with kernel lengths 5 and 3 and
dilations 1, 2, and 4, plus GroupNorm and SE gates. Adaptive average pooling
retains four ordered temporal bins (optionally eight), flattened and concatenated
with the global feature standard deviation. This preserves coarse temporal
order that a single global average discards; it does not recover aligned gait
cycles from arbitrary windows. Residual dilated convolutions are motivated by
the [TCN study by Bai, Kolter, and Koltun](https://arxiv.org/abs/1803.01271).
Here convolutions are symmetric and noncausal for whole-window embedding,
rather than a reproduction of its sequence prediction network.

All new backbones keep full temporal resolution, avoid large transformers,
and have fewer than 500,000 parameters at default width 32, with or without the
statistics branch. Their normalization layers have no batch running statistics.
The legacy backbone retains its historical BatchNorm behavior during training.
At C=9, width 32, and D=64, multiscale has 26,092 parameters (33,804 with
statistics); temporal with four bins has 42,412 (50,124 with statistics).

[Hoang et al.](https://arxiv.org/abs/1602.03199) motivate examining sensor
orientation sensitivity in inertial gait verification.
[IDNet](https://arxiv.org/abs/1606.03238) motivates learned inertial gait
features and explicitly handles walking cycles and subsequent decisions.
Neither paper's orientation correction, cycle extraction, SVM, or multi-stage
decision protocol is implemented here. Their reported error rates do not transfer
to the current HAR/WISDM within-recording protocol.

## Physical channels and fixed training normalization

For known contiguous 3- or 9-channel layouts, new encoders append each raw vector
magnitude before channel standardization. WISDM has one acceleration triad.
HAR has body acceleration `(0,1,2)`, gyroscope `(3,4,5)`, and total acceleration
`(6,7,8)`, in the existing adapter's order. A magnitude does not invent a gravity
estimate, orientation, or gyroscope channel for WISDM. Axis channels are retained;
appending magnitudes alone does not make the full model rotation invariant.

Use `add_magnitudes=False` for an ablation or `vector_groups=[[...], ...]` for
explicit disjoint three-axis groups. Other input channel counts have no inferred
triads. There are four feature signals for C=3 and twelve for C=9 by default.

`fit_normalization` computes channel means and population standard deviations
over the N and T axes of training X, including separate moments for appended
raw magnitudes. Chunked float64 moment merging bounds temporary memory and
avoids cancellation from large sensor offsets. Standard deviations clamp at
`norm_eps=1e-5`. Normalizer buffers are `channel_mean`, `channel_std`,
`magnitude_mean`, `magnitude_std`, `stats_mean`, `stats_std`, and the boolean
`normalization_fitted`; no training windows are retained. Fitting returns self.

Only an explicit call fits these buffers. `forward`, `eval`, and changing window
length never update them; fitting in eval mode raises an error. Fresh buffers
are identity transforms until fitted. Fit after the subject-heldout validation
carve, before making augmented training batches. With `normalization=False`,
raw channels and magnitudes are unchanged; a requested statistics branch still
fits its own training feature normalization.

## Optional statistical and spectral fusion

`stats_fusion=True` adds a learned width-dimensional projection, concatenated
with the convolutional representation before the final embedding head. Features
are differentiable and computed independently for each window from the same
input signals used by the backbone:

- `log1p` population standard deviation and RMS per signal;
- four relative powers from the centered real FFT, in normalized frequency bands
  `(0,1/16]`, `(1/16,1/8]`, `(1/8,1/4]`, `(1/4,1/2]` cycles/sample;
- centered autocorrelation at rounded lags T/16, T/8, and T/4, divided by
  zero-lag population variance and clamped to [-1,1];
- upper-triangular off-diagonal Pearson correlations between feature signals.

FFT bands exclude DC, normalize by total centered spectral power, and are not
specified in Hz: HAR/WISDM have different sampling rates. Autocorrelation lags
are fractions of the window, not clinically identified step or stride times.
Short windows can have duplicate lags or empty bands, represented consistently.
Epsilon floors keep constant windows finite. Statistics are standardized with
training-only moments, clamped to [-10,10], and passed through Linear,
LayerNorm, SiLU, and dropout. No BatchNorm running state is used.

For S input feature signals, the default statistics dimension is
`9*S + S*(S-1)/2`: 42 for C=3 and 174 for C=9 with magnitudes. Raw means are
excluded by default. `stats_include_mean=True` explicitly adds S means, giving
46 and 186 respectively. RMS and retained raw axes can still carry static
orientation or recording properties. A gain from this branch must therefore
be reported as a within-recording result and examined with no-statistics,
no-magnitudes, and orientation/mean ablations; it is not independent-session
authentication evidence.

## Explicit training augmentation

`augment_gait(x, config, generator=None)` accepts `(N,C,T)` or one `(C,T)` tensor,
returns new storage, and never changes the number/order of windows. Settings
default to zero; there is no automatic train/eval or test augmentation hook.

- `jitter_std`: additive independent small Gaussian sensor noise, in raw units.
- `scale_std`: positive lognormal scaling, one factor shared by all axes and
  timesteps of each vector. Different triads may receive different gains.
- `roll_fraction`: circular temporal shift bounded by a fraction of T (<= .5),
  one shift shared by all channels of a window. The wrap seam can be unnatural;
  keep the value small and include a no-roll temporal-bin ablation.
- `rotation_degrees`: bounded random axis-angle rotation (<= 180 degrees).
  The same proper SO(3) matrix applies to all related triads and all timesteps
  in each window. It preserves lengths, relative dot products, handedness, and
  linear relations between triads. It is a perturbation, not measured device
  orientation correction. Unknown layouts require explicit `vector_groups`.

Small angles and scale/noise levels are candidate choices, not validated
defaults. Large roll/rotation can remove useful phase or orientation information.
Use the same source-specific physical layout and transform every related vector
coherently. A seeded `torch.Generator` controls all augmentation randomness;
a CPU generator is accepted even when the input is on CUDA.

## Evaluation and local checks

Preserve walking label 0, T=128, stride=64, source-heldout test subjects,
four heldout validation subjects, and split seed 42 when comparing to the
established within-recording baseline. Keep independent enrollment/session
evaluation separate. Select configurations and epochs using validation only;
do not select using locked test results or promise a winning architecture.

The focused test module covers 3/9-channel shapes, varying T, normalized output,
finite backward paths including constant inputs, population moments and clamps,
frozen normalization, weights-only buffer reloads, raw legacy equality,
temporal order, default parameter budgets, and physically coherent rotations.
Run with the shared environment and one Torch/BLAS thread:

```bash
TMPDIR=/home/sherlock/Code/github/VariableThe/artifacts/tmp \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run --no-project --python /home/sherlock/Code/github/VariableThe/.venv/bin/python \
python -m pytest tests/test_gait_encoders.py -q
```

No real local training or GPU experiments are part of these checks.
