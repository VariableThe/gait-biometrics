# In-memory cross-day neural metric fitter

`engine.cross_day_metric.fit_cross_day_encoder` fits only the windows passed by
the caller. It reads no datasets or checkpoints, writes no files, performs no
validation, and returns the final epoch rather than selecting an epoch. The
caller must exclude held-out identities **before** calling it and owns subject
folds, candidate selection, calibration, checkpointing and evaluation.

The existing HAR, WISDM and Santos test cohorts have already been inspected.
Follow-ups using those cohorts are exploratory; a new fitter or output directory
does not create a fresh blind test. Synthetic checks here establish software
behavior only, with no measured gait recognition benefit.

```python
from engine.cross_day_metric import fit_cross_day_encoder
from models.gait_encoders import build_gait_encoder

encoder, training_record = fit_cross_day_encoder(
    X_train, subject_ids_train, day_ids_train,
    n_channels=6,
    model_args={
        "architecture": "multiscale",
        "width": 32,
        "embed_dim": 64,
        "stats_fusion": True,
        "normalization": True,
        # C=6 has no implicit physical grouping; declare the actual layout.
        "vector_groups": [[0, 1, 2], [3, 4, 5]],
    },
    seed=42,
    epochs=40,
    device="cpu",             # or explicit "cuda:0"; no automatic GPU choice
    torch_threads=2,
    objective="cross_day_supcon",  # "supcon" for the mask ablation
)
# encoder is in eval mode and exposes .n_channels and .embed_dim.
# The parent runner saves encoder.state_dict() and training_record separately.
restored = build_gait_encoder(6, training_record["config"]["model_args"])
# restored.load_state_dict(torch.load(path, weights_only=True))
restored.eval()
```

`X`, subjects and days must be in-memory NumPy arrays or Torch tensors. Windows
have nonempty floating `(N,C,T)` shape; integer signals, nonfinite values and
values that overflow FP32 are rejected. Subjects and days are one-dimensional
integer arrays with one entry per row. There must be at least two identities,
each represented on at least two distinct days. IDs need not be consecutive.
Only CPU and CUDA are supported. Training converts floating inputs to FP32 and
disables ambient autocast. The default architecture is multiscale; existing
multiscale, temporal and legacy factory configurations are accepted. Legacy
requires `T >= 4` and explicit `add_magnitudes=False`. Unknown model options and
nonintegral dimensions/seeds are rejected.

Normalization is fitted **once** on all unaugmented supplied training windows,
before creating AdamW. The existing encoder fits its channel, magnitude and
optional statistical-feature buffers in bounded chunks. Inference never refits
them. Training uses no augmentation (`augment={}`), so this ablation isolates
the pair mask while preserving the same architecture, initialization, batch
schedule, optimizer and auxiliary objective.

## Loss and sampler

For distinct original rows, cross-day positives have the same identity and
different recorded days. Negatives have different identities, regardless of
day. Same-identity/same-day pairs are neutral and excluded from both the positive
average and contrastive denominator. Copies of the same original row are always
excluded. A different augmentation of one row cannot establish a cross-day pair.

With unit-normalized embeddings, temperature `tau=0.1`, positives `P(i)` and
negatives `N(i)`, the per-anchor loss is

```text
logsumexp(z_i dot z_a / tau for a in P(i) union N(i))
    - mean(z_i dot z_p / tau for p in P(i))
```

Anchors without both a positive and a negative contribute no term; the public
loss returns differentiable zero if none are valid. The fitter's sampler ensures
every anchor has both. The loss is averaged over valid anchors. Both objectives
add `0.25 * cross_entropy(linear(z), contiguous_training_identity_label)`. This
small training-only classifier supplies identity discrimination alongside the
metric loss; it is discarded on return and does not guarantee absence of collapse.
The `supcon` baseline uses the existing ordinary same-ID supervised contrastive
loss with the same original-row exclusion. All other settings remain identical.

Each batch contains `P=min(8, identity_count)` distinct identities and `K=4`
windows per identity: two from each of two distinct days. All identities are
visited each epoch. With more than two days, shuffled day orders are cycled as
an identity recurs; the sampler does not guarantee every available day appears
in every epoch. If a selected day has fewer than two rows, sampling uses
replacement with its original row ID retained. Otherwise its two rows are
distinct. There is no cap on training N. Batch size is at most 32, pair matrices
are at most `32 x 32`, and epoch steps are the larger of
`ceil(N / (P*K))` and `ceil(identity_count / P)`. An epoch is a balanced sampling
budget, not a guarantee of visiting every original row.

Defaults are AdamW, learning rate `0.001`, weight decay `0.0001`, gradient norm
clip `1.0`, no scheduler, no early stopping and exactly 40 epochs. The `epochs`
argument permits a caller to predeclare another fixed budget; it does not enable
selection. CPU fitting with the same inputs/config is checked for exact
reproducibility. RNG state and thread count are restored after fitting. CUDA RNG
is seeded on the requested device, but nondeterministic CUDA kernels and hardware
or library changes can prevent bitwise repeatability.

## Records and serialization

The JSON-compatible record contains the expanded effective factory arguments,
all optimizer/objective/sampler settings, input shape and dtype, exact training
subject IDs and days, per-epoch total/metric/CE means, completed/final epoch,
encoder/classifier parameter counts, elapsed wall time, environment versions,
normalization-call count and finite-state checks. Each loss mean averages the
equal-size optimizer batches. Nonfinite losses, gradients or final states fail
the call rather than producing a successful record.

`training_window_ids_sha256` fingerprints local input row indices together with
subject/day IDs; this API receives no physical recording IDs. A second hash
includes FP32 training signal values, subjects and days. Parent experiment
registries must separately retain physical recording/window provenance and exact
source-file hashes. The fitter records SHA-256 fingerprints of Python marshalled
callable code objects without reading source files. Those interpreter/path-
dependent fingerprints complement, rather than replace, a source-file lock.
Neither hash proves the caller supplied only training data.

The returned object is the existing factory encoder without the classifier.
Raw `state_dict()` includes all fitted normalizer buffers. Build the same factory
with `record["config"]["model_args"]`, then load tensor state using
`torch.load(..., weights_only=True)` and `load_state_dict`; no custom pickle class
or refitting is needed. The small synthetic suite checks buffer/output equality
after this round trip, including six-axis inputs and statistical fusion.

## Research basis and limits

[Khosla et al., *Supervised Contrastive Learning*, NeurIPS 2020,
§3.2.2, Eq. 2](https://papers.nips.cc/paper_files/paper/2020/file/d89a66c7c80a29b1bdbab0f2a1a94af8-Paper.pdf)
defines the outside-log average over same-label positives. This implementation
adapts that reduction to existing inertial-window encoders, restricts positives
by recorded day, excludes same-day same-ID pairs from the denominator, and adds
a modest CE term. The paper's image results do not validate this adaptation.

[Nemes and Antal, *Feature Learning for Accelerometer based Gait Recognition*,
§6.2 and Table 2](https://arxiv.org/abs/2007.15958) evaluates cross-day gait
recognition using learned accelerometer features with per-person one-class
SVMs, illustrating a same-day/cross-day generalization gap. Its feature training,
classifier and datasets differ from this fitter; no reported error rate is
transferred here.

[Permatasari et al., *Adaptive 1-dimensional time invariant learning for inertial
sensor-based gait authentication*, §§3–5](https://doi.org/10.1007/s00521-022-07741-0)
studies two sessions a month apart and proposes domain adaptation. The authors
identify uncontrolled session covariates, so elapsed time alone should not be
treated as the cause of session differences. This fitter reproduces neither
their adaptation method nor their evaluation pipeline.

The cross-day mask and balanced sampler are proposals, with ordinary SupCon as
a matched ablation. A model trained on both days of training identities can be
evaluated on day1-to-day2 verification for held-out identities; this does not
establish generalization to a third, unseen acquisition day for trained users.
