# Angular-margin identity classification

`models.angular_margin.AngularMarginClassifier` implements an opt-in ArcFace
head for inertial gait embeddings. It is a face-recognition loss adaptation,
not a reproduction of the face system or evidence of improved gait
verification. The defaults `scale=30`, `margin=0.3` are experimental choices.
The head introduces no additional metric objective or pair-mining policy.

## Research basis and exact target transform

[Deng et al., *ArcFace: Additive Angular Margin Loss for Deep Face
Recognition*, Equation 3 and Algorithm 1](https://arxiv.org/abs/1801.07698v3)
normalize features and class weights, omit bias, and add a margin to the
ground-truth angle before cross entropy. For normalized dot product `c` and
margin `m` in radians, the target becomes
`cos(theta + m) = c*cos(m) - sqrt(1-c*c)*sin(m)`.

This implementation follows the monotonic extension in the
[authors' historical code, pinned at cadd367](https://github.com/deepinsight/insightface/blob/cadd367e00b3b6136c781cc8390ce8f261170e63/src/train_softmax.py#L199-L237),
with `easy_margin=False`:

```text
threshold = cos(pi - m)
phi(c) = c*cos(m) - sqrt(1-c*c)*sin(m)   if c > threshold
         c - sin(pi-m)*m                otherwise
logits[i, y[i]] = scale * phi(c[i, y[i]])
logits[i, j]    = scale * c[i, j]        for j != y[i]
```

The fallback avoids rewarding increasing angles beyond `pi-m`. It is
monotonic but has a downward jump as the angle crosses that boundary; the
threshold equality uses the fallback. The transformed target can fall below
`-1` before scaling, and is deliberately not clamped back to the cosine range.
The supported margin range `0 <= m < pi/2` follows that historical code.

## Module contract

```python
from models.angular_margin import AngularMarginClassifier
from torch.nn import functional as F

head = AngularMarginClassifier(
    in_features=64, n_classes=12, scale=30.0, margin=0.3
)
head = head.to(embeddings.device)
# labels: consecutive optimizer-training identity indices, shape (B,).
logits = head(embeddings, labels)
loss = F.cross_entropy(logits, labels.long())
# Omitting labels always removes the margin, including in train mode.
cosine_logits = head(embeddings)
```

Constructor dimensions must be positive integers, with booleans rejected.
Scale must be finite and positive; margin must be finite and in the range
above. `embeddings` must be a nonempty dense floating tensor of shape
`(B, in_features)`, finite after conversion to float32. Labels must be a
dense integer tensor of shape `(B,)` on the same device, with values in
`[0, n_classes)`. Supported label dtypes are int8/int16/int32/int64/uint8;
floating and boolean labels are rejected rather than silently truncated.
Embeddings and the head must share a device.

The only learned parameter is `weight`, of shape `(n_classes, in_features)`,
initialized with Xavier uniform and normalized on every forward pass. There
is no bias. Positive scaling of feature vectors or class weights does not
change logits outside the normalization epsilon region. Supplying labels
controls the margin independently of `train()`/`eval()`. At `margin=0`, both
paths return exactly the same scaled cosine logits.

## Numerical behavior

The head disables autocast locally and casts features and weights to float32
before normalization, similarity, and margin arithmetic. Output logits are
always float32, including under CPU autocast. This local precision choice
follows [PyTorch's guidance for autocast-disabled subregions](https://docs.pytorch.org/docs/2.14/amp.html).
It does not establish a throughput improvement; GPU behavior and throughput
need measurement by the trainer owner.

Computed cosine is clamped to `[-1, 1]` before the target transform.
Normalization uses epsilon `1e-12`, so zero features or class-weight rows
remain finite and normalize to zero. Those degenerate vectors have no unit
direction; finite logits alone do not imply a useful representation.
Before the L2 norm, rows with maximum absolute value above one are rescaled
by that maximum. This preserves their direction while avoiding fp32 norm
overflow for finite values such as `1e20`. The rescaling factor is detached;
unit normalization is invariant to it, so this does not alter its derivative.

For the sine term, `1-c*c` is floored at float32 epsilon **before** its square
root, then exact zero sine values are restored with `where`. Every strictly
interior representable float32 cosine already has `1-c*c >= epsilon`, so its
forward formula is unchanged. At exact `c=+/-1` the output also preserves the
formula's value, while choosing a finite zero subgradient for the square-root
term. This is an explicit endpoint stabilization, not an assertion that the
analytic derivative with respect to cosine is finite. Flooring before the
square root matters because a later mask can still propagate invalid
gradients through an already singular operation; see
[PyTorch autograd notes](https://docs.pytorch.org/docs/2.14/notes/autograd.html#gradients-for-non-differentiable-functions).
Feature and parameter gradients ultimately use their source tensor dtypes;
fp32 internal arithmetic cannot guarantee against overflow after conversion
back to a low-precision source gradient, especially at zero vectors.

## Trainer integration, selection, and persistence

The trainer owner integrates the requested configuration:

```yaml
classifier_type: arcface       # linear | arcface
angular_scale: 30.0
angular_margin: 0.3
```

These keys are an integration interface; this module alone does not parse
configuration or change the existing trainer. The trainer must include head
parameters in its optimizer and supply remapped **optimizer-training** labels
to the head before cross entropy. Validation/test identities must never
enter that class vocabulary or supervise this head. For unseen-subject
verification, use normalized encoder embeddings and the existing similarity
protocol rather than training-class head probabilities.

Select classifier type, scale, margin, epoch, and any other candidate settings
using training/validation identities only. Existing inspected test cohorts
must not guide those choices. Freeze the full candidate and any calibrated
threshold before a future locked evaluation. Report subject-fold variation
and independent-session enrollment/probe results where available. Good
within-recording validation does not establish generalization across subjects
or sessions, and overlapping windows are not independent recordings.

Save `head.state_dict()` together with all four constructor arguments in the
checkpoint configuration. The raw head state contains only `weight`; scale
and margin are configuration, not learned tensors. Restore with
`torch.load(..., weights_only=True)` and strict `load_state_dict` into a head
constructed from the saved configuration. Do not serialize the module object.

`tests/test_angular_margin.py` uses analytic and tiny synthetic CPU inputs
with one Torch thread. It covers target-only scalar-reference logits, the
threshold equality and monotonic fallback, exact zero-margin behavior,
train/eval label semantics, scaling invariance, normal and degenerate
gradients, endpoint angles, autocast and input dtypes, label/device/argument
errors, and a weights-only checkpoint round trip. These checks establish
module behavior, not gait accuracy.
