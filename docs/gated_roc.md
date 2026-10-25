# Failure-inclusive gated ROC

`engine.gated_roc` exports three pure functions, uses only NumPy and the standard
library, reads no files, and fits nothing. This is a general metric utility for
future quality-gated authentication. It implements no cycle detector or production
authentication system and changes no historical artifact. Verification uses only
synthetic claims; it supports no SOTA or performance claim.

## Inputs and decisions

`gated_roc(labels, scores, valid) -> dict` and
`calibrate_gated_threshold(labels, scores, valid, *, target_far=0.01) -> dict`
require aligned, unmasked, one-dimensional vectors. Labels contain integer or
boolean binary values: 1 is genuine, 0 is impostor. Both classes must occur in the
**original** labels, even if a class has no valid scores. Floating-point labels
are rejected. Validity elements must be Python or NumPy booleans; integer 0/1
masks are rejected, including mixed boolean/integer lists. Object arrays are
allowed when their individual elements satisfy these rules.

Each valid score must be a `numbers.Real`, excluding Python/NumPy booleans, and
must convert to a finite float64 value. Every invalid score must be **exactly
`None`**. Finite sentinels, NaN, infinities, complex values, numeric strings, and
booleans cannot represent invalid scores. A valid `None` is also rejected.
Validation observes the elements supplied to the API; types already lost through
caller-side NumPy coercion cannot be recovered.

Scores are converted to Python floats (IEEE binary64, the float64 representation
used for ranking here) **before** ranking, grouping ties, and applying policies.
Distinct higher-precision numbers or large integers can consequently become tied;
for example, `2**53` and `2**53 + 1`. Positive and negative zero tie. Overflow to
nonfinite float64 is rejected; underflow and finite rounding follow binary64.
Exact rate arithmetic does not recover score precision lost in this conversion.

Only these policies exist, each with exactly two keys:

```json
{"kind": "reject_all", "threshold": null}
{"kind": "score_at_least", "threshold": 0.6}
```

The first is an explicit **control action**; its null threshold is not missing
measurement data. The second accepts iff the claim is valid and its converted
score is at least the finite threshold. Numeric policy thresholds follow the
same Real, nonboolean, finite-float64 validation as scores. Invalid claims always
reject. ROC state 0 is reject-all, followed by one state for each distinct valid
score in descending order. Ties enter together. No threshold is invented above
float64 maximum; no endpoint accepts invalid original claims. The terminal state
accepts all valid claims. With all scores invalid, it is state 0 itself.

`apply_gated_policy(scores, valid, policy) -> numpy.ndarray` accepts either policy
dict and returns a fresh one-dimensional `np.bool_` array. It checks all inputs
even for reject-all, permits empty aligned vectors, and never mutates inputs.
All three functions raise `ValueError` for invalid inputs described here.

## Denominators, EER, and calibration

Every full ROC state uses all original genuine and impostor claims. Its counts
include invalid genuine claims in FN and invalid impostor claims in TN. FAR is
`FP / n_impostor`; FRR is `FN / n_genuine`. Class coverage and failures are
reported separately. If `fg` and `fi` are genuine and impostor failure fractions,
then, whenever both valid classes exist, at every state:

```text
FRR = fg + (1 - fg) * FRR_valid
FAR = (1 - fi) * FAR_valid
```

Reject-all starts with `FAR - FRR = -1`. The terminal rates are
`FAR = 1 - fi` and `FRR = fg`. This path is monotone, so a full EER crossing exists
iff `fg <= 1 - fi`, equivalently the two class coverages sum to at least 1.
It may require interpolation across tied-score jumps. If equality holds only at
the terminal endpoint, that endpoint is an exact threshold state. Either valid
class may be empty: full EER can still equal 0 or 1 at an endpoint, while the
conditional diagnostic is undefined.

Integer counts determine `Fraction` rates and the sign of `FAR - FRR`; float
display values never determine crossing status. An exact zero is
`exact_threshold_state`, including non-binary fractions such as 1/3. Otherwise
the first crossing segment is interpolated with exact fractions. An interpolated
EER describes a line segment, not an attainable deterministic threshold. With no
crossing, EER and its bracket are null and status is `undefined_no_crossing`;
terminal rates remain available. No full ROC AUC is reported because the
attainable FAR range may be incomplete.

`conditional.eer` uses only successful claims and has the same state indexes and
policy references as the full ROC. It is explicitly diagnostic and never drives
the full metric or calibration. If either valid class is absent, its status is
`undefined_empty_valid_class` and its EER and bracket are null. For example,
20% coverage in both classes can give conditional EER 0, terminal full FAR 1/5
and FRR 4/5, and no full EER crossing. For genuine scores
`[.9, .6, .3, None]` and impostor scores `[.8, .5, .2, None]`, full EER is 1/2,
while conditional EER is an **exact threshold state** at .6 with value 1/3.

Calibration checks a finite Real, nonboolean target in [0, 1], interpreting it as
`Fraction(str(target_far))`. A float target `1 / 3` therefore means the exact
decimal `0.3333333333333333`, which is below 1/3; pass `Fraction(1, 3)` for the
exact rational cutoff. The function selects the **last** state in the above
ordering with unconditional empirical FAR at most that exact target. This is the
most permissive attainable tied-score state, including later states with the
same FAR. Reject-all is eligible, and is selected when all claims are invalid or
no score state qualifies. The returned counts and rates describe that fixed
policy on the supplied calibration claims. Applying it to other claims requires
passing the nested `policy` dict to `apply_gated_policy`.

The caller must isolate held-out calibration from model/gate development and
final evaluation, freeze the gate and scoring system, and account for dependent
people, windows, sessions, and repeated claims. A smaller empirical FAR, exact
arithmetic, or a conditional EER provides no population FAR guarantee. These
functions perform no splitting, uncertainty estimation, independence checks,
identity balancing, or population inference.

## Exact JSON return schema

Both dict-returning functions produce JSON-native values and conform to the
following JSON Schema (2020-12). The root alternatives are `roc` for `gated_roc`
and `calibration` for `calibrate_gated_threshold`. Every listed field is required;
no extra keys are returned. The underscore-prefixed definitions are reusable
fragments, not extra outputs. Standard JSON numbers are finite.

All `*_exact` values are canonical reduced `str(Fraction(...))` strings, so
integers are `"0"` and `"1"`, not `"0/1"` or `"1/1"`. All corresponding numeric
fields are Python float displays. Count fields and indexes are Python integers.
Arithmetic invariants (including reduced fractions and consistency with counts)
are additional to the structural schema.

`terminal` repeats the complete last state and adds its zero-based `state_index`.
EER brackets reference `states` by zero-based indexes and repeat their policies.
For an exact state, both indexes/policies are identical and `upper_weight` is 0.
For interpolation, indexes are adjacent and the crossing is
`(1 - upper_weight) * lower_point + upper_weight * upper_point`, with the exact
rational weight authoritative. Undefined EERs have no bracket.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "oneOf": [{"$ref": "#/$defs/roc"}, {"$ref": "#/$defs/calibration"}],
  "$defs": {
    "probability": {"type": "number", "minimum": 0, "maximum": 1},
    "rational": {"type": "string", "pattern": "^(0|[1-9][0-9]*)(/[1-9][0-9]*)?$"},
    "count": {"type": "integer", "minimum": 0},
    "policy": {
      "oneOf": [
        {
          "type": "object", "additionalProperties": false,
          "required": ["kind", "threshold"],
          "properties": {
            "kind": {"const": "reject_all"},
            "threshold": {"type": "null"}
          }
        },
        {
          "type": "object", "additionalProperties": false,
          "required": ["kind", "threshold"],
          "properties": {
            "kind": {"const": "score_at_least"},
            "threshold": {"type": "number"}
          }
        }
      ]
    },
    "_totals": {
      "type": "object",
      "required": ["n_genuine", "n_impostor", "n_valid_genuine", "n_valid_impostor", "n_failed_genuine", "n_failed_impostor"],
      "properties": {
        "n_genuine": {"type": "integer", "minimum": 1},
        "n_impostor": {"type": "integer", "minimum": 1},
        "n_valid_genuine": {"$ref": "#/$defs/count"},
        "n_valid_impostor": {"$ref": "#/$defs/count"},
        "n_failed_genuine": {"$ref": "#/$defs/count"},
        "n_failed_impostor": {"$ref": "#/$defs/count"}
      }
    },
    "totals": {"$ref": "#/$defs/_totals", "unevaluatedProperties": false},
    "state_counts": {
      "allOf": [
        {"$ref": "#/$defs/_totals"},
        {
          "type": "object", "required": ["tp", "fp", "fn", "tn"],
          "properties": {
            "tp": {"$ref": "#/$defs/count"},
            "fp": {"$ref": "#/$defs/count"},
            "fn": {"$ref": "#/$defs/count"},
            "tn": {"$ref": "#/$defs/count"}
          }
        }
      ],
      "unevaluatedProperties": false
    },
    "_rates": {
      "type": "object", "required": ["far", "far_exact", "frr", "frr_exact"],
      "properties": {
        "far": {"$ref": "#/$defs/probability"},
        "far_exact": {"$ref": "#/$defs/rational"},
        "frr": {"$ref": "#/$defs/probability"},
        "frr_exact": {"$ref": "#/$defs/rational"}
      }
    },
    "_state": {
      "allOf": [
        {"$ref": "#/$defs/_rates"},
        {
          "type": "object", "required": ["policy", "counts"],
          "properties": {
            "policy": {"$ref": "#/$defs/policy"},
            "counts": {"$ref": "#/$defs/state_counts"}
          }
        }
      ]
    },
    "state": {"$ref": "#/$defs/_state", "unevaluatedProperties": false},
    "terminal": {
      "allOf": [
        {"$ref": "#/$defs/_state"},
        {
          "type": "object", "required": ["state_index"],
          "properties": {"state_index": {"$ref": "#/$defs/count"}}
        }
      ],
      "unevaluatedProperties": false
    },
    "coverage_class": {
      "type": "object", "additionalProperties": false,
      "required": ["valid_fraction", "valid_fraction_exact", "failure_fraction", "failure_fraction_exact"],
      "properties": {
        "valid_fraction": {"$ref": "#/$defs/probability"},
        "valid_fraction_exact": {"$ref": "#/$defs/rational"},
        "failure_fraction": {"$ref": "#/$defs/probability"},
        "failure_fraction_exact": {"$ref": "#/$defs/rational"}
      }
    },
    "bracket": {
      "type": "object", "additionalProperties": false,
      "required": ["state_indexes", "policies", "upper_weight", "upper_weight_exact"],
      "properties": {
        "state_indexes": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"$ref": "#/$defs/count"}},
        "policies": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"$ref": "#/$defs/policy"}},
        "upper_weight": {"$ref": "#/$defs/probability"},
        "upper_weight_exact": {"$ref": "#/$defs/rational"}
      }
    },
    "defined_eer": {
      "type": "object", "additionalProperties": false,
      "required": ["status", "eer", "eer_exact", "bracket"],
      "properties": {
        "status": {"enum": ["exact_threshold_state", "interpolated"]},
        "eer": {"$ref": "#/$defs/probability"},
        "eer_exact": {"$ref": "#/$defs/rational"},
        "bracket": {"$ref": "#/$defs/bracket"}
      }
    },
    "undefined_eer": {
      "type": "object", "additionalProperties": false,
      "required": ["status", "eer", "eer_exact", "bracket"],
      "properties": {
        "status": {"enum": ["undefined_no_crossing", "undefined_empty_valid_class"]},
        "eer": {"type": "null"},
        "eer_exact": {"type": "null"},
        "bracket": {"type": "null"}
      }
    },
    "full_eer": {
      "oneOf": [
        {"$ref": "#/$defs/defined_eer"},
        {"$ref": "#/$defs/undefined_eer", "properties": {"status": {"const": "undefined_no_crossing"}}}
      ]
    },
    "conditional_eer": {
      "oneOf": [
        {"$ref": "#/$defs/defined_eer"},
        {"$ref": "#/$defs/undefined_eer", "properties": {"status": {"const": "undefined_empty_valid_class"}}}
      ]
    },
    "roc": {
      "type": "object", "additionalProperties": false,
      "required": ["schema_version", "denominator", "counts", "coverage", "states", "terminal", "eer", "conditional"],
      "properties": {
        "schema_version": {"const": 1},
        "denominator": {"const": "all_original_claims"},
        "counts": {"$ref": "#/$defs/totals"},
        "coverage": {
          "type": "object", "additionalProperties": false,
          "required": ["genuine", "impostor"],
          "properties": {
            "genuine": {"$ref": "#/$defs/coverage_class"},
            "impostor": {"$ref": "#/$defs/coverage_class"}
          }
        },
        "states": {"type": "array", "minItems": 1, "items": {"$ref": "#/$defs/state"}},
        "terminal": {"$ref": "#/$defs/terminal"},
        "eer": {"$ref": "#/$defs/full_eer"},
        "conditional": {
          "type": "object", "additionalProperties": false,
          "required": ["denominator", "eer"],
          "properties": {
            "denominator": {"const": "valid_claims_only"},
            "eer": {"$ref": "#/$defs/conditional_eer"}
          }
        }
      }
    },
    "calibration": {
      "allOf": [
        {"$ref": "#/$defs/_state"},
        {
          "type": "object",
          "required": ["schema_version", "denominator", "target_far", "target_far_exact", "selection_contract", "state_index"],
          "properties": {
            "schema_version": {"const": 1},
            "denominator": {"const": "all_original_claims"},
            "target_far": {"$ref": "#/$defs/probability"},
            "target_far_exact": {"$ref": "#/$defs/rational"},
            "selection_contract": {"const": "most_permissive_attainable_state_with_empirical_far_at_most_target"},
            "state_index": {"$ref": "#/$defs/count"}
          }
        }
      ],
      "unevaluatedProperties": false
    }
  }
}
```

## Synthetic verification and review boundary

`tests/test_gated_roc.py` independently enumerates all original binary label
vectors of lengths 2–4 containing both classes and all combinations of `None`
and two tied score values. It checks decisions, original/valid/failure counts,
exact rates, coverage identities, full and conditional crossing references,
brackets, and calibration at three exact FAR caps. All-valid cases also compare
EER with `engine.robustness_selection.ranking_metrics`.

Explicit regressions cover the .6 exact-1/3 conditional state, low coverage with
no crossing, empty-valid-class boundary crossings, all-invalid reject-all, tied
states, extreme finite float64 thresholds, representation-induced ties, exact
FAR cutoffs, nullable/strict-mask/policy validation, and input immutability.
No real data, archives, score caches, study runners, or fits are needed. The
implementation requires independent root review before publication/integration;
these checks do not authorize a DEV cycle experiment.
