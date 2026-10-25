"""New full-support estimator; pure array APIs, no file/dataset access."""

import hashlib
import math
from contextlib import contextmanager
from fractions import Fraction

import common as c
import numpy as np
from engine.empirical import empirical_threshold
from engine.historical_metrics import report
from engine.robustness_selection import ranking_metrics, summarize_nuisance_study
from engine.secure_enrollment import score_secure_trials
from models.gait_canonical import extract_canonical_features
from models.gait_metric import WithinClassMetric
from models.lag_spectrum import extract_lagged_covariance_features
from planning import CONDITIONS
from sklearn.covariance import LedoitWolf


def array_hash(values):
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256(c.canonical([str(array.dtype), list(array.shape)]))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def physical_key(segment, row, channel):
    return c.canonical(
        [
            "physical-row-normal-v1",
            20261025,
            segment["recording_key"],
            segment["member_sha256"],
            int(row),
            channel,
        ]
    )


def physical_field(segment, rows):
    """One independent key per physical row/channel; no length/batch/cover key.

    The same key always yields the same float64 draw, including unordered and
    overlapping calls. This is a NEW realization, not historical window noise.
    """
    rows = np.asarray(rows)
    c.require(
        rows.ndim == 1 and rows.dtype.kind in "iu" and np.all(rows >= 0),
        "physical nonnegative row IDs required",
    )
    result = np.empty((6, len(rows)), dtype=np.float64)
    for j, row in enumerate(rows):
        for channel in range(6):
            digest = hashlib.sha256(physical_key(segment, row, channel)).digest()
            seed = int.from_bytes(digest[:16], "little")
            result[channel, j] = np.random.Generator(
                np.random.PCG64(seed)
            ).standard_normal()
    return result


def rotation(degrees):
    # Exactly the frozen helper's Rodrigues arithmetic at axis (1,0,0).
    x, y, z = (1.0, 0.0, 0.0)
    skew = np.array(((0, -z, y), (z, 0, -x), (-y, x, 0)), dtype=np.float64)
    angle = math.radians(degrees)
    return np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)


def perturb(segment, source, condition, scale, field):
    c.require(
        condition in CONDITIONS
        and source.shape == field.shape
        and source.dtype == np.float32
        and np.isfinite(source).all(),
        "finite float32 complete source and registered condition required",
    )
    if condition == "clean" or segment["day"] == 1:
        return source.copy()
    work = source.astype(np.float64)
    if condition.startswith("rotation"):
        matrix = rotation(float(condition[8:]))
        work[:3] = matrix @ work[:3]
        work[3:] = matrix @ work[3:]
    elif condition.startswith("gain"):
        work *= 0.9 if condition == "gain09" else 1.1
    else:
        c.require(
            np.asarray(scale).shape == (6,)
            and np.all(np.isfinite(scale))
            and np.all(np.asarray(scale) > 0),
            "positive TRAIN12 scale required",
        )
        work += (
            field
            * (np.asarray(scale) * (0.01 if condition == "noise01" else 0.05))[:, None]
        )
    result = work.astype(np.float32)
    c.require(np.isfinite(result).all(), "invalid perturbed source")
    return result


def slice_windows(metadata, segments):
    windows = []
    for w in metadata["windows"]:
        unit = w["unit"]
        start = metadata["eligible"][unit]["original"]["start_row"]
        a, b = w["rows"]
        values = segments[unit][:, a - start : b - start]
        c.require(values.shape == (6, 256), "required source interval missing")
        windows.append(values)
    return np.stack(windows)


def training_rms(windows, metadata, ids):
    mask = np.array(
        [w["historical"] and w["source_key"][0] in ids for w in metadata["windows"]]
    )
    c.require(
        {
            w["source_key"][0]
            for w, chosen in zip(metadata["windows"], mask, strict=True)
            if chosen
        }
        == set(ids),
        "all TRAIN12 source owners required",
    )
    values = windows[mask].astype(np.float64)
    values -= values.mean(axis=2, keepdims=True)
    scales = [
        float(np.sqrt(np.mean(values[:, start : start + 3] ** 2))) for start in (0, 3)
    ]
    result = np.repeat(scales, 3)
    return result, {
        "ids": list(ids),
        "historical_window_indices": np.flatnonzero(mask).tolist(),
        "historical_windows": int(mask.sum()),
        "source_columns": ["acc x", "acc y", "acc z", "gyro x", "gyro y", "gyro z"],
        "native_units": ["m/s^2"] * 3 + ["degrees/sec"] * 3,
        "valid": bool(np.isfinite(result).all() and np.all(result > 0)),
        "values_nullable": [float(v) if np.isfinite(v) else None for v in result],
    }


def preflight(windows, keys):
    """Never raise early on an individual window; persist every named diagnostic.

    Caller must write this complete result and only THEN enforce all valid.
    Geometry NaNs and reasons remain lossless arrays; JSON contains nullable
    values and masks. There is no success-subset path.
    """
    features = np.full((len(windows), 362), np.nan, dtype=np.float64)
    geometry = np.full((len(windows), 3), np.nan, dtype=np.float64)
    frames = np.full((len(windows), 2, 3, 3), np.nan, dtype=np.float64)
    valid = np.zeros(len(windows), dtype=bool)
    diagnostics = []
    for i, (window, key) in enumerate(zip(windows, keys, strict=True)):
        reason = "valid"
        try:
            value = extract_canonical_features(window[None], 100)
            g = value.geometry
            geometry[i] = [
                g.gravity_ratio[0],
                g.horizontal_energy_ratio[0],
                g.eigengap_ratio[0],
            ]
            frames[i] = g.frames[0]
            if not bool(g.valid[0]):
                reason = str(g.reason[0])
            else:
                features[i] = np.concatenate(
                    (
                        value.features.astype(np.float64),
                        extract_lagged_covariance_features(window[None]).astype(
                            np.float64
                        ),
                    ),
                    axis=1,
                )[0]
                if not np.isfinite(features[i]).all():
                    reason = "nonfinite_features"
                else:
                    valid[i] = True
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
            reason = type(error).__name__ + ": " + str(error)
        diagnostics.append(
            {
                "window_index": i,
                "source_key": key,
                "valid": bool(valid[i]),
                "reason": reason,
                "geometry_finite": np.isfinite(geometry[i]).tolist(),
                "geometry_nullable": [
                    float(v) if np.isfinite(v) else None for v in geometry[i]
                ],
            }
        )
    return {
        "features": features,
        "geometry": geometry,
        "frames": frames,
        "valid": valid,
    }, diagnostics


def aggregate(features, metadata, arm):
    c.require(
        arm in ("H", "F")
        and features.shape[0] == len(metadata["windows"])
        and features.dtype == np.float64
        and np.isfinite(features).all(),
        "finite float64 complete featurematrix required",
    )
    result = np.zeros((len(metadata["eligible"]), features.shape[1]), dtype=np.float64)
    for unit in range(len(result)):
        indices = [
            i
            for i, w in enumerate(metadata["windows"])
            if w["unit"] == unit and (arm == "F" or w["historical"])
        ]
        c.require(bool(indices), "missing required trial unit")
        if arm == "H":
            result[unit] = features[indices].mean(axis=0, dtype=np.float64)
        else:
            for (
                index
            ) in indices:  # deterministic registered cover order; no BLAS reduction
                result[unit] += features[index] * float(
                    Fraction(metadata["windows"][index]["weight_exact"])
                )
    c.require(np.isfinite(result).all(), "invalid trial descriptor")
    return result


def arm_data(features, metadata, arm):
    if arm == "R":
        indices = [i for i, w in enumerate(metadata["windows"]) if w["historical"]]
        labels = np.array(
            [metadata["windows"][i]["source_key"][:3] for i in indices], dtype=np.int64
        )
        values = features[indices]
    else:
        labels = np.asarray(metadata["trial_units"], dtype=np.int64)
        values = aggregate(features, metadata, arm)
    return values, {
        name: labels[:, i] for i, name in enumerate(("subjects", "days", "trials"))
    }


def score(model, values, data, ids, graph):
    mask = np.isin(data["subjects"], ids)
    c.require(
        set(data["subjects"][mask].tolist()) == set(ids), "missing scored identity"
    )
    labels, scores, metadata = score_secure_trials(
        model.transform(values[mask]),
        *(data[k][mask] for k in ("subjects", "days", "trials")),
        method="mean_multi",
    )
    c.same(labels.tolist(), graph["labels"], "all ordered claimlabels")
    for key in ("claim_subject_ids", "probe_subject_ids", "probe_trial_ids"):
        c.same(metadata[key], graph[key], "complete claim graph: " + key)
    c.same(
        {
            str(row["subject"]): row["enrollment_trial_ids"]
            for row in metadata["templates"]
        },
        graph["enrollment_trials"],
        "equal-day1-trial enrollment",
    )
    c.require(
        not metadata["coverage"].get("exclusions"),
        "cancelled trial/template: fail whole comparison",
    )
    return {
        "labels": labels,
        "scores": np.asarray(scores, dtype=np.float64),
        "metadata": metadata,
    }


class FitLedger:
    def __init__(self, assignments):
        self.assignments = assignments
        self.calls = []
        self.cutoffs = {}
        self.events = []

    def freeze_cutoffs(self, fold, cutoffs):
        c.require(
            set(cutoffs) == {"H", "F"} and fold not in self.cutoffs,
            "both own cutoffs must freeze once before either refit",
        )
        self.cutoffs[fold] = c.decode(c.canonical(cutoffs))
        self.events.append(
            {
                "event": "both_cutoffs_frozen",
                "fold": fold,
                "sha256": c.fingerprint(cutoffs),
            }
        )

    def fit(self, arm, fold, role, values, data):
        c.require(arm in ("H", "F"), "R fit forbidden")
        assigned = [
            a
            for a in self.assignments
            if (a["arm"], a["fold"], a["role"]) == (arm, fold, role)
        ]
        c.require(
            len(assigned) == 1
            and not any(
                (a["arm"], a["fold"], a["role"]) == (arm, fold, role)
                for a in self.calls
            ),
            "unique actual fit assignment required",
        )
        a = assigned[0]
        if role == "refit":
            c.require(
                fold in self.cutoffs, "BOTH cutoffs must be frozen BEFORE EITHER refit"
            )
        indices = np.flatnonzero(np.isin(data["subjects"], a["ids"]))
        c.same(
            indices.tolist(), a["unit_indices"], "actual optimizer training trial units"
        )
        c.require(
            len(indices) == a["training_units"]
            and set(data["subjects"][indices].tolist()) == set(a["ids"]),
            "TRAIN ownership/count mismatch",
        )
        record = {
            **a,
            "status": "fit_entered",
            "features_sha256": array_hash(values[indices]),
            "labels_sha256": array_hash(data["subjects"][indices]),
        }
        self.calls.append(record)  # invalid fits consume their slot; no retry
        self.events.append(
            {"event": "fit_entered", "fold": fold, "arm": arm, "role": role}
        )
        model = WithinClassMetric("wccn", "auto").fit(
            values[indices], data["subjects"][indices]
        )
        record["status"] = "fit_completed"
        return model


@contextmanager
def forbid_fits():
    def blocked(*args, **kwargs):
        raise RuntimeError("Estimator fit forbidden during restoration/replay")

    old_metric, old_covariance = WithinClassMetric.fit, LedoitWolf.fit
    WithinClassMetric.fit, LedoitWolf.fit = blocked, blocked
    try:
        yield
    finally:
        WithinClassMetric.fit, LedoitWolf.fit = old_metric, old_covariance


def restore(state, values, data, ids, *, check_statistics=True):
    c.require(
        set(state)
        == {"mean", "scale", "projection", "training_subject_ids", "shrinkage"},
        "full model numeric schema required",
    )
    with forbid_fits():
        model = WithinClassMetric.from_state_dict(state)
    c.same(model.training_subject_ids_, sorted(ids), "model TRAIN assignment")
    selected = np.asarray(values[np.isin(data["subjects"], ids)], dtype=np.float64)
    if check_statistics:
        c.require(
            np.array_equal(model.mean_, selected.mean(axis=0))
            and np.array_equal(model.scale_, selected.std(axis=0).clip(min=1e-6)),
            "restored standardizer differs from assigned clean TRAIN units",
        )
    alpha = state["shrinkage"]
    c.require(
        alpha.shape == () and np.isfinite(alpha) and 0 <= float(alpha) <= 1,
        "invalid auto shrinkage state",
    )
    model.shrinkage_ = float(alpha)
    return model


def replay(fresh, cached, threshold):
    c.require(
        fresh["scores"].dtype == cached["scores"].dtype == np.float64
        and fresh["labels"].dtype == cached["labels"].dtype,
        "lossless score/label dtype mismatch",
    )
    c.require(
        np.array_equal(fresh["labels"], cached["labels"])
        and fresh["scores"].shape == cached["scores"].shape
        and np.isfinite(cached["scores"]).all()
        and np.allclose(fresh["scores"], cached["scores"], atol=1e-6, rtol=1e-6),
        "score replay tolerance mismatch",
    )
    c.same(fresh["metadata"], cached["metadata"], "complete scorer metadata")
    c.require(
        np.array_equal(fresh["scores"] >= threshold, cached["scores"] >= threshold),
        "STOP: frozen decisions changed; no roundoff exception",
    )
    return cached


def pool(items):
    return {
        "labels": np.concatenate([x["labels"] for x in items]),
        "scores": np.concatenate([x["scores"] for x in items]),
    }


def cutoff(items):
    values = pool(items)
    return empirical_threshold(values["labels"], values["scores"])


def row(fold, arm, condition, scored, threshold):
    labels, scores, metadata = scored["labels"], scored["scores"], scored["metadata"]
    pairs = {}
    for claim, probe, label, value in zip(
        metadata["claim_subject_ids"],
        metadata["probe_subject_ids"],
        labels,
        scores,
        strict=True,
    ):
        if not label:
            item = pairs.setdefault(
                f"{claim}<-{probe}", {"claims": 0, "false_accepts": 0}
            )
            item["claims"] += 1
            item["false_accepts"] += int(value >= threshold)
    return {
        "fold": fold,
        "arm": arm,
        "condition": condition,
        "metrics": report(scored, threshold),
        "pair_errors": pairs,
        "ranking": ranking_metrics(
            labels, scores, np.asarray(metadata["probe_subject_ids"], dtype=np.int64)
        ),
        "decision_sha256": hashlib.sha256(
            np.asarray(scores >= threshold, dtype=np.bool_).tobytes()
        ).hexdigest(),
    }


def descriptive_summaries(rows, metadata):
    """All-arm equal-fold/person/count diagnostics; no H nomination rule."""
    result = {}
    mean = lambda values: sum(values, Fraction()) / len(values)
    for arm in ("R", "H", "F"):
        result[arm] = {}
        for condition in CONDITIONS:
            group = [r for r in rows if r["arm"] == arm and r["condition"] == condition]
            fixed = [r["metrics"]["fixed_threshold"] for r in group]
            people, claimants = {}, {}
            for subject in metadata["source_metadata"]["subject_ids"]:
                s = str(subject)
                present = [r for r in group if s in r["metrics"]["per_subject_errors"]]
                c.require(
                    len(present) == 4, "four dependent person appearances required"
                )
                counts = [r["metrics"]["per_subject_errors"][s] for r in present]
                people[s] = {
                    "frr_exact": str(
                        mean(
                            [
                                Fraction(v["false_rejects"], v["n_genuine"])
                                for v in counts
                            ]
                        )
                    ),
                    "eer_exact": str(
                        mean(
                            [
                                Fraction(r["ranking"]["per_subject"][s]["eer_exact"])
                                for r in present
                            ]
                        )
                    ),
                    "auc_exact": str(
                        mean(
                            [
                                Fraction(r["ranking"]["per_subject"][s]["auc_exact"])
                                for r in present
                            ]
                        )
                    ),
                    "repeated_rejects": sum(v["false_rejects"] for v in counts),
                    "repeated_genuine_claims": sum(v["n_genuine"] for v in counts),
                }
                claimants[s] = {
                    "repeated_false_accepts": sum(v["false_accepts"] for v in counts),
                    "repeated_impostor_claims": sum(v["n_impostor"] for v in counts),
                }
            result[arm][condition] = {
                "far_exact": str(
                    mean([Fraction(v["false_accepts"], v["n_impostor"]) for v in fixed])
                ),
                "frr_exact": str(
                    mean([Fraction(v["false_rejects"], v["n_genuine"]) for v in fixed])
                ),
                "eer_exact": str(
                    mean([Fraction(r["ranking"]["eer_exact"]) for r in group])
                ),
                "auc_exact": str(
                    mean([Fraction(r["ranking"]["auc_exact"]) for r in group])
                ),
                "splits_far_above_2pct": sum(
                    Fraction(v["false_accepts"], v["n_impostor"]) > Fraction(1, 50)
                    for v in fixed
                ),
                "repeated_false_accepts": sum(v["false_accepts"] for v in fixed),
                "repeated_false_rejects": sum(v["false_rejects"] for v in fixed),
                "repeated_genuine_claims": sum(v["n_genuine"] for v in fixed),
                "repeated_impostor_claims": sum(v["n_impostor"] for v in fixed),
                "per_person": people,
                "per_claimant": claimants,
            }
    return result


def exact_difference(left, right):
    if type(left) is dict:
        c.require(left.keys() == right.keys(), "matched diagnostic fields required")
        return {key: exact_difference(left[key], right[key]) for key in left}
    if type(left) is str:
        return str(Fraction(left) - Fraction(right))
    c.require(type(left) is type(right) is int, "exact diagnostic counts required")
    return left - right


def comparison(rows, metadata):
    for arm in ("R", "H", "F"):
        group = [r for r in rows if r["arm"] == arm]
        c.require(
            len(group) == 140
            and {(r["fold"], r["condition"]) for r in group}
            == {(f, condition) for f in range(20) for condition in CONDITIONS},
            "complete20x7 arm rows required",
        )
        for condition in CONDITIONS:
            fixed = [
                r["metrics"]["fixed_threshold"]
                for r in group
                if r["condition"] == condition
            ]
            c.require(
                sum(v["n_genuine"] for v in fixed) == 596
                and sum(v["n_impostor"] for v in fixed) == 1192,
                "no partial-success claim denominator",
            )

    def rates(arm, condition):
        group = [
            r["metrics"]["fixed_threshold"]
            for r in rows
            if r["arm"] == arm and r["condition"] == condition
        ]
        return tuple(
            sum((Fraction(v[k], v[n]) for v in group), Fraction()) / 20
            for k, n in (
                ("false_accepts", "n_impostor"),
                ("false_rejects", "n_genuine"),
            )
        )

    far_h, frr_h = rates("H", "clean")
    far_f, frr_f = rates("F", "clean")
    remapped = [
        {**r, "arm": "baseline" if r["arm"] == "R" else "nuisance"}
        for r in rows
        if r["arm"] in ("R", "F")
    ]
    replacement = summarize_nuisance_study(
        remapped,
        dev_ids=metadata["source_metadata"]["subject_ids"],
        folds=metadata["folds"],
    )
    c.require(len(replacement["gates"]) == 33, "authoritative gate inventory changed")
    summaries = descriptive_summaries(rows, metadata)
    secondary = {
        arm + "-R": exact_difference(summaries[arm], summaries["R"])
        for arm in ("H", "F")
    }
    return {
        "primary": {
            "F_minus_H_clean_equalfold_FRR_exact": str(frr_f - frr_h),
            "F_clean_FAR_exact": str(far_f),
            "H_clean_FAR_exact": str(far_h),
            "favors_support": frr_f < frr_h
            and far_f <= far_h
            and far_f <= Fraction(1, 100),
        },
        "replacement": replacement,
        "nominee": "F" if replacement["winner"] == "nuisance" else "R",
        "arm_summaries": summaries,
        "secondary": secondary,
        "H_nomination_permitted": False,
    }
