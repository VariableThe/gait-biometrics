"""Reference normalization, graph preservation, and cohort separation."""

import numpy as np
import pytest

from engine.claimant_normalization import normalize_claimant_scores
from engine.secure_enrollment import score_secure_trials
from models.gait_metric import WithinClassMetric


def population(ids):
    rng = np.random.default_rng(sum(ids))
    values, subjects, days, trials = [], [], [], []
    for subject in ids:
        center = rng.normal(size=4)
        for day in (1, 2):
            for trial in (1, 2, 3):
                values.append(center + rng.normal(scale=0.3, size=4))
                subjects.append(subject)
                days.append(day)
                trials.append(trial)
    return tuple(np.array(value) for value in (values, subjects, days, trials))


def kwargs(data):
    return dict(
        zip(
            ("cohort_embeddings", "cohort_subjects", "cohort_days", "cohort_trials"),
            data,
            strict=True,
        )
    )


def test_raw_exactly_matches_existing_blend():
    target = population([1, 2])
    expected = score_secure_trials(*target, method="mean_multi")
    actual = normalize_claimant_scores(*target)
    assert np.array_equal(actual[0], expected[0])
    assert np.array_equal(actual[1], expected[1])


@pytest.mark.parametrize("mode", ["z", "s"])
def test_scalar_reference_and_common_graph(mode):
    target, cohort = population([1, 2]), population([11, 12, 13])
    labels, scores, meta = normalize_claimant_scores(
        *target, normalization=mode, **kwargs(cohort)
    )
    baseline = score_secure_trials(*target, method="mean_multi")
    assert np.array_equal(labels, baseline[0])
    assert meta["coverage"] == baseline[2]["coverage"]
    for index, raw in enumerate(meta["raw_scores"]):
        claim = str(meta["claim_subject_ids"][index])
        z = meta["z_stats"][claim]
        expected = (raw - z["mean"]) / z["denominator"]
        if mode == "s":
            key = f"{meta['probe_subject_ids'][index]}:{meta['probe_trial_ids'][index]}"
            t = meta["t_stats"][key]
            expected = 0.5 * expected + 0.5 * (raw - t["mean"]) / t["denominator"]
        assert scores[index] == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("mode", ["z", "s"])
def test_other_target_probes_do_not_change_existing_scores(mode):
    target, cohort = population([1, 2]), population([11, 12, 13])
    _labels, scores, meta = normalize_claimant_scores(
        *target, normalization=mode, **kwargs(cohort)
    )
    changed = list(target)
    changed[0] = target[0].copy()
    changed[0][(target[1] == 2) & (target[2] == 2)] *= -1
    _, updated, updated_meta = normalize_claimant_scores(
        *changed, normalization=mode, **kwargs(cohort)
    )
    mask = np.array(meta["probe_subject_ids"]) == 1
    assert np.array_equal(scores[mask], updated[mask])
    assert meta["z_stats"] == updated_meta["z_stats"]


@pytest.mark.parametrize("mode", ["z", "s"])
def test_hand_calculated_orthogonal_population(mode):
    # Independent reference: each cohort contributes two day2 probes. Target
    # scores against them are [-2,-2,-1,-1], mean=-1.5 and population std=.5.
    def fixed_population(ids, vectors):
        values, subjects, days, trials = [], [], [], []
        for subject, vector in zip(ids, vectors, strict=True):
            for day in (1, 2):
                for trial in (1, 2):
                    values.append(vector)
                    subjects.append(subject)
                    days.append(day)
                    trials.append(trial)
        return tuple(np.asarray(item) for item in (values, subjects, days, trials))

    target = fixed_population([1, 2], [(1, 0), (0, 1)])
    cohort = fixed_population([11, 12], [(-1, 0), (0, -1)])
    labels, scores, metadata = normalize_claimant_scores(
        *target, normalization=mode, **kwargs(cohort)
    )
    assert np.array_equal(scores, np.where(labels == 1, 3.0, 1.0))
    for stats in metadata["z_stats"].values():
        assert stats["mean"] == -1.5
        assert stats["std"] == 0.5
        assert stats["n_scores"] == 4
    for stats in metadata["t_stats"].values():
        assert stats["mean"] == -1.5
        assert stats["std"] == 0.5
        assert stats["n_scores"] == 2


@pytest.mark.parametrize("floor", [False, 0, -1, float("nan"), float("inf")])
def test_invalid_floor_rejected(floor):
    with pytest.raises(ValueError):
        normalize_claimant_scores(
            *population([1, 2]),
            normalization="z",
            std_floor=floor,
            **kwargs(population([11, 12])),
        )


def test_overlapping_cohort_ids_rejected():
    with pytest.raises(ValueError, match="disjoint"):
        normalize_claimant_scores(
            *population([1, 2]), normalization="z", **kwargs(population([2, 3]))
        )


@pytest.mark.parametrize("mode", ["z", "s"])
@pytest.mark.parametrize("day", [1, 2])
def test_one_cancelled_cohort_trial_cannot_silently_shrink_population(mode, day):
    cohort = list(population([11, 12, 13]))
    row = np.flatnonzero((cohort[1] == 11) & (cohort[2] == day) & (cohort[3] == 1))[0]
    cohort[0] = np.concatenate((cohort[0], -cohort[0][row : row + 1]))
    for column in (1, 2, 3):
        cohort[column] = np.concatenate((cohort[column], cohort[column][row : row + 1]))
    with pytest.raises(ValueError, match="complete source cohort trial"):
        normalize_claimant_scores(
            *population([1, 2]), normalization=mode, **kwargs(cohort)
        )


@pytest.mark.parametrize(
    "shrinkage",
    [
        True,
        np.bool_(True),
        "0.2",
        [0.2],
        np.array(0.2),
        float("nan"),
        float("inf"),
        0,
        -1,
        1.1,
    ],
)
def test_metric_shrinkage_requires_real_valid_scalar(shrinkage):
    with pytest.raises(ValueError, match="shrinkage"):
        WithinClassMetric(shrinkage=shrinkage)
