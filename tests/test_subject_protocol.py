"""NumPy-only controls: no data loading, optimizer work, or test cohorts."""

import json

import numpy as np
import pytest

from engine.subject_protocol import (
    fold_summary,
    make_subject_folds,
    per_subject_error_rates,
)


def test_partition_is_order_and_window_multiplicity_invariant():
    ids = np.arange(10, 31, dtype=np.int64) * 7
    first = make_subject_folds(ids)
    repeated = make_subject_folds(np.repeat(ids[::-1], np.arange(1, 22)))
    assert first == repeated
    assert first == make_subject_folds(ids, np.int64(4), np.int64(3), np.int64(2026))
    json.dumps(first, allow_nan=False)
    assert first != make_subject_folds(ids, seed=2027)


@pytest.mark.parametrize(
    "n_subjects,n_calibration,n_folds", [(10, 4, 3), (21, 4, 3), (13, 3, 5), (6, 2, 2)]
)
def test_exact_coverage_balanced_folds_and_calibration_exclusion(
    n_subjects, n_calibration, n_folds
):
    # Deliberately noncontiguous authentic encoded values: never replace by ranks.
    ids = np.arange(n_subjects, dtype=np.uint32) * 11 + 100
    split = make_subject_folds(ids, n_calibration, n_folds)
    all_ids = set(map(int, ids))
    development = set(split["development_subject_ids"])
    calibration = set(split["calibration_subject_ids"])
    assert development | calibration == all_ids
    assert not development & calibration
    assert len(calibration) == n_calibration
    seen, sizes = [], []
    for index, fold in enumerate(split["folds"]):
        training, validation = (
            set(fold["train_subject_ids"]),
            set(fold["val_subject_ids"]),
        )
        assert fold["fold"] == index
        assert len(training) >= 2 and len(validation) >= 2
        assert not training & validation
        assert not (training | validation) & calibration
        assert training | validation == development
        assert fold["train_subject_ids"] == sorted(training)
        assert fold["val_subject_ids"] == sorted(validation)
        seen.extend(validation)
        sizes.append(len(validation))
    assert sorted(seen) == sorted(development)
    assert max(sizes) - min(sizes) <= 1


def test_calibration_is_first_seeded_permutation_and_independent_of_fold_count():
    ids = np.arange(100, 124)
    expected = sorted(map(int, np.random.default_rng(2026).permutation(ids)[:4]))
    for n_folds in (2, 3, 4, 5):
        assert (
            make_subject_folds(ids, n_folds=n_folds)["calibration_subject_ids"]
            == expected
        )


@pytest.mark.parametrize(
    "ids",
    [
        [],
        [[0, 1]],
        [0.0, 1.0],
        [True, False],
        [-1, 2],
        ["1", "2"],
        np.array([1], dtype=object),
    ],
)
def test_invalid_subject_labels_rejected(ids):
    with pytest.raises(ValueError):
        make_subject_folds(ids)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"calibration_subjects": 0},
        {"calibration_subjects": 1},
        {"calibration_subjects": True},
        {"calibration_subjects": 2.5},
        {"n_folds": 1},
        {"n_folds": 2.0},
        {"n_folds": False},
        {"seed": -1},
        {"seed": "2026"},
        {"seed": True},
    ],
)
def test_invalid_partition_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        make_subject_folds(np.arange(24), **kwargs)


def test_insufficient_unique_subjects_fail_without_silent_fold_reduction():
    with pytest.raises(ValueError, match="at least 10 unique subjects"):
        make_subject_folds(np.repeat(np.arange(9), 100))


def test_summary_equal_fold_and_seed_weighting_preserves_all_runs():
    rows = [
        {"fold": 1, "seed": 43, "eer": 0.8, "roc_auc": 0.2},
        {"fold": 0, "seed": 42, "eer": 0.1, "roc_auc": 0.9},
        {"fold": 1, "seed": 42, "eer": 0.6, "roc_auc": 0.4},
        {"fold": 0, "seed": 43, "eer": 0.3, "roc_auc": 0.7},
    ]
    original = json.dumps(rows, sort_keys=True)
    result = fold_summary(rows)
    assert result["mean"] == pytest.approx({"eer": 0.45, "roc_auc": 0.55})
    assert [r["mean"]["eer"] for r in result["per_fold"]] == pytest.approx([0.2, 0.7])
    assert [r["mean"]["eer"] for r in result["per_seed"]] == pytest.approx([0.35, 0.55])
    assert result["fold_sample_sd"]["eer"] == pytest.approx(0.5 / 2**0.5)
    assert result["seed_sample_sd"]["eer"] == pytest.approx(0.2 / 2**0.5)
    assert result["all_fold_results"] == sorted(
        rows, key=lambda r: (r["fold"], r["seed"])
    )
    assert (result["n_folds"], result["n_seeds"], result["n_runs"]) == (2, 2, 4)
    assert "CI" in result["uncertainty"]
    assert fold_summary(rows[::-1]) == result
    assert json.dumps(rows, sort_keys=True) == original
    json.dumps(result, allow_nan=False)


def test_single_run_summary_uses_null_for_undefined_sample_sd():
    result = fold_summary(
        [{"fold": np.int64(0), "seed": np.int32(42), "eer": np.float32(0.25)}]
    )
    assert result["mean"]["eer"] == 0.25
    assert result["fold_sample_sd"] == result["seed_sample_sd"] == {"eer": None}
    json.dumps(result, allow_nan=False)


def test_finite_large_metrics_do_not_overflow_summary_arithmetic():
    rows = [{"fold": fold, "seed": 42, "metric": 1e308} for fold in range(3)]
    result = fold_summary(rows)
    assert result["mean"]["metric"] == 1e308
    assert result["fold_sample_sd"]["metric"] == 0
    json.dumps(result, allow_nan=False)


def test_unrepresentable_sample_sd_fails_instead_of_emitting_json_infinity():
    largest = np.finfo(np.float64).max
    rows = [
        {"fold": 0, "seed": 42, "metric": -largest},
        {"fold": 1, "seed": 42, "metric": largest},
    ]
    with pytest.raises(ValueError, match="finite JSON range"):
        fold_summary(rows)


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [{}],
        [{"fold": 0, "seed": 42}],
        [{"fold": True, "seed": 42, "eer": 0.1}],
        [{"fold": 0, "seed": 42, "eer": True}],
        [{"fold": 0, "seed": 42, "eer": np.nan}],
        [{"fold": 0, "seed": 42, "eer": np.inf}],
        [{"fold": 0, "seed": 42, "eer": "0.1"}],
        [{"fold": 0, "seed": 42, "eer": [0.1]}],
        [{"fold": 0, "seed": 42, "eer": 0.1}] * 2,
        [{"fold": 0, "seed": 42, "eer": 0.1}, {"fold": 1, "seed": 43, "eer": 0.2}],
        [{"fold": 0, "seed": 42, "eer": 0.1}, {"fold": 1, "seed": 42, "auc": 0.8}],
    ],
)
def test_summary_rejects_invalid_duplicate_or_incomplete_runs(rows):
    with pytest.raises(ValueError):
        fold_summary(rows)


def test_subject_errors_attribute_false_accepts_to_both_different_roles():
    # Two probes from ID 10, one from ID 25; every probe claims both IDs.
    claims = [10, 25, 10, 25, 10, 25]
    probes = [10, 10, 10, 10, 25, 25]
    labels = [1, 0, 1, 0, 0, 1]
    scores = [0.5, 0.8, 0.4, 0.2, 0.3, 0.9]
    result = per_subject_error_rates(labels, scores, claims, probes, 0.5)
    assert result["micro"] == {
        "n_genuine": 3,
        "n_impostor": 3,
        "false_accepts": 1,
        "false_rejects": 1,
        "far": 1 / 3,
        "frr": 1 / 3,
    }
    claimed = result["by_claim_subject"]["per_subject"]
    probed = result["by_probe_subject"]["per_subject"]
    assert claimed["25"]["far"] == 0.5 and claimed["10"]["far"] == 0
    assert probed["10"]["far"] == 0.5 and probed["25"]["far"] == 0
    assert claimed["10"]["frr"] == probed["10"]["frr"] == 0.5
    assert result["by_claim_subject"]["macro"]["far"] == 0.25
    assert result["by_probe_subject"]["macro"]["frr"] == 0.25
    json.dumps(result, allow_nan=False)


def test_subject_errors_missing_denominators_are_null_not_zero():
    result = per_subject_error_rates([0, 0], [0.1, 0.9], [10, 25], [25, 10], 0.5)
    assert result["micro"]["frr"] is None
    assert result["by_probe_subject"]["per_subject"]["10"]["frr"] is None
    assert result["by_probe_subject"]["macro"]["n_subjects_with_frr"] == 0
    assert result["by_probe_subject"]["macro"]["frr"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "change",
    [
        {"labels": [0, 0]},
        {"labels": [2, 0]},
        {"labels": [[1, 0]]},
        {"scores": [0.1]},
        {"scores": [np.nan, 0.5]},
        {"claim_subject_ids": [10.0, 25.0]},
        {"probe_subject_ids": [10, -1]},
        {"threshold": np.inf},
        {"threshold": True},
        {"threshold": "0.5"},
    ],
)
def test_subject_errors_reject_malformed_comparisons(change):
    args = {
        "labels": [1, 0],
        "scores": [0.5, 0.2],
        "claim_subject_ids": [10, 25],
        "probe_subject_ids": [10, 10],
        "threshold": 0.5,
    }
    args.update(change)
    with pytest.raises(ValueError):
        per_subject_error_rates(**args)
