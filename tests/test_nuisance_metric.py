"""Tiny synthetic tests only; no datasets, NPZ files, CUDA, or study runner."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from models.gait_metric import WithinClassMetric
from models.nuisance_metric import fit_nuisance_metric


def _clean():
    return np.array([[-2.0, -2.0], [0.0, 2.0], [2.0, -2.0], [0.0, 2.0]])


def _subjects():
    return np.array([3, 9, 9, 9])


def _views(clean, delta):
    return clean[:, None, :] + delta * clean.std(axis=0).clip(min=1e-6)


@pytest.mark.parametrize(
    ("features", "subjects"),
    [
        (_clean(), _subjects()),
        (_clean().astype(np.float32), _subjects()),
        (_clean().astype(np.int64), _subjects()),
        (np.array([[0, 0], [2, 0], [5, 0], [7, 0]]), [2, 2, 7, 7]),
        (np.zeros((4, 2)), [0, 1, 1, 1]),
        (np.array([[1.0], [3.0]]), [0, 1]),
        (
            np.array([[1.5, 2, -1], [3, -4, 2], [-2, 1, 4], [7, 3, -3], [2, 1, 2]]),
            [7, 2, 7, 2, 7],
        ),
    ],
)
def test_baseline_is_bitwise_identical_to_existing_automatic_metric(features, subjects):
    expected = WithinClassMetric().fit(features, subjects)
    actual, diagnostics = fit_nuisance_metric(features, subjects)
    assert type(actual) is WithinClassMetric
    assert actual.mode == "wccn"
    assert actual.shrinkage == "auto"
    for field in ("mean_", "scale_", "projection_"):
        first, second = getattr(actual, field), getattr(expected, field)
        assert first.dtype == second.dtype
        assert first.shape == second.shape
        assert first.tobytes() == second.tobytes()
    assert (
        np.float64(actual.shrinkage_).tobytes()
        == np.float64(expected.shrinkage_).tobytes()
    )
    assert actual.training_subject_ids_ == expected.training_subject_ids_
    np.testing.assert_array_equal(
        diagnostics["nuisance_second_moment"],
        np.zeros_like(diagnostics["clean_covariance"]),
    )
    assert diagnostics["views_per_row"] == 0


def test_hand_computed_identity_balanced_four_view_second_moment():
    clean = _clean()
    delta = np.array(
        [
            [[2, 0], [-2, 0], [0, 2], [0, -2]],
            [[0, 1]] * 4,
            [[1, 1]] * 4,
            [[0, 2]] * 4,
        ],
        dtype=np.float64,
    )
    views = _views(clean, delta)
    _, diagnostics = fit_nuisance_metric(clean, _subjects(), views)
    # ID 3: diag(2,2). ID 9: [[1/3,1/3],[1/3,2]]. Average IDs.
    expected = np.array([[7 / 6, 1 / 6], [1 / 6, 2]])
    np.testing.assert_allclose(
        diagnostics["nuisance_second_moment"], expected, atol=1e-15
    )
    row_weighted = np.einsum("nvi,nvj->ij", delta, delta) / 16
    assert not np.allclose(expected, row_weighted)
    # Neither clean rows nor views are changed; read-only arrays also work.
    clean.setflags(write=False)
    views.setflags(write=False)
    _, repeated = fit_nuisance_metric(clean, _subjects(), views)
    np.testing.assert_array_equal(
        diagnostics["nuisance_second_moment"], repeated["nuisance_second_moment"]
    )


def test_constant_corruption_bias_is_not_centered_away():
    clean = _clean()
    delta = np.broadcast_to([2.0, -1.0], (4, 4, 2))
    _, diagnostics = fit_nuisance_metric(clean, _subjects(), _views(clean, delta))
    np.testing.assert_allclose(
        diagnostics["nuisance_second_moment"], [[4, -2], [-2, 1]], atol=1e-15
    )


def test_clean_statistics_covariance_and_shrinkage_are_unchanged_between_arms():
    clean = _clean()
    baseline, first = fit_nuisance_metric(clean, _subjects())
    delta = np.arange(32, dtype=np.float64).reshape(4, 4, 2) / 7
    augmented, second = fit_nuisance_metric(clean, _subjects(), _views(clean, delta))
    for field in ("mean_", "scale_"):
        np.testing.assert_array_equal(
            getattr(baseline, field), getattr(augmented, field)
        )
    assert baseline.shrinkage_ == augmented.shrinkage_
    np.testing.assert_array_equal(first["clean_covariance"], second["clean_covariance"])
    assert augmented.training_subject_ids_ == baseline.training_subject_ids_ == [3, 9]
    assert not np.allclose(augmented.projection_, baseline.projection_)


def test_zero_displacements_preserve_baseline_projection():
    clean = _clean()
    baseline, first = fit_nuisance_metric(clean, _subjects())
    augmented, second = fit_nuisance_metric(
        clean, _subjects(), np.repeat(clean[:, None, :], 4, axis=1)
    )
    np.testing.assert_array_equal(baseline.projection_, augmented.projection_)
    np.testing.assert_array_equal(first["eigenvalues"], second["eigenvalues"])
    np.testing.assert_array_equal(second["nuisance_second_moment"], np.zeros((2, 2)))
    assert second["views_per_row"] == 4


def test_analytic_whitening_and_existing_transform_state_roundtrip():
    clean = np.array([[-1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [1.0, -1.0]])
    subjects = [0, 0, 1, 1]
    views = _views(clean, np.broadcast_to([1.0, 2.0], (4, 4, 2)))
    metric, diagnostics = fit_nuisance_metric(clean, subjects, views)
    np.testing.assert_allclose(diagnostics["clean_covariance"], np.eye(2), atol=1e-15)
    # I + uu^T for u=(1,2) has eigenvalues 1 and 6, so its inverse
    # square root is I + (1/sqrt(6)-1) uu^T / 5.
    expected = np.eye(2) + (1 / np.sqrt(6) - 1) * np.array([[1, 2], [2, 4]]) / 5
    np.testing.assert_allclose(metric.projection_, expected, atol=1e-15)
    np.testing.assert_allclose(diagnostics["eigenvalues"], [1, 6], atol=1e-15)
    assert diagnostics["eigenvalue_floor"] == pytest.approx(6e-5)
    covariance = diagnostics["clean_covariance"] + diagnostics["nuisance_second_moment"]
    np.testing.assert_allclose(expected @ covariance @ expected, np.eye(2), atol=1e-15)
    query = np.array([[3.0, 1.0], [-2.0, 4.0]])
    transformed = query @ expected
    transformed /= np.linalg.norm(transformed, axis=1, keepdims=True)
    np.testing.assert_array_equal(
        metric.transform(query), transformed.astype(np.float32)
    )
    state = metric.state_dict()
    restored = WithinClassMetric.from_state_dict(state)
    np.testing.assert_array_equal(restored.transform(query), metric.transform(query))
    np.testing.assert_allclose(
        np.linalg.norm(restored.transform(query), axis=1), 1, atol=1e-7
    )
    for key, value in state.items():
        np.testing.assert_array_equal(value, metric.state_dict()[key])


def test_zero_clean_covariance_is_not_reconstructed_from_floored_projection():
    clean = np.array([[1.0, 0.0], [3.0, 0.0]])
    _, diagnostics = fit_nuisance_metric(clean, [3, 9])
    np.testing.assert_array_equal(diagnostics["clean_covariance"], np.zeros((2, 2)))
    np.testing.assert_array_equal(diagnostics["eigenvalues"], [0, 0])
    assert diagnostics["eigenvalue_floor"] == 1e-6
    views = _views(clean, np.broadcast_to([0.0, 1.0], (2, 4, 2)))
    metric, diagnostics = fit_nuisance_metric(clean, [3, 9], views)
    np.testing.assert_array_equal(diagnostics["clean_covariance"], np.zeros((2, 2)))
    np.testing.assert_array_equal(
        diagnostics["nuisance_second_moment"], np.diag([0, 1])
    )
    np.testing.assert_array_equal(diagnostics["eigenvalues"], [0, 1])
    assert diagnostics["eigenvalue_floor"] == 1e-5
    np.testing.assert_allclose(
        metric.projection_, np.diag([np.sqrt(1e5), 1]), atol=1e-12
    )


@pytest.mark.parametrize("with_views", [False, True])
def test_diagnostics_are_lossless_finite_arrays_with_exact_keys_and_original_counts(
    with_views,
):
    clean = _clean()
    views = _views(clean, np.arange(32).reshape(4, 4, 2) / 8) if with_views else None
    metric, diagnostics = fit_nuisance_metric(clean, _subjects(), views)
    assert set(diagnostics) == {
        "clean_covariance",
        "nuisance_second_moment",
        "eigenvalues",
        "eigenvalue_floor",
        "original_rows",
        "original_subjects",
        "views_per_row",
    }
    for key, value in diagnostics.items():
        assert isinstance(value, np.ndarray), key
        assert np.isfinite(value).all(), key
        assert value.dtype == (
            np.int64
            if key in ("original_rows", "original_subjects", "views_per_row")
            else np.float64
        ), key
    for key, expected in (
        ("original_rows", 4),
        ("original_subjects", 2),
        ("views_per_row", 4 if with_views else 0),
    ):
        assert diagnostics[key].shape == ()
        assert diagnostics[key] == expected
    assert diagnostics["eigenvalue_floor"].shape == ()
    assert diagnostics["eigenvalues"].shape == (2,)
    for key in ("clean_covariance", "nuisance_second_moment"):
        matrix = diagnostics[key]
        assert matrix.shape == (2, 2)
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-15)
        assert np.linalg.eigvalsh(matrix).min() >= -1e-14
    total = diagnostics["clean_covariance"] + diagnostics["nuisance_second_moment"]
    np.testing.assert_allclose(
        diagnostics["eigenvalues"], np.linalg.eigvalsh(total), atol=1e-14
    )
    np.testing.assert_allclose(metric.projection_, metric.projection_.T, atol=1e-14)


@pytest.mark.parametrize(
    "subjects",
    [
        None,
        [0, 0, 0, 0],
        [0, 1, 2],
        [[0], [1], [2], [3]],
        [0.0, 0.0, 1.0, 1.0],
        [False, False, True, True],
        [-1, 0, 1, 1],
        np.array([0, 0, 1, 2**63], dtype=np.uint64),
        [0j, 0j, 1j, 1j],
        ["0", "0", "1", "1"],
        np.array([0, 0, 1, 1], dtype=object),
    ],
)
def test_rejects_invalid_subject_ids(subjects):
    with pytest.raises(ValueError):
        fit_nuisance_metric(_clean(), subjects)


@pytest.mark.parametrize(
    "features",
    [
        None,
        1,
        np.empty((0, 2)),
        np.empty((4, 0)),
        np.ones(4),
        np.ones((4, 1, 2)),
        [[1, 2], [3]],
        np.full((4, 2), np.nan),
        np.full((4, 2), np.inf),
        np.full((4, 2), -np.inf),
        np.ones((4, 2), dtype=bool),
        np.ones((4, 2), dtype=complex),
        np.full((4, 2), "1"),
        np.ones((4, 2), dtype=object),
    ],
)
def test_rejects_invalid_clean_feature_types_shapes_and_missing_values(features):
    with pytest.raises(ValueError):
        fit_nuisance_metric(features, _subjects())


@pytest.mark.parametrize(
    "views",
    [
        [],
        np.ones((4, 3, 2)),
        np.ones((4, 5, 2)),
        np.ones((4, 4, 1)),
        np.ones((3, 4, 2)),
        np.ones((16, 2)),
        np.ones((4, 2, 4)),
        np.full((4, 4, 2), np.nan),
        np.full((4, 4, 2), np.inf),
        np.full((4, 4, 2), -np.inf),
        np.ones((4, 4, 2), dtype=bool),
        np.ones((4, 4, 2), dtype=complex),
        np.full((4, 4, 2), "1"),
        np.full((4, 4, 2), None, dtype=object),
    ],
)
def test_rejects_invalid_view_types_dimensions_counts_and_missing_values(views):
    with pytest.raises(ValueError):
        fit_nuisance_metric(_clean(), _subjects(), views)


@pytest.mark.parametrize(
    "features",
    [
        np.full((4, 2), 1e308),
        np.array([[-1e200, 0], [1e200, 0], [0, 1], [0, 2]]),
    ],
)
def test_rejects_clean_training_statistics_overflow(features):
    with pytest.raises(ValueError, match="training statistics"):
        fit_nuisance_metric(features, _subjects())


@pytest.mark.parametrize("amplitude", [1e200, 1e308])
def test_rejects_displacement_or_second_moment_overflow(amplitude):
    clean = _clean()
    clean[:, 1] = 1  # std floor makes 1e308 / 1e-6 overflow before the moment.
    with pytest.raises(ValueError, match="finite"):
        fit_nuisance_metric(clean, _subjects(), np.full((4, 4, 2), amplitude))


def test_rejects_extended_float_inputs_outside_float64_range():
    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        pytest.skip("platform longdouble has no wider exponent range")
    huge = np.finfo(np.longdouble).max
    with pytest.raises(ValueError, match="finite"):
        fit_nuisance_metric(np.full((4, 2), huge, dtype=np.longdouble), _subjects())
    with pytest.raises(ValueError, match="finite"):
        fit_nuisance_metric(
            _clean(), _subjects(), np.full((4, 4, 2), huge, dtype=np.longdouble)
        )


@pytest.mark.parametrize(
    ("covariance", "shrinkage", "message"),
    [
        (np.array([[np.nan, 0], [0, 1]]), 0.5, "finite"),
        (np.array([[1.0, 1], [0, 1]]), 0.5, "symmetric"),
        (np.diag([1.0, -1]), 0.5, "positive semidefinite"),
        (np.eye(2), np.nan, "shrinkage"),
        (np.eye(2), np.inf, "shrinkage"),
        (np.eye(2), -0.1, "shrinkage"),
        (np.eye(2), 1.1, "shrinkage"),
    ],
)
def test_rejects_invalid_estimator_results_before_returning(
    covariance, shrinkage, message
):
    fitted = SimpleNamespace(covariance_=covariance, shrinkage_=shrinkage)
    with patch("models.nuisance_metric.LedoitWolf") as estimator:
        estimator.return_value.fit.return_value = fitted
        with pytest.raises(ValueError, match=message):
            fit_nuisance_metric(_clean(), _subjects())
