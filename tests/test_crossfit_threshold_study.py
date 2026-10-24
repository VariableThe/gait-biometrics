"""DEV role separation, fixed promotion gates, and numerical replay guards."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from engine.secure_enrollment import score_secure_trials
from scripts import crossfit_threshold_study as study


def test_all_inner_calibration_scores_exclude_validation_and_fit_people():
    for fold in study.folds():
        outer = set(fold["validation_ids"])
        held = []
        for inner in fold["inner"]:
            train, calibration = set(inner["train_ids"]), set(inner["held_ids"])
            assert len(train) == 9 and len(calibration) == len(outer) == 3
            assert (
                not train & calibration
                and not train & outer
                and not calibration & outer
            )
            assert train | calibration | outer == set(study.prior.DEV_IDS)
            held.extend(calibration)
        assert len(held) == len(set(held)) == 12
        assert set(held) == set(fold["refit_ids"])
        assert not set(fold["refit_ids"]) & outer


def test_directed_pair_counts_use_claimant_and_probe():
    scored = {
        "labels": np.array([1, 0, 0, 0]),
        "scores": np.array([0.9, 0.5, 0.4, 0.6]),
        "metadata": {
            "claim_subject_ids": [1, 1, 2, 1],
            "probe_subject_ids": [1, 2, 1, 2],
        },
    }
    assert study.pair_errors(scored, 0.5) == {
        "1<-2": {"claims": 2, "false_accepts": 2},
        "2<-1": {"claims": 1, "false_accepts": 0},
    }


def test_missing_trial_cannot_silently_change_registered_claim_graph():
    subjects, days, trials = np.array(
        [
            (s, d, t)
            for s in study.prior.DEV_IDS[:3]
            for d in (1, 2)
            for t in range(1, 11)
        ]
    ).T
    data = {"subjects": subjects, "days": days, "trials": trials}
    expected = study.expected_graph(data, study.prior.DEV_IDS[:3])
    assert (expected["n_genuine"], expected["n_impostor"]) == (30, 60)
    keep = ~((subjects == study.prior.DEV_IDS[0]) & (days == 2) & (trials == 10))
    changed = {key: value[keep] for key, value in data.items()}
    vectors = np.column_stack(
        (np.ones(keep.sum()), subjects[keep] / 30, trials[keep] / 20)
    )
    labels, scores, metadata = score_secure_trials(
        vectors,
        *(changed[key] for key in ("subjects", "days", "trials")),
        method="mean_multi",
    )
    scored = {"labels": labels, "scores": scores, "metadata": metadata}
    with pytest.raises(ValueError, match="registered graph"):
        study.validate_graph(scored, expected)
    registered = study.expected_graph(changed, study.prior.DEV_IDS[:3])
    assert (registered["n_genuine"], registered["n_impostor"]) == (29, 58)
    study.validate_graph(scored, registered)


def test_synthetic_run_replay_and_no_original_calib_or_test_reads(tmp_path):
    rng = np.random.default_rng(28)
    signals, subjects, days, trials = [], [], [], []
    for subject in study.prior.DEV_IDS:
        center = rng.normal(size=4)
        for day in (1, 2):
            for trial in range(1, 11):
                signals.append(center + rng.normal(scale=0.1, size=4))
                subjects.append(subject)
                days.append(day)
                trials.append(trial)
    data = {
        "X": np.array(signals),
        "subjects": np.array(subjects),
        "days": np.array(days),
        "trials": np.array(trials),
    }
    keep = ~(
        (data["subjects"] == study.prior.DEV_IDS[0])
        & (data["days"] == 2)
        & (data["trials"] == 10)
    )
    data = {key: value[keep] for key, value in data.items()}
    calls = []

    def load(_archive, ids):
        assert tuple(ids) == tuple(study.prior.DEV_IDS)
        calls.append(tuple(ids))
        return data

    args = SimpleNamespace(archive_path="synthetic.zip", out_root=tmp_path / "study")
    with (
        patch.object(
            study.prior._previous(), "audit_archive", return_value={"fixture": True}
        ),
        patch.object(study, "hashes", return_value={"fixture": "frozen"}),
        patch.object(study.prior, "load_data", side_effect=load),
        patch.object(
            study.prior,
            "_manifest",
            side_effect=lambda values: {"sha": study.prior.object_hash(values)},
        ),
        patch.object(study.secure, "_features", side_effect=lambda values: values["X"]),
        patch.object(study.secure, "_fit", wraps=study.secure._fit) as fitted,
    ):
        study.run(args)
        assert fitted.call_count == 100
        original_score = study.secure._score

        def noisy_score(*args, **kwargs):
            scored = original_score(*args, **kwargs)
            return {**scored, "scores": np.nextafter(scored["scores"], np.inf)}

        with patch.object(study.secure, "_score", side_effect=noisy_score):
            result = study.verify(args)
        assert fitted.call_count == 100
        assert result["verified"] and result["fits"] == 100
        assert len(calls) == 2
        evidence = study.prior.read(args.out_root / "results.json")
        evidence["rows"][0]["threshold"]["threshold"] += 0.1
        (args.out_root / "results.json").write_bytes(study.prior.canonical(evidence))
        with pytest.raises(ValueError, match="threshold policy"):
            study.verify(args)
        with pytest.raises(ValueError, match="fresh empty"):
            study.run(args)
