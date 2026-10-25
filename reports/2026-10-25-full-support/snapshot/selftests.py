"""Synthetic implementation tests only; no real signals/states/scores are read."""

import hashlib
import os
import subprocess
import sys
import time
from fractions import Fraction
from zipfile import ZipFile, ZipInfo

import common as c
import guard
import kernel as k
import numpy as np
import payload
import planning
import review_tests
import worker
from models.gait_metric import WithinClassMetric
from sklearn.covariance import LedoitWolf


def must_fail(function, contains=None):
    try:
        function()
    except (RuntimeError, ValueError, FileExistsError) as error:
        if contains is not None:
            c.require(contains in str(error), "wrong failure: " + str(error))
        return str(error)
    raise RuntimeError("expected fail-stop did not occur")


def intervals_and_aggregation():
    for start, stop in ((0, 256), (0, 512), (17, 463), (2000, 2931), (301, 903)):
        old, cover, weights, _multiplicity = planning.intervals(start, stop)
        union = {r for a, b in cover for r in range(a, b)}
        c.require(
            union == set(range(start, stop)) and len(cover) == len(set(cover)),
            "exact union/dedup",
        )
        c.require(
            sum(weights) == 1
            and cover == sorted(cover)
            and cover[-1] == (stop - 256, stop),
            "endcap order/weights",
        )
        for j, (a, b) in enumerate(cover):
            expected = sum(
                (Fraction(1, sum(x <= r < y for x, y in cover)) for r in range(a, b)),
                Fraction(),
            ) / (stop - start)
            c.require(weights[j] == expected, "independent rational-weight reference")
        meta = {
            "eligible": [{}],
            "windows": [
                {"unit": 0, "historical": interval in old, "weight_exact": str(weight)}
                for interval, weight in zip(cover, weights, strict=True)
            ],
        }
        features = np.arange(len(cover) * 3, dtype=np.float64).reshape(-1, 3) + 1
        h, f = k.aggregate(features, meta, "H"), k.aggregate(features, meta, "F")
        c.require(h.shape == f.shape == (1, 3), "one trial unit, not one per cover")
        c.require(
            np.array_equal(
                h[0],
                features[
                    [i for i, w in enumerate(meta["windows"]) if w["historical"]]
                ].mean(axis=0),
            ),
            "H exact1/K featuremean",
        )
        reference = np.zeros(3)
        for value, weight in zip(features, weights, strict=True):
            reference += value * float(weight)
        c.require(
            np.array_equal(f[0], reference),
            "F fixed float64 deterministic weighted sum",
        )
    # Two fragments of one trial are NEVER joined into an eligible 256 window.
    must_fail(lambda: planning.intervals(0, 200))
    must_fail(lambda: planning.intervals(350, 550))
    a = planning.intervals(17, 463)[1]
    b = planning.intervals(700, 1200)[1]
    c.require(
        all(right <= 463 for _, right in a) and all(left >= 700 for left, _ in b),
        "no crossing fragmented gap",
    )


def stress_fields():
    item = {
        "recording_key": [7, 1, 2, 1, "MCU", "processed_data"],
        "member_sha256": "a" * 64,
        "day": 2,
    }
    rows = np.arange(71, 649, dtype=np.int64)
    full = k.physical_field(item, rows)
    order = np.arange(len(rows))[::-1]
    c.require(
        np.array_equal(full[:, order], k.physical_field(item, rows[order])),
        "source-row field order stability",
    )
    c.require(
        np.array_equal(
            full,
            np.concatenate(
                [
                    k.physical_field(item, rows[:191]),
                    k.physical_field(item, rows[191:]),
                ],
                axis=1,
            ),
        ),
        "field batching stability",
    )
    c.require(
        np.array_equal(full[:, 322:], k.physical_field(item, rows[322:])),
        "tail field is the same complete-source field",
    )
    changed = {**item, "member_sha256": "b" * 64}
    c.require(
        not np.array_equal(full, k.physical_field(changed, rows)),
        "source identity participates in key",
    )
    source = np.arange(6 * len(rows), dtype=np.float32).reshape(6, -1) / 70
    scale = np.array([1, 1, 1, 4, 4, 4], dtype=np.float64)
    noise1 = k.perturb(item, source, "noise01", scale, full)
    noise5 = k.perturb(item, source, "noise05", scale, full)
    expected1 = (source.astype(np.float64) + full * (scale * 0.01)[:, None]).astype(
        np.float32
    )
    expected5 = (source.astype(np.float64) + full * (scale * 0.05)[:, None]).astype(
        np.float32
    )
    c.require(
        np.array_equal(noise1, expected1) and np.array_equal(noise5, expected5),
        "shared standardnormal noise1/5 draws with native-unit scale",
    )
    # Historical window2 [256,512) and endcap [322,578) overlap exactly.
    r_old = noise1[:, 256:512]
    f_tail = noise1[:, 322:578]
    c.require(
        np.array_equal(r_old[:, 66:], f_tail[:, :190]),
        "R/H/F overlap/tail physical noise equality",
    )
    c.require(
        np.array_equal(
            noise1[:, :256],
            k.perturb(
                item,
                source[:, :256],
                "noise01",
                scale,
                k.physical_field(item, rows[:256]),
            ),
        ),
        "no hidden whole-cover state",
    )
    for condition in planning.CONDITIONS:
        enrollment = k.perturb({**item, "day": 1}, source, condition, scale, full)
        c.require(np.array_equal(enrollment, source), "day1 enrollment must stay clean")
    for angle in (30, 90):
        matrix = k.rotation(angle)
        c.require(
            np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-15)
            and np.linalg.det(matrix) > 0,
            "fixed proper common-triad rotation",
        )
        rotated = k.perturb(item, source, f"rotation{angle}", scale, full)
        c.require(
            np.array_equal(
                rotated[:3], (matrix @ source[:3].astype(np.float64)).astype(np.float32)
            )
            and np.array_equal(
                rotated[3:], (matrix @ source[3:].astype(np.float64)).astype(np.float32)
            ),
            "same physical rotation both triads",
        )


def diagnostics_and_storage(directory):
    ticks = np.arange(256, dtype=np.float64) / 100
    fixture = np.stack(
        (
            np.sin(2 * np.pi * 1.7 * ticks),
            0.2 * np.cos(2 * np.pi * 1.7 * ticks),
            9.8 + 0.1 * np.cos(4 * np.pi * 1.7 * ticks),
            2 * np.sin(2 * np.pi * ticks),
            np.cos(2 * np.pi * ticks),
            0.2 * np.sin(4 * np.pi * ticks),
        )
    ).astype(np.float32)
    windows = np.stack((fixture, np.zeros_like(fixture), fixture.copy()))
    windows[2, 0, 0] = np.nan
    arrays, diagnostics = k.preflight(windows, [[1, 1, i + 1, 0] for i in range(3)])
    c.require(
        arrays["valid"].tolist() == [True, False, False] and len(diagnostics) == 3,
        "complete fail-stop diagnostic denominator",
    )
    c.require(
        np.isnan(arrays["features"][~arrays["valid"]]).all()
        and diagnostics[1]["geometry_nullable"][1] is None,
        "NaN geometry must remain lossless with nullable JSON",
    )
    artifact = c.save_arrays(directory, "diagnostics", arrays)
    c.write(directory / "diagnostics.json", diagnostics)
    restored = c.load_arrays(directory, artifact)
    for key in arrays:
        c.require(
            np.array_equal(arrays[key], restored[key], equal_nan=True)
            and arrays[key].dtype == restored[key].dtype,
            "lossless diagnostic arrays/dtypes",
        )
    must_fail(lambda: c.canonical({1: "bad"}), "keys")
    must_fail(lambda: c.canonical({"x": float("nan")}), "Nonfinite")
    must_fail(lambda: c.decode('{"x":1,"x":2}'), "Duplicate")
    must_fail(lambda: c.same({"x": 0.0}, {"x": -0.0}, "signed-zero"), "Mismatch")
    must_fail(lambda: c.same({"x": 1}, {"x": True}, "bool"), "Mismatch")
    indexed = directory / "index-fixture"
    indexed.mkdir()
    c.write(indexed / "payload.json", {"finite": 1, "signed": -0.0})
    c.seal(indexed)
    c.check_index(indexed)
    c.write(indexed / "extra.json", {"unexpected": True})
    must_fail(lambda: c.check_index(indexed), "whole-file")


def training_and_replay(directory, metadata):
    # Small six-dimensional SYNTHETIC trial descriptors using public labels only.
    labels = np.asarray(metadata["trial_units"], dtype=np.int64)
    random = np.random.Generator(np.random.PCG64(7781))
    identity = {subject: random.normal(size=6) for subject in planning.DEV}
    values = np.stack(
        [
            identity[int(s)]
            + 0.03 * random.normal(size=6)
            + np.array([0.001 * d, 0.001 * t, 0, 0, 0, 0])
            for s, d, t in labels
        ]
    ).astype(np.float64)
    data = {key: labels[:, i] for i, key in enumerate(("subjects", "days", "trials"))}
    old_indices = [w["unit"] for w in metadata["windows"] if w["historical"]]
    old_values = values[old_indices]
    old_labels = labels[old_indices]
    old_data = {
        key: old_labels[:, i] for i, key in enumerate(("subjects", "days", "trials"))
    }
    ledger = k.FitLedger(metadata["fit_assignments"])
    metric_calls, covariance_calls = [], []
    metric_fit, covariance_fit = WithinClassMetric.fit, LedoitWolf.fit

    def metric_spy(self, features, subject_ids):
        metric_calls.append((features.shape, tuple(np.unique(subject_ids).tolist())))
        return metric_fit(self, features, subject_ids)

    def covariance_spy(self, residuals, *args, **kwargs):
        covariance_calls.append(residuals.shape)
        return covariance_fit(self, residuals, *args, **kwargs)

    WithinClassMetric.fit, LedoitWolf.fit = metric_spy, covariance_spy
    rows, restore_count = [], 0
    try:
        for fold in metadata["folds"]:
            f = fold["fold"]
            must_fail(
                lambda f=f: ledger.fit("H", f, "refit", values, data), "BOTH cutoffs"
            )
            cutoffs, states = {}, {}
            for arm in ("H", "F"):
                pools = []
                for role, inner in enumerate(fold["inner"]):
                    model = ledger.fit(arm, f, role, values, data)
                    pools.append(
                        k.score(
                            model,
                            values,
                            data,
                            inner["held_ids"],
                            metadata["coverage"][str(f)]["inner"][role],
                        )
                    )
                    states[arm, role] = worker.model_state(model)
                cutoffs[arm] = k.cutoff(pools)
            ledger.freeze_cutoffs(f, cutoffs)
            models = {}
            for arm in ("H", "F"):
                models[arm] = ledger.fit(arm, f, "refit", values, data)
                states[arm, "refit"] = worker.model_state(models[arm])
            calls_before = len(metric_calls), len(covariance_calls)
            with k.forbid_fits():
                for role, ids in [
                    *[(i, x["train_ids"]) for i, x in enumerate(fold["inner"])],
                    ("refit", fold["refit_ids"]),
                ]:
                    k.restore(
                        states["H", role],
                        old_values,
                        old_data,
                        ids,
                        check_statistics=False,
                    )
                    restore_count += 1
                # Fit spies must see ZERO optimizer calls on transform/replay.
                for arm in ("H", "F"):
                    restored = k.restore(
                        states[arm, "refit"], values, data, fold["refit_ids"]
                    )
                    fresh = k.score(
                        restored,
                        values,
                        data,
                        fold["validation_ids"],
                        metadata["coverage"][str(f)]["validation"],
                    )
                    cached = k.score(
                        models[arm],
                        values,
                        data,
                        fold["validation_ids"],
                        metadata["coverage"][str(f)]["validation"],
                    )
                    k.replay(fresh, cached, cutoffs[arm]["threshold"])
                    for condition in planning.CONDITIONS:
                        rows.append(
                            k.row(f, arm, condition, cached, cutoffs[arm]["threshold"])
                        )
                    if arm == "H":
                        for condition in planning.CONDITIONS:
                            rows.append(
                                k.row(
                                    f, "R", condition, cached, cutoffs[arm]["threshold"]
                                )
                            )
                    if f == 0 and arm == "H":
                        record = c.save_arrays(
                            directory, "tiny-model", states[arm, "refit"]
                        )
                        serialized = worker.store_score(directory, "tiny-score", cached)
                        loaded = worker.load_score(directory, serialized)
                        k.replay(fresh, loaded, cutoffs[arm]["threshold"])
                        k.restore(
                            c.load_arrays(directory, record),
                            values,
                            data,
                            fold["refit_ids"],
                        )
                        # Within numeric tolerance but across the threshold MUST stop.
                        boundary = {
                            "labels": np.array([1, 0], dtype=np.int64),
                            "scores": np.array([0.0, -1.0]),
                            "metadata": {},
                        }
                        changed = {**boundary, "scores": np.array([-1e-12, -1.0])}
                        must_fail(
                            lambda changed=changed, boundary=boundary: k.replay(
                                changed, boundary, 0.0
                            ),
                            "decisions changed",
                        )
                        tampered = {**cached, "scores": cached["scores"] + 0.01}
                        must_fail(
                            lambda fresh=fresh, tampered=tampered, threshold=cutoffs[arm]["threshold"]: (
                                k.replay(fresh, tampered, threshold)
                            ),
                            "tolerance",
                        )
                        broken_graph = {
                            **metadata["coverage"][str(f)]["validation"],
                            "labels": [0],
                        }
                        must_fail(
                            lambda model=models[arm], ids=fold["validation_ids"], graph=broken_graph: (
                                k.score(model, values, data, ids, graph)
                            ),
                            "claimlabels",
                        )
                        zero_model = WithinClassMetric.from_state_dict(
                            states[arm, "refit"]
                        )
                        must_fail(
                            lambda model=zero_model: model.transform(model.mean_[None]),
                            "no cosine direction",
                        )
                        bad_state = {
                            **states[arm, "refit"],
                            "mean": states[arm, "refit"]["mean"] + 0.1,
                        }
                        must_fail(
                            lambda state=bad_state, ids=fold["refit_ids"]: k.restore(
                                state, values, data, ids
                            ),
                            "standardizer",
                        )
            c.require(
                calls_before == (len(metric_calls), len(covariance_calls)),
                "R/compact restore invoked estimator fit",
            )
        c.require(
            len(metric_calls) == len(covariance_calls) == len(ledger.calls) == 200
            and restore_count == 100,
            "actual fit spies require200 autoWCCN/LedoitWolf fits and100 fit-free R restores",
        )
        c.require(
            {shape[0] for shape, _ in metric_calls} == {179, 180, 239, 240}
            and all(shape[1] == 6 for shape, _ in metric_calls),
            "actual optimizer TRAIN unit counts/dimensions",
        )
        c.require(
            all(len(ids) in (9, 12) for _, ids in metric_calls),
            "actual optimizer TRAIN ownership",
        )
        c.require(
            all(
                shape[0] == call["training_units"]
                and tuple(ids) == tuple(sorted(call["ids"]))
                for (shape, ids), call in zip(metric_calls, ledger.calls, strict=True)
            ),
            "actual coefficient-fit spy assignments match every registered unit count/owner",
        )
        summary = k.comparison(rows, metadata)
        c.require(
            summary["primary"]["F_minus_H_clean_equalfold_FRR_exact"] == "0"
            and not summary["primary"]["favors_support"]
            and summary["nominee"] == "R"
            and len(summary["replacement"]["gates"]) == 33,
            "exact no-promotion-on-ties/fullgate test",
        )
        c.write(
            directory / "tiny-end-to-end.json",
            {
                "synthetic_only": True,
                "rows": rows,
                "summary": summary,
                "actual_WCCN_fits": len(metric_calls),
                "actual_LedoitWolf_fits": len(covariance_calls),
                "fit_free_R_restores": restore_count,
                "actual_training_shapes": [list(shape) for shape, _ in metric_calls],
                "events": ledger.events,
            },
        )
    finally:
        WithinClassMetric.fit, LedoitWolf.fit = metric_fit, covariance_fit
    # Inner fitting does not consult any12TRAIN RMS or held-triple features.
    inner = metadata["folds"][0]["inner"][0]
    mask = np.isin(data["subjects"], inner["train_ids"])
    a = metric_fit(WithinClassMetric(), values[mask], data["subjects"][mask])
    changed = values.copy()
    changed[~mask] += 1e6
    b = metric_fit(WithinClassMetric(), changed[mask], data["subjects"][mask])
    c.require(
        all(
            np.array_equal(a.state_dict()[key], b.state_dict()[key])
            for key in a.state_dict()
        ),
        "held features polluted inner9 fit",
    )
    return {
        "actual_candidate_test_fits": 200,
        "additional_pollution_test_fits": 2,
        "actual_covariance_spy_calls": 200,
        "R_restore_fit_calls": 0,
        "R_restores": 100,
        "ordered_outer_rows": len(rows),
        "genuine_claims_percondition_perarm": 596,
        "impostor_claims_percondition_perarm": 1192,
        "gate_count": 33,
    }


def rms_owner(metadata):
    # Tail windows and held people have enormous synthetic energy; neither enters RMS.
    random = np.random.Generator(np.random.PCG64(331))
    windows = random.normal(size=(885, 6, 256)).astype(np.float32)
    ids = metadata["folds"][0]["refit_ids"]
    mask = np.array(
        [w["historical"] and w["source_key"][0] in ids for w in metadata["windows"]]
    )
    before, record = k.training_rms(windows, metadata, ids)
    windows[~mask] *= 1e8
    after, _ = k.training_rms(windows, metadata, ids)
    c.require(
        np.array_equal(before, after) and record["valid"], "held/tail RMS pollution"
    )
    owned = windows[mask].astype(np.float64)
    owned -= owned.mean(axis=2, keepdims=True)
    reference = np.repeat(
        [np.sqrt(np.mean(owned[:, j : j + 3] ** 2)) for j in (0, 3)], 3
    )
    c.require(
        np.array_equal(before, reference), "inherited dynamic window RMS arithmetic"
    )
    zeros, diag = k.training_rms(np.zeros_like(windows), metadata, ids)
    c.require(
        not diag["valid"] and np.array_equal(zeros, np.zeros(6)),
        "zero RMS diagnostics before stop",
    )


def preflight_failure_persistence(directory):
    meta = {
        "eligible": [],
        "windows": [],
        "folds": [{"fold": 0, "refit_ids": [1], "validation_ids": [3]}],
    }
    sources = []
    for subject, day in ((1, 1), (1, 2), (3, 1), (3, 2)):
        unit = len(sources)
        item = {
            "subject": subject,
            "day": day,
            "trial": 1,
            "recording_key": [7, subject, day, 1, "MCU", "processed_data"],
            "member_sha256": "c" * 64,
            "original": {"start_row": 0, "stop_row": 446},
            "length": 446,
        }
        old, cover, weights, _ = planning.intervals(0, 446)
        meta["eligible"].append(item)
        for interval, weight in zip(cover, weights, strict=True):
            meta["windows"].append(
                {
                    "unit": unit,
                    "rows": list(interval),
                    "historical": interval in old,
                    "source_key": [subject, day, 1, interval[0]],
                    "weight_exact": str(weight),
                }
            )
        sources.append(
            {
                "values": np.zeros((6, 446), dtype=np.float32),
                "frames": np.arange(446, dtype=np.int64),
                "rows": np.arange(446, dtype=np.int64),
            }
        )
    failed = directory / "complete-preflight-failure"
    failed.mkdir()
    with k.forbid_fits():
        must_fail(
            lambda: worker.preflight_all(failed, meta, sources),
            "complete preflight diagnostic ledger persisted",
        )
    record, diagnostics, rms = (
        c.read(failed / "preflight.json"),
        c.read(failed / "preflight-diagnostics.json"),
        c.read(failed / "train-rms.json"),
    )
    c.require(
        record["context_count"] == 7
        and record["invalid_windows"] == len(diagnostics) == 20
        and record["no_fit_entered"] is True
        and not rms["0"]["valid"],
        "complete clean8 plus six stressed2 window names and zero-RMS ledger before abort",
    )
    c.require(
        sum(d["historical_R_required"] for d in diagnostics) == 10,
        "R old windows retained in failed coverage ledger",
    )
    c.require(
        not list(failed.glob("models/*")),
        "no optimizer/model state before full preflight",
    )


def synthetic_zip_allowlist(directory):
    """Synthetic ZIP fixture only; never inspect the official central directory."""
    path = directory / "SYNTHETIC-members.zip"
    body = (
        "frame,acc x,acc y,acc z,gyro x,gyro y,gyro z\n"
        + "".join(f"{i},1.0,2.0,9.8,0.1,0.2,0.3\n" for i in range(446))
    ).encode()

    def member(subject):
        return f"A multi-sensor human gait dataset/processed_data/user{subject:02}/capture_user{subject:02}_day1_0001_imu_walk.csv"

    with ZipFile(path, "x") as archive:
        for subject in planning.DEV:
            archive.writestr(member(subject), body)
        archive.writestr(member(21), b"SYNTHETIC CALIB sentinel must never be read")
        archive.writestr("raw/synthetic.csv", b"raw sentinel must never be read")
    metadata = {
        "records": [],
        "eligible": [],
        "official_authenticity": {"size_bytes": path.stat().st_size},
    }
    with ZipFile(path) as archive:
        for index, subject in enumerate(planning.DEV):
            info = archive.getinfo(member(subject))
            record = payload.santos._member_record((subject, 1, 1), info, body)
            payload.santos._parse_recording(body, record, 256, 256, n_channels=6)
            metadata["records"].append(record)
            metadata["eligible"].append(
                {
                    "recording_index": index,
                    "length": 446,
                    "original": {
                        "start_row": 0,
                        "stop_row": 446,
                        "frame_start": 0,
                        "frame_end": 445,
                    },
                }
            )
    opened = []

    class SelectedOnlyZip(ZipFile):
        def read(self, name, *args, **kwargs):
            c.require(
                isinstance(name, ZipInfo),
                "validated ZipInfo rather than arbitrary member name required",
            )
            opened.append(name.filename)
            c.require(
                name.filename in {member(subject) for subject in planning.DEV},
                "nonDEV/raw synthetic payload opened",
            )
            return super().read(name, *args, **kwargs)

    original_zip = payload.ZipFile
    payload.ZipFile = SelectedOnlyZip  # ty: ignore[invalid-assignment] -- deliberate synthetic IO spy
    try:
        sources, ledger = payload.load(path, metadata)
        c.require(
            opened == [member(subject) for subject in planning.DEV]
            and len(sources) == 15
            and all(s["values"].shape == (6, 446) for s in sources),
            "exact allowed members/complete tails only",
        )
        c.require(
            all(r["payload_read"] for r in ledger),
            "every loaded synthetic member validated",
        )
        corrupt = {
            **metadata,
            "records": [
                {**metadata["records"][0], "sha256": "0" * 64},
                *metadata["records"][1:],
            ],
        }
        with k.forbid_fits():
            must_fail(
                lambda: payload.load(path, corrupt),
                "allowed member payload identity/hash",
            )
        c.write(
            directory / "synthetic-member-access.json",
            {
                "fixture_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "opened_before_first_stop": opened,
                "real_archive_opened": False,
                "nonDEV_payload_reads": 0,
            },
        )
    finally:
        payload.ZipFile = original_zip


def launcher_guards(directory):
    for argv in (
        ["--archive", "/none", "execute"],
        ["execute", "--out-root", "/x", "--archive", "/y"],
        ["execute", "--archive", "/x", "--out-root", "/a/../b"],
    ):
        must_fail(lambda argv=argv: guard.grammar(argv))
    script = directory / "guard-fixture.txt"
    script.write_text(
        "import sys\nsys.path.insert(0,"
        + repr(str(c.ROOT))
        + ")\nfrom guard import bootstrap, deadlines\nidentity=bootstrap(sys.argv[1:])\nif '--expired' in sys._xoptions: deadlines((identity[0],'1'))\nraise RuntimeError('synthetic pre-scientific-import failure after worker receipt')\n"
    )
    attempt = directory / "single-use"
    attempt.mkdir()
    argv = [
        sys.executable,
        "-B",
        str(script),
        "execute",
        "--archive",
        "/synthetic-not-opened.zip",
        "--out-root",
        str(attempt),
    ]
    first = subprocess.run(
        argv, capture_output=True, text=True, timeout=10, check=False
    )
    c.require(
        first.returncode != 0 and "pre-scientific-import failure" in first.stderr,
        "synthetic failed startup",
    )
    original = (attempt / "attempt.started.json").read_bytes()
    c.require(
        (attempt / "worker.started.json").is_file()
        and c.read(attempt / "attempt.started.json")["cpu_budget_seconds"] == 1800,
        "worker receipt before imports/coherent reservation",
    )
    second = subprocess.run(
        argv, capture_output=True, text=True, timeout=10, check=False
    )
    c.require(
        second.returncode != 0
        and "FileExistsError" in second.stderr
        and original == (attempt / "attempt.started.json").read_bytes(),
        "failed attempt refusal preserves original",
    )
    expired = directory / "expired"
    expired.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-X",
            "--expired=1",
            str(script),
            "execute",
            "--archive",
            "/synthetic-not-opened.zip",
            "--out-root",
            str(expired),
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    c.require(
        result.returncode != 0
        and "deadline already exhausted" in result.stderr
        and (expired / "worker.started.json").exists(),
        "deadline refusal still consumes attempt/receipt",
    )
    fake = directory / "fake-handoff"
    fake.mkdir()
    (fake / "attempt.started.json").write_text(
        guard.attempt_text("shell", os.getpid(), "1")
    )
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-X",
            f"full_support_reserved={os.getpid()}:1",
            str(script),
            "execute",
            "--archive",
            "/synthetic-not-opened.zip",
            "--out-root",
            str(fake),
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    c.require(
        result.returncode != 0
        and "PID:startticks" in result.stderr
        and not (fake / "worker.started.json").exists(),
        "PID reuse/startticks protection",
    )
    # Native imported source has no bytecode writes, including failed launches.
    c.require(not list(c.ROOT.rglob("*.pyc")), "no owned bytecode writes")


def run():
    base = c.ROOT / "synthetic-tests"
    base.mkdir(exist_ok=True)
    directory = base / f"attempt-{time.time_ns()}"
    directory.mkdir(exist_ok=False)
    started = time.process_time()
    completed = []
    try:
        metadata = planning.metadata()
        for name, function in (
            (
                "loaded_singlefile_bytecode_transitive_source_cache_tamper_missingbinding_lateexec",
                lambda: review_tests.runtime_closure(directory),
            ),
            (
                "original_R_full_worker_verify_complete_synthetic_reference_rehashed_tampering_nofit",
                lambda: review_tests.copied_R(directory, metadata),
            ),
            (
                "exact_intervals_weights_union_fragmentation_oneunit",
                intervals_and_aggregation,
            ),
            ("physical_source_noise_overlap_order_batch_tail_allarms", stress_fields),
            (
                "complete_failstop_diagnostics_strict_IO_tamper",
                lambda: diagnostics_and_storage(directory),
            ),
            (
                "TRAIN12_historical_RMS_owner_and_zero_diagnostics",
                lambda: rms_owner(metadata),
            ),
            (
                "preflight_all_required_names_R_windows_zeroRMS_persist_before_abort",
                lambda: preflight_failure_persistence(directory),
            ),
            (
                "SYNTHETIC_ZipInfo_DEV_allowlist_memberhash_continuity_tail_nofallback",
                lambda: synthetic_zip_allowlist(directory),
            ),
            (
                "R3_stage_PID_startticks_workerreceipt_singleuse_deadline",
                lambda: launcher_guards(directory),
            ),
        ):
            function()
            completed.append(name)
        fitting = training_and_replay(directory, metadata)
        completed.append(
            "actual200_coefficient_fits_counts_ownership_cutoff_order_Rfitforbidden_endtoend_claims_replay_tamper"
        )
        record = {
            "status": "passed",
            "synthetic_only": True,
            "groups": completed,
            "fitting": fitting,
            "test_process_cpu_seconds": time.process_time() - started,
            "real_DEV_or_R_payload_reads": 0,
            "directory": str(directory.relative_to(c.ROOT)),
        }
        c.write(directory / "result.json", record)
        return record
    except BaseException as error:
        c.write(
            directory / "failure.json",
            {
                "status": "synthetic_implementation_failure_preserved",
                "completed_groups": completed,
                "error_type": type(error).__name__,
                "message": str(error),
            },
        )
        raise
