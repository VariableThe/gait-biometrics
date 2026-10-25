"""Only fixed handcases and tiny fixtures, never the study grid or primary panel."""

import json
import time

import common as c
import fixtures
import kernel
import legacy
import plateau_helpers
import regression
import numpy as np


def run(*, include_systems=False):
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    rows = []

    def log(name, **fields):
        rows.append({"name": name, "pass": True, **fields})

    fixed, vectors = legacy.fixed_checks()
    c.require(
        len(fixed["rows"]) == 18 and vectors.shape == (18, 121), "Fixed checks count"
    )
    plateau = plateau_helpers.synthetic_helper_checks()
    c.require(plateau["fixed_case_count"] == 16, "Plateau helper count")
    x = np.linspace(-2, 3, 25)
    for name, y, expected in (
        ("identical", x, 1.0),
        ("negated", -x, -1.0),
        ("positive_affine", 3 * x + 7, 1.0),
    ):
        value, _, good, _ = kernel.pearson(x, y)
        c.require(
            good.all() and abs(value - expected) < 5e-15, "Pearson identity " + name
        )
        log(name, value=value)
    for name, vector, expected in (
        ("zero", np.zeros(32), False),
        ("constant", np.ones(32), False),
        ("unit_dc_5e-13", 1 + 5e-13 * np.tile([-1.0, 1.0], 16), False),
        ("unit_dc_2e-12", 1 + 2e-12 * np.tile([-1.0, 1.0], 16), True),
        ("NaN", np.r_[np.ones(31), np.nan], False),
        ("positive_inf", np.r_[np.ones(31), np.inf], False),
        ("negative_inf", np.r_[np.ones(31), -np.inf], False),
    ):
        good, stats, finite = kernel.vector_guard(vector)
        c.require(good == expected, "Guard handcase " + name)
        value, _, _, _ = kernel.pearson(vector, vector)
        c.require(np.isfinite(value) == expected, "Guarded correlation " + name)
        log(
            name,
            valid=good,
            finite=finite,
            q=float(stats[0]) if np.isfinite(stats[0]) else None,
            std=float(stats[1]) if np.isfinite(stats[1]) else None,
        )
    for n in c.LENGTHS:
        theta = 2 * np.pi * (np.arange(n, dtype=float) / 180)
        vectors = np.array([10 + np.cos(2 * theta), 4 + 0.5 * np.cos(theta)])
        for lag, expected in ((180, 1.0), (90, 0.0)):
            rs = [kernel.pearson(v[:-lag], v[lag:])[0] for v in vectors]
            joint = float(sum(rs) / 2)
            c.require(
                abs(joint - expected) < 1e-14, "Unfiltered analytic joint identity"
            )
            distances = []
            dots = []
            for v in vectors:
                left, right = v[:-lag].copy(), v[lag:].copy()
                left -= left.mean()
                right -= right.mean()
                left /= np.linalg.norm(left)
                right /= np.linalg.norm(right)
                dots.append(float(left @ right))
                distances.append(float(np.sum((left - right) ** 2)))
            dot_j = float(sum(dots) / 2)
            distance = float(sum(distances) / 2)
            c.require(
                abs(dot_j - joint) < 1e-14 and abs(distance - 2 * (1 - joint)) < 2e-14,
                "Centered dot/distance identity",
            )
            log(
                "analytic_P180_phase0",
                length=n,
                lag=lag,
                joint=joint,
                dot_joint=dot_j,
                distance_squared=distance,
            )
    # A single analytic N256 case exercises the full candidate and the no-fallback path.
    theta = 2 * np.pi * np.arange(256, dtype=float) / 180
    tiny_mags = np.array([10 + np.cos(2 * theta), 4 + 0.5 * np.cos(theta)])
    candidate, _, _, _ = kernel.joint_correlations(tiny_mags)
    base, feature = legacy.extract(
        tiny_mags, tiny_mags[0], candidate[2], legacy.RULES[1]
    )
    c.require(
        base["selected_lag"] == 180 and feature is not None,
        "Tiny joint extraction identity",
    )
    item = next(
        r
        for r in fixtures.keys()
        if r["kind"] == "stress"
        and r["period_samples"] == 180
        and r["phase_index"] == 0
        and r["length"] == 256
        and r["condition"] == "clean"
    )
    record = kernel.record(
        item,
        2,
        base,
        feature,
        candidate[2],
        kernel.whole_window(tiny_mags, tiny_mags),
        candidate,
    )
    c.require(
        record["quality_cutoff"] is None
        and record["lag_resolves_declared_P_within_one_sample"],
        "Tiny candidate record",
    )
    flat = np.stack((tiny_mags[0], np.full(256, 4.0)))
    invalid, _, guards, _ = kernel.joint_correlations(flat)
    c.require(
        np.isfinite(invalid[0]).all()
        and np.isnan(invalid[1:]).all()
        and not guards[1].any(),
        "No gyro fallback",
    )
    failure, feature = legacy.extract(flat, flat[0], invalid[2], legacy.RULES[1])
    c.require(
        failure["failure_reason"] == "no_lag" and feature is None,
        "Undefined joint failure",
    )
    log(
        "tiny_candidate_and_degenerate_no_fallback",
        selected_lag=base["selected_lag"],
        pair=base["pair_bounds_inclusive"],
    )
    diag, _, _, _, _ = kernel.diagnostic(tiny_mags, tiny_mags)
    c.require(
        diag.shape == (6, 161) and abs(float(diag[5, 180 - 40]) - 1) < 1e-14,
        "Tiny common-anchor diagnostic",
    )
    log(
        "tiny_common_anchor_diagnostic",
        length=256,
        anchors_first=0,
        anchors_last=55,
        anchor_count=56,
    )
    # One predetermined legacy fixture, two lengths, two controls: NO balanced primary-panel sweep.
    parent = c.ROOT.parent / "cycle-length-feasibility-r1"
    report = json.loads((parent / "report.json").read_text())
    cached = {
        r["row_id"]: r
        for r in map(json.loads, (parent / "records.jsonl").read_text().splitlines())
        if r["row_id"] < 4
    }
    parent_features = np.load(
        parent / "arrays/features.npy", mmap_mode="r", allow_pickle=False
    )
    parent_corr = np.load(
        parent / "arrays/extractor_correlations.npy", mmap_mode="r", allow_pickle=False
    )
    for li, n in enumerate(c.LENGTHS):
        item = {
            "family": legacy.FAMILIES[0],
            "period_samples": 80,
            "phase_index": 0,
            "length": n,
            "source_fixture_index": 0,
            "source_length_index": li,
        }
        window = legacy.fixture(legacy.FAMILIES[0], 80, 0, n)
        mags = kernel.magnitudes(window)
        filtered = kernel.filter_modalities(mags)
        rs = legacy.correlations(filtered[0])
        c.require(
            rs.tobytes() == parent_corr[li].tobytes(),
            "Tiny cached correlation mismatch",
        )
        for ri, rule in enumerate(legacy.RULES):
            base, feature = legacy.extract(mags, filtered[0], rs, rule)
            compatible = kernel.compatible_record(
                base,
                feature,
                item,
                ri,
                window,
                mags,
                filtered[0],
                rs,
                report["analytic_minimum_root_phases_cycles"],
            )
            c.require(
                c.typed(compatible) == c.typed(cached[li * 2 + ri]),
                "Tiny complete-field legacy mismatch",
            )
            c.require(
                feature.tobytes() == parent_features[li * 2 + ri].tobytes(),
                "Tiny cached feature bits mismatch",
            )
        log(
            "tiny_legacy_control_fixture",
            length=n,
            controls=2,
            all_cached_fields_and_bits_equal=True,
        )
    # One stress seed/setting only checks fixed equations and shared prefix, never outcomes.
    stress = fixtures.stress_long(160, 0)
    noise = np.random.Generator(np.random.PCG64(20261025)).uniform(
        -np.sqrt(3), np.sqrt(3), size=(768, 2)
    )
    c.require(
        stress[c.CONDITIONS[4]][2].tobytes()
        == (10 + 4 * (1 / np.sqrt(2)) * noise[:, 0]).tobytes(),
        "Stress replacement amplitude",
    )
    for condition in c.CONDITIONS:
        item = {
            "kind": "stress",
            "length": 256,
            "period_samples": 160,
            "phase_index": 0,
            "condition": condition,
        }
        c.require(
            fixtures.generate(item).tobytes()
            == stress[condition][:, :256].copy().tobytes(),
            "Stress prefix mismatch",
        )
    log("tiny_stress_equation_and_prefix", settings=1, extraction_runs=0)
    # Rotation/gain construction only on a single gravity-family fixture, no candidate sweep.
    base = legacy.fixture(legacy.FAMILIES[0], 160, 0, 768)
    for transform in c.TRANSFORMS[:2]:
        r = fixtures.rotation(transform)
        c.require(
            np.allclose(r.T @ r, np.eye(3), atol=3e-16, rtol=0)
            and abs(np.linalg.det(r) - 1) < 3e-16,
            "Proper rotation",
        )
        item = {
            "kind": "invariance",
            "family": legacy.FAMILIES[0],
            "length": 256,
            "period_samples": 160,
            "phase_index": 0,
            "transformation": transform,
        }
        actual = fixtures.generate(item)
        c.require(
            np.allclose(
                kernel.magnitudes(actual),
                kernel.magnitudes(base[:, :256].copy()),
                atol=1e-14,
                rtol=0,
            ),
            "Rotation magnitude construction",
        )
    log("tiny_rotation_construction", settings=1, extraction_runs=0)
    comparisons = []
    baseline_mags = kernel.magnitudes(base[:, :256].copy())
    baseline_filtered = kernel.filter_modalities(baseline_mags)
    bj, _, _, _ = kernel.joint_correlations(baseline_filtered)
    br, bf = legacy.extract(baseline_mags, baseline_filtered[0], bj[2], legacy.RULES[1])
    for transform in c.TRANSFORMS:
        item = {
            "kind": "invariance",
            "family": legacy.FAMILIES[0],
            "length": 256,
            "period_samples": 160,
            "phase_index": 0,
            "transformation": transform,
        }
        tm = kernel.magnitudes(fixtures.generate(item))
        tf = kernel.filter_modalities(tm)
        tj, _, _, _ = kernel.joint_correlations(tf)
        tr, feat = legacy.extract(tm, tf[0], tj[2], legacy.RULES[1])
        comparisons.append(
            {
                "transformation": transform,
                "lag_equal": br["selected_lag"] == tr["selected_lag"],
                "pair_equal": br["pair_bounds_inclusive"]
                == tr["pair_bounds_inclusive"],
                "validity_equal": (bf is not None) == (feat is not None),
                "feature_max_abs_delta": float(np.max(np.abs(bf - feat)))
                if bf is not None and feat is not None
                else None,
                "repair_applied": False,
            }
        )
    log("tiny_invariance_extraction_logs", settings=1, comparisons=comparisons)
    # Canonical serialization explicitly rejects old integer-key bug and preserves -0/type.
    value = {
        "string_hist": {"1": 2},
        "float": -0.0,
        "int": 1,
        "bool": True,
        "null": None,
    }
    c.require(
        c.typed(json.loads(c.canonical(value))) == c.typed(value), "Strict round-trip"
    )
    try:
        c.canonical({1: 2})
    except RuntimeError:
        pass
    else:
        raise RuntimeError("Integer keys accepted")
    log("canonical_strict_types_and_string_keys")
    c.require(len(fixtures.keys()) == 2880, "Inventory metadata count")
    provenance = c.provenance_check()
    systems = regression.run() if include_systems else None
    return {
        "status": "pass",
        "scope": "fixed helpers, handcases, tiny predetermined fixtures only; no study grid, no96 primary panel, no freeze, no measurement",
        "legacy_18": fixed,
        "plateau_16": plateau,
        "new_checks": rows,
        "copy_provenance": provenance,
        "systems_regressions": systems,
        "source_files_sha256": c.source_index(),
        "cpu_seconds": time.process_time() - start_cpu,
        "wall_seconds": time.perf_counter() - start_wall,
    }


def main():
    result = run(include_systems=True)
    result["invocation"] = ["artifacts/joint-repeat-feasibility-r1/run.sh", "selftest"]
    c.save_json("selftests.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "new_checks": len(result["new_checks"]),
                "cpu_seconds": result["cpu_seconds"],
                "wall_seconds": result["wall_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
