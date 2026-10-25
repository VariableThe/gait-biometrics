"""Own protocol/coverage registration from completed public metadata only."""

import gzip
import hashlib
from fractions import Fraction

import common as c

DEV = (1, 3, 5, 7, 9, 11, 13, 14, 16, 17, 19, 24, 25, 27, 29)
CONDITIONS = (
    "clean",
    "rotation30",
    "rotation90",
    "gain09",
    "gain11",
    "noise01",
    "noise05",
)
CPU_SECONDS, WALL_SECONDS = 1800, 3600
SUPPORT_SHA = "80be0bc4e54ff1244a2ff8cf909dccf5ca7c3c2ee5189f58f64e9991e32aa25b"


def intervals(start, stop):
    c.require(
        type(start) is int
        and type(stop) is int
        and 0 <= start < stop
        and stop - start >= 256,
        "eligible source interval required",
    )
    old = [(a, a + 256) for a in range(start, stop - 255, 256)]
    cover = sorted({*old, (stop - 256, stop)})
    multiplicity = [sum(a <= row < b for a, b in cover) for row in range(start, stop)]
    c.require(min(multiplicity) > 0, "uncovered complete-source row")
    weights = [
        sum((Fraction(1, multiplicity[row - start]) for row in range(a, b)), Fraction())
        / (stop - start)
        for a, b in cover
    ]
    c.require(sum(weights) == 1, "nonunit exact weights")
    return old, cover, weights, multiplicity


def graph(trials, ids):
    ids = sorted(ids)
    enrollment, probes = {}, {}
    for subject in ids:
        for day, target in ((1, enrollment), (2, probes)):
            target[str(subject)] = sorted(
                t[2] for t in trials if t[:2] == (subject, day)
            )
            c.require(
                bool(target[str(subject)]), "missing registered enrollment/probe day"
            )
    labels, claims, people, trial_ids = [], [], [], []
    for subject in ids:
        for trial in probes[str(subject)]:
            for claim in ids:
                labels.append(int(claim == subject))
                claims.append(claim)
                people.append(subject)
                trial_ids.append(trial)
    return {
        "labels": labels,
        "claim_subject_ids": claims,
        "probe_subject_ids": people,
        "probe_trial_ids": trial_ids,
        "enrollment_trials": enrollment,
        "probe_trials": probes,
        "n_genuine": sum(labels),
        "n_impostor": len(labels) - sum(labels),
    }


def metadata():
    inputs = c.ROOT / "inputs"
    raw = gzip.decompress((inputs / "support-result.input.json.gz").read_bytes())
    c.require(
        hashlib.sha256(raw).hexdigest() == SUPPORT_SHA, "support metadata SHA mismatch"
    )
    source = c.decode(raw)["inventory"]
    registration = c.read(inputs / "historical-registration.input.json")
    development = c.read(inputs / "development-metadata.input.json.gz")
    secondary = c.read(inputs / "secondary-analysis.input.json")
    c.same(source["dev_ids"], list(DEV), "DEV allowlist")
    c.same(
        source["original_folds"],
        registration["plan"]["folds"],
        "unchanged historical folds/order",
    )
    records = development["source_coverage"]["recordings"]
    c.require(0 < len(records) <= 300, "at most300 metadata recording entries")
    selected, excluded, windows, trial_units = [], [], [], []
    for item in source["segments"]:
        if item["length"] < 256:
            excluded.append(item)
            continue
        a, b = item["original"]["start_row"], item["original"]["stop_row"]
        old, cover, weights, mult = intervals(a, b)
        unit = len(selected)
        selected.append(
            {
                **item,
                "historical_intervals": [list(x) for x in old],
                "cover_intervals": [list(x) for x in cover],
                "weights_exact": [str(w) for w in weights],
                "multiplicity": mult,
            }
        )
        trial_units.append([item["subject"], item["day"], item["trial"]])
        for j, ((left, right), weight) in enumerate(zip(cover, weights, strict=True)):
            windows.append(
                {
                    "unit": unit,
                    "cover_index": j,
                    "historical": (left, right) in old,
                    "rows": [left, right],
                    "source_key": [item["subject"], item["day"], item["trial"], left],
                    "member": item["member"],
                    "member_sha256": item["member_sha256"],
                    "segment_index": item["segment_index"],
                    "recording_index": item["recording_index"],
                    "weight_exact": str(weight),
                    "physical_frame_interval": [
                        item["original"]["frame_start"] + left - a,
                        item["original"]["frame_start"] + right - a,
                    ],
                }
            )
    c.require(
        len(selected) == len(set(map(tuple, trial_units))) == 299,
        "exactly299 historical trial units, no new graph",
    )
    old = [w for w in windows if w["historical"]]
    c.same(
        [w["source_key"] for w in old],
        [w["row_key"] for w in source["original_windows"]],
        "ordered588 historical row keys",
    )
    c.require(
        len(windows) == 885
        and len(old) == 588
        and sum(s["length"] for s in selected) == 183128,
        "registered885/588/183128 support",
    )
    c.require(
        len(excluded) == 29 and sum(s["length"] for s in excluded) == 1210,
        "common excluded fragments",
    )
    c.require(
        sum(
            s["length"]
            for s in excluded
            if (s["subject"], s["day"], s["trial"]) == (19, 2, 1)
        )
        == 470,
        "subject19D2T1 stays outside",
    )
    c.same(
        [s["cover_intervals"] for s in selected],
        [s["cover_intervals"] for s in secondary["eligible_segments"]],
        "secondary cover cross-check",
    )
    c.same(
        [s["weights_exact"] for s in selected],
        [s["weights_exact"] for s in secondary["eligible_segments"]],
        "secondary rational-weight cross-check",
    )
    folds, coverage, fits = source["original_folds"], {}, []
    trials = list(map(tuple, trial_units))
    for f in folds:
        number = f["fold"]
        c.require(
            type(number) is int
            and len(f["inner"]) == 4
            and len(f["refit_ids"]) == 12
            and len(f["validation_ids"]) == 3,
            "registered20x4 fold structure",
        )
        coverage[str(number)] = {
            "inner": [graph(trials, x["held_ids"]) for x in f["inner"]],
            "validation": graph(trials, f["validation_ids"]),
        }
        for arm in ("H", "F"):
            for role, ids in [
                *[(i, x["train_ids"]) for i, x in enumerate(f["inner"])],
                ("refit", f["refit_ids"]),
            ]:
                units = [i for i, t in enumerate(trials) if t[0] in ids]
                c.require(
                    len(units) in ((239, 240) if role == "refit" else (179, 180)),
                    "TRAIN trial count differs",
                )
                fits.append(
                    {
                        "fold": number,
                        "arm": arm,
                        "role": role,
                        "ids": ids,
                        "unit_indices": units,
                        "training_units": len(units),
                    }
                )
    c.require(
        len(folds) == 20
        and len(fits) == 200
        and all(sum(s in f["validation_ids"] for f in folds) == 4 for s in DEV),
        "balanced fixed folds",
    )
    c.require(
        sum(g["validation"]["n_genuine"] for g in coverage.values()) == 596
        and sum(g["validation"]["n_impostor"] for g in coverage.values()) == 1192,
        "fixed repeated596/1192 claim denominators",
    )
    for number, group in coverage.items():
        original = source["original_graphs"][number]
        # Completed inventory carries the authoritative larger original coverage.
        for role in ("inner", "validation"):
            c.same(
                group[role], original[role], "historical claim graph " + number + role
            )
    return {
        "schema": 1,
        "support_decompressed_sha256": SUPPORT_SHA,
        "historical_source_commit": "108a559bc1a83355dc85335e9f640786846f894a",
        "historical_registration_sha256": c.sha(
            inputs / "historical-registration.input.json"
        ),
        "records": records,
        "source_metadata": {
            k: v
            for k, v in development["source_coverage"].items()
            if k not in ("recordings", "archive_path")
        },
        "eligible": selected,
        "excluded_fragments": excluded,
        "windows": windows,
        "trial_units": trial_units,
        "folds": folds,
        "coverage": coverage,
        "fit_assignments": fits,
        "historical_data_binding": development["data"],
        "historical_members_sha256": development["source_coverage"]["members_sha256"],
        "official_authenticity": c.read(inputs / "official-authenticity.input.json"),
    }


def proposal():
    return {
        "schema": 1,
        "protocol": "full-support-canonical-v1",
        "status": "PREPARATION_ONLY_ROOT_REVIEW_REQUIRED",
        "systems_review_revision": {
            "revision": "R2 preparation fixes for R1 P1/P2; scientific protocol unchanged",
            "R1_archive": "../full-support-canonical-preparation-reviewed-r1",
            "R1_archive_inventory_sha256": "e198bf29b3cbd3eba01238c4f8fd15fdb72dc875716818c4086b280181a44706",
            "R1_index_sha256": "2573688d3b8b5d268bfa694ae413b28783e571fa771bb8427df132201c7a2908",
            "runtime_binding": "loaded module file/cache/spec-origin, all loaded transitive package roots, existing bytecode/native/source; checked late executable imports; no OS/kernel closure claim",
            "original_R_verification": "compact/full explicitly compare saved models/all arrays dtype-shape-bits, historical caches/metadata, cutoff types/values and original results digest against frozen reference_files and original registration/development/results/assignments; no fit; no raw archive in compact",
        },
        "candidate": "F only; H diagnostic historical-support aggregation control",
        "arms": {
            "R": "frozen100 original states/thresholds; unchanged588 window embeddings then normalized trial mean",
            "H": "historical588 mean features per old trial BEFORE own clean autoWCCN",
            "F": "full885 tiles+deduplicated finalendcap per trial; exact source-row multiplicity weights before own clean autoWCCN",
        },
        "fit_budget": {
            "new_actual_fits": 200,
            "new_inner_scores": 160,
            "new_outer_scores": 280,
            "R_restores_no_fit": 100,
            "R_clean_inner_replays": 80,
            "R_matched_outer_scores": 140,
            "R_historical_nonnoise_parity_checks": 100,
        },
        "runtime": {
            "process_cpu_seconds": CPU_SECONDS,
            "wall_seconds": WALL_SECONDS,
            "cpus": 1,
            "numerical_threads": 1,
            "cuda": False,
            "offline": True,
            "automatic_retry": False,
            "budget_expansion": False,
            "performance_budget_sufficiency": "unknown until one actual attempt",
        },
        "stress": list(CONDITIONS),
        "stress_field": {
            "version": "physical-row-normal-v1",
            "seed": 20261025,
            "key": "SHA256 canonical JSON [version,seed,[7,subject,day,trial,MCU,processed_data],member_sha256,zero_based_source_row,channel_index] -> first128bits little-endian PCG64 -> one float64 standard_normal",
            "noise_pair": "same unit standardnormal field for noise01/noise05 and all arms/covers/folds",
            "RMS_owner": "each12TRAIN clean HISTORICAL588 subset windows bothdays nativeunits; never complete tails or held probes; inner9 clean only",
            "historical_noise_caches": "different WINDOW-keyed realization; diagnostic-only, never paired gates",
            "nonnoise_parity": "Rclean/rotation30/rotation90/gain09/gain11 ordered caches exact decisions and1e-6 allclose; metadata exact",
        },
        "primary": "equal fold mean(FcleanFRR-HcleanFRR), exact fractions; support-favoring iff strictlynegative and FcleanFAR<=HcleanFAR and<=1%",
        "replacement": "FvsR through unchanged engine.robustness_selection.summarize_nuisance_study; all33 gates authoritative; no H nomination",
        "score_replay": {
            "atol": 1e-6,
            "rtol": 1e-6,
            "decision_identity": True,
            "metric_arithmetic": "use cached scores only; threshold exact; never recompute cutoff from freshscores",
        },
        "R_clean_reference_display_percent": {
            "FAR": "0.5",
            "FRR": "14.787356",
            "EER": "5.876437",
        },
        "preflight": "persist complete source/geometry/features/diagnostics for clean885 plus all required validation-day2 stresswindows in20fold6conditions; clean enrollment reused; complete named failures before any new fit, no abstention/drop/impute",
        "payload_access": "future execute/fullverify only after full freeze validation; validated exact DEVZipInfo only, no archive testzip/fullhash/unselected memberhash, no downloads/fallback",
        "permanent_stop": "STOPPED N446 P200phase2/18 fails4/960; STOP_NO_RETRY_NO_PARAMETER_CHANGE remains FINAL; this is an independent nonsegmented hypothesis",
        "interpretation": "adaptive exploratory reusedDEV, dependent repeated claims and overlapping windows; R-H changes aggregation/trainingunit; H-F supportestimator includesendcap/weights; F-R is not puretaileffect; featuremeans not uniform nonlinear rowequivalence, no independence/fresh confirmation/SOTA/fullsystemreplication/deployment claim",
        "research": [
            {
                "url": "https://arxiv.org/html/1606.03238v3#S6",
                "scope": "IDNet evidence accumulation; not this featuremean estimator",
            },
            {
                "url": "https://www.isca-archive.org/interspeech_2006/hatch06_interspeech.pdf",
                "scope": "section2 TRAIN within-class covariance",
            },
            {
                "url": "https://research.ibm.com/publications/an-evaluation-of-error-confidence-interval-estimation-methods",
                "scope": "biometric repeated claims are dependent",
            },
        ],
        "unresolved": [
            "Current arithmetic/validation source differs from original registration. No historical runtimeclosure assertion; actual R replay must satisfy exact decisions/counts and inherited1e-6 scores.",
            "1800CPU/3600wall feasibility unknown; no real timing available and no retry/expansion authorized.",
        ],
    }
