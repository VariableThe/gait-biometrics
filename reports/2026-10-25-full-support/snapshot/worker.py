"""Future one-attempt comparison and fit-free compact/full replay.

No work runs on import. The driver owns freeze/attempt/runtime validation.
"""

import time
from fractions import Fraction
from pathlib import Path

import common as c
import kernel as k
import numpy as np
import payload
import planning
import reference_binding


def store_score(root, name, scored):
    artifact = c.save_arrays(
        root, name, {key: scored[key] for key in ("labels", "scores")}
    )
    return {"artifact": artifact, "metadata": scored["metadata"]}


def load_score(root, record):
    return {**c.load_arrays(root, record["artifact"]), "metadata": record["metadata"]}


def model_state(model):
    return {
        **model.state_dict(),
        "shrinkage": np.asarray(model.shrinkage_, dtype=np.float64),
    }


def old_arrays(path):
    with np.load(path, allow_pickle=False) as file:
        return {key: file[key] for key in file.files}


def old_artifact(root, record):
    path = Path(root) / record["path"]
    c.require(
        not path.is_symlink()
        and path.resolve().is_relative_to(Path(root).resolve())
        and c.sha(path) == record["sha256"],
        "Frozen historical artifact provenance mismatch",
    )
    return path


def required_contexts(metadata):
    result = [("clean", None, "clean", list(range(len(metadata["windows"]))))]
    for fold in metadata["folds"]:
        selected = [
            i
            for i, w in enumerate(metadata["windows"])
            if w["source_key"][0] in fold["validation_ids"] and w["source_key"][1] == 2
        ]
        for condition in planning.CONDITIONS[1:]:
            result.append(
                (f"fold{fold['fold']:02}_{condition}", fold, condition, selected)
            )
    return result


def preflight_all(root, metadata, sources):
    values = [s["values"] for s in sources]
    clean = k.slice_windows(metadata, values)
    scales, scale_records, diagnostics, records = {}, {}, [], {}
    for f in metadata["folds"]:
        scale, record = k.training_rms(clean, metadata, f["refit_ids"])
        scales[str(f["fold"])], scale_records[str(f["fold"])] = scale, record
    # Persist every RMS even if zero/NaN BEFORE enforcing validity.
    c.write(root / "train-rms.json", scale_records)
    c.save_arrays(root, "train-rms", {str(f): scale for f, scale in scales.items()})
    fields = []
    for unit, (source, item) in enumerate(
        zip(sources, metadata["eligible"], strict=True)
    ):
        fields.append(k.physical_field(item, source["rows"]))
    source_record = c.save_arrays(
        root,
        "source-segments",
        {
            f"u{i:03}_{key}": value
            for i, source in enumerate(sources)
            for key, value in source.items()
        },
    )
    field_record = c.save_arrays(
        root, "source-row-fields", {f"u{i:03}": field for i, field in enumerate(fields)}
    )
    clean_features = None
    invalid = []
    for name, fold, condition, indices in required_contexts(metadata):
        if condition == "clean":
            windows = clean
        else:
            # Generate each complete validation segment ONCE, then slice all covers.
            # Enrollment is clean; unscored TRAIN/held probes are never perturbed.
            perturbed = list(values)
            if scale_records[str(fold["fold"])]["valid"]:
                for unit, (source, item) in enumerate(
                    zip(values, metadata["eligible"], strict=True)
                ):
                    if item["subject"] in fold["validation_ids"] and item["day"] == 2:
                        try:
                            perturbed[unit] = k.perturb(
                                item,
                                source,
                                condition,
                                scales[str(fold["fold"])],
                                fields[unit],
                            )
                        except (RuntimeError, ValueError, ArithmeticError):
                            # Continue every required window's diagnostic ledger.
                            perturbed[unit] = np.full_like(source, np.nan)
                windows = k.slice_windows(metadata, perturbed)[indices]
            else:
                windows = np.full((len(indices), 6, 256), np.nan, dtype=np.float32)
        arrays, diagnostic = k.preflight(
            windows, [metadata["windows"][i]["source_key"] for i in indices]
        )
        for i, item in zip(indices, diagnostic, strict=True):
            item["global_window_index"] = i
            item["context"] = name
            item["historical_R_required"] = metadata["windows"][i]["historical"]
            if not item["valid"]:
                invalid.append(item)
        diagnostics.extend(diagnostic)
        if condition == "clean":
            clean_features = arrays["features"].copy()
        if clean_features is None:
            raise RuntimeError("Clean preflight must precede every stress context")
        features = clean_features.copy()
        features[indices] = arrays["features"]
        arrays["complete_features"] = features
        arrays["global_indices"] = np.asarray(indices, dtype=np.int64)
        records[name] = c.save_arrays(root, "features/" + name, arrays)
        # This is output progress, not a scientific interpretation.
        if condition in ("clean", "noise05"):
            print("preflight " + name, flush=True)
    record = {
        "sources": source_record,
        "physical_fields": field_record,
        "contexts": records,
        "context_count": len(records),
        "diagnostics": "preflight-diagnostics.json",
        "required_windows": len(diagnostics),
        "invalid_windows": len(invalid),
        "invalid_scale_folds": [f for f, r in scale_records.items() if not r["valid"]],
        "all_invalid_names": invalid,
        "no_fit_entered": True,
    }
    c.write(root / "preflight-diagnostics.json", diagnostics)
    c.write(root / "preflight.json", record)
    c.require(
        not invalid and not record["invalid_scale_folds"],
        "ABORT whole comparison: complete preflight diagnostic ledger persisted before any new fit",
    )
    return record


def features(root, preflight, name):
    return c.load_arrays(root, preflight["contexts"][name])["complete_features"]


def reference_restore(root, frozen, metadata, clean_features):
    """Read historical R only during actual attempt. Never regenerate a fit."""
    reference_root, original, _indexes, _cutoffs = reference_binding.originals(
        frozen, metadata
    )
    c.require(
        c.sha(reference_root / "registration.json") == original["registration_sha256"]
        and c.sha(reference_root / "development.json")
        == original["development_sha256"],
        "Historical R report bindings invalid",
    )
    rms_records = c.read(root / "train-rms.json")
    c.same(
        {f: r["values_nullable"] for f, r in rms_records.items()},
        original["scales"],
        "inherited TRAIN12 historical-window triad RMS source/units/arithmetic",
    )
    model_records = {
        (r["fold"], r["role"]): r
        for r in original["models"]
        if r["arm"] == "canonical362"
    }
    inner = {
        (r["fold"], r["role"]): r
        for r in original["inner"]
        if r["arm"] == "canonical362"
    }
    outer = {
        (r["fold"], r["condition"]): r
        for r in original["outer"]
        if r["arm"] == "canonical362"
    }
    expected_models = {(f, role) for f in range(20) for role in (*range(4), "refit")}
    c.require(
        set(model_records) == expected_models
        and set(inner) == {(f, i) for f in range(20) for i in range(4)}
        and set(outer)
        == {(f, condition) for f in range(20) for condition in planning.CONDITIONS},
        "Complete100 R states/80inner/140outer required",
    )
    values, data = k.arm_data(clean_features, metadata, "R")
    models, saved_models, saved_inner, saved_original_outer, cutoffs, rows, restored = (
        {},
        [],
        [],
        [],
        {},
        [],
        [],
    )
    with k.forbid_fits():
        for fold in metadata["folds"]:
            f = fold["fold"]
            cutoff = c.read(reference_root / f"thresholds_{f:02}.json")["canonical362"]
            cutoffs[str(f)] = cutoff
            pools = []
            for role, ids in [
                *[(i, item["train_ids"]) for i, item in enumerate(fold["inner"])],
                ("refit", fold["refit_ids"]),
            ]:
                record = model_records[f, role]
                c.same(record["train_ids"], ids, "R original TRAIN assignment")
                state = old_arrays(old_artifact(reference_root, record["artifact"]))
                model = k.restore(state, values, data, ids, check_statistics=False)
                models[f, role] = model
                saved_models.append(
                    {
                        "fold": f,
                        "arm": "R",
                        "role": role,
                        "ids": ids,
                        "artifact": c.save_arrays(
                            root, f"models/R_{f:02}_{role}", state
                        ),
                        "historical_stats": record["stats"],
                        "current_feature_stats_match": record["stats"][
                            "train_features_sha256"
                        ]
                        == k.array_hash(values[np.isin(data["subjects"], ids)]),
                    }
                )
                restored.append({"fold": f, "role": role, "fit_calls": 0})
                if role != "refit":
                    source = inner[f, role]
                    cache = {
                        **old_arrays(old_artifact(reference_root, source["artifact"])),
                        "metadata": source["metadata"],
                    }
                    fresh = k.score(
                        model,
                        values,
                        data,
                        fold["inner"][role]["held_ids"],
                        metadata["coverage"][str(f)]["inner"][role],
                    )
                    k.replay(fresh, cache, cutoff["threshold"])
                    saved_inner.append(
                        {
                            "fold": f,
                            "arm": "R",
                            "role": role,
                            **store_score(root, f"scores/R_{f:02}_inner{role}", cache),
                        }
                    )
                    pools.append(cache)
                else:
                    source = outer[f, "clean"]
                    cache = {
                        **old_arrays(old_artifact(reference_root, source["artifact"])),
                        "metadata": source["metadata"],
                    }
                    fresh = k.score(
                        model,
                        values,
                        data,
                        fold["validation_ids"],
                        metadata["coverage"][str(f)]["validation"],
                    )
                    k.replay(fresh, cache, cutoff["threshold"])
                    rows.append(k.row(f, "R", "clean", cache, cutoff["threshold"]))
            c.same(
                k.cutoff(pools),
                cutoff,
                "R exact original cutoff from cached clean inner pools",
            )
            for condition in planning.CONDITIONS:
                source = outer[f, condition]
                cache = {
                    **old_arrays(old_artifact(reference_root, source["artifact"])),
                    "metadata": source["metadata"],
                }
                saved_original_outer.append(
                    {
                        "fold": f,
                        "arm": "R_historical",
                        "condition": condition,
                        **store_score(
                            root, f"historical-scores/R_{f:02}_{condition}", cache
                        ),
                        "paired_gate_ground_truth": condition
                        not in ("noise01", "noise05"),
                    }
                )
    historical_rows = [
        r
        for r in original["rows"]
        if r["arm"] == "canonical362" and r["condition"] == "clean"
    ]
    for current, original_row in zip(rows, historical_rows, strict=True):
        c.same(
            current["metrics"],
            original_row["metrics"],
            "R all exact original cached clean metrics/counts",
        )
        c.same(
            current["pair_errors"],
            original_row["pair_errors"],
            "R original claimant counts",
        )
    mean = lambda name: (
        sum((Fraction(str(r["metrics"][name])) for r in rows), Fraction()) / 20
    )
    far = (
        sum(
            (
                Fraction(
                    r["metrics"]["fixed_threshold"]["false_accepts"],
                    r["metrics"]["fixed_threshold"]["n_impostor"],
                )
                for r in rows
            ),
            Fraction(),
        )
        / 20
    )
    frr = (
        sum(
            (
                Fraction(
                    r["metrics"]["fixed_threshold"]["false_rejects"],
                    r["metrics"]["fixed_threshold"]["n_genuine"],
                )
                for r in rows
            ),
            Fraction(),
        )
        / 20
    )
    # Published six-decimal percent displays are an additional recognizable audit.
    c.require(
        far == Fraction(1, 200)
        and abs(float(frr) * 100 - 14.787356) <= 0.0000005
        and abs(float(mean("eer")) * 100 - 5.876437) <= 0.0000005,
        "R clean reference FAR/FRR/EER mismatch: STOP before candidate fits",
    )
    record = {
        "clean_gate_passed": True,
        "models": saved_models,
        "inner": saved_inner,
        "historical_outer": saved_original_outer,
        "cutoffs": cutoffs,
        "rows": rows,
        "restores": restored,
        "fit_calls": 0,
        "original_results_sha256": c.sha(reference_root / "results.json"),
        "inherited_score_atol": 1e-6,
        "inherited_score_rtol": 1e-6,
        "historical_runtime_equivalence_claimed": False,
    }
    reference_binding.verify_copies(root, frozen, metadata, record, clean_features)
    c.write(root / "R-restore-gate.json", record)
    return models, record


def execute(root, frozen):
    root = Path(root)
    metadata = c.read(c.ROOT / "metadata.json")
    start_cpu, start_wall = time.process_time(), time.monotonic()
    models, scores, rows, threshold_records, calls = [], [], [], {}, []
    ledger = None
    try:
        c.write(
            root / "registration.json",
            {
                "proposal": c.read(c.ROOT / "proposal.json"),
                "preparation_index_sha256": frozen["preparation_index_sha256"],
                "freeze_sha256": c.sha(root / "freeze.json"),
                "metadata_sha256": c.sha(c.ROOT / "metadata.json"),
                "physical_coordinates": payload.physical_registration(metadata),
                "claim_graphs": metadata["coverage"],
                "fit_assignments": metadata["fit_assignments"],
            },
        )
        sources, members = payload.load(frozen["archive"], metadata)
        c.write(
            root / "member-access.json",
            {
                "members": members,
                "nonDEV_payload_reads": 0,
                "archive_testzip": False,
                "whole_archive_hash": False,
            },
        )
        preflight = preflight_all(root, metadata, sources)
        del sources
        clean = features(root, preflight, "clean")
        reference_models, reference = reference_restore(root, frozen, metadata, clean)
        # Complete historical clean-control replay has passed before FitLedger exists.
        ledger = k.FitLedger(metadata["fit_assignments"])
        rows.extend(reference["rows"])
        original_outer = {
            (r["fold"], r["condition"]): r for r in reference["historical_outer"]
        }
        for fold in metadata["folds"]:
            f = fold["fold"]
            graphs = metadata["coverage"][str(f)]
            clean_arms = {
                arm: k.arm_data(clean, metadata, arm) for arm in ("R", "H", "F")
            }
            cutoffs = {}
            for arm in ("H", "F"):
                values, data = clean_arms[arm]
                pools = []
                for role, inner in enumerate(fold["inner"]):
                    model = ledger.fit(arm, f, role, values, data)
                    models.append(
                        {
                            "fold": f,
                            "arm": arm,
                            "role": role,
                            "ids": inner["train_ids"],
                            "artifact": c.save_arrays(
                                root, f"models/{arm}_{f:02}_{role}", model_state(model)
                            ),
                        }
                    )
                    scored = k.score(
                        model, values, data, inner["held_ids"], graphs["inner"][role]
                    )
                    pools.append(scored)
                    scores.append(
                        {
                            "fold": f,
                            "arm": arm,
                            "role": role,
                            **store_score(
                                root, f"scores/{arm}_{f:02}_inner{role}", scored
                            ),
                        }
                    )
                cutoffs[arm] = k.cutoff(pools)
            # Persist BOTH cutoffs before either of the two twelve-person refits.
            c.write(root / f"cutoffs_{f:02}.json", cutoffs)
            threshold_records[str(f)] = cutoffs
            ledger.freeze_cutoffs(f, cutoffs)
            refits = {"R": reference_models[f, "refit"]}
            for arm in ("H", "F"):
                values, data = clean_arms[arm]
                model = ledger.fit(arm, f, "refit", values, data)
                refits[arm] = model
                models.append(
                    {
                        "fold": f,
                        "arm": arm,
                        "role": "refit",
                        "ids": fold["refit_ids"],
                        "artifact": c.save_arrays(
                            root, f"models/{arm}_{f:02}_refit", model_state(model)
                        ),
                    }
                )
            for condition in planning.CONDITIONS:
                current = (
                    clean
                    if condition == "clean"
                    else features(root, preflight, f"fold{f:02}_{condition}")
                )
                for arm in ("R", "H", "F"):
                    threshold = (
                        reference["cutoffs"][str(f)]["threshold"]
                        if arm == "R"
                        else cutoffs[arm]["threshold"]
                    )
                    values, data = (
                        clean_arms[arm]
                        if condition == "clean"
                        else k.arm_data(current, metadata, arm)
                    )
                    if arm == "R" and condition == "clean":
                        scored = load_score(root, original_outer[f, "clean"])
                    else:
                        scored = k.score(
                            refits[arm],
                            values,
                            data,
                            fold["validation_ids"],
                            graphs["validation"],
                        )
                        if arm == "R" and condition not in ("noise01", "noise05"):
                            scored = k.replay(
                                scored,
                                load_score(root, original_outer[f, condition]),
                                threshold,
                            )
                    record = {
                        "fold": f,
                        "arm": arm,
                        "condition": condition,
                        **store_score(root, f"scores/{arm}_{f:02}_{condition}", scored),
                    }
                    scores.append(record)
                    if not (arm == "R" and condition == "clean"):
                        rows.append(k.row(f, arm, condition, scored, threshold))
            print(
                f"fold {f + 1}/20: {len(ledger.calls)} actual candidate fits",
                flush=True,
            )
        c.require(
            len(ledger.calls) == 200
            and all(x["status"] == "fit_completed" for x in ledger.calls),
            "actual200 complete fits required",
        )
        summary = k.comparison(rows, metadata)
        c.same(
            c.runtime_identity(),
            c.read(c.ROOT / "runtime.preparation.json"),
            "postrun source/bytecode/native/runtime closure",
        )
        result = {
            "schema": 1,
            "status": "completed_one_attempt",
            "models": models,
            "scores": scores,
            "rows": rows,
            "cutoffs": threshold_records,
            "R_gate_sha256": c.sha(root / "R-restore-gate.json"),
            "preflight_sha256": c.sha(root / "preflight.json"),
            "registration_sha256": c.sha(root / "registration.json"),
            "freeze_sha256": c.sha(root / "freeze.json"),
            "attempt_sha256": c.sha(root / "attempt.started.json"),
            "worker_receipt_sha256": c.sha(root / "worker.started.json"),
            "summary": summary,
            "accounting": {
                "new_actual_fits": len(ledger.calls),
                "new_inner_scores": sum("role" in s for s in scores),
                "new_outer_scores": sum(
                    "condition" in s and s["arm"] != "R" for s in scores
                ),
                "R_matched_outer_scores": sum(s["arm"] == "R" for s in scores),
                "R_clean_inner_replays": len(reference["inner"]),
                "R_restores": len(reference["models"]),
            },
            "timing": {
                "worker_cpu_seconds": time.process_time() - start_cpu,
                "worker_wall_seconds": time.monotonic() - start_wall,
                "budget_cpu_seconds": 1800,
                "budget_wall_seconds": 3600,
            },
        }
        c.same(
            result["accounting"],
            {
                "new_actual_fits": 200,
                "new_inner_scores": 160,
                "new_outer_scores": 280,
                "R_matched_outer_scores": 140,
                "R_clean_inner_replays": 80,
                "R_restores": 100,
            },
            "exact fit/score accounting",
        )
        c.write(root / "results.json", result)
        c.write(
            root / "report.json",
            {
                "status": result["status"],
                "summary": summary,
                "interpretation": planning.proposal()["interpretation"],
                "bindings": {
                    key: result[key]
                    for key in (
                        "freeze_sha256",
                        "registration_sha256",
                        "R_gate_sha256",
                        "preflight_sha256",
                        "attempt_sha256",
                        "worker_receipt_sha256",
                    )
                },
            },
        )
    except BaseException as error:
        c.write(
            root / "failure.json",
            {
                "status": "STOP_no_automatic_retry_or_parameter_change",
                "error_type": type(error).__name__,
                "message": str(error),
                "attempt_consumed": True,
                "completed_models": models,
                "completed_scores": scores,
                "completed_rows": rows,
            },
        )
        raise
    finally:
        calls = [] if ledger is None else ledger.calls
        c.write(
            root / "fit-ledger.json",
            {
                "actual_candidate_fit_entries": len(calls),
                "calls": calls,
                "R_fit_entries": 0,
                "events": [] if ledger is None else ledger.events,
                "both_cutoffs_before_either_refit": {}
                if ledger is None
                else {str(f): value for f, value in ledger.cutoffs.items()},
            },
        )
        c.seal(root)


def verify(root, *, archive=None):
    """Compact includes original R numeric equality; raw archive replay is separate."""
    root = Path(root)
    c.check_index(root)
    result, reference, preflight = (
        c.read(root / name)
        for name in ("results.json", "R-restore-gate.json", "preflight.json")
    )
    metadata = c.read(c.ROOT / "metadata.json")
    frozen = c.read(root / "freeze.json")
    for key, name in (
        ("freeze_sha256", "freeze.json"),
        ("registration_sha256", "registration.json"),
        ("R_gate_sha256", "R-restore-gate.json"),
        ("preflight_sha256", "preflight.json"),
        ("attempt_sha256", "attempt.started.json"),
        ("worker_receipt_sha256", "worker.started.json"),
    ):
        c.require(result[key] == c.sha(root / name), "Result binding mismatch: " + name)
    original_binding = reference_binding.verify_copies(
        root,
        frozen,
        metadata,
        reference,
        features(root, preflight, "clean"),
        matched_scores=result["scores"],
    )
    c.require(
        preflight["invalid_windows"] == 0
        and not preflight["invalid_scale_folds"]
        and reference["clean_gate_passed"] is True,
        "required preflight/R gate not passed",
    )
    c.require(
        reference["fit_calls"] == 0
        and len(reference["restores"]) == 100
        and all(r["fit_calls"] == 0 for r in reference["restores"]),
        "R restoration must be fit free",
    )
    c.same(
        result["accounting"],
        {
            "new_actual_fits": 200,
            "new_inner_scores": 160,
            "new_outer_scores": 280,
            "R_matched_outer_scores": 140,
            "R_clean_inner_replays": 80,
            "R_restores": 100,
        },
        "saved exact fit/score budget",
    )
    attempt, receipt = (
        c.read(root / "attempt.started.json"),
        c.read(root / "worker.started.json"),
    )
    c.same(attempt, receipt["attempt"], "single-use worker receipt/attempt binding")
    c.require(
        attempt["cpu_budget_seconds"] == planning.CPU_SECONDS
        and attempt["wall_budget_seconds"] == planning.WALL_SECONDS
        and attempt["automatic_retry"] is False,
        "attempt deadline/retry policy",
    )
    clean = features(root, preflight, "clean")
    expected_contexts = {
        name: indices for name, _, _, indices in required_contexts(metadata)
    }
    c.same(
        sorted(preflight["contexts"]),
        sorted(expected_contexts),
        "complete preflight context assignments",
    )
    diagnostics = c.read(root / "preflight-diagnostics.json")
    c.require(
        len(diagnostics)
        == sum(map(len, expected_contexts.values()))
        == preflight["required_windows"]
        and all(d["valid"] is True for d in diagnostics),
        "complete valid preflight diagnostic denominator",
    )
    for name, indices in expected_contexts.items():
        arrays = c.load_arrays(root, preflight["contexts"][name])
        c.same(
            arrays["global_indices"].tolist(),
            indices,
            "registered preflight physical interval order",
        )
        c.require(
            arrays["valid"].all()
            and np.isfinite(arrays["features"]).all()
            and np.array_equal(
                arrays["features"], arrays["complete_features"][indices]
            ),
            "preflight feature/mask binding",
        )
        unchanged = np.ones(len(metadata["windows"]), dtype=bool)
        unchanged[indices] = False
        c.require(
            np.array_equal(arrays["complete_features"][unchanged], clean[unchanged]),
            "nonprobe and clean enrollment features unchanged",
        )
    models = {
        (r["fold"], r["arm"], r["role"]): r
        for r in [*reference["models"], *result["models"]]
    }
    inner = {
        (r["fold"], r["arm"], r["role"]): r
        for r in [*reference["inner"], *[s for s in result["scores"] if "role" in s]]
    }
    outer = {
        (r["fold"], r["arm"], r["condition"]): r
        for r in result["scores"]
        if "condition" in r
    }
    c.require(
        len(models) == 300 and len(inner) == 240 and len(outer) == 420,
        "replay complete R/H/F models/scores required",
    )
    ledger = c.read(root / "fit-ledger.json")
    c.require(
        ledger["actual_candidate_fit_entries"] == len(ledger["calls"]) == 200
        and ledger["R_fit_entries"] == 0,
        "actual optimizer ledger counts",
    )
    expected_events = []
    for fold in metadata["folds"]:
        f = fold["fold"]
        expected_events.extend(
            {"event": "fit_entered", "fold": f, "arm": arm, "role": role}
            for arm in ("H", "F")
            for role in range(4)
        )
        c.same(
            c.read(root / f"cutoffs_{f:02}.json"),
            result["cutoffs"][str(f)],
            "persisted both cutoffs",
        )
        expected_events.append(
            {
                "event": "both_cutoffs_frozen",
                "fold": f,
                "sha256": c.fingerprint(result["cutoffs"][str(f)]),
            }
        )
        expected_events.extend(
            {"event": "fit_entered", "fold": f, "arm": arm, "role": "refit"}
            for arm in ("H", "F")
        )
    c.same(
        ledger["events"],
        expected_events,
        "both cutoff freezes precede both refits, actual event order",
    )
    c.same(
        ledger["both_cutoffs_before_either_refit"],
        result["cutoffs"],
        "cutoff freeze ledger",
    )
    assigned = {
        (a["fold"], a["arm"], a["role"]): a for a in metadata["fit_assignments"]
    }
    for call in ledger["calls"]:
        identity = call["fold"], call["arm"], call["role"]
        c.same(
            {key: call[key] for key in assigned[identity]},
            assigned[identity],
            "actual fit assignment",
        )
        values, data = k.arm_data(clean, metadata, call["arm"])
        indices = assigned[identity]["unit_indices"]
        c.require(
            call["status"] == "fit_completed"
            and call["features_sha256"] == k.array_hash(values[indices])
            and call["labels_sha256"] == k.array_hash(data["subjects"][indices]),
            "actual clean training-unit feature/label provenance",
        )
    rows = []
    with k.forbid_fits():
        for fold in metadata["folds"]:
            f = fold["fold"]
            for arm in ("R", "H", "F"):
                values, data = k.arm_data(clean, metadata, arm)
                threshold = (
                    reference["cutoffs"][str(f)]
                    if arm == "R"
                    else result["cutoffs"][str(f)][arm]
                )
                pools = []
                for role, assigned in enumerate(fold["inner"]):
                    model = k.restore(
                        c.load_arrays(root, models[f, arm, role]["artifact"]),
                        values,
                        data,
                        assigned["train_ids"],
                        check_statistics=arm != "R",
                    )
                    fresh = k.score(
                        model,
                        values,
                        data,
                        assigned["held_ids"],
                        metadata["coverage"][str(f)]["inner"][role],
                    )
                    cached = load_score(root, inner[f, arm, role])
                    pools.append(k.replay(fresh, cached, threshold["threshold"]))
                c.same(
                    k.cutoff(pools),
                    threshold,
                    "threshold arithmetic from CACHED inner scores",
                )
                model = k.restore(
                    c.load_arrays(root, models[f, arm, "refit"]["artifact"]),
                    values,
                    data,
                    fold["refit_ids"],
                    check_statistics=arm != "R",
                )
                for condition in planning.CONDITIONS:
                    current = (
                        clean
                        if condition == "clean"
                        else features(root, preflight, f"fold{f:02}_{condition}")
                    )
                    values_c, data_c = k.arm_data(current, metadata, arm)
                    fresh = k.score(
                        model,
                        values_c,
                        data_c,
                        fold["validation_ids"],
                        metadata["coverage"][str(f)]["validation"],
                    )
                    cached = k.replay(
                        fresh,
                        load_score(root, outer[f, arm, condition]),
                        threshold["threshold"],
                    )
                    rows.append(
                        k.row(f, arm, condition, cached, threshold["threshold"])
                    )
    c.same(
        sorted(rows, key=lambda r: (r["fold"], r["arm"], r["condition"])),
        sorted(result["rows"], key=lambda r: (r["fold"], r["arm"], r["condition"])),
        "cached metrics/ordered decisions/rational ranking",
    )
    c.same(
        k.comparison(rows, metadata),
        result["summary"],
        "primary and unchanged complete replacement gates",
    )
    report = c.read(root / "report.json")
    c.same(
        report,
        {
            "status": result["status"],
            "summary": result["summary"],
            "interpretation": planning.proposal()["interpretation"],
            "bindings": {
                key: result[key]
                for key in (
                    "freeze_sha256",
                    "registration_sha256",
                    "R_gate_sha256",
                    "preflight_sha256",
                    "attempt_sha256",
                    "worker_receipt_sha256",
                )
            },
        },
        "report/result/freeze/attempt/source/runtime crosslinks",
    )
    if archive is not None:
        verify_raw(root, metadata, preflight, archive)
    return {
        "verified": True,
        "scope": "full DEV raw feature/source replay"
        if archive
        else "compact fit-free replay from saved raw FEATURES plus explicit frozen original R model/cache/metadata/threshold comparison; no source signal arrays/archive loaded",
        "original_R_binding": original_binding,
        "fit_calls": 0,
        "score_sets_replayed": 660,
        "index_sha256": c.sha(root / "index.json"),
        "summary": result["summary"],
    }


def verify_raw(root, metadata, preflight, archive):
    sources, ledger = payload.load(archive, metadata)
    c.same(
        {
            "members": ledger,
            "nonDEV_payload_reads": 0,
            "archive_testzip": False,
            "whole_archive_hash": False,
        },
        c.read(root / "member-access.json"),
        "exact allowed raw member access ledger",
    )
    saved_sources = c.load_arrays(root, preflight["sources"])
    fields = c.load_arrays(root, preflight["physical_fields"])
    for unit, (source, item) in enumerate(
        zip(sources, metadata["eligible"], strict=True)
    ):
        for key, value in source.items():
            c.require(
                np.array_equal(value, saved_sources[f"u{unit:03}_{key}"]),
                "raw source arrays/frame/row identity",
            )
        c.require(
            np.array_equal(
                k.physical_field(item, source["rows"]), fields[f"u{unit:03}"]
            ),
            "registered physical noise field mismatch",
        )
    values = [s["values"] for s in sources]
    clean = k.slice_windows(metadata, values)
    for name, fold, condition, indices in required_contexts(metadata):
        if condition == "clean":
            windows = clean
        else:
            scale, _ = k.training_rms(clean, metadata, fold["refit_ids"])
            perturbed = [
                k.perturb(item, value, condition, scale, fields[f"u{unit:03}"])
                if item["subject"] in fold["validation_ids"] and item["day"] == 2
                else value
                for unit, (item, value) in enumerate(
                    zip(metadata["eligible"], values, strict=True)
                )
            ]
            windows = k.slice_windows(metadata, perturbed)[indices]
        fresh, _ = k.preflight(
            windows, [metadata["windows"][i]["source_key"] for i in indices]
        )
        cached = c.load_arrays(root, preflight["contexts"][name])
        for key, value in fresh.items():
            c.require(
                value.dtype == cached[key].dtype
                and np.array_equal(value, cached[key], equal_nan=True),
                "full required source/geometry/feature replay: " + name + "/" + key,
            )
