"""R2 systems regressions: existing caches and complete tiny synthetic R bundles."""

import sys
from pathlib import Path
from types import SimpleNamespace

import common as c
import kernel as k
import numpy as np
import payload
import planning
import worker


def refuses(function, text):
    try:
        function()
    except (RuntimeError, ValueError, FileNotFoundError) as error:
        c.require(text in str(error), "Unexpected regression refusal: " + str(error))
        return str(error)
    raise RuntimeError("Regression unexpectedly accepted tampering: " + text)


def runtime_closure(directory):
    binding = c._runtime_binding
    if binding is None:
        raise RuntimeError("Test only after runtime registration")
    helper, transitive = sys.modules["threadpoolctl"], sys.modules["pandas"]
    c.require(
        not hasattr(helper, "__path__"), "single-file cache regression precondition"
    )
    for module in (helper, transitive):
        if not isinstance(module.__file__, str):
            raise TypeError("Expected real module source path")
        c.require(
            Path(module.__cached__).is_file(), "Existing cache regression precondition"
        )
        paths = c.module_paths({"fixture": module})
        c.require(
            str(Path(module.__file__).absolute()) in paths
            and str(Path(module.__cached__).absolute()) in paths,
            "loaded source and bytecode must both be bound",
        )
        c.assert_runtime_paths(paths, binding)
        missing = {
            "files": {
                name: record
                for name, record in binding["files"].items()
                if name != str(Path(module.__cached__).absolute())
            }
        }
        refuses(
            lambda paths=paths, missing=missing: c.assert_runtime_paths(paths, missing),
            "Unregistered executable",
        )
    # __spec__.origin can differ from __file__ and __cached__.
    synthetic_module = SimpleNamespace(
        __file__=helper.__file__,
        __cached__=helper.__cached__,
        __spec__=SimpleNamespace(origin=transitive.__file__),
    )
    paths = c.module_paths({"synthetic": synthetic_module})
    c.require(
        str(Path(str(transitive.__file__)).absolute()) in paths, "spec-origin binding"
    )
    c.assert_runtime_paths(paths, binding)
    fixture = directory / "existing-bytecode-copy.fixture"
    fixture.write_bytes(Path(helper.__cached__).read_bytes())
    stat = fixture.stat()
    mini = {
        "files": {
            str(fixture): {
                "resolved": str(fixture.resolve()),
                "sha256": c.sha(fixture),
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "mode": stat.st_mode,
            }
        }
    }
    c.assert_runtime_paths([fixture], mini)
    raw = fixture.read_bytes()
    fixture.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    tamper = refuses(
        lambda: c.assert_runtime_paths([fixture], mini), "bound executable bytes"
    )
    fixture.unlink()
    missing = refuses(
        lambda: c.assert_runtime_paths([fixture], mini), "Missing bound executable"
    )
    # Exercise the live import/exec guard with a code filename outside the binding.
    code = compile(
        "raise AssertionError('unregistered code executed')",
        str(directory / "late-import.fixture"),
        "exec",
    )
    late = refuses(lambda: exec(code, {}), "Unregistered executable")  # noqa: S102 -- exercise the execution audit refusal
    record = {
        "single_file_source": helper.__file__,
        "single_file_cache": helper.__cached__,
        "loaded_transitive_source": transitive.__file__,
        "loaded_transitive_cache": transitive.__cached__,
        "module_file_cache_spec_origin_bound": True,
        "tampering_refused": tamper,
        "missing_file_refused": missing,
        "unregistered_late_execution_refused": late,
        "package_cache_writes": 0,
        "owned_pyc_writes": 0,
    }
    c.write(directory / "runtime-closure-regressions.json", record)
    return record


def replace(path, value):
    path.write_bytes(c.canonical(value) + b"\n")


def synthetic_reference(directory, metadata):
    """Complete 20-fold fixture, analytic states, real scoring/storage/replay, zero fits.

    Every array is invented here. Public metadata supplies only labels/assignments.
    This never calls real study execute, freeze, raw IO, or R clean-display gates.
    """
    base = directory / "original-R-binding"
    prep, original_root, root = (
        base / "preparation",
        base / "canonical-orientation-v1",
        base / "replay",
    )
    for path in (prep, original_root, root):
        path.mkdir(parents=True)
    metadata = c.decode(c.canonical(metadata))
    registration = {"synthetic_only": True, "plan": {"folds": metadata["folds"]}}
    c.write(original_root / "registration.json", registration)
    metadata["historical_registration_sha256"] = c.sha(
        original_root / "registration.json"
    )
    metadata["historical_data_binding"] = {"synthetic_only": True, "dimension": 3}
    development = {
        "registration_sha256": metadata["historical_registration_sha256"],
        "data": metadata["historical_data_binding"],
        "coverage": metadata["coverage"],
        "source_coverage": {
            **metadata["source_metadata"],
            "recordings": metadata["records"],
        },
    }
    c.write(original_root / "development.json", development)
    c.write(prep / "metadata.json", metadata)
    random = np.random.Generator(np.random.PCG64(1227))
    # All historical/full windows of a trial use the same invented feature value.
    trial_values = random.normal(size=(len(metadata["trial_units"]), 3))
    clean = np.stack([trial_values[w["unit"]] for w in metadata["windows"]])
    original = {
        "registration_sha256": metadata["historical_registration_sha256"],
        "development_sha256": c.sha(original_root / "development.json"),
        "models": [],
        "inner": [],
        "outer": [],
        "rows": [],
        "threshold_files": [],
    }
    saved = {
        "models": [],
        "inner": [],
        "historical_outer": [],
        "cutoffs": {},
        "rows": [],
        "restores": [],
        "fit_calls": 0,
        "clean_gate_passed": True,
        "inherited_score_atol": 1e-6,
        "inherited_score_rtol": 1e-6,
        "historical_runtime_equivalence_claimed": False,
    }
    result = {
        "status": "synthetic_complete_replay",
        "models": [],
        "scores": [],
        "rows": [],
        "cutoffs": {},
    }
    arm_values = {arm: k.arm_data(clean, metadata, arm) for arm in ("R", "H", "F")}
    ledger = {
        "calls": [],
        "events": [],
        "both_cutoffs_before_either_refit": {},
        "actual_candidate_fit_entries": 200,
        "R_fit_entries": 0,
        "synthetic_analytic_fixture_no_fits": True,
    }

    def original_artifact(record):
        return {key: record[key] for key in ("path", "sha256")}

    for fold in metadata["folds"]:
        f = fold["fold"]
        models, pools, cutoffs = {}, {}, {}
        for arm in ("R", "H", "F"):
            values, data = arm_values[arm]
            pools[arm] = []
            for role, ids in [
                *[(i, item["train_ids"]) for i, item in enumerate(fold["inner"])],
                ("refit", fold["refit_ids"]),
            ]:
                selected = values[np.isin(data["subjects"], ids)]
                state = {
                    "mean": selected.mean(axis=0),
                    "scale": selected.std(axis=0).clip(min=1e-6),
                    "projection": np.eye(3),
                    "training_subject_ids": np.asarray(sorted(ids), dtype=np.int64),
                    "shrinkage": np.asarray(0.1, dtype=np.float64),
                }
                model = k.restore(state, values, data, ids)
                models[arm, role] = model
                if arm == "R":
                    name = f"fold{f:02}_canonical362_" + (
                        "refit" if role == "refit" else f"inner{role}"
                    )
                    original_record = original_artifact(
                        c.save_arrays(original_root, name + "_model", state)
                    )
                    stats = {
                        "train_features_sha256": k.array_hash(selected),
                        "shrinkage": 0.1,
                        "synthetic_only": True,
                        "training_units": len(selected),
                    }
                    original["models"].append(
                        {
                            "fold": f,
                            "arm": "canonical362",
                            "role": role,
                            "train_ids": ids,
                            "stats": stats,
                            "artifact": original_record,
                        }
                    )
                    saved["models"].append(
                        {
                            "fold": f,
                            "arm": "R",
                            "role": role,
                            "ids": ids,
                            "artifact": c.save_arrays(
                                root, f"models/R_{f:02}_{role}", state
                            ),
                            "historical_stats": stats,
                            "current_feature_stats_match": True,
                        }
                    )
                    saved["restores"].append({"fold": f, "role": role, "fit_calls": 0})
                else:
                    result["models"].append(
                        {
                            "fold": f,
                            "arm": arm,
                            "role": role,
                            "ids": ids,
                            "artifact": c.save_arrays(
                                root, f"models/{arm}_{f:02}_{role}", state
                            ),
                        }
                    )
                if role != "refit":
                    scored = k.score(
                        model,
                        values,
                        data,
                        fold["inner"][role]["held_ids"],
                        metadata["coverage"][str(f)]["inner"][role],
                    )
                    pools[arm].append(scored)
                    record = {
                        "fold": f,
                        "arm": arm,
                        "role": role,
                        **worker.store_score(
                            root, f"scores/{arm}_{f:02}_inner{role}", scored
                        ),
                    }
                    if arm == "R":
                        saved["inner"].append(record)
                        orig = worker.store_score(original_root, name, scored)
                        orig["artifact"] = original_artifact(orig["artifact"])
                        original["inner"].append(
                            {"fold": f, "arm": "canonical362", "role": role, **orig}
                        )
                    else:
                        result["scores"].append(record)
            cutoffs[arm] = k.cutoff(pools[arm])
        c.write(
            original_root / f"thresholds_{f:02}.json", {"canonical362": cutoffs["R"]}
        )
        original["threshold_files"].append(
            {
                "path": f"thresholds_{f:02}.json",
                "sha256": c.sha(original_root / f"thresholds_{f:02}.json"),
            }
        )
        saved["cutoffs"][str(f)] = cutoffs["R"]
        result["cutoffs"][str(f)] = {arm: cutoffs[arm] for arm in ("H", "F")}
        c.write(root / f"cutoffs_{f:02}.json", result["cutoffs"][str(f)])
        ledger["both_cutoffs_before_either_refit"][str(f)] = result["cutoffs"][str(f)]
        ledger["events"].extend(
            {"event": "fit_entered", "fold": f, "arm": arm, "role": role}
            for arm in ("H", "F")
            for role in range(4)
        )
        ledger["events"].append(
            {
                "event": "both_cutoffs_frozen",
                "fold": f,
                "sha256": c.fingerprint(result["cutoffs"][str(f)]),
            }
        )
        ledger["events"].extend(
            {"event": "fit_entered", "fold": f, "arm": arm, "role": "refit"}
            for arm in ("H", "F")
        )
        for condition in planning.CONDITIONS:
            for arm in ("R", "H", "F"):
                values, data = arm_values[arm]
                scored = k.score(
                    models[arm, "refit"],
                    values,
                    data,
                    fold["validation_ids"],
                    metadata["coverage"][str(f)]["validation"],
                )
                row = k.row(f, arm, condition, scored, cutoffs[arm]["threshold"])
                result["rows"].append(row)
                result["scores"].append(
                    {
                        "fold": f,
                        "arm": arm,
                        "condition": condition,
                        **worker.store_score(
                            root, f"scores/{arm}_{f:02}_{condition}", scored
                        ),
                    }
                )
                if arm == "R":
                    orig = worker.store_score(
                        original_root,
                        f"fold{f:02}_canonical362_refit_{condition}",
                        scored,
                    )
                    orig["artifact"] = original_artifact(orig["artifact"])
                    original["outer"].append(
                        {
                            "fold": f,
                            "arm": "canonical362",
                            "condition": condition,
                            **orig,
                        }
                    )
                    original["rows"].append({**row, "arm": "canonical362"})
                    saved["historical_outer"].append(
                        {
                            "fold": f,
                            "arm": "R_historical",
                            "condition": condition,
                            **worker.store_score(
                                root, f"historical-scores/R_{f:02}_{condition}", scored
                            ),
                            "paired_gate_ground_truth": condition
                            not in ("noise01", "noise05"),
                        }
                    )
                    if condition == "clean":
                        saved["rows"].append(row)
    c.write(original_root / "results.json", original)
    frozen = {
        "reference_root": str(original_root),
        "reference_files": payload.reference_freeze(original_root),
        "synthetic_only": True,
    }
    c.write(root / "freeze.json", frozen)
    saved["original_results_sha256"] = frozen["reference_files"]["results.json"][
        "sha256"
    ]
    c.write(root / "R-restore-gate.json", saved)
    for assigned in metadata["fit_assignments"]:
        values, data = arm_values[assigned["arm"]]
        indices = assigned["unit_indices"]
        ledger["calls"].append(
            {
                **assigned,
                "status": "fit_completed",
                "features_sha256": k.array_hash(values[indices]),
                "labels_sha256": k.array_hash(data["subjects"][indices]),
            }
        )
    c.write(root / "fit-ledger.json", ledger)
    contexts, diagnostics = {}, []
    for name, _, _, indices in worker.required_contexts(metadata):
        contexts[name] = c.save_arrays(
            root,
            "features/" + name,
            {
                "features": clean[indices],
                "complete_features": clean,
                "global_indices": np.asarray(indices, dtype=np.int64),
                "valid": np.ones(len(indices), dtype=bool),
            },
        )
        diagnostics.extend({"valid": True, "synthetic_only": True} for _ in indices)
    c.write(
        root / "preflight.json",
        {
            "contexts": contexts,
            "invalid_windows": 0,
            "invalid_scale_folds": [],
            "required_windows": len(diagnostics),
        },
    )
    c.write(root / "preflight-diagnostics.json", diagnostics)
    c.write(root / "registration.json", {"synthetic_only": True})
    attempt = {
        "cpu_budget_seconds": planning.CPU_SECONDS,
        "wall_budget_seconds": planning.WALL_SECONDS,
        "automatic_retry": False,
        "synthetic_only": True,
    }
    c.write(root / "attempt.started.json", attempt)
    c.write(root / "worker.started.json", {"attempt": attempt})
    result["accounting"] = {
        "new_actual_fits": 200,
        "new_inner_scores": 160,
        "new_outer_scores": 280,
        "R_matched_outer_scores": 140,
        "R_clean_inner_replays": 80,
        "R_restores": 100,
    }
    result["summary"] = k.comparison(result["rows"], metadata)
    for key, name in (
        ("freeze_sha256", "freeze.json"),
        ("registration_sha256", "registration.json"),
        ("R_gate_sha256", "R-restore-gate.json"),
        ("preflight_sha256", "preflight.json"),
        ("attempt_sha256", "attempt.started.json"),
        ("worker_receipt_sha256", "worker.started.json"),
    ):
        result[key] = c.sha(root / name)
    c.write(root / "results.json", result)
    c.write(
        root / "report.json",
        {
            "status": result["status"],
            "summary": result["summary"],
            "interpretation": planning.proposal()["interpretation"],
            "bindings": {key: result[key] for key in result if key.endswith("_sha256")},
        },
    )
    c.seal(root)
    return prep, root, original_root


def copied_R(directory, metadata):
    prep, root, original_root = synthetic_reference(directory, metadata)
    print("synthetic R binding: complete invented bundle stored", flush=True)
    original_files = c.file_index(original_root)
    baseline = {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file()
    }
    failures = []
    old_root = c.ROOT
    c.ROOT = prep
    try:
        with k.forbid_fits():
            clean_result = worker.verify(root)
            c.write(directory / "R-binding-baseline.json", clean_result)
            print("synthetic R binding: full fit-free worker.verify passed", flush=True)
            c.require(
                clean_result["verified"] and clean_result["fit_calls"] == 0,
                "complete synthetic compact replay",
            )

            def case(name, mutation, expected):
                saved = c.read(root / "R-restore-gate.json")
                mutation(saved)
                replace(root / "R-restore-gate.json", saved)
                result = c.read(root / "results.json")
                result["R_gate_sha256"] = c.sha(root / "R-restore-gate.json")
                # Rehash ALL local crosslinks and index, reproducing review attack.
                replace(root / "results.json", result)
                report = c.read(root / "report.json")
                report["bindings"]["R_gate_sha256"] = result["R_gate_sha256"]
                replace(root / "report.json", report)
                (root / "index.json").unlink()
                c.seal(root)
                c.check_index(root)
                error = refuses(lambda: worker.verify(root), expected)
                failures.append(
                    {
                        "case": name,
                        "error": error,
                        "local_index_sha256": c.sha(root / "index.json"),
                        "R_gate_sha256": result["R_gate_sha256"],
                        "all_local_crosslinks_rehashed": True,
                        "fit_calls": 0,
                    }
                )
                c.write(
                    directory / f"R-binding-refusal-{len(failures):02}.json",
                    failures[-1],
                )
                print("synthetic R binding refused: " + name, flush=True)
                for rel, raw in baseline.items():
                    (root / rel).write_bytes(raw)
                c.same(
                    c.file_index(original_root),
                    original_files,
                    "synthetic immutable originals unchanged",
                )

            def state(saved, field, transform):
                record = saved["models"][0]["artifact"]
                arrays = c.load_arrays(root, record)
                arrays[field] = transform(arrays[field])
                path = root / record["path"]
                with path.open("wb") as handle:
                    np.savez(handle, **arrays)
                saved["models"][0]["artifact"] = {
                    **c.artifact(root, path),
                    "schema": {
                        key: {"dtype": value.dtype.str, "shape": list(value.shape)}
                        for key, value in arrays.items()
                    },
                }

            def two_fields(saved):
                state(saved, "shrinkage", lambda _: np.asarray(0.9))
                saved["original_results_sha256"] = "0" * 64

            case(
                "exact_review_two_field_shrinkage_0.1_to_0.9_and_zero_original_digest",
                two_fields,
                "original_results_sha256",
            )
            case(
                "unused_shrinkage_alone",
                lambda saved: state(saved, "shrinkage", lambda _: np.asarray(0.9)),
                "array bits/dtype/shape",
            )
            for field in ("mean", "scale", "projection", "training_subject_ids"):
                case(
                    "model_" + field,
                    lambda saved, field=field: state(
                        saved, field, lambda values: values + 1
                    ),
                    "array bits/dtype/shape",
                )
            case(
                "model_dtype",
                lambda saved: state(
                    saved, "mean", lambda values: values.astype(np.float32)
                ),
                "array bits/dtype/shape",
            )
            case(
                "model_shape",
                lambda saved: state(
                    saved, "projection", lambda values: values.reshape(-1)
                ),
                "array bits/dtype/shape",
            )
            case(
                "original_digest_alone",
                lambda saved: saved.update(original_results_sha256="0" * 64),
                "original_results_sha256",
            )
            case(
                "state_signed_zero_bits",
                lambda saved: state(
                    saved,
                    "projection",
                    lambda values: np.where(values == 0, -0.0, values),
                ),
                "array bits/dtype/shape",
            )
            case(
                "threshold_value",
                lambda saved: saved["cutoffs"]["0"].update(threshold=0.9),
                "copied thresholds",
            )
            case(
                "threshold_type",
                lambda saved: saved["cutoffs"]["0"].update(threshold=1),
                "copied thresholds",
            )
            case(
                "training_owner",
                lambda saved: saved["models"][0]["ids"].reverse(),
                "training owner",
            )
            case(
                "historical_training_metadata",
                lambda saved: saved["models"][0]["historical_stats"].update(
                    shrinkage=0.9
                ),
                "training metadata",
            )
            case(
                "current_training_metadata",
                lambda saved: saved["models"][0].update(
                    current_feature_stats_match=False
                ),
                "feature hash diagnostic",
            )
            case(
                "model_assignment_mapping",
                lambda saved: saved["models"][0].update(role=1),
                "Duplicate R assignment",
            )
            case(
                "cache_assignment_mapping",
                lambda saved: saved["historical_outer"][0].update(condition="noise05"),
                "Duplicate R assignment",
            )
            case(
                "cache_metadata",
                lambda saved: saved["historical_outer"][0]["metadata"].update(
                    synthetic_tamper=True
                ),
                "scorer metadata/mapping",
            )
            case(
                "inner_metadata",
                lambda saved: saved["inner"][0]["metadata"].update(
                    synthetic_tamper=True
                ),
                "scorer metadata/mapping",
            )

            def score(saved, group, field, index=0):
                record = saved[group][index]["artifact"]
                arrays = c.load_arrays(root, record)
                arrays[field] = arrays[field].copy()
                arrays[field][0] += 1 if field == "labels" else 1e-12
                path = root / record["path"]
                with path.open("wb") as handle:
                    np.savez(handle, **arrays)
                record.update(c.artifact(root, path))

            for group in ("inner", "historical_outer"):
                for field in ("labels", "scores"):
                    case(
                        group + "_" + field,
                        lambda saved, group=group, field=field: score(
                            saved, group, field
                        ),
                        "array bits/dtype/shape",
                    )
            case(
                "unused_historical_noise_cache",
                lambda saved: score(saved, "historical_outer", "scores", 5),
                "array bits/dtype/shape",
            )
            case(
                "historical_noise_scope",
                lambda saved: saved["historical_outer"][5].update(
                    paired_gate_ground_truth=True
                ),
                "diagnostic scope",
            )

            def matched_cache(saved):
                result = c.read(root / "results.json")
                record = next(
                    r
                    for r in result["scores"]
                    if r["arm"] == "R" and r["condition"] == "clean"
                )
                score({"copy": [record]}, "copy", "scores")
                replace(root / "results.json", result)

            case(
                "copied_matched_R_clean_cache", matched_cache, "array bits/dtype/shape"
            )
            case(
                "clean_gate_metadata",
                lambda saved: saved["rows"][0].update(decision_sha256="0" * 64),
                "complete original R clean gate row",
            )
            # Frozen inventory omissions and root substitution are independent of local hashing.
            frozen = c.read(root / "freeze.json")
            for name, mutation, expected in (
                (
                    "missing_reference_binding",
                    lambda value: value["reference_files"].pop("results.json"),
                    "frozen R states",
                ),
                (
                    "reference_path_substitution",
                    lambda value: value.update(reference_root=str(root)),
                    "reference path/owner",
                ),
                (
                    "original_result_inventory_digest",
                    lambda value: value["reference_files"]["results.json"].update(
                        sha256="0" * 64
                    ),
                    "frozen R states",
                ),
            ):
                changed = c.decode(c.canonical(frozen))
                mutation(changed)
                replace(root / "freeze.json", changed)

                def noop(saved):
                    result = c.read(root / "results.json")
                    result["freeze_sha256"] = c.sha(root / "freeze.json")
                    replace(root / "results.json", result)
                    report = c.read(root / "report.json")
                    report["bindings"]["freeze_sha256"] = result["freeze_sha256"]
                    replace(root / "report.json", report)

                case(name, noop, expected)
            # Repacking is allowed: identical arrays, different container encoding.
            saved = c.read(root / "R-restore-gate.json")
            record = saved["models"][0]["artifact"]
            arrays = c.load_arrays(root, record)
            with (root / record["path"]).open("wb") as handle:
                np.savez_compressed(handle, **arrays)
            record.update(c.artifact(root, root / record["path"]))
            replace(root / "R-restore-gate.json", saved)
            result = c.read(root / "results.json")
            result["R_gate_sha256"] = c.sha(root / "R-restore-gate.json")
            replace(root / "results.json", result)
            report = c.read(root / "report.json")
            report["bindings"]["R_gate_sha256"] = result["R_gate_sha256"]
            replace(root / "report.json", report)
            (root / "index.json").unlink()
            c.seal(root)
            c.require(
                worker.verify(root)["verified"], "lossless NPZ repacking should pass"
            )
    finally:
        c.ROOT = old_root
    c.same(
        c.file_index(original_root),
        original_files,
        "original synthetic bundle unchanged after all attacks",
    )
    record = {
        "synthetic_only": True,
        "baseline_full_worker_verify": True,
        "lossless_repack_passed": True,
        "no_scoring_restore_aggregation_context_or_summary_stubs": True,
        "fit_calls": 0,
        "original_R_arrays_read": "synthetic fixture only",
        "actual_R_payload_reads": 0,
        "original_inventory_sha256": c.fingerprint(original_files),
        "refused_cases": failures,
        "compact_raw_archive_reads": 0,
    }
    c.write(directory / "copied-R-binding-regressions.json", record)
    return record
