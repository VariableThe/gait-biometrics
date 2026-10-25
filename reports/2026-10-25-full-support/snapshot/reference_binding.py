"""Fit-free original-R binding. Numeric access is future execute/verify ONLY.

Preparation calls this exclusively with complete, isolated synthetic references.
Local crosslinks cannot substitute for the immutable frozen reference inventory.
"""

from pathlib import Path

import common as c
import kernel as k
import numpy as np
import payload
import planning


def exact_arrays(actual, expected, name):
    c.same(sorted(actual), sorted(expected), name + " array keys")
    for key, original in expected.items():
        copied = actual[key]
        c.require(
            not copied.dtype.hasobject
            and not original.dtype.hasobject
            and copied.dtype == original.dtype
            and copied.shape == original.shape
            and copied.tobytes(order="C") == original.tobytes(order="C"),
            "Original R array bits/dtype/shape mismatch: " + name + "/" + key,
        )


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        c.require(
            len(archive.files) == len(set(archive.files)), "Duplicate R NPZ array"
        )
        return {name: archive[name] for name in archive.files}


def indexed(records, fields, expected, name):
    result = {}
    for record in records:
        identity = [record[key] for key in fields]
        key = c.fingerprint(identity)
        c.require(key not in result, "Duplicate R assignment: " + name)
        result[key] = record
    c.same(
        sorted(result),
        sorted(c.fingerprint(list(key)) for key in expected),
        "complete typed R assignments: " + name,
    )
    return {key: result[c.fingerprint(list(key))] for key in expected}


def originals(frozen, metadata):
    root = Path(frozen["reference_root"])
    expected_root = c.ROOT.parent / "canonical-orientation-v1"
    c.require(
        root.is_absolute()
        and str(root) == str(root.resolve())
        and root == expected_root.resolve()
        and not root.is_symlink(),
        "Frozen R reference path/owner mismatch",
    )
    payload.reference_check(root, frozen["reference_files"])

    def bound(name, record=None):
        inventory = frozen["reference_files"]
        c.require(
            name in inventory and Path(name).name == name,
            "Missing frozen R file binding",
        )
        if record is not None:
            c.same(
                record,
                {"path": name, "sha256": inventory[name]["sha256"]},
                "original R result artifact assignment/inventory",
            )
        return c.checked(root, {"path": name, **inventory[name]})

    registration = c.read(bound("registration.json"))
    development = c.read(bound("development.json"))
    original = c.read(bound("results.json"))
    registration_sha = frozen["reference_files"]["registration.json"]["sha256"]
    c.same(
        registration_sha,
        metadata["historical_registration_sha256"],
        "public original registration",
    )
    c.same(
        registration["plan"]["folds"],
        metadata["folds"],
        "original registered fold owners",
    )
    c.same(
        original["registration_sha256"],
        registration_sha,
        "original results/registration link",
    )
    c.same(
        development["registration_sha256"],
        registration_sha,
        "original development/registration link",
    )
    c.same(
        original["development_sha256"],
        frozen["reference_files"]["development.json"]["sha256"],
        "original results/development link",
    )
    c.same(
        development["data"],
        metadata["historical_data_binding"],
        "original development data binding",
    )
    c.same(
        {
            f: {role: group[role] for role in ("inner", "validation")}
            for f, group in development["coverage"].items()
        },
        metadata["coverage"],
        "original development claim graph",
    )
    c.same(
        development["source_coverage"]["members_sha256"],
        metadata["historical_members_sha256"],
        "original development member provenance",
    )
    c.same(
        development["source_coverage"]["recordings"],
        metadata["records"],
        "original development member mapping",
    )
    c.same(
        {
            key: value
            for key, value in development["source_coverage"].items()
            if key not in ("recordings", "archive_path")
        },
        metadata["source_metadata"],
        "original development source metadata",
    )
    expected_models = [
        (f["fold"], role) for f in metadata["folds"] for role in (*range(4), "refit")
    ]
    expected_inner = [(f["fold"], role) for f in metadata["folds"] for role in range(4)]
    expected_outer = [
        (f["fold"], condition)
        for f in metadata["folds"]
        for condition in planning.CONDITIONS
    ]
    indexes = {}
    for name, field, expected in (
        ("models", "role", expected_models),
        ("inner", "role", expected_inner),
        ("outer", "condition", expected_outer),
    ):
        records = [r for r in original[name] if r["arm"] == "canonical362"]
        indexes[name] = indexed(records, ("fold", field), expected, "original " + name)
        for (fold, role), record in indexes[name].items():
            suffix = (
                ("refit_" + role)
                if name == "outer"
                else ("refit" if role == "refit" else f"inner{role}")
            )
            filename = (
                f"fold{fold:02}_canonical362_{suffix}"
                + ("_model" if name == "models" else "")
                + ".npz"
            )
            bound(filename, record["artifact"])
    c.same(
        original["threshold_files"],
        [
            {
                "path": f"thresholds_{f['fold']:02}.json",
                "sha256": frozen["reference_files"][f"thresholds_{f['fold']:02}.json"][
                    "sha256"
                ],
            }
            for f in metadata["folds"]
        ],
        "original threshold file assignments/inventory",
    )
    cutoffs = {
        str(f["fold"]): c.read(bound(f"thresholds_{f['fold']:02}.json"))["canonical362"]
        for f in metadata["folds"]
    }
    for fold in metadata["folds"]:
        for role, ids in [
            *[(i, item["train_ids"]) for i, item in enumerate(fold["inner"])],
            ("refit", fold["refit_ids"]),
        ]:
            c.same(
                indexes["models"][fold["fold"], role]["train_ids"],
                ids,
                "original R training owner",
            )
    return root, original, indexes, cutoffs


def verify_copies(root, frozen, metadata, saved, clean, *, matched_scores=None):
    """Compare every saved original state/cache/metadata/cutoff before replay.

    Repacked NPZ containers are permitted; their arrays must match bit for bit.
    Both compact and full verification read the frozen R DEV reference bundle.
    Neither mode opens the raw archive here, and neither invokes estimator fit.
    """
    reference_root, original, indexes, cutoffs = originals(frozen, metadata)
    c.same(
        saved["original_results_sha256"],
        frozen["reference_files"]["results.json"]["sha256"],
        "original_results_sha256 anchored to reference_files",
    )
    c.same(saved["cutoffs"], cutoffs, "original R copied thresholds exact types/values")
    for field, expected in (
        ("clean_gate_passed", True),
        ("inherited_score_atol", 1e-6),
        ("inherited_score_rtol", 1e-6),
        ("historical_runtime_equivalence_claimed", False),
    ):
        c.same(saved[field], expected, "original R gate metadata: " + field)
    values, data = k.arm_data(clean, metadata, "R")
    copied_models = indexed(
        saved["models"], ("fold", "role"), indexes["models"], "copied models"
    )
    for identity, source in indexes["models"].items():
        copied = copied_models[identity]
        c.same(copied["arm"], "R", "copied R model arm")
        c.same(copied["ids"], source["train_ids"], "copied R training owner")
        c.same(
            copied["historical_stats"],
            source["stats"],
            "copied R historical training metadata",
        )
        c.same(
            copied["current_feature_stats_match"],
            source["stats"]["train_features_sha256"]
            == k.array_hash(values[np.isin(data["subjects"], source["train_ids"])]),
            "copied R current/historical feature hash diagnostic",
        )
        exact_arrays(
            c.load_arrays(root, copied["artifact"]),
            arrays(reference_root / source["artifact"]["path"]),
            "model " + str(identity),
        )
    for name, saved_name, arm, field in (
        ("inner", "inner", "R", "role"),
        ("outer", "historical_outer", "R_historical", "condition"),
    ):
        copied_scores = indexed(
            saved[saved_name], ("fold", field), indexes[name], "copied " + name
        )
        for identity, source in indexes[name].items():
            copied = copied_scores[identity]
            c.same(copied["arm"], arm, "copied R cache arm")
            c.same(
                copied["metadata"],
                source["metadata"],
                "copied R historical scorer metadata/mapping",
            )
            exact_arrays(
                c.load_arrays(root, copied["artifact"]),
                arrays(reference_root / source["artifact"]["path"]),
                name + " " + str(identity),
            )
            if name == "outer":
                c.same(
                    copied["paired_gate_ground_truth"],
                    identity[1] not in ("noise01", "noise05"),
                    "historical noise diagnostic scope",
                )
    rows = indexed(
        [
            r
            for r in original["rows"]
            if r["arm"] == "canonical362" and r["condition"] == "clean"
        ],
        ("fold",),
        [(f["fold"],) for f in metadata["folds"]],
        "original clean rows",
    )
    copied_rows = indexed(saved["rows"], ("fold",), rows, "copied R clean rows")
    for identity, source in rows.items():
        copied = copied_rows[identity]
        c.same(
            [copied["arm"], copied["condition"]],
            ["R", "clean"],
            "copied clean row assignment",
        )
        for field in ("metrics", "pair_errors"):
            c.same(copied[field], source[field], "original R clean cached " + field)
        cache_record = indexes["outer"][identity[0], "clean"]
        cache = {
            **arrays(reference_root / cache_record["artifact"]["path"]),
            "metadata": cache_record["metadata"],
        }
        c.same(
            copied,
            k.row(
                identity[0], "R", "clean", cache, cutoffs[str(identity[0])]["threshold"]
            ),
            "complete original R clean gate row",
        )
    if matched_scores is not None:
        copied_outer = indexed(
            [r for r in matched_scores if r["arm"] == "R"],
            ("fold", "condition"),
            indexes["outer"],
            "matched R outer",
        )
        for identity, source in indexes["outer"].items():
            if identity[1] not in ("noise01", "noise05"):
                copied = copied_outer[identity]
                c.same(
                    copied["metadata"],
                    source["metadata"],
                    "copied nonnoise R scorer metadata",
                )
                exact_arrays(
                    c.load_arrays(root, copied["artifact"]),
                    arrays(reference_root / source["artifact"]["path"]),
                    "matched nonnoise " + str(identity),
                )
    c.same(
        saved["restores"],
        [{"fold": f, "role": role, "fit_calls": 0} for f, role in indexes["models"]],
        "original R complete fit-free restoration owners",
    )
    c.same(saved["fit_calls"], 0, "original R zero fit calls")
    return {
        "models": len(copied_models),
        "inner_caches": len(indexes["inner"]),
        "historical_outer_caches": len(indexes["outer"]),
        "thresholds": len(cutoffs),
        "original_results_sha256": frozen["reference_files"]["results.json"]["sha256"],
        "array_equality": "keys/dtype/shape/C-order bits; JSON exact types/values",
    }
