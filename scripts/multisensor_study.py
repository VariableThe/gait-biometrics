"""Registered exploratory Santos sensor/objective/template ablations.

All historical TEST people have been inspected. This runner preserves their
roles but does not describe a subsequent evaluation as independent confirmation.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from scripts import cross_day_verification as previous

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ("mean", "robust", "multi")
DESCRIPTORS = (
    ("accel", "axes", "auto"),
    ("accel_gyro", "axes", "auto"),
    ("accel", "invariant", "auto"),
    ("accel_gyro", "invariant", "auto"),
    ("accel_gyro", "fusion", "auto"),
    ("accel_gyro", "fusion", 0.2),
)
NEURAL = tuple(
    f"{c}_{o}" for c in ("accel", "accel_gyro") for o in ("supcon", "cross_day_supcon")
)
FAMILIES = ("descriptors", *NEURAL)
SOURCES = (
    "scripts/multisensor_study.py",
    "scripts/cross_day_verification.py",
    "data/santos.py",
    "models/gait_features.py",
    "models/gait_invariants.py",
    "models/gait_metric.py",
    "models/gait_encoders.py",
    "models/siamese.py",
    "models/metric_learning.py",
    "data/gait_transforms.py",
    "engine/cross_day_metric.py",
    "engine/template_scoring.py",
    "engine/evaluate.py",
    "engine/subject_protocol.py",
)


def plan():
    return {
        "protocol": "santos_multisensor_exploratory_v2",
        "prior_test_inspection": True,
        "cohorts": {
            "development": list(previous.DEV_IDS),
            "calibration": list(previous.CALIBRATION_IDS),
            "test": list(previous.TEST_IDS),
        },
        "folds": previous.development_folds(),
        "seeds": [42, 43],
        "epochs": 40,
        "descriptors": [list(x) for x in DESCRIPTORS],
        "neural_families": list(NEURAL),
        "templates": list(TEMPLATES),
        "window_size": 256,
        "stride": 256,
        "sampling_rate": 100,
        "selection": "minimum equal-fold EER; ties fixed candidate order",
        "n_candidates": 30,
        "n_fold_neural_runs": 40,
        "reproduction": "lossless validation embedding caches; model replay atol=rtol=1e-5; metrics exactly recomputed from frozen caches",
        "final_neural_runs": "two only if the selected method is neural",
        "threshold": "separately calibrate winner and baseline on reserved CALIB; FAR<=1%",
        "baseline": "descriptor_accel_axes_auto/mean",
        "test_comparators": "winner and registered baseline only, matched claim graph",
        "uncertainty": "descriptive per-user errors; five TEST subjects, dependent claims",
    }


def source_hashes():
    return {p: previous.digest(ROOT / p) for p in SOURCES}


def load_data(archive, ids):
    from data.santos import load_santos_windows

    data = load_santos_windows(archive, ids, channels="accel_gyro")
    if data["X"].shape[1:] != (6, 256):
        raise ValueError("expected aligned authentic six-axis windows")
    if set(data["subjects"].tolist()) != set(ids):
        raise ValueError("requested identities must all have retained windows")
    return data


def subset(data, ids):
    mask = np.isin(data["subjects"], ids)
    return {
        k: v[mask]
        if isinstance(v, np.ndarray) and v.ndim and len(v) == len(mask)
        else v
        for k, v in data.items()
    }


def windows(data, channels):
    return data["X"][:, :3] if channels == "accel" else data["X"]


def descriptor_features(data, spec):
    from models.gait_features import extract_gait_features
    from models.gait_invariants import extract_invariant_features

    x = windows(data, spec[0])
    axes = extract_gait_features(x, 100) if spec[1] != "invariant" else None
    invariants = extract_invariant_features(x, 100) if spec[1] != "axes" else None
    return (
        axes
        if invariants is None
        else invariants
        if axes is None
        else np.concatenate((axes, invariants), axis=1)
    )


def descriptor_name(spec):
    return "descriptor_" + "_".join(map(str, spec))


def model_args(channels):
    count = 3 if channels == "accel" else 6
    return {
        "architecture": "multiscale",
        "embed_dim": 64,
        "width": 32,
        "dropout": 0.1,
        "stats_fusion": True,
        "normalization": True,
        "vector_groups": [list(range(i, i + 3)) for i in range(0, count, 3)],
    }


def family_config(family):
    if family not in NEURAL:
        raise ValueError("unknown neural family")
    channels, objective = family.split("_supcon")[0], "supcon"
    # Longer suffix must be removed before the shorter one.
    if family.endswith("_cross_day_supcon"):
        channels, objective = (
            family.removesuffix("_cross_day_supcon"),
            "cross_day_supcon",
        )
    else:
        channels = family.removesuffix("_supcon")
    return channels, objective


def validate_training_record(record, family, ids, seed):
    """Validate the complete effective optimizer budget, not a sidecar subset."""
    channels, objective = family_config(family)
    count = 3 if channels == "accel" else 6
    args = {
        **model_args(channels),
        "add_magnitudes": True,
        "stats_include_mean": False,
        "norm_eps": 1e-5,
    }
    n = record["training_window_count"]
    p = min(8, len(ids))
    steps = max(int(np.ceil(n / (p * 4))), int(np.ceil(len(ids) / p)))
    expected = {
        "n_channels": count,
        "model_args": args,
        "seed": seed,
        "epochs": 40,
        "epoch_selection": "fixed_final_epoch",
        "device": record["config"]["device"],
        "precision": "float32",
        "torch_threads": 2,
        "objective": objective,
        "temperature": 0.1,
        "ce_weight": 0.25,
        "classifier": "linear_training_id_only",
        "label_smoothing": 0.0,
        "optimizer": "AdamW",
        "optimizer_args": {
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "amsgrad": False,
            "foreach": None,
            "maximize": False,
            "capturable": False,
            "differentiable": False,
            "fused": False,
        },
        "lr": 0.001,
        "weight_decay": 0.0001,
        "gradient_clip": 1.0,
        "scheduler": "none",
        "augment": {},
        "sampler": {
            "requested_identities_per_batch": 8,
            "identities_per_batch": p,
            "windows_per_identity": 4,
            "days_per_identity_per_batch": 2,
            "windows_per_selected_day": 2,
            "steps_per_epoch": steps,
            "batch_size": p * 4,
            "subject_coverage_per_epoch_guaranteed": True,
            "row_replacement": "only_if_selected_day_has_fewer_than_K_over_2_rows",
            "original_row_ids_preserved": True,
        },
    }
    previous._same(record["config"], expected, "complete checkpoint configuration")
    if (
        record["epochs_completed"] != 40
        or record["selected_epoch"] != 40
        or [h["epoch"] for h in record["history"]] != list(range(1, 41))
        or any(h["steps"] != steps for h in record["history"])
        or set(record["training_subject_ids"]) != set(ids)
        or record["normalization_fit_calls"] != 1
        or not record["all_finite"]
        or record["input_shape"] != [n, count, 256]
    ):
        raise ValueError("checkpoint does not match complete registered training")
    return args


def score(z, data, template):
    from engine.template_scoring import score_trials

    labels, scores, metadata = score_trials(
        z, data["subjects"], data["days"], data["trials"], template_method=template
    )
    return {"labels": labels, "scores": scores, "metadata": metadata}


def metrics(scored, threshold=None):
    from engine.evaluate import compute_eer, fixed_threshold_metrics

    result = {
        "eer": float(compute_eer(scored["labels"], scored["scores"])),
        "roc_auc": float(roc_auc_score(scored["labels"], scored["scores"])),
        "coverage": scored["metadata"]["coverage"],
        "claim_graph_sha256": previous._object_hash(
            {
                k: scored["metadata"][k]
                for k in ("claim_subject_ids", "probe_subject_ids", "probe_trial_ids")
            }
        ),
    }
    if threshold is not None:
        from engine.subject_protocol import per_subject_error_rates

        result["fixed_threshold"] = fixed_threshold_metrics(
            scored["labels"], scored["scores"], threshold
        )
        result["per_subject_errors"] = per_subject_error_rates(
            scored["labels"],
            scored["scores"],
            scored["metadata"]["claim_subject_ids"],
            scored["metadata"]["probe_subject_ids"],
            threshold,
        )
    return result


def write(path, value):
    previous._write_new(Path(path), value)


def check_registration(args):
    root = Path(args.out_root)
    registration = previous._read(root / "registration.json")
    previous._same(registration["plan"], plan(), "registered study plan")
    previous._same(
        registration["sources"], source_hashes(), "registered implementation"
    )
    previous._same(
        registration["archive"], previous.audit_archive(args.archive_path), "archive"
    )
    return root, registration


def development(args, registration):
    data = load_data(args.archive_path, previous.DEV_IDS)
    previous._same(
        previous.data_fingerprint(data),
        registration["development_fingerprint"],
        "DEV values",
    )
    return data


def register(args):
    root = Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("registration requires a completely empty study root")
    archive, sources = previous.audit_archive(args.archive_path), source_hashes()
    # No fitting or embeddings before the plan and initial data fingerprint.
    data = load_data(args.archive_path, previous.DEV_IDS)
    registration = {
        "plan": plan(),
        "sources": sources,
        "archive": archive,
        "development_fingerprint": previous.data_fingerprint(data),
        "development_members_sha256": data["metadata"]["members_sha256"],
    }
    write(root / "registration.json", registration)
    return {"registered": True, "plan": plan()}


def save_npz(path, **arrays):
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def cache_validation(path, embeddings, data):
    save_npz(
        path,
        embeddings=np.asarray(embeddings),
        data_fingerprint=np.asarray(previous.data_fingerprint(data)),
    )


def verified_validation(path, replayed, data):
    with np.load(path, allow_pickle=False) as saved:
        previous._same(
            str(saved["data_fingerprint"]),
            previous.data_fingerprint(data),
            "cached validation inputs",
        )
        z = saved["embeddings"].copy()
    if (
        z.shape != replayed.shape
        or not np.isfinite(z).all()
        or not np.allclose(z, replayed, atol=1e-5, rtol=1e-5)
    ):
        raise ValueError("model replay differs from frozen validation embeddings")
    return z


def encode(model, x, device):
    import torch

    old_matmul, old_cudnn = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.inference_mode():
            return np.concatenate(
                [
                    model(torch.as_tensor(x[i : i + 64], device=device))
                    .float()
                    .cpu()
                    .numpy()
                    for i in range(0, len(x), 64)
                ],
                axis=0,
            )
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = (
            old_matmul,
            old_cudnn,
        )


def fit_neural(data, family, seed, device):
    from engine.cross_day_metric import fit_cross_day_encoder

    channels, objective = family_config(family)
    x = windows(data, channels)
    return fit_cross_day_encoder(
        x,
        data["subjects"],
        data["days"],
        n_channels=x.shape[1],
        model_args=model_args(channels),
        seed=seed,
        epochs=40,
        device=device,
        torch_threads=2,
        objective=objective,
    )


def save_model(path, model, record):
    import torch

    with Path(path).open("xb") as stream:
        torch.save(
            {
                "state_dict": {
                    k: v.detach().cpu() for k, v in model.state_dict().items()
                },
                "record": record,
                "implementation_sha256": source_hashes(),
            },
            stream,
        )


def load_model(path, family, ids, seed, device, training_data=None):
    import torch

    from models.gait_encoders import build_gait_encoder

    saved = torch.load(path, map_location=device, weights_only=True)
    previous._same(
        saved["implementation_sha256"], source_hashes(), "checkpoint implementation"
    )
    record = saved["record"]
    validate_training_record(record, family, ids, seed)
    if training_data is not None:
        from engine.cross_day_metric import _array_hash

        channels, _ = family_config(family)
        x = np.ascontiguousarray(windows(training_data, channels), dtype=np.float32)
        subjects, days = training_data["subjects"], training_data["days"]
        previous._same(
            record["training_input_sha256"],
            _array_hash(x, subjects, days),
            "checkpoint training values",
        )
        previous._same(
            record["training_window_ids_sha256"],
            _array_hash(np.arange(len(x), dtype=np.int64), subjects, days),
            "checkpoint row identities",
        )
    channels, objective = family_config(family)
    config = record["config"]
    if (
        record["epochs_completed"] != 40
        or record["selected_epoch"] != 40
        or [h["epoch"] for h in record["history"]] != list(range(1, 41))
        or set(record["training_subject_ids"]) != set(ids)
        or config["seed"] != seed
        or config["objective"] != objective
    ):
        raise ValueError("checkpoint does not match complete registered training")
    count = 3 if channels == "accel" else 6
    expected_args = model_args(channels)
    if any(config["model_args"].get(k) != v for k, v in expected_args.items()):
        raise ValueError("checkpoint architecture changed")
    model = build_gait_encoder(count, config["model_args"]).to(device)
    model.load_state_dict(saved["state_dict"], strict=True)
    return model.eval(), record


def train_family(args):
    root, registration = check_registration(args)
    data = development(args, registration)
    family = args.family
    folder = root / family
    folder.mkdir()
    started = time.perf_counter()
    rows, artifacts = [], {}
    if family == "descriptors":
        from models.gait_metric import WithinClassMetric

        for spec in DESCRIPTORS:
            name = descriptor_name(spec)
            features = descriptor_features(data, spec)
            folds = {template: [] for template in TEMPLATES}
            for fold in previous.development_folds():
                train = np.isin(data["subjects"], fold["train_subject_ids"])
                val = np.isin(data["subjects"], fold["val_subject_ids"])
                metric = WithinClassMetric("wccn", spec[2]).fit(
                    features[train], data["subjects"][train]
                )
                path = folder / f"{name}_fold{fold['fold']}.npz"
                save_npz(path, **metric.state_dict())
                artifacts[str(path.relative_to(root))] = previous.digest(path)
                z = metric.transform(features[val])
                cache = folder / f"{name}_fold{fold['fold']}_validation.npz"
                cache_validation(cache, z, subset(data, fold["val_subject_ids"]))
                artifacts[str(cache.relative_to(root))] = previous.digest(cache)
                for template in TEMPLATES:
                    folds[template].append(
                        {
                            "fold": fold["fold"],
                            **metrics(
                                score(
                                    z, subset(data, fold["val_subject_ids"]), template
                                )
                            ),
                        }
                    )
            for template in TEMPLATES:
                rows.append(
                    {
                        "name": name + "/" + template,
                        "family": "descriptors",
                        "descriptor": list(spec),
                        "template": template,
                        "folds": folds[template],
                        "mean_eer": float(np.mean([r["eer"] for r in folds[template]])),
                    }
                )
    else:
        channels, _ = family_config(family)
        folds = {template: [] for template in TEMPLATES}
        for fold in previous.development_folds():
            train, val = (
                subset(data, fold["train_subject_ids"]),
                subset(data, fold["val_subject_ids"]),
            )
            embeddings, seed_reports = [], []
            for seed in (42, 43):
                model, record = fit_neural(train, family, seed, args.device)
                path = folder / f"fold{fold['fold']}_seed{seed}.pt"
                save_model(path, model, record)
                artifacts[str(path.relative_to(root))] = previous.digest(path)
                z = encode(model, windows(val, channels), args.device)
                embeddings.append(z)
                seed_reports.append(
                    {
                        "seed": seed,
                        "metrics": metrics(score(z, val, "mean")),
                        "training_record": record,
                    }
                )
                del model
            ensemble = np.concatenate(embeddings, axis=1) / np.sqrt(2)
            cache = folder / f"fold{fold['fold']}_validation.npz"
            cache_validation(cache, ensemble, val)
            artifacts[str(cache.relative_to(root))] = previous.digest(cache)
            for template in TEMPLATES:
                folds[template].append(
                    {
                        "fold": fold["fold"],
                        "seeds": seed_reports,
                        **metrics(score(ensemble, val, template)),
                    }
                )
            print(
                json.dumps({"family": family, "fold_complete": fold["fold"]}),
                flush=True,
            )
        for template in TEMPLATES:
            rows.append(
                {
                    "name": family + "/" + template,
                    "family": family,
                    "template": template,
                    "folds": folds[template],
                    "mean_eer": float(np.mean([r["eer"] for r in folds[template]])),
                }
            )
    result = {
        "family": family,
        "registration_sha256": previous.digest(root / "registration.json"),
        "rows": rows,
        "artifacts": artifacts,
        "wall_seconds": time.perf_counter() - started,
    }
    write(folder / "result.json", result)
    return {
        "family": family,
        "rows": [{"name": r["name"], "mean_eer": r["mean_eer"]} for r in rows],
        "wall_seconds": result["wall_seconds"],
    }


def expected_names():
    return [descriptor_name(s) + "/" + t for s in DESCRIPTORS for t in TEMPLATES] + [
        f + "/" + t for f in NEURAL for t in TEMPLATES
    ]


def select(rows):
    names = expected_names()
    if len(rows) != len(names) or {r["name"] for r in rows} != set(names):
        raise ValueError("all 30 registered candidates must complete before selection")
    for row in rows:
        expected = canonical_candidate(row["name"])
        previous._same(
            {k: row[k] for k in expected}, expected, "candidate method identity"
        )
        if [f["fold"] for f in row["folds"]] != list(range(5)):
            raise ValueError("five-fold coverage incomplete")
        previous._same(
            row["mean_eer"],
            float(np.mean([f["eer"] for f in row["folds"]])),
            "fold summary",
        )
    return min(rows, key=lambda r: (r["mean_eer"], names.index(r["name"])))


def canonical_candidate(name):
    for spec in DESCRIPTORS:
        for template in TEMPLATES:
            if name == descriptor_name(spec) + "/" + template:
                return {
                    "name": name,
                    "family": "descriptors",
                    "descriptor": list(spec),
                    "template": template,
                }
    for family in NEURAL:
        for template in TEMPLATES:
            if name == family + "/" + template:
                return {"name": name, "family": family, "template": template}
    raise ValueError("unregistered method")


def expected_artifacts(family):
    if family == "descriptors":
        return {
            f"descriptors/{descriptor_name(s)}_fold{f}.npz"
            for s in DESCRIPTORS
            for f in range(5)
        } | {
            f"descriptors/{descriptor_name(s)}_fold{f}_validation.npz"
            for s in DESCRIPTORS
            for f in range(5)
        }
    return {f"{family}/fold{f}_seed{s}.pt" for f in range(5) for s in (42, 43)} | {
        f"{family}/fold{f}_validation.npz" for f in range(5)
    }


def verified_results(root):
    rows, hashes = [], {}
    for family in FAMILIES:
        path = root / family / "result.json"
        result = previous._read(path)
        previous._same(
            result["registration_sha256"],
            previous.digest(root / "registration.json"),
            "run registration",
        )
        if result["family"] != family:
            raise ValueError("wrong run family")
        previous._same(
            set(result["artifacts"]),
            expected_artifacts(family),
            "complete artifact inventory",
        )
        previous._same(
            {r["name"] for r in result["rows"]},
            {n for n in expected_names() if canonical_candidate(n)["family"] == family},
            "family candidate coverage",
        )
        for name, digest in result["artifacts"].items():
            previous._same(previous.digest(root / name), digest, "fitted artifact")
        rows.extend(result["rows"])
        hashes[str(path.relative_to(root))] = previous.digest(path)
        hashes.update(result["artifacts"])
    return rows, hashes


def fit_final(row, dev, root, device):
    if row["family"] == "descriptors":
        from models.gait_metric import WithinClassMetric

        spec = row["descriptor"]
        metric = WithinClassMetric("wccn", spec[2]).fit(
            descriptor_features(dev, spec), dev["subjects"]
        )
        path = root / (row["name"].replace("/", "_") + "_final.npz")
        save_npz(path, **metric.state_dict())
        return {
            "kind": "descriptor",
            "paths": [str(path.relative_to(root))],
            "row": row,
        }
    paths = []
    for seed in (42, 43):
        model, record = fit_neural(dev, row["family"], seed, device)
        path = root / f"{row['family']}_final_seed{seed}.pt"
        save_model(path, model, record)
        paths.append(str(path.relative_to(root)))
        del model
    return {"kind": "neural", "paths": paths, "row": row}


def final_embeddings(fitted, data, root, device):
    row = fitted["row"]
    if fitted["kind"] == "descriptor":
        from models.gait_metric import WithinClassMetric

        with np.load(root / fitted["paths"][0], allow_pickle=False) as saved:
            metric = WithinClassMetric.from_state_dict(dict(saved))
        if set(metric.training_subject_ids_) != set(previous.DEV_IDS):
            raise ValueError(
                "final metric did not fit development identities exclusively"
            )
        return metric.transform(descriptor_features(data, row["descriptor"]))
    channels, _ = family_config(row["family"])
    zs = []
    for seed, path in zip((42, 43), fitted["paths"], strict=True):
        model, _ = load_model(
            root / path, row["family"], previous.DEV_IDS, seed, device
        )
        zs.append(encode(model, windows(data, channels), device))
    return np.concatenate(zs, axis=1) / np.sqrt(2)


def lock(args):
    root, registration = check_registration(args)
    if (root / "lock.json").exists():
        raise ValueError("study already locked")
    dev = development(args, registration)
    rows, hashes = verified_results(root)
    winner = select(rows)
    baseline = next(r for r in rows if r["name"] == plan()["baseline"])
    # Recompute every candidate from the registered fitted artifacts before any
    # calibration read. This catches changed summaries, incomplete runs/caches.
    for row in rows:
        for fold, report in zip(
            previous.development_folds(), row["folds"], strict=True
        ):
            val = subset(dev, fold["val_subject_ids"])
            if row["family"] == "descriptors":
                from models.gait_metric import WithinClassMetric

                path = (
                    root
                    / "descriptors"
                    / f"{descriptor_name(row['descriptor'])}_fold{fold['fold']}.npz"
                )
                with np.load(path, allow_pickle=False) as saved:
                    metric = WithinClassMetric.from_state_dict(dict(saved))
                if set(metric.training_subject_ids_) != set(fold["train_subject_ids"]):
                    raise ValueError("fold metric identity leakage")
                z = metric.transform(descriptor_features(val, row["descriptor"]))
                cache = (
                    root
                    / "descriptors"
                    / f"{descriptor_name(row['descriptor'])}_fold{fold['fold']}_validation.npz"
                )
            else:
                channels, _ = family_config(row["family"])
                zs = []
                for seed in (42, 43):
                    model, _ = load_model(
                        root / row["family"] / f"fold{fold['fold']}_seed{seed}.pt",
                        row["family"],
                        fold["train_subject_ids"],
                        seed,
                        args.device,
                        subset(dev, fold["train_subject_ids"]),
                    )
                    zs.append(encode(model, windows(val, channels), args.device))
                z = np.concatenate(zs, axis=1) / np.sqrt(2)
                cache = root / row["family"] / f"fold{fold['fold']}_validation.npz"
            z = verified_validation(cache, z, val)
            actual = metrics(score(z, val, row["template"]))
            previous._same(
                actual["coverage"], report["coverage"], "reported fold coverage"
            )
            previous._same(
                actual["claim_graph_sha256"],
                report["claim_graph_sha256"],
                "reported fold graph",
            )
            previous._same(
                actual["claim_graph_sha256"],
                baseline["folds"][fold["fold"]]["claim_graph_sha256"],
                "matched candidate validation graph",
            )
            if (
                abs(actual["eer"] - report["eer"]) > 1e-10
                or abs(actual["roc_auc"] - report["roc_auc"]) > 1e-10
            ):
                raise ValueError(
                    "fold scores do not reproduce registered candidate summary"
                )
    # Selection is frozen before this CALIB load; no TEST values in this stage.
    selected = {"baseline": fit_final(baseline, dev, root, args.device)}
    selected["winner"] = (
        selected["baseline"]
        if winner["name"] == baseline["name"]
        else fit_final(winner, dev, root, args.device)
    )
    calibration = load_data(args.archive_path, previous.CALIBRATION_IDS)
    from engine.evaluate import calibrate_threshold

    for fitted in selected.values():
        scored = score(
            final_embeddings(fitted, calibration, root, args.device),
            calibration,
            fitted["row"]["template"],
        )
        fitted["threshold"] = float(
            calibrate_threshold(scored["labels"], scored["scores"], target_far=0.01)[
                "threshold"
            ]
        )
        fitted["calibration"] = metrics(scored, fitted["threshold"])
        for path in fitted["paths"]:
            hashes[path] = previous.digest(root / path)
    hashes["registration.json"] = previous.digest(root / "registration.json")
    value = {
        "registration_sha256": hashes["registration.json"],
        "artifact_hashes": hashes,
        "winner": winner["name"],
        "selected": selected,
        "candidates": rows,
        "calibration_fingerprint": previous.data_fingerprint(calibration),
        "prior_test_inspection": True,
    }
    value["self_hash"] = previous._object_hash(value)
    write(root / "lock.json", value)
    return {
        "winner": winner["name"],
        "mean_eer": winner["mean_eer"],
        "baseline_mean_eer": baseline["mean_eer"],
        "self_hash": value["self_hash"],
    }


def evaluate(args):
    root, registration = check_registration(args)
    value = previous._read(root / "lock.json")
    hash_value = value.pop("self_hash")
    previous._same(previous._object_hash(value), hash_value, "study lock")
    for path, digest in value["artifact_hashes"].items():
        previous._same(previous.digest(root / path), digest, "locked artifact")
    development(args, registration)
    calibration = load_data(args.archive_path, previous.CALIBRATION_IDS)
    previous._same(
        previous.data_fingerprint(calibration),
        value["calibration_fingerprint"],
        "CALIB values",
    )
    write(
        root / "exploratory_test_attempt.json",
        {"lock_hash": hash_value, "prior_test_inspection": True},
    )
    test = load_data(args.archive_path, previous.TEST_IDS)
    results, scores = {}, {}
    for key, fitted in value["selected"].items():
        scored = score(
            final_embeddings(fitted, test, root, args.device),
            test,
            fitted["row"]["template"],
        )
        scores[key] = scored
        results[key] = {
            "name": fitted["row"]["name"],
            **metrics(scored, fitted["threshold"]),
        }
    for key in ("claim_subject_ids", "probe_subject_ids", "probe_trial_ids"):
        previous._same(
            scores["baseline"]["metadata"][key],
            scores["winner"]["metadata"][key],
            "matched TEST claim graph",
        )
    if not np.array_equal(scores["baseline"]["labels"], scores["winner"]["labels"]):
        raise ValueError("candidate exclusions changed comparison support")
    from engine.template_scoring import paired_subject_diagnostics

    paired = paired_subject_diagnostics(
        scores["baseline"]["labels"],
        scores["baseline"]["scores"],
        scores["winner"]["scores"],
        scores["baseline"]["metadata"]["claim_subject_ids"],
        scores["baseline"]["metadata"]["probe_subject_ids"],
        {k: v["threshold"] for k, v in value["selected"].items() if k == "baseline"}
        | {"new": value["selected"]["winner"]["threshold"]},
    )
    result = {
        "prior_test_inspection": True,
        "lock_hash": hash_value,
        "test_fingerprint": previous.data_fingerprint(test),
        "results": results,
        "paired_subject_diagnostics": paired,
        "interpretation": "Exploratory reused-TEST comparison; no new independent confirmation.",
    }
    write(root / "exploratory_test_result.json", result)
    save_npz(
        root / "exploratory_test_scores.npz",
        labels=scores["baseline"]["labels"],
        baseline=scores["baseline"]["scores"],
        winner=scores["winner"]["scores"],
        claim_subject_ids=np.asarray(
            scores["baseline"]["metadata"]["claim_subject_ids"]
        ),
        probe_subject_ids=np.asarray(
            scores["baseline"]["metadata"]["probe_subject_ids"]
        ),
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-path", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("register", "train", "lock", "evaluate"), required=True
    )
    parser.add_argument("--family", choices=FAMILIES, default="descriptors")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    result = {
        "register": register,
        "train": train_family,
        "lock": lock,
        "evaluate": evaluate,
    }[args.stage](args)
    print(json.dumps(previous._jsonable(result), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
