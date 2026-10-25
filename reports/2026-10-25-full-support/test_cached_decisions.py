"""Tiny synthetic/refusal checks only; no study cache replay and no fits."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import sys
from pathlib import Path

import common as c
import verify_cached_decisions as replay


def run_tests(bundle, out):
    c.enforce_environment()
    c.require(not out.exists() and out.parent.is_dir(), "Fresh test output required")
    c.require(not out.is_relative_to(bundle), "Keep tests outside immutable bundle")
    out.mkdir()
    sys.path.insert(0, str(bundle / "authority"))
    np = importlib.import_module("numpy")
    empirical = importlib.import_module("engine.empirical")
    ranking = importlib.import_module("engine.robustness_selection")
    checks = []

    def check(name, action, *, rejection=False):
        try:
            action()
        except (RuntimeError, ValueError, FileNotFoundError, SystemExit) as error:
            if not rejection:
                raise
            checks.append(
                {
                    "name": name,
                    "passed": True,
                    "expected_rejection": type(error).__name__,
                    "reason": str(error),
                }
            )
        else:
            c.require(not rejection, "Expected rejection: " + name)
            checks.append({"name": name, "passed": True})

    check("duplicate JSON keys", lambda: c.decode(b'{"x":1,"x":2}'), rejection=True)
    check("NaN JSON", lambda: c.decode(b'{"x":NaN}'), rejection=True)
    check("overflow JSON float", lambda: c.decode(b'{"x":1e999}'), rejection=True)
    check("non-string mapping keys", lambda: c.canonical({1: "x"}), rejection=True)
    check(
        "boolean is not integer", lambda: c.same(True, 1, "typed bool"), rejection=True
    )
    check(
        "float is not integer", lambda: c.same(1.0, 1, "typed number"), rejection=True
    )
    check(
        "signed zero identity", lambda: c.same(-0.0, 0.0, "signed zero"), rejection=True
    )
    check(
        "lossless signed-zero roundtrip",
        lambda: c.same(c.decode(c.canonical({"z": -0.0})), {"z": -0.0}, "roundtrip"),
    )
    for name in ("../escape", "/absolute", "a//b", "a/./b", "a/../b", "a\\b"):
        check(
            "unsafe relative " + name,
            lambda name=name: c.relative(name),
            rejection=True,
        )
    check(
        "unsafe absolute path",
        lambda: c.absolute(str(out) + "/../escape"),
        rejection=True,
    )
    check(
        "typed assignment bool",
        lambda: replay.unique_records(
            [{"fold": False}], ("fold",), [(0,)], "synthetic fold"
        ),
        rejection=True,
    )
    check(
        "duplicate assignment",
        lambda: replay.unique_records(
            [{"fold": 0}, {"fold": 0}], ("fold",), [(0,)], "synthetic fold"
        ),
        rejection=True,
    )
    check(
        "missing assignment",
        lambda: replay.unique_records([], ("fold",), [(0,)], "synthetic fold"),
        rejection=True,
    )

    labels = np.asarray([1, 0, 1, 0], dtype=np.int64)
    tied = np.asarray([0.0, 0.0, 1.0, -1.0], dtype=np.float64)
    threshold = empirical.empirical_threshold(labels, tied)
    check(
        "whole tied-score group cutoff",
        lambda: c.same(threshold["threshold"], 1.0, "tie threshold"),
    )
    check(
        "threshold equality accepted",
        lambda: c.same(
            (tied >= threshold["threshold"]).tolist(),
            [False, False, True, False],
            "equality decisions",
        ),
    )
    reject_all = empirical.empirical_threshold(
        np.asarray([1, 0], dtype=np.int64), np.asarray([0.0, 0.0])
    )
    check(
        "finite nextafter reject-all cutoff",
        lambda: c.same(
            reject_all["threshold"], float(np.nextafter(0.0, np.inf)), "finite edge"
        ),
    )
    check(
        "exact tied EER/AUC",
        lambda: c.same(
            ranking._ranking(np.asarray([1, 0]), np.asarray([0.0, 0.0])),
            {"eer_exact": "1/2", "auc_exact": "1/2"},
            "rational tied ranking",
        ),
    )
    check(
        "typed threshold rejects integer",
        lambda: replay.threshold_record(0, "R", {"threshold": 1}),
        rejection=True,
    )
    check(
        "IEEE signed-zero threshold binding",
        lambda: c.require(
            replay.threshold_record(0, "R", {"threshold": -0.0})["float64_be_hex"]
            == "8000000000000000",
            "threshold bits",
        ),
    )

    fixture = out / "synthetic"
    (fixture / "evidence/scores").mkdir(parents=True)
    trials = [[s, day, trial] for s in (1, 2, 3) for day in (1, 2) for trial in (1, 2)]
    graph = replay.rebuild_graph(trials, [3, 1, 2])
    metadata = {
        key: graph[key]
        for key in ("claim_subject_ids", "probe_subject_ids", "probe_trial_ids")
    }
    metadata["coverage"] = {key: graph[key] for key in ("n_genuine", "n_impostor")}
    path = fixture / "evidence/scores/tiny.npz"
    arrays = {
        "labels": np.asarray(graph["labels"], dtype=np.int64),
        "scores": np.linspace(-1, 1, len(graph["labels"]), dtype=np.float64),
    }
    with path.open("xb") as handle:
        np.savez(handle, labels=arrays["labels"], scores=arrays["scores"])
    record = {
        "artifact": {
            "path": "scores/tiny.npz",
            "sha256": c.sha(path),
            "bytes": path.stat().st_size,
            "schema": {
                k: {"dtype": a.dtype.str, "shape": list(a.shape)}
                for k, a in arrays.items()
            },
        },
        "metadata": metadata,
    }
    check(
        "tiny exact ordered claim/label cache",
        lambda: replay.load_score(fixture, record, graph, np),
    )
    check(
        "tiny full claim denominators",
        lambda: c.same(
            [graph["n_genuine"], graph["n_impostor"]], [6, 12], "denominators"
        ),
    )
    changed = c.decode(c.canonical(record))
    changed["metadata"]["probe_trial_ids"][0] = 2
    check(
        "ordered claim metadata tamper",
        lambda: replay.load_score(fixture, changed, graph, np),
        rejection=True,
    )
    changed = c.decode(c.canonical(graph))
    changed["labels"][0] = 0
    check(
        "ordered label tamper",
        lambda: replay.load_score(fixture, record, changed, np),
        rejection=True,
    )
    changed = c.decode(c.canonical(record))
    changed["metadata"]["coverage"]["n_genuine"] = 6.0
    check(
        "denominator type change",
        lambda: replay.load_score(fixture, changed, graph, np),
        rejection=True,
    )
    with path.open("wb") as handle:
        np.savez(
            handle, labels=arrays["labels"], scores=arrays["scores"].astype(np.float32)
        )
    changed = c.decode(c.canonical(record))
    changed["artifact"].update(sha256=c.sha(path), bytes=path.stat().st_size)
    changed["artifact"]["schema"]["scores"]["dtype"] = "<f4"
    check(
        "rehashed score dtype tamper",
        lambda: replay.load_score(fixture, changed, graph, np),
        rejection=True,
    )

    mini = out / "inventory-fixture"
    mini.mkdir()
    c.write(mini / "payload.json", {"x": 1})
    index = {
        "schema": 1,
        "excludes_only": "inventory.json",
        "files": c.file_index(mini),
    }
    c.write(mini / "inventory.json", index)
    check("exact tiny inventory", lambda: c.check_inventory(mini))
    c.write(mini / "extra.json", {})
    check("unknown file", lambda: c.check_inventory(mini), rejection=True)
    changed = c.decode(c.canonical(index))
    changed["files"]["payload.json"]["bytes"] = float(
        changed["files"]["payload.json"]["bytes"]
    )
    check(
        "rehash inventory scalar type",
        lambda: c.same(changed, index, "inventory types"),
        rejection=True,
    )
    (mini / "empty-directory").mkdir()
    check("unknown empty directory", lambda: c.file_index(mini), rejection=True)
    (mini / "empty-directory").rmdir()
    symlink = out / "symlink-fixture"
    symlink.mkdir()
    (symlink / "link").symlink_to(mini / "payload.json")
    check("symlink inventory path", lambda: c.file_index(symlink), rejection=True)
    check(
        "symlink absolute path",
        lambda: c.absolute(str(symlink / "link")),
        rejection=True,
    )
    (symlink / "link").unlink()
    symlink.rmdir()

    def cli(arguments):
        old = sys.argv
        sys.argv = ["verify_cached_decisions.py", *arguments]
        try:
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                replay.main()
        finally:
            sys.argv = old

    check(
        "CLI existing output refused",
        lambda: cli(["--bundle", str(bundle), "--out", str(out)]),
        rejection=True,
    )
    check(
        "CLI unsafe output refused",
        lambda: cli(["--bundle", str(bundle), "--out", str(out) + "/../unsafe"]),
        rejection=True,
    )
    check(
        "CLI output inside bundle refused",
        lambda: cli(
            ["--bundle", str(bundle), "--out", str(bundle / "forbidden-output")]
        ),
        rejection=True,
    )
    check(
        "CLI unknown option refused",
        lambda: cli(["--archive", "forbidden"]),
        rejection=True,
    )
    missing = out / "missing-bundle"
    check(
        "CLI missing bundle refused",
        lambda: cli(
            ["--bundle", str(missing), "--out", str(out / "missing-bundle-output")]
        ),
        rejection=True,
    )
    check(
        "failed helper attempt retained",
        lambda: c.require(
            (out / "missing-bundle-output/failure.json").is_file(),
            "missing failure receipt",
        ),
    )
    result = {
        "passed": True,
        "checks": checks,
        "count": len(checks),
        "scope": "tiny synthetic/refusal checks only",
        "fits": 0,
        "real_score_cache_reads": 0,
    }
    c.write(out / "tests.json", result)
    c.write(
        out / "inventory.json",
        {"schema": 1, "excludes_only": "inventory.json", "files": c.file_index(out)},
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = run_tests(c.absolute(args.bundle), c.absolute(args.out))
    print(
        c.canonical(
            {
                "passed": result["passed"],
                "checks": result["count"],
                "fits": 0,
                "real_score_cache_reads": 0,
            }
        ).decode()
    )


if __name__ == "__main__":
    main()
