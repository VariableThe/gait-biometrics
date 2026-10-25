"""Independent count-only reconstruction from frozen synthetic JSON records."""

import collections
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def count(rows):
    return {
        "inputs": len(rows),
        "finite": sum(r["valid_pre_quality"] for r in rows),
        "finite_one_P": sum(r["finite_one_P_extraction"] for r in rows),
        "lag_resolved_within_one": sum(r["lag_resolves_declared_P_within_one_sample"] for r in rows),
        "failures": dict(sorted(collections.Counter(r["failure_reason"] or "none" for r in rows).items())),
        "finite_pair_aliases": dict(sorted(collections.Counter(r["selected_pair_alias"] for r in rows if r["valid_pre_quality"]).items())),
    }


def main():
    root = Path("artifacts/joint-repeat-feasibility-r1")
    anchor = json.loads(Path("artifacts/joint-repeat-root-completion-anchors-r3.json").read_bytes())
    assert digest(root / "report.json") == anchor["report_sha256"]
    assert digest(root / "summary.json") == anchor["summary_sha256"]
    rows = [json.loads(line) for line in (root / "records.jsonl").read_text().splitlines()]
    summary = json.loads((root / "summary.json").read_bytes())
    assert len(rows) == 8640
    assert len({(r["input_index"], r["arm"]) for r in rows}) == 8640
    legacy, primary, stress, invariance = [], [], [], []
    by_input = {(r["input_index"], r["arm"]): r for r in rows}
    by_key = {r["key"]: r["input_index"] for r in rows}
    keys = json.loads((root / "keys.json").read_bytes())
    for length in (256, 768):
        for arm in ("control_exact", "control_near", "candidate_joint"):
            subset = [r for r in rows if r["length"] == length and r["arm"] == arm and r["kind"] == "legacy"]
            assert len(subset) == 1280
            legacy.append({"length": length, "arm": arm, **count(subset)})
            selected = [r for r in subset if r["family"] == "pure_periodic_balanced_peaks" and r["period_samples"] in (160, 180, 200)]
            assert len(selected) == 96
            report = {"length": length, "arm": arm, **count(selected)}
            frozen = next(r for r in summary["primary_96_by_length_arm"] if r["length"] == length and r["arm"] == arm)
            assert all(report[k] == frozen[k] for k in report)
            primary.append(report)
        conditions = sorted({r["condition"] for r in rows if r["kind"] == "stress"})
        for condition in conditions:
            subset = [r for r in rows if r["kind"] == "stress" and r["length"] == length and r["arm"] == "candidate_joint" and r["condition"] == condition]
            assert len(subset) == 12
            stress.append({"length": length, "condition": condition, **count(subset)})
    for item in keys:
        if item["kind"] != "invariance":
            continue
        base = next(x for x in keys if x["kind"] == "legacy" and x["family"] == item["family"] and x["period_samples"] == 160 and x["phase_index"] == item["phase_index"] and x["length"] == item["length"])
        for arm in ("control_exact", "control_near", "candidate_joint"):
            a, b = by_input[(base["input_index"], arm)], by_input[(item["input_index"], arm)]
            invariance.append({"arm": arm, "key": item["key"],
                               "lag_equal": a["selected_lag"] == b["selected_lag"],
                               "pair_equal": a["pair_bounds_inclusive"] == b["pair_bounds_inclusive"],
                               "validity_equal": a["valid_pre_quality"] == b["valid_pre_quality"]})
    assert len(invariance) == 384 and all(r["lag_equal"] and r["pair_equal"] and r["validity_equal"] for r in invariance)
    assert len(by_key) == 2880
    result = {"scope": "Independent count-only reconstruction; no production summary helper, signal generation or fits.",
              "script_sha256": digest(Path(__file__)), "records_sha256": digest(root / "records.jsonl"),
              "summary_sha256": digest(root / "summary.json"), "legacy": legacy, "primary": primary,
              "candidate_stress": stress, "invariance_decisions_checked": len(invariance),
              "all_invariance_lags_pairs_validity_unchanged": True}
    output = Path("artifacts/joint-repeat-root-count-reconstruction-20261025.json")
    with output.open("x") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps({"legacy": legacy, "primary": primary, "invariance_checks": len(invariance)}, sort_keys=True))


if __name__ == "__main__":
    main()
