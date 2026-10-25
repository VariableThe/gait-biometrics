"""Secondary planning arithmetic on a completed public metadata result only."""

import collections
import gzip
import hashlib
import json
from fractions import Fraction
from pathlib import Path


def main():
    base = Path("gait-biometrics/reports/2026-10-25-support")
    raw = gzip.decompress((base / "evidence/result.json.gz").read_bytes())
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == json.loads((base / "provenance.json").read_text())["result_sha256"]
    inventory = json.loads(raw)["inventory"]
    selected, excluded = [], []
    for segment in inventory["segments"]:
        length = segment["length"]
        if length < 256:
            excluded.append(segment)
            continue
        start = segment["original"]["start_row"]
        stop = segment["original"]["stop_row"]
        assert stop - start == length
        tiles = [(i, i + 256) for i in range(start, stop - 255, 256)]
        covers = sorted(set(tiles + [(stop - 256, stop)]))
        multiplicity = [sum(a <= r < b for a, b in covers) for r in range(start, stop)]
        assert min(multiplicity) >= 1
        weights = [
            sum((Fraction(1, multiplicity[r - start]) for r in range(a, b)), Fraction()) / length
            for a, b in covers
        ]
        assert sum(weights) == 1
        selected.append({
            "subject": segment["subject"], "day": segment["day"], "trial": segment["trial"],
            "recording_index": segment["recording_index"], "segment_index": segment["segment_index"],
            "length": length, "source_rows": [start, stop],
            "historical_tiles": [list(x) for x in tiles], "cover_intervals": [list(x) for x in covers],
            "weights_exact": [str(w) for w in weights],
            "multiplicity_counts": {str(k): v for k, v in sorted(collections.Counter(multiplicity).items())},
        })
    old_keys = {(t["subject"], t["day"], t["trial"]) for t in inventory["trials"] if t["in_original_trial_graph"]}
    new_keys = {(s["subject"], s["day"], s["trial"]) for s in selected}
    assert old_keys == new_keys and len(selected) == len(new_keys)
    rows = sum(s["length"] for s in selected)
    historical_rows = sum(s["support"]["historical256"]["unique_original_row_coverage"] for s in inventory["segments"])
    covers = sum(len(s["cover_intervals"]) for s in selected)
    tiles = sum(len(s["historical_tiles"]) for s in selected)
    totals = {"eligible_segments": len(selected), "eligible_rows": rows,
              "historical_rows": historical_rows, "recovered_rows": rows - historical_rows,
              "relative_row_increase_exact": str(Fraction(rows - historical_rows, historical_rows)),
              "excluded_fragments": len(excluded), "excluded_rows": sum(s["length"] for s in excluded),
              "excluded_rows_in_historical_trials": sum(s["length"] for s in excluded if (s["subject"],s["day"],s["trial"]) in old_keys),
              "historical_tiles": tiles, "endcaps": covers-tiles, "cover_windows": covers,
              "window_row_observations": covers*256, "duplicate_row_observations": covers*256-rows,
              "eligible_length_min": min(s["length"] for s in selected),
              "eligible_length_max": max(s["length"] for s in selected),
              "subject_day_cells": len({(s["subject"], s["day"]) for s in selected})}
    assert [len(selected), rows, covers, tiles] == [299,183128,885,588]
    result = {"status": "secondary_metadata_planning_arithmetic", "input_decompressed_sha256": digest,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "no_signals_features_scores_fits_or_new_registered_inventory": True,
              "interpretation": "Observation coverage only; overlap weighting does not make nonlinear descriptors row-equivalent or observations independent.",
              "totals": totals, "eligible_segments": selected}
    output = Path("artifacts/complete-support-root-metadata-analysis-20261025.json")
    with output.open("x") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(totals, sort_keys=True))


if __name__ == "__main__":
    main()
