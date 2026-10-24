"""Replay external score/gate JSONL through one causal authentication episode.

Use ``python -m scripts.stream_authentication --threshold VALUE`` from the repo
root. The threshold has no default: trial-level study thresholds are not valid
single-window policy calibration. See docs/stream_authentication.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import TextIO

from engine.continuous_evaluation import AuthenticationStream


def run_stream(stream: AuthenticationStream, source: TextIO, output: TextIO) -> int:
    """Emit/flush one JSON-safe decision per line; malformed input exits 2.

    Continue after errors so a later valid event visibly remains challenged.
    EOF never invents a clock tick or clears a challenge.
    """
    failed = False
    for line_number, line in enumerate(source, start=1):
        result = stream.process_line(line)
        result["line_number"] = line_number
        failed = failed or not result["input_valid"]
        output.write(json.dumps(result, allow_nan=False, sort_keys=True) + "\n")
        output.flush()
    return 2 if failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--required-accepts", type=int, default=2)
    parser.add_argument("--required-rejects", type=int, default=2)
    parser.add_argument("--walking-threshold", type=float, default=0.8)
    parser.add_argument("--max-evidence-age-seconds", type=float, default=30)
    parser.add_argument("--max-gap-seconds", type=float, default=10)
    args = parser.parse_args(argv)
    try:
        stream = AuthenticationStream(
            args.threshold,
            history_size=args.history_size,
            required_accepts=args.required_accepts,
            required_rejects=args.required_rejects,
            walking_threshold=args.walking_threshold,
            max_evidence_age_seconds=args.max_evidence_age_seconds,
            max_gap_seconds=args.max_gap_seconds,
        )
    except ValueError as error:
        parser.error(str(error))
    return run_stream(stream, sys.stdin, sys.stdout)


if __name__ == "__main__":
    raise SystemExit(main())
