"""One-time preparation utility: copy only AST-delimited parent pure functions."""

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENTS = ROOT.parent
SPECS = {
    "legacy.py": (
        "cycle-length-feasibility-r1/prototype.py",
        (
            "fixture",
            "correlations",
            "select",
            "correlation_summary",
            "minima_and_pair",
            "features_from_pair",
            "extract",
            "alias",
            "truth",
            "geometric_reference",
            "fixed_checks",
        ),
        """# Verbatim pure functions; parent top-level code is never imported.
import itertools
import math
import warnings
import numpy as np
from scipy.interpolate import CubicSpline
PERIODS = (80, 90, 100, 110, 120, 128, 140, 160, 180, 200)
FAMILIES = ("gravity_periodic_6axis", "alternating_step_harmonic", "pure_periodic_balanced_peaks", "weak_degenerate_gyro")
RULES = ("exact_max_smallest_exact_tie", "nearmax_atol_1e-10_smallest")
LAGS = np.arange(80, 201)
GUARD = 1e-12
ATOL = 1e-10
""",
    ),
    "plateau_helpers.py": (
        "cycle-plateau-regression-r1/runner.py",
        (
            "exact_plateau_runs",
            "ranked_pairs",
            "plateau_minima_and_pair",
            "independent_groupby_reference",
            "synthetic_helper_checks",
        ),
        """# Helper-only regression. Main extraction NEVER uses these minima.
import itertools
import numpy as np
import legacy as fa
""",
    ),
}


def main():
    provenance = []
    for destination, (source, names, header) in SPECS.items():
        path = PARENTS / source
        text = path.read_text()
        nodes = {
            n.name: n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)
        }
        segments = []
        for name in names:
            node = nodes[name]
            segment = ast.get_source_segment(text, node)
            segments.append(segment)
            provenance.append(
                {
                    "destination": destination,
                    "function": name,
                    "parent": source,
                    "parent_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "line": node.lineno,
                    "end_line": node.end_lineno,
                    "function_utf8_sha256": hashlib.sha256(
                        segment.encode()
                    ).hexdigest(),
                }
            )
        with (ROOT / destination).open("x") as handle:
            handle.write(header + "\n\n" + "\n\n\n".join(segments) + "\n")
    with (ROOT / "copy_provenance.json").open("x") as handle:
        json.dump(
            {"copies": provenance, "parent_top_level_executed": False},
            handle,
            sort_keys=True,
            indent=2,
        )
        handle.write("\n")


if __name__ == "__main__":
    main()
