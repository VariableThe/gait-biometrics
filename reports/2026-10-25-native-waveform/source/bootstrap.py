"""Preparation construction only: text/AST copies from explicitly allowed SOURCE.

Never imports or executes any parent module. Refuses existing destinations.
"""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent / "joint-repeat-feasibility-r1"
COPIES = []
PARENTS = {}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(name):
    path = PARENT / name
    data = path.read_bytes()
    PARENTS[str(path)] = digest(data)
    return data.decode()


def write(name, text):
    with (ROOT / name).open("x") as handle:
        handle.write(text)


def extract(source, names, header, destination):
    text = read(source)
    nodes = {n.name: n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}
    segments = []
    for name in names:
        segment = ast.get_source_segment(text, nodes[name])
        segments.append(segment)
        COPIES.append({"parent": str(PARENT / source), "parent_sha256": digest(text.encode()),
                       "function": name, "function_utf8_sha256": digest(segment.encode()),
                       "destination": destination, "mode": "verbatim_AST_function"})
    tail = "\nstage = contextmanager(stage)\n" if destination == "common.py" else ""
    write(destination, header + "\n\n" + "\n\n\n".join(segments) + "\n" + tail)


def main():
    extract("legacy.py", ("fixture", "correlations", "select", "correlation_summary", "minima_and_pair", "fixed_checks"),
            '"""Verbatim legacy generator and fixed helper controls; no parent imports."""\n'
            'import itertools\nimport warnings\nimport numpy as np\n'
            'PERIODS = (80,90,100,110,120,128,140,160,180,200)\n'
            'FAMILIES = ("gravity_periodic_6axis", "alternating_step_harmonic", "pure_periodic_balanced_peaks", "weak_degenerate_gyro")\n'
            'RULES = ("exact_max_smallest_exact_tie", "nearmax_atol_1e-10_smallest")\n'
            'LAGS = np.arange(80,201)\nGUARD = 1e-12\nATOL = 1e-10\n', "legacy.py")
    extract("plateau_helpers.py", ("exact_plateau_runs", "independent_groupby_reference", "synthetic_helper_checks"),
            '"""Reused 16 convention handchecks; candidate pairing is implemented separately."""\n'
            'import itertools\nimport numpy as np\nimport legacy as fa\n', "plateau_helpers.py")
    extract("kernel.py", ("magnitudes", "filter_modalities", "vector_guard", "pearson", "joint_correlations", "whole_window"),
            '"""Verbatim reviewed numerical pure components; no parent imports."""\n'
            'import common as c\nimport legacy\nimport numpy as np\nfrom scipy.signal import sosfiltfilt\n', "guards.py")
    extract("common.py", ("require", "typed", "canonical", "fingerprint", "sha", "save_json", "load_json", "decode_json", "runtime_identity", "validate_runtime", "utc", "elapsed", "stage", "arm_budget"),
            '"""Reviewed runtime closure/serialization; only explicit allowed parent source hashes."""\n'
            'from __future__ import annotations\n'
            'import os, sys, time\nfrom pathlib import Path\n'
            'START_WALL = time.perf_counter()\n'
            'if os.environ.get("JOINT_RUN_LAUNCH_EPOCH"):\n'
            '    START_WALL -= max(0.,time.time()-float(os.environ["JOINT_RUN_LAUNCH_EPOCH"]))\n'
            'START_CPU = time.process_time()\nROOT = Path(__file__).resolve().parent\n'
            'THREAD_KEYS = ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS","BLIS_NUM_THREADS","OMP_THREAD_LIMIT")\n'
            'for key in THREAD_KEYS: os.environ[key] = "1"\n'
            'os.environ["CUDA_VISIBLE_DEVICES"] = ""\nos.environ["PYTHONDONTWRITEBYTECODE"] = "1"\nsys.dont_write_bytecode = True\n'
            'os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})\n'
            'import hashlib, json, math, platform, resource, signal, shutil, sysconfig\n'
            'from contextlib import contextmanager\nimport numpy as np\nimport scipy\nfrom scipy.signal import butter\n'
            'EXPECTED_VERSIONS = {"python":"3.14.8","numpy":"2.5.4","scipy":"1.18.1"}\n'
            'SOS = butter(2,3.,btype="lowpass",fs=100.,output="sos")\n'
            'CHECK = lambda: None\nclass BudgetExceeded(RuntimeError): pass\n', "common.py")
    launcher = read("run.sh")
    write("run.sh", launcher)
    COPIES.append({"parent": str(PARENT / "run.sh"), "parent_sha256": digest(launcher.encode()),
                   "destination": "run.sh", "mode": "verbatim_file"})
    entry = read("study.py")
    prefix = entry.split("from collections import Counter", 1)[0]
    write("entry_guard.source", prefix)
    COPIES.append({"parent": str(PARENT / "study.py"), "parent_sha256": digest(entry.encode()),
                   "destination": "entry_guard.source", "mode": "verbatim_prefix", "bytes": len(prefix.encode())})
    # Original generator-bearing SOURCE bytes retained as inert text, never imported.
    write("legacy_parent.source", read("legacy.py"))
    COPIES.append({"parent": str(PARENT / "legacy.py"), "parent_sha256": PARENTS[str(PARENT / "legacy.py")],
                   "destination": "legacy_parent.source", "mode": "verbatim_file"})
    for relative in ("joint-repeat-preparation-reviewed-r2/snapshot/BRIEF.md",):
        path = ROOT.parent / relative
        data = path.read_bytes()
        PARENTS[str(path)] = digest(data)
        write("stress_brief.source", data.decode())
    # Explicit completed public-support metadata only. No measurement outputs are read.
    metadata = ROOT.parent / "complete-support-root-metadata-analysis-20261025.json"
    PARENTS[str(metadata)] = digest(metadata.read_bytes())
    write("copy_provenance.json", json.dumps({"copies": COPIES, "parents_sha256": PARENTS,
          "restored_decorators": {"common.py": {"stage": "contextmanager"}},
          "parent_top_level_executed": False, "active_measurement_outputs_read": False}, sort_keys=True, indent=2)+"\n")


if __name__ == "__main__":
    main()
