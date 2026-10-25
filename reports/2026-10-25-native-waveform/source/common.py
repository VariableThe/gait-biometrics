"""Reviewed runtime closure/serialization; only explicit allowed parent source hashes."""
from __future__ import annotations
import os, sys, time
from pathlib import Path
START_WALL = time.perf_counter()
if os.environ.get("JOINT_RUN_LAUNCH_EPOCH"):
    START_WALL -= max(0.,time.time()-float(os.environ["JOINT_RUN_LAUNCH_EPOCH"]))
START_CPU = time.process_time()
ROOT = Path(__file__).resolve().parent
THREAD_KEYS = ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS","BLIS_NUM_THREADS","OMP_THREAD_LIMIT")
for key in THREAD_KEYS: os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
import hashlib, json, math, platform, resource, signal, shutil, sysconfig
from contextlib import contextmanager
import numpy as np
import scipy
from scipy.signal import butter
EXPECTED_VERSIONS = {"python":"3.14.8","numpy":"2.5.4","scipy":"1.18.1"}
SOS = butter(2,3.,btype="lowpass",fs=100.,output="sos")
CHECK = lambda: None
class BudgetExceeded(RuntimeError): pass


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def typed(value):
    kind = type(value)
    if kind is dict:
        require(
            all(type(k) is str for k in value), "Canonical keys must already be strings"
        )
        return ("dict", tuple((k, typed(value[k])) for k in sorted(value)))
    if kind is list:
        return ("list", tuple(typed(v) for v in value))
    if kind is float:
        require(
            math.isfinite(value),
            "Nonfinite JSON forbidden; use masks and lossless arrays",
        )
        return ("float", value.hex())
    require(kind in (str, int, bool, type(None)), f"Unsupported canonical type {kind}")
    return (kind.__name__, value)


def canonical(value):
    original = typed(value)
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    require(
        typed(json.loads(encoded)) == original,
        "Canonical type/float round-trip mismatch",
    )
    return encoded


def fingerprint(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            CHECK()
            h.update(block)
    return h.hexdigest()


def save_json(name, value):
    encoded = canonical(value)
    with (ROOT / name).open("xb") as handle:
        handle.write(encoded + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    require(
        typed(json.loads((ROOT / name).read_bytes())) == typed(value),
        "Saved JSON changed types/bits",
    )


def load_json(name):
    return decode_json((ROOT / name).read_bytes())


def decode_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key " + key)
            result[key] = value
        return result

    def bad_constant(value):
        raise RuntimeError("Nonfinite JSON constant " + value)

    return json.loads(data, object_pairs_hook=pairs, parse_constant=bad_constant)


def runtime_identity():
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    require(versions == EXPECTED_VERSIONS, f"Runtime mismatch: {versions}")
    # Conservative cache closure: -B suppresses writes, but does NOT suppress cache reads.
    files, metadata = {}, {}

    def include(label, path):
        path = Path(path)
        resolved = path.resolve(strict=True)
        stat = resolved.stat()
        files[label] = sha(resolved)
        metadata[label] = {
            "path": str(path.absolute()),
            "resolved": str(resolved),
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "mode": stat.st_mode,
        }

    launchers = {
        "python": sys.executable,
        "uv_launcher": "/home/sherlock/.local/bin/uv",
        "wall_timeout_launcher": "/usr/bin/timeout",
        "shebang_env": "/usr/bin/env",
        "shebang_bash": shutil.which("bash"),
    }
    for label, path in launchers.items():
        require(path is not None, "Required launcher missing " + label)
        include(label, path)
    stdlib = Path(sysconfig.get_path("stdlib"))
    for path in sorted(stdlib.rglob("*")):
        relative = path.relative_to(stdlib)
        if (
            "site-packages" not in relative.parts
            and path.is_file()
            and path.suffix in (".py", ".pyc", ".so")
        ):
            include("stdlib/" + relative.as_posix(), path)
    for label, package in (("numpy", np), ("scipy", scipy)):
        base = Path(package.__file__).parent
        for path in sorted(base.rglob("*")):
            if path.is_file() and (
                path.suffix in (".py", ".pyc", ".so") or ".so." in path.name
            ):
                include(f"{label}/{path.relative_to(base).as_posix()}", path)
        for path in sorted((base.parent / (label + ".libs")).glob("*")):
            if path.is_file():
                include(f"{label}.libs/{path.name}", path)
    for path in sorted(ROOT.rglob("*.pyc")):
        # Includes unused stale caches and retained regression-fixture caches conservatively.
        include("local_cache/" + path.relative_to(ROOT).as_posix(), path)
    # Record any consumed Python/cache/native module outside the named package trees too.
    # A module's __cached__ may name a nonexistent cache under -B; record that distinction.
    modules = {}
    for name, module in sorted(sys.modules.items()):
        origin = getattr(module, "__file__", None)
        cached = getattr(module, "__cached__", None)
        if origin and Path(origin).is_file():
            include("imported_source:" + name, origin)
        if cached and Path(cached).is_file():
            include("imported_cache:" + name, cached)
        if origin or cached:
            modules[name] = {
                "origin": origin,
                "cache": cached,
                "cache_exists": bool(cached and Path(cached).is_file()),
            }
    # Exact loaded interpreter/OS numerical dependencies as mapped before any extraction.
    mapped = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        path = line.split()[-1]
        if path.startswith("/") and (
            ".so" in Path(path).name or "ld-linux" in Path(path).name
        ):
            mapped.add(path)
    for path in sorted(mapped):
        if Path(path).is_file():
            include("mapped:" + path, path)
    process_threads = int(
        next(
            line.split(":", 1)[1]
            for line in Path("/proc/self/status").read_text().splitlines()
            if line.startswith("Threads:")
        )
    )
    require(
        process_threads == 1, "Scientific Python process must have exactly one thread"
    )
    return {
        "versions": versions,
        "machine": platform.machine(),
        "byteorder": sys.byteorder,
        "float64": np.dtype(np.float64).str,
        "implementation": sys.implementation.name,
        "files_sha256": files,
        "file_metadata": metadata,
        "imported_modules": modules,
        "coverage_policy": {
            "bytecode": "Conservative ALL existing .pyc under interpreter stdlib, NumPy, SciPy and owned job; plus existing __cached__ of loaded modules. Includes unused caches. Source/cache metadata binds Python timestamp invalidation inputs; -B only disables writes.",
            "launchers": "Python, uv, timeout, /usr/bin/env and PATH-resolved bash; run.sh path resolution uses only bash cd/pwd/parameter expansion/printf builtins (no dirname).",
            "native": "Named package sources/native libraries and their .libs, plus currently mapped shared libraries. No complete OS, kernel, firmware, unloaded dynamic dependency or external service closure claim.",
        },
        "environment": {
            "executable": sys.executable,
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
            "sys_path": list(sys.path),
            "cwd": str(Path.cwd()),
            "python_flags": repr(sys.flags),
            "python_xoptions": {
                key: value
                for key, value in sys._xoptions.items()
                if key != "joint_run_reserved"
            },
            "dont_write_bytecode": sys.dont_write_bytecode,
            "variables": {
                key: os.environ.get(key)
                for key in (
                    "PATH",
                    "PYTHONPATH",
                    "PYTHONHOME",
                    "VIRTUAL_ENV",
                    "LD_LIBRARY_PATH",
                    "LD_PRELOAD",
                    "UV_OFFLINE",
                    "UV_PYTHON_DOWNLOADS",
                    "PYTHONPYCACHEPREFIX",
                    "PYTHONDONTWRITEBYTECODE",
                    "PYTHONHASHSEED",
                    "PYTHONSAFEPATH",
                )
            },
        },
        "threads": {k: os.environ[k] for k in THREAD_KEYS},
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "affinity": sorted(os.sched_getaffinity(0)),
        "process_threads": process_threads,
        "child_processes_created_by_study": 0,
    }


def validate_runtime(saved, current=None):
    if current is None:
        current = runtime_identity()
    require(typed(current) == typed(saved), "Runtime changed since preparation")
    return current


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def elapsed():
    return {
        "cpu_seconds": time.process_time(),
        "wall_seconds": time.perf_counter() - START_WALL,
        "wall_origin": "shell_before_uv_launch"
        if os.environ.get("JOINT_RUN_LAUNCH_EPOCH")
        else "common_module_start",
    }


def stage(timings, name):
    before_cpu, before_wall = time.process_time(), time.perf_counter()
    try:
        yield
    finally:
        timings[name] = {
            "cpu_seconds": time.process_time() - before_cpu,
            "wall_seconds": time.perf_counter() - before_wall,
        }


def arm_budget():
    global CHECK

    def check():
        if time.process_time() >= 595 or time.perf_counter() - START_WALL >= 895:
            raise BudgetExceeded(
                "Exclusive attempt exceeded cooperative 595CPU/895wall cutoff; budget 600/900"
            )

    def stop(_signum, _frame):
        raise BudgetExceeded("Exclusive attempt reached CPU or wall deadline")

    CHECK = check
    signal.signal(signal.SIGXCPU, stop)
    signal.signal(signal.SIGALRM, stop)
    resource.setrlimit(resource.RLIMIT_CPU, (599, 600))
    signal.setitimer(
        signal.ITIMER_REAL, max(0.001, 900 - (time.perf_counter() - START_WALL))
    )
    CHECK()


stage = contextmanager(stage)
