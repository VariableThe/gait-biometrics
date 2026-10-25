"""Strict evidence IO and the fresh, conservatively enumerated runtime closure."""

from __future__ import annotations

import gzip
import hashlib
import importlib
import importlib.util
import json
import math
import os
import shutil
import sys
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parent
THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "OMP_THREAD_LIMIT",
)
_runtime_binding: dict | None = None
_runtime_audit_installed = False
_runtime_checking = False
_consumed_executables = set()


def executable_file(path):
    return (
        path.suffix in (".py", ".pyc", ".so", ".pyd", ".dll", ".dylib")
        or ".so." in path.name
    )


def assert_runtime_paths(paths, binding):
    """Check bytes as well as membership; -B still permits cache consumption."""
    global _runtime_checking
    previous = _runtime_checking
    _runtime_checking = True
    try:
        for name in paths:
            path = Path(name).absolute()
            record = binding["files"].get(str(path))
            require(record is not None, "Unregistered executable path: " + str(path))
            require(path.is_file(), "Missing bound executable: " + str(path))
            resolved = path.resolve(strict=True)
            stat = resolved.stat()
            same(
                {
                    "resolved": str(resolved),
                    "sha256": sha(resolved),
                    "bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "mode": stat.st_mode,
                },
                record,
                "bound executable bytes/identity: " + str(path),
            )
    finally:
        _runtime_checking = previous


def start_runtime_audit():
    """Observe cache reads, imported natives and Python execution before science imports.

    After binding, reject a late executable import before its code runs. Dynamic
    in-memory code has no on-disk origin; OS/kernel closure is not claimed.
    """
    global _runtime_audit_installed
    if _runtime_audit_installed:
        return

    def audit(event, args):
        if _runtime_checking:
            return
        name = None
        if event == "open" and args and isinstance(args[0], (str, bytes)):
            candidate = Path(os.fsdecode(args[0]))
            # Missing candidate caches are normal importlib probes.
            if candidate.suffix == ".pyc" and candidate.is_file():
                name = candidate
        elif event == "exec" and args:
            filename = args[0].co_filename
            if not filename.startswith("<"):
                name = Path(filename)
        elif event == "import" and len(args) > 1 and args[1]:
            name = Path(args[1])
        elif event == "ctypes.dlopen" and args and args[0]:
            candidate = Path(args[0])
            # Unresolved sonames cannot establish a file binding; refuse late.
            if _runtime_binding is not None:
                require(
                    candidate.is_absolute(),
                    "Unresolved late native library: " + str(candidate),
                )
            if candidate.is_absolute():
                name = candidate
        if name is not None:
            name = str(name.absolute())
            if _runtime_binding is not None:
                assert_runtime_paths([name], _runtime_binding)
            _consumed_executables.add(name)

    sys.addaudithook(audit)
    _runtime_audit_installed = True


def module_paths(modules=None):
    """Every loaded module file/cache/spec origin, including single-file helpers."""
    result = set()
    for module in tuple((sys.modules if modules is None else modules).values()):
        if module is None:
            continue
        spec = getattr(module, "__spec__", None)
        for name, required in (
            (getattr(module, "__file__", None), True),
            (getattr(module, "__cached__", None), False),
            (getattr(spec, "origin", None), True),
        ):
            if (
                not isinstance(name, str)
                or name in ("built-in", "frozen")
                or name.startswith("<")
            ):
                continue
            path = Path(name).absolute()
            require(
                not required or path.is_file(),
                "Missing loaded executable origin: " + str(path),
            )
            if path.is_file():
                result.add(str(path))
                if path.suffix == ".py":
                    cached = Path(importlib.util.cache_from_source(str(path)))
                    if cached.is_file():
                        result.add(str(cached))
                    # Bind other existing interpreter/optimization variants too.
                    result.update(
                        str(p.absolute())
                        for p in (path.parent / "__pycache__").glob(
                            path.stem + ".*.pyc"
                        )
                    )
    return result


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def typed(value):
    kind = type(value)
    if kind is dict:
        require(all(type(k) is str for k in value), "JSON keys must already be strings")
        return ("dict", tuple((k, typed(value[k])) for k in sorted(value)))
    if kind is list:
        return ("list", tuple(typed(v) for v in value))
    if kind is float:
        require(
            math.isfinite(value),
            "Nonfinite JSON: persist masks/NPZ and nullable diagnostics",
        )
        return ("float", value.hex())
    require(kind in (str, int, bool, type(None)), "unsupported JSON type: " + str(kind))
    return (kind.__name__, value)


def pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def decode(raw):
    def bad(token):
        raise RuntimeError("Nonfinite JSON token: " + token)

    result = json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)
    typed(result)
    return result


def canonical(value):
    original = typed(value)
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    require(typed(decode(raw)) == original, "Lossy JSON type or signed-zero round trip")
    return raw


def fingerprint(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    with Path(path).open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256")
    return digest.hexdigest()


def read(path):
    path = Path(path)
    raw = path.read_bytes()
    return decode(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def write(path, value):
    with Path(path).open("xb") as handle:
        handle.write(canonical(value) + b"\n")


def same(actual, expected, name):
    require(typed(actual) == typed(expected), "Mismatch: " + name)


def artifact(root, path):
    path = Path(path)
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha(path),
        "bytes": path.stat().st_size,
    }


def checked(root, record):
    path = Path(root) / record["path"]
    require(
        not path.is_symlink() and path.resolve().is_relative_to(Path(root).resolve()),
        "Unsafe artifact path",
    )
    require(
        path.is_file()
        and path.stat().st_size == record["bytes"]
        and sha(path) == record["sha256"],
        "Artifact hash/size mismatch: " + str(path),
    )
    return path


def save_arrays(root, name, arrays):
    import numpy as np

    require(all(type(k) is str for k in arrays), "NPZ string keys required")
    for key, array in arrays.items():
        require(
            isinstance(array, np.ndarray) and not array.dtype.hasobject,
            "Lossless nonobject ndarray required: " + key,
        )
    path = Path(root) / (name + ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez(handle, **arrays)
    return {
        **artifact(root, path),
        "schema": {
            k: {"dtype": a.dtype.str, "shape": list(a.shape)} for k, a in arrays.items()
        },
    }


def load_arrays(root, record):
    import numpy as np

    with np.load(checked(root, record), allow_pickle=False) as file:
        arrays = {k: file[k] for k in file.files}
    same(
        {k: {"dtype": a.dtype.str, "shape": list(a.shape)} for k, a in arrays.items()},
        record["schema"],
        "NPZ schema/dtypes",
    )
    return arrays


def file_index(root):
    root = Path(root)
    result = {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), "Symlink forbidden in owned evidence tree")
        if path.is_file() and path != root / "index.json":
            result[path.relative_to(root).as_posix()] = {
                "sha256": sha(path),
                "bytes": path.stat().st_size,
            }
    return result


def seal(root):
    write(
        Path(root) / "index.json",
        {"schema": 1, "excludes_only": "index.json", "files": file_index(root)},
    )


def check_index(root):
    record = read(Path(root) / "index.json")
    same(
        record,
        {"schema": 1, "excludes_only": "index.json", "files": file_index(root)},
        "exact whole-file index",
    )


def enforce_environment():
    require(
        sys.dont_write_bytecode and os.environ.get("PYTHONDONTWRITEBYTECODE") == "1",
        "Use -B and PYTHONDONTWRITEBYTECODE=1",
    )
    require(
        os.environ.get("UV_OFFLINE") == "1"
        and os.environ.get("UV_PYTHON_DOWNLOADS") == "never",
        "Offline existing environment required",
    )
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and all(os.environ.get(k) == "1" for k in THREAD_KEYS),
        "CPU-only numerical thread environment required",
    )
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    threads = next(
        line
        for line in Path("/proc/self/status").read_text().splitlines()
        if line.startswith("Threads:")
    )
    require(int(threads.split()[1]) == 1, "Exactly one process thread required")


def runtime_identity():
    """Loaded executable origins/caches and complete loaded package roots.

    This is a new closure, not an assertion of historical runtime equivalence.
    No kernel/firmware/unloaded external service closure is asserted.
    """
    global _runtime_binding
    enforce_environment()
    import numpy as np
    import scipy.linalg

    np.linalg.eigh(np.eye(2))
    scipy.linalg.eigh(np.eye(2))
    files = {}

    def include(path):
        path = Path(path)
        if path.is_file():
            resolved = path.resolve(strict=True)
            stat = resolved.stat()
            files[str(path.absolute())] = {
                "resolved": str(resolved),
                "sha256": sha(resolved),
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "mode": stat.st_mode,
            }

    stdlib = Path(sysconfig.get_path("stdlib"))
    for path in sorted(stdlib.rglob("*")):
        if "site-packages" not in path.relative_to(stdlib).parts and path.suffix in (
            ".py",
            ".pyc",
            ".so",
        ):
            include(path)
    versions = {}
    for name in ("numpy", "scipy", "sklearn", "joblib", "threadpoolctl"):
        module = importlib.import_module(name)
        versions[name] = module.__version__
    # Version labels are informational. File binding follows ALL loaded modules,
    # not a hand-maintained dependency-name list. This includes pandas/dateutil
    # and caches beside single-file modules such as threadpoolctl and six.
    roots = set()
    for module in tuple(sys.modules.values()):
        for name in getattr(module, "__path__", ()):
            path = Path(name).absolute()
            if path.is_dir():
                roots.add(path)
    roots = {p for p in roots if not any(p != q and p.is_relative_to(q) for q in roots)}
    for base in sorted(roots):
        for path in sorted(base.rglob("*")):
            if executable_file(path):
                include(path)
        # Wheels can keep native dependencies outside their package directory.
        for sibling in sorted(base.parent.glob("*.libs")):
            for path in sorted(sibling.rglob("*")):
                if path.is_file():
                    include(path)
    for name in sorted(module_paths() | _consumed_executables):
        include(name)
    include(Path(sys.prefix) / "pyvenv.cfg")
    for base in (ROOT.parent.parent, ROOT.parent.parent / "gait-biometrics"):
        for name in (".python-version", "pyproject.toml", "uv.lock"):
            include(base / name)
    for path in sorted(ROOT.rglob("*")):
        if executable_file(path) or path.suffix == ".sh":
            include(path)
    for line in Path("/proc/self/maps").read_text().splitlines():
        name = line.split()[-1]
        if name.startswith("/") and (".so" in Path(name).name or "ld-linux" in name):
            include(name)
    launchers = (
        sys.executable,
        "/home/sherlock/.local/bin/uv",
        "/usr/bin/timeout",
        "/usr/bin/env",
        shutil.which("bash"),
    )
    for path in launchers:
        require(path is not None, "Missing launcher")
        include(path)
    result = {
        "schema": 2,
        "files": files,
        "versions": versions,
        "python": sys.version,
        "implementation": sys.implementation.name,
        "byteorder": sys.byteorder,
        "executable": sys.executable,
        "prefix": sys.prefix,
        "sys_path": list(sys.path),
        "cwd": str(Path.cwd()),
        "flags": repr(sys.flags),
        "xoptions": {
            k: v for k, v in sys._xoptions.items() if k != "full_support_reserved"
        },
        "affinity": sorted(os.sched_getaffinity(0)),
        "environment": {
            k: os.environ.get(k)
            for k in (
                *THREAD_KEYS,
                "PATH",
                "PYTHONPATH",
                "PYTHONHOME",
                "LD_PRELOAD",
                "LD_LIBRARY_PATH",
                "PYTHONPYCACHEPREFIX",
                "PYTHONHASHSEED",
                "PYTHONSAFEPATH",
                "VIRTUAL_ENV",
                "UV_OFFLINE",
                "UV_PYTHON_DOWNLOADS",
                "CUDA_VISIBLE_DEVICES",
                "PYTHONDONTWRITEBYTECODE",
            )
        },
        "coverage": "all loaded module __file__/__cached__/spec-origin and observed executable paths; complete loaded transitive package roots and existing caches, stdlib, sibling .libs, owned code, mapped shared objects and launchers; late executable import bytes checked before execution; no OS/kernel/firmware, in-memory generated code, unloaded service or historical-runtime closure claim",
    }
    assert_runtime_paths(module_paths() | _consumed_executables, result)
    if _runtime_binding is not None:
        same(result, _runtime_binding, "stable fresh-process executable closure")
    _runtime_binding = result
    return result
