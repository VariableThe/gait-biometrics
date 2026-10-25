"""Strict, fit-free publication IO. No archive, feature or model reader."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import stat
import sys
from pathlib import Path


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
        require(math.isfinite(value), "Nonfinite JSON")
        return ("float", value.hex())
    require(kind in (str, int, bool, type(None)), "Unsupported JSON type")
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
    require(typed(decode(raw)) == original, "Lossy JSON type/signed-zero round trip")
    return raw


def fingerprint(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def same(actual, expected, name):
    require(typed(actual) == typed(expected), "Mismatch: " + name)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    path = Path(path)
    raw = path.read_bytes()
    return decode(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def write(path, value):
    with Path(path).open("xb") as handle:
        handle.write(canonical(value) + b"\n")


def relative(name):
    require(type(name) is str and name != "", "Nonempty relative path required")
    path = Path(name)
    require(
        not path.is_absolute()
        and path.as_posix() == name
        and all(p not in (".", "..", "") for p in name.split("/"))
        and "\\" not in name,
        "Unsafe relative path: " + name,
    )
    return path


def absolute(name):
    require(type(name) is str and name.startswith("/"), "Absolute path required")
    require("//" not in name and "\\" not in name, "Unsafe absolute path")
    require(all(p not in (".", "..", "") for p in name[1:].split("/")), "Unsafe path")
    path = Path(name)
    for component in (path, *path.parents):
        require(not component.is_symlink(), "Symlink path component forbidden")
    return path


def file_index(root):
    root = Path(root)
    result, directories = {}, set()
    require(
        root.is_dir() and not root.is_symlink(), "Regular bundle directory required"
    )
    for path in sorted(root.rglob("*")):
        mode = path.lstat().st_mode
        require(not stat.S_ISLNK(mode), "Symlink forbidden: " + str(path))
        name = path.relative_to(root).as_posix()
        relative(name)
        if stat.S_ISDIR(mode):
            directories.add(name)
        else:
            require(stat.S_ISREG(mode), "Only regular files permitted")
            if name != "inventory.json":
                result[name] = {"bytes": path.stat().st_size, "sha256": sha(path)}
    expected_directories = {
        str(parent)
        for name in result
        for parent in Path(name).parents
        if str(parent) != "."
    }
    require(directories == expected_directories, "Unknown or empty directory")
    return result


def check_inventory(root):
    actual = {"schema": 1, "excludes_only": "inventory.json", "files": file_index(root)}
    same(read(Path(root) / "inventory.json"), actual, "exact publication inventory")
    return actual


def enforce_environment():
    require(sys.dont_write_bytecode, "Use Python -B")
    require(os.environ.get("PYTHONDONTWRITEBYTECODE") == "1", "Disable bytecode writes")
    require(os.environ.get("UV_OFFLINE") == "1", "UV_OFFLINE=1 required")
    require(
        os.environ.get("UV_PYTHON_DOWNLOADS") == "never", "Disable Python downloads"
    )
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CUDA must be hidden")
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
        "OMP_THREAD_LIMIT",
    ):
        require(os.environ.get(key) == "1", "One numerical thread required: " + key)
    require(len(os.sched_getaffinity(0)) == 1, "Exactly one CPU affinity required")


def mapped_path(kind, name):
    relative(name)
    prefix = "snapshot" if kind == "source" else "evidence"
    return prefix + "/" + name + (".gz" if name.endswith(".json") else "")


def source_selected(name):
    path = relative(name)
    return (
        (len(path.parts) == 1 and path.suffix in (".py", ".sh", ".md", ".json"))
        or (path.parts[0] == "vendor" and path.suffix == ".py")
        or path.parts[0] == "inputs"
        or (path.parts[0] == "preparation-history" and path.suffix in (".json", ".log"))
    )
