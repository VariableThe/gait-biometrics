"""Extract the WISDM and UCI HAR datasets from CI-2026.zip into datasets/.

Reads (never modifies) the pre-existing CI-2026.zip and produces:

    datasets/wisdm/train.csv
    datasets/wisdm/test.csv
    datasets/uci_har/UCI HAR Dataset/   (README.txt, features.txt,
        activity_labels.txt, train/, test/, each split's Inertial Signals/)

The HAR archive is nested: human+activity+recognition+using+smartphones.zip
contains UCI HAR Dataset.zip, which is extracted one further level.

Idempotent: complete datasets are skipped. Partial datasets are repaired from
the existing archive, preserving ordinary nonempty files. Required files are
checked before and after extraction; README alone is not a completion marker.
"""

from __future__ import annotations

import io
import ntpath
import os
import shutil
import stat
import sys
import tempfile
import zipfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVE = os.path.join(PROJECT_ROOT, "CI-2026.zip")
DATASETS_ROOT = os.path.join(PROJECT_ROOT, "datasets")

WISDM_MEMBER = "CI-2026/Dataset/WISDM_Dataset.zip"
HAR_OUTER_MEMBER = "CI-2026/Dataset/human+activity+recognition+using+smartphones.zip"

HAR_TOP_DIR = "UCI HAR Dataset"
WISDM_REQUIRED_FILES = ("train.csv", "test.csv")
HAR_REQUIRED_FILES = (
    *(
        f"{HAR_TOP_DIR}/{name}"
        for name in ("README.txt", "features.txt", "activity_labels.txt")
    ),
    *(
        f"{HAR_TOP_DIR}/{split}/{prefix}_{split}.txt"
        for split in ("train", "test")
        for prefix in ("X", "y", "subject")
    ),
    *(
        f"{HAR_TOP_DIR}/{split}/Inertial Signals/{sensor}_{axis}_{split}.txt"
        for split in ("train", "test")
        for sensor in ("body_acc", "body_gyro", "total_acc")
        for axis in "xyz"
    ),
)


def _is_within(root: str, path: str) -> bool:
    """True when `path` is `root` or lives underneath it."""
    try:
        return os.path.commonpath([root, path]) == root
    except ValueError:  # different drives / mixed relative+absolute
        return False


def _safe_member_target(dest: str, name: str) -> str:
    """Resolve archive member `name` under `dest`, or raise `ValueError`.

    Rejects anything that would write outside `dest`: absolute paths (POSIX,
    Windows drive-letter and UNC/backslash forms), ``..`` traversal, and any
    path that resolves outside `dest` through a symlink -- including a
    *pre-existing* symlink component such as ``dest/link -> /outside``, which a
    lexical check cannot see. `dest` itself may be a symlink; its resolved
    target is the containment root.
    """
    if not name or name.endswith("/"):
        raise ValueError(f"not a file member: {name!r}")

    # Zip members are '/'-separated; treat '\' as a separator too so a
    # Windows-style traversal cannot slip through on a POSIX host.
    normalized = name.replace("\\", "/")
    if os.path.isabs(name) or normalized.startswith("/"):
        raise ValueError(f"absolute archive member escapes destination: {name!r}")
    # `ntpath` catches Windows drive-letter (C:\) and UNC (\\host\share) members,
    # which `os.path` on a POSIX host does not treat as absolute.
    if ntpath.splitdrive(normalized)[0] or ntpath.splitdrive(name)[0]:
        raise ValueError(f"absolute archive member escapes destination: {name!r}")

    parts = [part for part in normalized.split("/") if part not in ("", ".")]
    if not parts or ".." in parts:
        raise ValueError(f"archive member escapes destination: {name!r}")

    dest_abs = os.path.abspath(dest)
    dest_real = os.path.realpath(dest_abs)
    target = os.path.join(dest_abs, *parts)

    # Lexical containment ...
    if not _is_within(dest_abs, os.path.abspath(target)):
        raise ValueError(f"archive member escapes destination: {name!r}")
    # ... and real containment, which resolves any pre-existing symlink in the
    # component chain (realpath is non-strict, so not-yet-created parts are kept
    # as plain names).
    if not _is_within(dest_real, os.path.realpath(target)):
        raise ValueError(
            f"archive member escapes destination through a symlink: {name!r}"
        )

    # Reject pre-existing symlink components outright rather than relying on the
    # realpath comparison alone; this also keeps the write below from following
    # a link that a later member could otherwise plant.
    current = dest_abs
    for part in parts:
        current = os.path.join(current, part)
        if os.path.islink(current):
            raise ValueError(
                f"refusing to extract through symlink component: {name!r} "
                f"({current!r} is a symlink)"
            )
    return target


def _is_symlink_member(info: zipfile.ZipInfo) -> bool:
    """True when the member is a symlink (unix mode stored in external_attr)."""
    return stat.S_ISLNK(info.external_attr >> 16)


def _open_no_follow(target: str):
    """Open `target` for writing without following a symlink at the leaf.

    `O_NOFOLLOW` closes the gap between `_safe_member_target`'s checks and the
    actual write; it is skipped where the platform does not provide it, in which
    case the pre-checks are the only guard.
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
    return os.fdopen(os.open(target, flags, 0o666), "wb")


def _extract_zipfile_to(
    zip_bytes: bytes,
    dest: str,
    skip_prefixes=(),
    *,
    preserve_existing: bool = False,
    required_files=(),
) -> list[str]:
    """Extract a zip archive given as bytes into dest, skipping junk paths.

    Path safety: members that would land outside `dest` (``../`` traversal,
    absolute names, or a pre-existing symlink component) and symlink members are
    rejected with `ValueError` instead of being written. Directory entries and
    the archives' macOS metadata junk (``__MACOSX/``, ``._*``, ``.DS_Store``) are
    skipped as before. All member paths and required archive entries are checked
    before writing. Each file is replaced atomically only after its CRC passes.
    Repair callers may preserve existing ordinary nonempty files.
    """
    written: list[str] = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        members = {}
        for info in zf.infolist():
            name = info.filename
            if name.startswith(skip_prefixes) or name.endswith("/"):
                continue
            if os.path.basename(name).startswith("._") or name.endswith(".DS_Store"):
                continue
            if _is_symlink_member(info):
                raise ValueError(
                    f"refusing to extract symlink archive member: {name!r}"
                )
            kind = stat.S_IFMT(info.external_attr >> 16)
            if kind not in (0, stat.S_IFREG):
                raise ValueError(f"expected an ordinary archive file: {name!r}")
            target = _safe_member_target(dest, name)
            if os.path.lexists(target) and not os.path.isfile(target):
                raise ValueError(f"expected an ordinary destination file: {target!r}")
            if target in members:
                raise ValueError(f"duplicate archive member destination: {name!r}")
            members[target] = info
        for name in required_files:
            target = _safe_member_target(dest, name)
            if target not in members or members[target].file_size <= 0:
                raise ValueError(f"archive is missing a nonempty required file: {name}")
        for target, info in members.items():
            if (
                preserve_existing
                and os.path.isfile(target)
                and os.path.getsize(target) > 0
            ):
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            temporary = None
            try:
                with (
                    zf.open(info) as src,
                    tempfile.NamedTemporaryFile(
                        dir=os.path.dirname(target), delete=False
                    ) as dst,
                ):
                    temporary = dst.name
                    shutil.copyfileobj(src, dst)
                # Recheck the destination after writing the temporary file.
                _safe_member_target(dest, info.filename)
                os.replace(temporary, target)
            finally:
                if temporary is not None and os.path.exists(temporary):
                    os.unlink(temporary)
            written.append(target)
    return written


def _dataset_complete(dataset: str, required_files) -> bool:
    # Validate relative to DATASETS_ROOT, so even a complete directory symlink
    # cannot silently move the extraction boundary outside the dataset root.
    targets = [
        _safe_member_target(DATASETS_ROOT, f"{dataset}/{name}")
        for name in required_files
    ]
    return all(os.path.isfile(path) and os.path.getsize(path) > 0 for path in targets)


def setup_wisdm() -> bool:
    """Extract WISDM_Dataset.zip -> datasets/wisdm/{train,test}.csv."""
    dest = os.path.join(DATASETS_ROOT, "wisdm")
    if _dataset_complete("wisdm", WISDM_REQUIRED_FILES):
        print("[wisdm] already present, skipping")
        return False
    with zipfile.ZipFile(ARCHIVE) as outer:
        inner_bytes = outer.read(WISDM_MEMBER)
    written = _extract_zipfile_to(
        inner_bytes, dest, preserve_existing=True, required_files=WISDM_REQUIRED_FILES
    )
    if not _dataset_complete("wisdm", WISDM_REQUIRED_FILES):
        raise ValueError("WISDM extraction did not produce all required files")
    print(f"[wisdm] extracted {len(written)} files -> {dest}")
    return True


def setup_har() -> bool:
    """Extract the nested HAR archive -> datasets/uci_har/UCI HAR Dataset/."""
    dest = os.path.join(DATASETS_ROOT, "uci_har")
    if _dataset_complete("uci_har", HAR_REQUIRED_FILES):
        print("[har] already present, skipping")
        return False
    with zipfile.ZipFile(ARCHIVE) as outer:
        outer_bytes = outer.read(HAR_OUTER_MEMBER)
    with zipfile.ZipFile(io.BytesIO(outer_bytes)) as mid:
        inner_bytes = mid.read(f"{HAR_TOP_DIR}.zip")
    written = _extract_zipfile_to(
        inner_bytes,
        dest,
        skip_prefixes=("__MACOSX/",),
        preserve_existing=True,
        required_files=HAR_REQUIRED_FILES,
    )
    if not _dataset_complete("uci_har", HAR_REQUIRED_FILES):
        raise ValueError("HAR extraction did not produce all required files")
    print(f"[har] extracted {len(written)} files -> {dest}")
    return True


def _count_rows(path: str) -> int:
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def print_layout() -> None:
    print("\nFinal layout:")
    for dirpath, dirnames, filenames in os.walk(DATASETS_ROOT):
        dirnames.sort()
        rel = os.path.relpath(dirpath, PROJECT_ROOT)
        print(f"{rel}/")
        for fn in sorted(filenames):
            full = os.path.join(dirpath, fn)
            size = os.path.getsize(full)
            rows = _count_rows(full)
            print(f"  {fn}  ({rows} rows, {size} bytes)")


def main() -> int:
    if not os.path.exists(ARCHIVE):
        print(f"ERROR: archive not found: {ARCHIVE}", file=sys.stderr)
        return 1
    setup_wisdm()
    setup_har()
    print_layout()
    return 0


if __name__ == "__main__":
    sys.exit(main())
