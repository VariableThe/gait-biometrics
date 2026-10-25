"""Stage-first entrypoint. The root reviewer alone invokes freeze/execute."""

import sys

from guard import bootstrap, deadlines

if __name__ != "__main__":
    raise RuntimeError("Use CLI stages; do not import this entrypoint")

_identity = bootstrap(sys.argv[1:])  # receipt before any other import
deadlines(_identity)

import os
from pathlib import Path

_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_root / "vendor"))
import common as c

c.enforce_environment()
c.start_runtime_audit()

if sys.argv[1] in ("prepare", "check-preparation", "selftest"):

    def _prep_audit(event, args):
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).absolute()
            if not path.is_relative_to(_root):
                if path.suffix in (
                    ".zip",
                    ".npz",
                    ".npy",
                    ".csv",
                    ".pt",
                    ".pth",
                    ".pkl",
                    ".pickle",
                ):
                    raise RuntimeError(
                        "PREPARATION forbids rawarchive/data/model/feature/score reads: "
                        + str(path)
                    )
                if (
                    "canonical-orientation-v1" in path.parts
                    and path.name != "registration.json"
                ):
                    raise RuntimeError(
                        "PREPARATION forbids R numeric/outcome payload access: "
                        + str(path)
                    )

    sys.addaudithook(_prep_audit)

import payload
import planning
import selftests
import worker


def source_bindings():
    origins = c.read(c.ROOT / "source-origins.json")
    for item in origins:
        c.require(
            c.sha(item["origin"]) == item["sha256"],
            "Read-only source/public input changed: " + item["origin"],
        )
        if item["destination"] is not None and item["method"] in (
            "byte-identical copy",
            "public metadata, no signal/feature/model/score payload",
        ):
            c.require(
                c.sha(c.ROOT / item["destination"]) == item["sha256"],
                "Copied origin mismatch",
            )
    return {item["origin"]: item["sha256"] for item in origins}


def review_fixes(tests):
    history = planning.proposal()["systems_review_revision"]
    archive = c.ROOT.parent / "full-support-canonical-preparation-reviewed-r1"
    c.require(
        c.sha(archive / "inventory.json") == history["R1_archive_inventory_sha256"],
        "R1 preservation inventory changed",
    )
    inventory = c.read(archive / "inventory.json")
    changed = {}
    for path in sorted(c.ROOT.rglob("*")):
        if path.is_file() and (
            path.suffix in (".py", ".sh") or path.name == "README.md"
        ):
            relative = path.relative_to(c.ROOT).as_posix()
            prior = inventory["files"].get(relative)
            digest = c.sha(path)
            if prior is None or digest != prior["sha256"]:
                changed[relative] = {
                    "before_sha256": None if prior is None else prior["sha256"],
                    "after_sha256": digest,
                }
    directory = c.ROOT / tests["directory"]
    regressions = c.read(directory / "copied-R-binding-regressions.json")
    return {
        "schema": 1,
        "status": "PREPARATION_FIXED_FOR_INDEPENDENT_R2_NOT_FROZEN_NOT_EXECUTED",
        "archive_history": {
            "path": str(archive),
            "inventory_sha256": c.sha(archive / "inventory.json"),
            "preserved_files": inventory["file_count"],
            "reviewed_anchors": inventory["reviewed_anchors"],
            "verification": "complete source/snapshot hash equality verified before changes; final archive recheck recorded by reviewer handoff",
        },
        "changed_source_files": changed,
        "fixes": {
            "P1": "all loaded module file/cache/spec-origin and transitive package roots bound; source/native/bytecode/launcher/PATH rechecks; late executable imports checked before running",
            "P2": "explicit original-R numeric dtype/shape/bit equality, exact metadata/cutoffs/training/mappings, immutable reference inventory and original registration/development/results links; zero fits; compact keeps archive unopened",
        },
        "tests": {
            "synthetic": tests,
            "new_rehashed_tamper_refusals": len(regressions["refused_cases"]),
            "full_worker_verify_without_scientific_stubs": True,
            "lossless_NPZ_repack": regressions["lossless_repack_passed"],
            "evidence": [
                c.artifact(c.ROOT, directory / name)
                for name in (
                    "runtime-closure-regressions.json",
                    "copied-R-binding-regressions.json",
                    "result.json",
                )
            ],
            "static_checks": "review-validation.json; Ruff and ty, existing executables, no installs/cache writes",
        },
        "derived_anchors": {
            name: c.sha(c.ROOT / name)
            for name in (
                "proposal.json",
                "source-bindings.json",
                "metadata.json",
                "physical-coordinates.json",
                "runtime.preparation.json",
            )
        },
        "anchor_graph": "review_fixes binds sources/runtime/proposal/tests; preparation binds review_fixes; exact whole-file index binds all except itself (no cyclic self-hash)",
        "provenance": "R1 seven source copies and 23 AST selections/source-origins unchanged; all original scientific files unchanged; only preparation-owned sources/derived manifests and exclusive R1 archive written",
        "scope": {
            "real_DEV_numeric_reads": 0,
            "real_R_numeric_reads": 0,
            "raw_archive_or_central_directory_reads": 0,
            "actual_attempt_consumed": False,
            "actual_study_fits": 0,
            "freeze_execute_invocations": 0,
            "network_installs_git_ssh_child_agents": 0,
            "package_cache_mutations": 0,
            "OS_kernel_runtime_closure_claimed": False,
        },
        "unchanged": "scientific arithmetic, physical noise realization, primary/gates/support/fit budget and final N446 STOP",
        "unresolved": planning.proposal()["unresolved"]
        + [
            "Actual data/R parity, availability, original numeric equality and full-run budget remain untested; later root-only stages fail closed.",
            "No OS/kernel/firmware or in-memory generated-code closure assertion; on-disk imported executable paths are bound and checked.",
        ],
    }


def prepare():
    c.require(
        not (c.ROOT / "preparation.json").exists()
        and not (c.ROOT / "freeze.json").exists(),
        "Refuse existing preparation; preserve previous work",
    )
    before = source_bindings()
    # Protocol/source/runtime registration before own source-derived metadata.
    c.write(c.ROOT / "proposal.json", planning.proposal())
    c.write(c.ROOT / "source-bindings.json", before)
    runtime = c.runtime_identity()
    c.write(c.ROOT / "runtime.preparation.json", runtime)
    metadata = planning.metadata()
    c.write(c.ROOT / "metadata.json", metadata)
    c.write(
        c.ROOT / "physical-coordinates.json", payload.physical_registration(metadata)
    )
    tests = selftests.run()
    c.same(
        source_bindings(),
        before,
        "original frozen sources unchanged after synthetic tests",
    )
    c.same(
        c.runtime_identity(),
        runtime,
        "scientific imports/native closure stable after tiny tests",
    )
    c.write(c.ROOT / "review_fixes.json", review_fixes(tests))
    c.write(
        c.ROOT / "preparation.json",
        {
            "schema": 1,
            "status": "PREPARED_NOT_FROZEN_NOT_EXECUTED",
            "proposal_sha256": c.sha(c.ROOT / "proposal.json"),
            "metadata_sha256": c.sha(c.ROOT / "metadata.json"),
            "physical_coordinate_sha256": c.sha(c.ROOT / "physical-coordinates.json"),
            "source_bindings_sha256": c.sha(c.ROOT / "source-bindings.json"),
            "runtime_sha256": c.sha(c.ROOT / "runtime.preparation.json"),
            "review_fixes_sha256": c.sha(c.ROOT / "review_fixes.json"),
            "synthetic_tests": tests,
            "actual_DEV_reads": 0,
            "archive_central_directory_reads": 0,
            "R_numeric_payload_reads": 0,
            "actual_new_study_fits": 0,
            "earlier_entrypoints_imported_or_executed": 0,
            "freeze_or_execute_called": False,
            "unresolved": planning.proposal()["unresolved"],
        },
    )
    c.seal(c.ROOT)
    return {
        "status": "prepared",
        "synthetic_tests": tests,
        "source_tree_index_sha256": c.sha(c.ROOT / "index.json"),
        "runtime_sha256": c.sha(c.ROOT / "runtime.preparation.json"),
        "proposal_sha256": c.sha(c.ROOT / "proposal.json"),
    }


def check_preparation():
    c.check_index(c.ROOT)
    prepared = c.read(c.ROOT / "preparation.json")
    c.same(
        review_fixes(prepared["synthetic_tests"]),
        c.read(c.ROOT / "review_fixes.json"),
        "review fixes archive/source/runtime/test provenance",
    )
    c.same(
        source_bindings(),
        c.read(c.ROOT / "source-bindings.json"),
        "all original source/public metadata unchanged",
    )
    c.same(planning.proposal(), c.read(c.ROOT / "proposal.json"), "full proposal")
    c.same(
        planning.metadata(),
        c.read(c.ROOT / "metadata.json"),
        "own complete metadata/cover/fit/claim ledgers",
    )
    c.same(
        payload.physical_registration(c.read(c.ROOT / "metadata.json")),
        c.read(c.ROOT / "physical-coordinates.json"),
        "source row/channel physical coordinate registration",
    )
    c.same(
        c.runtime_identity(),
        c.read(c.ROOT / "runtime.preparation.json"),
        "fresh source/bytecode/native/launcher/flags/PATH runtime closure",
    )
    for key, name in (
        ("proposal_sha256", "proposal.json"),
        ("metadata_sha256", "metadata.json"),
        ("physical_coordinate_sha256", "physical-coordinates.json"),
        ("source_bindings_sha256", "source-bindings.json"),
        ("runtime_sha256", "runtime.preparation.json"),
        ("review_fixes_sha256", "review_fixes.json"),
    ):
        c.require(
            prepared[key] == c.sha(c.ROOT / name), "preparation crosslink: " + name
        )
    return {
        "status": "preparation_verified",
        "freeze_or_execution": False,
        "index_sha256": c.sha(c.ROOT / "index.json"),
        "runtime_sha256": prepared["runtime_sha256"],
    }


def freeze(archive, root):
    check_preparation()
    c.require(
        not root.exists()
        and not root.is_relative_to(c.ROOT)
        and not c.ROOT.is_relative_to(root),
        "Fresh separate output root required; no old artifact edits",
    )
    # This root-only stage first anchors R byte payloads; preparation never does.
    reference_root = c.ROOT.parent / "canonical-orientation-v1"
    reference = payload.reference_freeze(reference_root)
    frozen = {
        "schema": 1,
        "protocol": "full-support-canonical-v1",
        "archive": str(archive),
        "out_root": str(root),
        "reference_root": str(reference_root),
        "reference_files": reference,
        "preparation_index_sha256": c.sha(c.ROOT / "index.json"),
        "proposal_sha256": c.sha(c.ROOT / "proposal.json"),
        "metadata_sha256": c.sha(c.ROOT / "metadata.json"),
        "runtime_sha256": c.sha(c.ROOT / "runtime.preparation.json"),
        "cpu_budget_seconds": planning.CPU_SECONDS,
        "wall_budget_seconds": planning.WALL_SECONDS,
        "automatic_retry": False,
        "archive_opened": False,
        "root_review_stage": "freeze",
    }
    root.mkdir(parents=True, exist_ok=False)
    c.write(root / "freeze.json", frozen)
    return {
        "status": "frozen_no_DEV_read",
        "freeze_sha256": c.sha(root / "freeze.json"),
        "out_root": str(root),
    }


def validate_freeze(root, archive):
    check_preparation()
    frozen = c.read(root / "freeze.json")
    c.same(
        {
            key: frozen[key]
            for key in (
                "archive",
                "out_root",
                "reference_root",
                "preparation_index_sha256",
                "proposal_sha256",
                "metadata_sha256",
                "runtime_sha256",
                "cpu_budget_seconds",
                "wall_budget_seconds",
                "automatic_retry",
            )
        },
        {
            "archive": str(archive),
            "out_root": str(root),
            "reference_root": str(
                (c.ROOT.parent / "canonical-orientation-v1").resolve()
            ),
            "preparation_index_sha256": c.sha(c.ROOT / "index.json"),
            "proposal_sha256": c.sha(c.ROOT / "proposal.json"),
            "metadata_sha256": c.sha(c.ROOT / "metadata.json"),
            "runtime_sha256": c.sha(c.ROOT / "runtime.preparation.json"),
            "cpu_budget_seconds": 1800,
            "wall_budget_seconds": 3600,
            "automatic_retry": False,
        },
        "complete preaccess freeze",
    )
    payload.reference_check(Path(frozen["reference_root"]), frozen["reference_files"])
    return frozen


def main():
    stage = sys.argv[1]
    if stage == "prepare":
        return prepare()
    if stage == "check-preparation":
        return check_preparation()
    if stage == "selftest":
        c.require(
            not (c.ROOT / "index.json").exists(),
            "Do not mutate a sealed preparation; synthetic tests already indexed",
        )
        c.runtime_identity()
        return selftests.run()
    if stage in ("freeze", "execute"):
        archive, root = Path(sys.argv[3]), Path(sys.argv[5])
        if stage == "freeze":
            return freeze(archive, root)
        frozen = validate_freeze(
            root, archive
        )  # complete closure BEFORE any DEV access
        c.require(
            {p.name for p in root.iterdir()}
            == {"freeze.json", "attempt.started.json", "worker.started.json"},
            "Refuse partial/failed/extra-file attempt; no automatic retry",
        )
        worker.execute(root, frozen)
        return {
            "status": "completed_one_attempt",
            "results_sha256": c.sha(root / "results.json"),
        }
    root = Path(sys.argv[3])
    frozen = c.read(root / "freeze.json")
    archive = Path(sys.argv[5]) if len(sys.argv) == 6 else None
    validate_freeze(root, Path(frozen["archive"]))
    if archive is not None:
        c.require(
            str(archive) == frozen["archive"],
            "Full replay requires the frozen archive path",
        )
    return worker.verify(root, archive=archive)


if __name__ == "__main__":
    print(c.canonical(main()).decode(), flush=True)
