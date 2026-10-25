"""PREFREEZE systems regressions: six fixed synthetic inputs and isolated stub workers only."""

import copy
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from unittest.mock import patch

import common as c
import evidence
import fixtures
import kernel
import legacy
import numpy as np
import replay


def write_json(path, value):
    with path.open("xb") as handle:
        handle.write(c.canonical(value) + b"\n")


def six_fixtures(directory):
    chosen = []
    for n in (256, 768):
        chosen.append(
            next(
                k
                for k in fixtures.keys()
                if k["kind"] == "legacy"
                and k["family"] == legacy.FAMILIES[0]
                and k["period_samples"] == 100
                and k["phase_index"] == 0
                and k["length"] == n
            )
        )
    for n, condition, period in (
        (256, "clean", 180),
        (768, "clean", 180),
        (256, "constant_gyro_4", 160),
        (256, "near_constant_gyro_4_plus_1e-13_cos", 160),
    ):
        chosen.append(
            next(
                k
                for k in fixtures.keys()
                if k["kind"] == "stress"
                and k["period_samples"] == period
                and k["phase_index"] == 0
                and k["length"] == n
                and k["condition"] == condition
            )
        )
    inventory = [{**item, "input_index": i} for i, item in enumerate(chosen)]
    arrays = {}
    for name, (shape, dtype) in evidence.storage_specs(6, 2).items():
        value = np.empty(shape, dtype=dtype)
        value[:] = (
            np.nan
            if np.issubdtype(dtype, np.floating)
            else (-1 if np.issubdtype(dtype, np.integer) else False)
        )
        arrays[name] = value
    arrays.update(evidence.metadata_arrays(inventory))
    rows, diagnostics = [], []
    for item in inventory:
        i, n = item["input_index"], item["length"]
        window = fixtures.generate(item)
        mags = kernel.magnitudes(window)
        filtered = kernel.filter_modalities(mags)
        joint, stats, valid, finite = kernel.joint_correlations(filtered)
        old = legacy.correlations(filtered[0])
        diag, ds, dv, df, summaries = kernel.diagnostic(mags, filtered)
        arrays["windows"][i, :, :n] = window
        arrays["magnitudes"][i, :, :n] = mags
        arrays["filtered"][i, :, :n] = filtered
        arrays["extraction_correlations"][i] = np.vstack((old, joint))
        arrays["joint_guard_stats"][i] = stats
        arrays["joint_guard_valid"][i] = valid
        arrays["joint_guard_finite"][i] = finite
        arrays["diagnostic_correlations"][i] = diag
        arrays["diagnostic_guard_stats"][i] = ds
        arrays["diagnostic_guard_valid"][i] = dv
        arrays["diagnostic_guard_finite"][i] = df
        whole = kernel.whole_window(mags, filtered)
        for ai in range(3):
            rs = old if ai < 2 else joint[2]
            base, feature = legacy.extract(
                mags, filtered[0], rs, legacy.RULES[0 if ai == 0 else 1]
            )
            row = kernel.record(item, ai, base, feature, rs, whole, joint)
            rowid = row["row_id"]
            rows.append(row)
            if feature is not None:
                arrays["features"][rowid] = feature
                arrays["feature_valid"][rowid] = True
            if base["pair_bounds_inclusive"] is not None:
                arrays["selected_pairs"][rowid] = base["pair_bounds_inclusive"]
            if base["selected_lag"] is not None:
                arrays["selected_lags"][rowid] = base["selected_lag"]
        diagnostics.append(evidence.diagnostic_record(item, summaries))
    for name in (
        "input_present",
        "record_present",
        "joint_present",
        "diagnostic_present",
        "baseline_present",
    ):
        arrays[name][:] = True
    arrays["extraction_correlation_finite"][:] = np.isfinite(
        arrays["extraction_correlations"]
    )
    arrays["diagnostic_finite"][:] = np.isfinite(arrays["diagnostic_correlations"])
    schema = evidence.array_schema(inventory)
    with (directory / "six_fixed_evidence.npz").open("xb") as handle:
        np.savez(handle, allow_pickle=False, **arrays)
    for name, value in (
        ("inventory", inventory),
        ("records", rows),
        ("diagnostics", diagnostics),
        ("array_schema", schema),
    ):
        write_json(directory / (name + ".json"), value)
    return rows, arrays, inventory, diagnostics, schema


def run():
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    cases = c.ROOT / "regression_cases"
    cases.mkdir(exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="review-r1-", dir=cases))
    tests = []

    def log(name, **data):
        tests.append({"name": name, "pass": True, **data})

    def reject(name, function, contains=None):
        try:
            function()
        except (RuntimeError, ValueError, KeyError, FileNotFoundError) as exc:
            if contains is not None:
                c.require(contains in str(exc), name + " wrong rejection: " + str(exc))
            log(name, expected_rejection=type(exc).__name__, error=str(exc))
        else:
            raise RuntimeError("Tampering accepted: " + name)

    def worker(
        name,
        *,
        shell,
        args,
        body=None,
        limits=None,
        outer_deadline=False,
        fail_initial_import=False,
        fail_worker_start=False,
        shim_wait=False,
        fail_launcher=False,
    ):
        job = directory / name
        job.mkdir()
        source = (
            (c.ROOT / "study.py")
            .read_text()
            .split("from collections import Counter", 1)[0]
        )
        if fail_initial_import:
            source = source.replace(
                "import json\n",
                "raise ImportError('forced first stdlib import failure')\n",
                1,
            )
        if fail_worker_start:
            source = source.replace(
                '    with open(_reservation_root + "/worker.started.json", "x") as _receipt:',
                "    raise RuntimeError('forced worker-start failure')\n"
                '    with open(_reservation_root + "/worker.started.json", "x") as _receipt:',
                1,
            )
        if limits:
            source = source.replace("(599, 600)", limits[0]).replace(
                "900 - _launch_elapsed", limits[1] + " - _launch_elapsed"
            )
        if body is None:
            body = "with (_ROOT / 'dispatched.json').open('x') as f: json.dump({'args':sys.argv[1:]}, f)\n"
        (job / "study.py").write_text(source + body)
        launcher = (c.ROOT / "run.sh").read_text()
        shim = job / "uv-stub"
        shim.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\nshift 5\nexec "
            + shlex.quote(sys.executable)
            + ' "$@"\n'
        )
        if shim_wait:
            shim.write_text(shim.read_text().replace("\nexec ", "\n") + 'exit "$?"\n')
        if fail_launcher:
            shim.write_text("#!/usr/bin/env bash\nexit 47\n")
        shim.chmod(0o755)
        launcher = launcher.replace(
            "uv_bin=/home/sherlock/.local/bin/uv", "uv_bin=" + shlex.quote(str(shim))
        )
        if outer_deadline:
            launcher = launcher.replace("--signal=KILL 900", "--signal=KILL 0.2")
        (job / "run.sh").write_text(launcher)
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("JOINT_RUN_RESERVED", "JOINT_RUN_LAUNCH_EPOCH")
        }
        command = (
            ["/usr/bin/bash", str(job / "run.sh")]
            if shell
            else [sys.executable, "-B", str(job / "study.py")]
        )
        before = time.perf_counter()
        completed = subprocess.run(
            [*command, *args],
            env=env,
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        result = {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "wall_seconds": time.perf_counter() - before,
            "shell": shell,
            "args": args,
            "mock": "source preamble plus stub worker, launcher shim; no study execute function",
        }
        write_json(job / "result.json", result)
        return job, result

    def replay_token(job, name):
        """Repair only the failed isolated stub; the consumed anchors stay byte exact."""
        marker = (job / "attempt.started.json").read_bytes()
        attempt = c.decode_json(marker)
        receipt = job / "worker.started.json"
        before_receipt = receipt.read_bytes() if receipt.exists() else None
        shutil.copy2(job / "study.py", job / "failed-study.source.txt")
        preamble = (
            (c.ROOT / "study.py")
            .read_text()
            .split("from collections import Counter", 1)[0]
        )
        (job / "study.py").write_text(preamble + "print('STUB_DISPATCH', flush=True)\n")
        token = str(attempt["launcher_pid"]) + ":" + attempt["launcher_starttime_ticks"]
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                "-X",
                "joint_run_reserved=" + token,
                str(job / "study.py"),
                "execute",
            ],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        write_json(
            job / "old_token_replay.json",
            {
                "token": token,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "marker_sha256_before": c.fingerprint(attempt),
                "receipt_existed": before_receipt is not None,
            },
        )
        c.require(
            completed.returncode != 0
            and "STUB_DISPATCH" not in completed.stdout
            and "live launcher ancestor/starttime" in completed.stderr
            and (job / "attempt.started.json").read_bytes() == marker
            and (receipt.read_bytes() if receipt.exists() else None) == before_receipt,
            "Old shell token redispatched or mutated anchors after " + name,
        )
        log(
            "old_token_refused_after_" + name,
            stub_only=True,
            receipt_existed=before_receipt is not None,
        )

    try:
        for shell in (False, True):
            prefix = "shell" if shell else "python"
            for index, args in enumerate(
                (
                    [],
                    ["--replay-correlations", "execute"],
                    ["execute", "--replay-correlations"],
                    ["prepare", "--replay-correlations"],
                    ["verify", "execute"],
                    ["verify", "--replay-correlations", "extra"],
                    ["selftest", "--replay-correlations"],
                    ["unknown"],
                    ["--", "execute"],
                )
            ):
                job, result = worker(
                    f"{prefix}-invalid-{index}",
                    shell=shell,
                    args=args,
                    body="raise RuntimeError('STARTUP MUST NOT BE REACHED')\n",
                )
                c.require(
                    result["returncode"] == 2
                    and "STARTUP MUST NOT" not in result["stderr"]
                    and not (job / "attempt.started.json").exists(),
                    "Invalid grammar reached startup/reservation",
                )
                log(f"{prefix}_invalid_grammar_{index}", argv=args)
            for index, args in enumerate(
                (
                    ["prepare"],
                    ["check-preparation"],
                    ["freeze"],
                    ["selftest"],
                    ["verify"],
                    ["verify", "--replay-correlations"],
                    ["execute"],
                )
            ):
                job, result = worker(f"{prefix}-valid-{index}", shell=shell, args=args)
                c.require(
                    result["returncode"] == 0
                    and c.decode_json((job / "dispatched.json").read_bytes())["args"]
                    == args,
                    "Valid grammar not dispatched",
                )
                c.require(
                    (job / "attempt.started.json").exists() == (args == ["execute"]),
                    "Reservation grammar mismatch",
                )
                log(f"{prefix}_valid_grammar_{index}", argv=args, stub_only=True)
            for index, body in enumerate(
                (
                    "raise ImportError('forced dependency import failure')\n",
                    "raise RuntimeError('forced source validation failure')\n",
                    "raise RuntimeError('forced runtime validation failure')\n",
                )
            ):
                job, result = worker(
                    f"{prefix}-failure-{index}",
                    shell=shell,
                    args=["execute"],
                    body=body,
                )
                c.require(
                    result["returncode"] != 0
                    and (job / "attempt.started.json").is_file(),
                    "Failed startup did not consume reservation",
                )
                before = (job / "attempt.started.json").read_bytes()
                command = (
                    ["/usr/bin/bash", str(job / "run.sh")]
                    if shell
                    else [sys.executable, "-B", str(job / "study.py")]
                )
                retry = subprocess.run(
                    [*command, "execute"],
                    capture_output=True,
                    text=True,
                    timeout=8,
                    check=False,
                )
                write_json(
                    job / "retry_result.json",
                    {"returncode": retry.returncode, "stderr": retry.stderr},
                )
                c.require(
                    retry.returncode != 0
                    and (job / "attempt.started.json").read_bytes() == before,
                    "Retry changed marker",
                )
                log(
                    f"{prefix}_failure_reserves_and_retry_refused_{index}",
                    stub_only=True,
                )
                if shell:
                    replay_token(
                        job, ("dependency", "source", "runtime")[index] + "_failure"
                    )
            job, result = worker(
                f"{prefix}-first-import-failure",
                shell=shell,
                args=["execute"],
                fail_initial_import=True,
            )
            c.require(
                result["returncode"] != 0 and (job / "attempt.started.json").is_file(),
                "First stdlib import failure was not reserved",
            )
            log(f"{prefix}_reservation_precedes_first_stdlib_import", stub_only=True)
            c.require(
                (job / "worker.started.json").is_file(),
                "Receipt must precede first stdlib import",
            )
            log(f"{prefix}_worker_receipt_precedes_first_stdlib_import", stub_only=True)
            if shell:
                replay_token(job, "first_stdlib_import_failure")
            for index, flag in enumerate(("-h", "--help")):
                job, result = worker(
                    f"{prefix}-help-{index}",
                    shell=shell,
                    args=[flag],
                    body="raise RuntimeError('STARTUP MUST NOT BE REACHED')\n",
                )
                c.require(
                    result["returncode"] == 0
                    and "usage:" in result["stdout"]
                    and not (job / "attempt.started.json").exists(),
                    "Help dispatched startup",
                )
                log(f"{prefix}_help_before_startup_{index}")
        job, result = worker(
            "shell-source-syntax-failure",
            shell=True,
            args=["execute"],
            body="this is invalid python syntax !\n",
        )
        c.require(
            result["returncode"] != 0
            and "SyntaxError" in result["stderr"]
            and (job / "attempt.started.json").exists(),
            "Shell source syntax failure not reserved",
        )
        log("shell_reserves_before_python_source_parse", stub_only=True)
        c.require(
            not (job / "worker.started.json").exists(), "Syntax failure ran prefix"
        )
        replay_token(job, "main_source_syntax_failure")
        for name, options in (
            ("worker_start_failure", {"fail_worker_start": True}),
            ("launcher_startup_failure", {"fail_launcher": True}),
        ):
            job, result = worker(name, shell=True, args=["execute"], **options)
            c.require(
                result["returncode"] != 0
                and not (job / "worker.started.json").exists(),
                "Pre-receipt failure not injected",
            )
            replay_token(job, name)
        job, result = worker(
            "shell-uv-intermediary", shell=True, args=["execute"], shim_wait=True
        )
        c.require(
            result["returncode"] == 0, "Live launcher not found through uv intermediary"
        )
        evidence.validate_attempt(
            job
        )  # Launcher is already dead; persisted bindings remain valid.
        log(
            "shell_live_ancestor_through_intermediary_once_and_postexit_validation",
            stub_only=True,
        )
        replay_token(job, "successful_shell_worker")

        # Parent process is a genuine live ancestor; use the exact same token twice.
        preamble = (
            (c.ROOT / "study.py")
            .read_text()
            .split("from collections import Counter", 1)[0]
        )

        def proc(pid):
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
            return fields[19]

        def handoff_case(name, pid, ticks, *, token=None, seed=True):
            job = directory / name
            job.mkdir()
            (job / "study.py").write_text(
                preamble + "print('STUB_DISPATCH', flush=True)\n"
            )
            if seed:
                write_json(
                    job / "attempt.started.json",
                    {
                        "automatic_retry": False,
                        "cpu_budget_seconds": 600,
                        "launcher_pid": pid,
                        "launcher_starttime_ticks": ticks,
                        "route": "shell",
                        "schema": 2,
                        "status": "exclusive_attempt_consumed",
                        "wall_budget_seconds": 900,
                    },
                )
            command = [
                sys.executable,
                "-B",
                "-X",
                "joint_run_reserved="
                + (token if token is not None else f"{pid}:{ticks}"),
                str(job / "study.py"),
                "execute",
            ]
            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=8, check=False
            )
            write_json(
                job / "result.json",
                {
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                    "command": command,
                },
            )
            return job, completed, command

        pid, ticks = os.getpid(), proc("self")
        job, completed, command = handoff_case("live-ancestor-once", pid, ticks)
        c.require(
            completed.returncode == 0 and completed.stdout == "STUB_DISPATCH\n",
            "Valid ancestor handoff refused",
        )
        anchors = {
            n: (job / n).read_bytes()
            for n in ("attempt.started.json", "worker.started.json")
        }
        again = subprocess.run(
            command, capture_output=True, text=True, timeout=8, check=False
        )
        write_json(
            job / "second_worker_result.json",
            {
                "returncode": again.returncode,
                "stdout": again.stdout,
                "stderr": again.stderr,
            },
        )
        c.require(
            again.returncode != 0
            and "STUB_DISPATCH" not in again.stdout
            and "FileExistsError" in again.stderr
            and all((job / n).read_bytes() == b for n, b in anchors.items()),
            "Live ancestor handoff reused",
        )
        log("live_ancestor_token_consumed_by_one_worker_receipt", stub_only=True)
        job, completed, _ = handoff_case(
            "token-attempt-mismatch", pid, ticks, token=f"{pid}:{int(ticks) + 1}"
        )
        c.require(
            completed.returncode != 0
            and "marker missing or mismatched" in completed.stderr
            and "STUB_DISPATCH" not in completed.stdout
            and not (job / "worker.started.json").exists(),
            "Token accepted for a different persisted attempt",
        )
        log("token_bound_to_exact_attempt_marker", stub_only=True)
        for name, bad_pid, bad_ticks in (
            ("pid-reuse-starttime", pid, str(int(ticks) + 1)),
            ("stale-pid", 2147483647, "1"),
        ):
            job, completed, _ = handoff_case(name, bad_pid, bad_ticks)
            c.require(
                completed.returncode != 0
                and "STUB_DISPATCH" not in completed.stdout
                and not (job / "worker.started.json").exists(),
                "Stale/reused PID accepted",
            )
            log(name + "_refused", stub_only=True)
        foreign = subprocess.Popen(
            [sys.executable, "-B", "-c", "import time; time.sleep(8)"]
        )
        try:
            job, completed, _ = handoff_case(
                "foreign-live-pid", foreign.pid, proc(foreign.pid)
            )
            c.require(
                completed.returncode != 0
                and "STUB_DISPATCH" not in completed.stdout
                and not (job / "worker.started.json").exists(),
                "Foreign live PID accepted",
            )
            log("foreign_live_nonancestor_pid_refused", stub_only=True)
        finally:
            foreign.terminate()
            foreign.wait(timeout=2)
        for index, token in enumerate(
            ("1", "0:1", "1:0", "01:1", "1:01", "1:2:3", "-1:2", "١:2", ":", "")
        ):
            job, completed, _ = handoff_case(
                f"bad-token-{index}", pid, ticks, token=token, seed=False
            )
            c.require(
                completed.returncode != 0
                and "Invalid shell reservation token" in completed.stderr
                and not (job / "attempt.started.json").exists()
                and not (job / "worker.started.json").exists(),
                "Malformed handoff token executed",
            )
            log(f"malformed_handoff_token_{index}_refused", stub_only=True)
        closure_job = directory / "four-stage-import-closure"
        closure_job.mkdir()
        for path in c.ROOT.glob("*.py"):
            if path.name != "study.py":
                shutil.copy2(path, closure_job / path.name)
        source = (c.ROOT / "study.py").read_text().split("PREPARATION_INPUTS =", 1)[0]
        (closure_job / "study.py").write_text(
            source
            + "with (_ROOT / (sys.argv[1] + '.runtime.json')).open('x') as f: json.dump(c.runtime_identity(), f, sort_keys=True)\n"
        )
        closures = {}
        for stage in ("prepare", "freeze", "execute", "verify"):
            completed = subprocess.run(
                [sys.executable, "-B", str(closure_job / "study.py"), stage],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            write_json(
                closure_job / (stage + ".result.json"),
                {
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                    "scope": "actual import preamble plus runtime dump stub; no study stage function",
                },
            )
            c.require(completed.returncode == 0, "Stage import stub failed " + stage)
            closures[stage] = c.decode_json(
                (closure_job / (stage + ".runtime.json")).read_bytes()
            )
        c.require(
            all(c.typed(v) == c.typed(closures["prepare"]) for v in closures.values()),
            "Stage runtime/import closures differ",
        )
        log(
            "four_stage_import_runtime_closures_match_without_stage_execution",
            stub_only=True,
        )
        for name, shell, limits, outer in (
            ("wall_deadline", False, ("(599, 600)", "0.2"), False),
            ("cpu_deadline", False, ("(1, 2)", "4"), False),
            ("shell_outer_deadline", True, None, True),
        ):
            job, result = worker(
                name,
                shell=shell,
                args=["execute"],
                body="while True: pass\n",
                limits=limits,
                outer_deadline=outer,
            )
            c.require(
                result["returncode"] < 0 and (job / "attempt.started.json").exists(),
                "Stub deadline failed",
            )
            c.require(
                result["wall_seconds"] < (3 if name == "cpu_deadline" else 2),
                "Stub deadline not bounded",
            )
            log(name, **result, actual_job_marker_consumed=False)
        rows, arrays, inventory, diagnostics, schema = six_fixtures(directory)

        def replay_saved(full=False, ar=None, rr=None, dd=None, ss=None, ii=None):
            with (
                ExitStack() as stack,
                patch.object(
                    fixtures,
                    "generate",
                    side_effect=AssertionError("replay generated input"),
                ),
                patch.object(
                    kernel,
                    "filter_modalities",
                    side_effect=AssertionError("replay filtered input"),
                ),
            ):
                if not full:
                    for module, name in (
                        (kernel, "joint_correlations"),
                        (kernel, "diagnostic"),
                        (legacy, "correlations"),
                    ):
                        stack.enter_context(
                            patch.object(
                                module,
                                name,
                                side_effect=AssertionError(
                                    "default replay recomputed correlations"
                                ),
                            )
                        )
                return replay.verify_tiny(
                    rows if rr is None else rr,
                    arrays if ar is None else ar,
                    inventory if ii is None else ii,
                    diagnostics if dd is None else dd,
                    schema=schema if ss is None else ss,
                    full=full,
                )

        for full in (False, True):
            c.require(
                replay_saved(full)["status"] == "pass", "Unaltered six-input replay"
            )
            log(
                "six_fixed_saved_replay_full_" + str(full),
                inputs=6,
                records=18,
                primary_panel_inputs=0,
            )
        reject(
            "production_replay_refuses_tiny_counts",
            lambda: replay.verify(rows, arrays, inventory, diagnostics, schema=schema),
            "inventory",
        )
        mutations = (
            (
                "root_probe_raw_prefix_plus123",
                "windows",
                lambda v: v.__setitem__((0, 0, 0), v[0, 0, 0] + 123),
                "norm bits",
            ),
            (
                "raw_prefix_nonfinite",
                "windows",
                lambda v: v.__setitem__((0, 0, 0), np.inf),
                "Nonfinite raw",
            ),
            (
                "window_padding",
                "windows",
                lambda v: v.__setitem__((0, 0, 256), 0),
                "padding",
            ),
            (
                "magnitude_padding",
                "magnitudes",
                lambda v: v.__setitem__((0, 0, 256), 0),
                "padding",
            ),
            (
                "filtered_padding",
                "filtered",
                lambda v: v.__setitem__((0, 0, 256), 0),
                "padding",
            ),
            (
                "magnitude_binding",
                "magnitudes",
                lambda v: v.__setitem__((0, 0, 0), v[0, 0, 0] + 1),
                "norm bits",
            ),
            ("lag_axis", "lags", lambda v: v.__setitem__(0, 79), "mapping/axis"),
            (
                "diagnostic_lag_axis",
                "diagnostic_lags",
                lambda v: v.__setitem__(0, 39),
                "mapping/axis",
            ),
            (
                "anchor256",
                "diagnostic_anchors_256",
                lambda v: v.__setitem__(0, 1),
                "mapping/axis",
            ),
            (
                "anchor768",
                "diagnostic_anchors_768",
                lambda v: v.__setitem__(0, 1),
                "mapping/axis",
            ),
            (
                "row_mapping",
                "row_mapping",
                lambda v: v.__setitem__((0, 1), 1),
                "mapping/axis",
            ),
            ("arms", "arms", lambda v: v.__setitem__(0, "bad"), "mapping/axis"),
            ("key_mapping", "keys", lambda v: v.__setitem__(0, "bad"), "mapping/axis"),
            (
                "length_mapping",
                "lengths",
                lambda v: v.__setitem__(0, 768),
                "mapping/axis",
            ),
            (
                "baseline_mask",
                "baseline_present",
                lambda v: v.__setitem__((0, 0), False),
                "present mask",
            ),
            (
                "joint_guard_valid",
                "joint_guard_valid",
                lambda v: v.__setitem__((0, 0, 0, 0), False),
                "threshold/mask",
            ),
            (
                "joint_guard_finite",
                "joint_guard_finite",
                lambda v: v.__setitem__((0, 0, 0, 0), False),
                "finite masks",
            ),
            (
                "joint_guard_stats",
                "joint_guard_stats",
                lambda v: v.__setitem__((0, 0, 0, 0, 1), 0),
                "threshold/mask",
            ),
            (
                "diagnostic_guard_valid",
                "diagnostic_guard_valid",
                lambda v: v.__setitem__((0, 0, 0, 0), False),
                "threshold/mask",
            ),
            (
                "diagnostic_guard_finite",
                "diagnostic_guard_finite",
                lambda v: v.__setitem__((0, 0, 0, 0), False),
                "finite masks",
            ),
            (
                "diagnostic_guard_stats",
                "diagnostic_guard_stats",
                lambda v: v.__setitem__((0, 0, 0, 0, 1), 0),
                "threshold/mask",
            ),
        )
        for name, key, mutate, message in mutations:
            changed = {**arrays, key: arrays[key].copy()}
            mutate(changed[key])
            reject(name, lambda: replay_saved(ar=changed), message)
            if name == "root_probe_raw_prefix_plus123":
                reject(name + "_FULL", lambda: replay_saved(True, ar=changed), message)
        for label, changed in (
            ("unknown_array", {**arrays, "unknown": np.zeros(1)}),
            ("missing_array", {k: v for k, v in arrays.items() if k != "arms"}),
            ("wrong_dtype", {**arrays, "lags": arrays["lags"].astype(np.int32)}),
            ("wrong_shape", {**arrays, "features": arrays["features"][:, :399]}),
        ):
            reject(label, lambda: replay_saved(ar=changed))
        for field in (
            "anchor_count",
            "anchors_first",
            "anchors_last",
            "lags_first",
            "lags_last",
            "length",
            "input_index",
            "key",
            "extraction_effect",
            "channels",
            "unexpected",
        ):
            changed = copy.deepcopy(diagnostics)
            changed[0][field] = "tampered"
            reject(
                "diagnostic_metadata_" + field,
                lambda: replay_saved(dd=changed),
                "Diagnostic complete",
            )
        for field in rows[0]:
            changed = copy.deepcopy(rows)
            changed[0][field] = "tampered"
            reject("record_metadata_" + field, lambda: replay_saved(rr=changed))
        changed = copy.deepcopy(rows)
        changed[0]["unexpected"] = True
        reject(
            "record_unknown_metadata", lambda: replay_saved(rr=changed), "record replay"
        )
        changed = copy.deepcopy(schema)
        changed["row_mapping"] = "tampered"
        reject("schema_mapping", lambda: replay_saved(ss=changed), "schema")
        reject(
            "missing_schema",
            lambda: replay.verify_tiny(
                rows, arrays, inventory, diagnostics, schema=None
            ),
            "schema",
        )

        # Index component fixtures exercise the exact production inventory checker, not study.verify.
        index_base = directory / "index-base"
        index_base.mkdir()
        write_json(index_base / "array_schema.json", schema)
        (index_base / "nested").mkdir()
        (index_base / "nested/artifact_index.json").write_text(
            "nested index is covered\n"
        )
        original_index = evidence.index_value(index_base)
        write_json(index_base / "artifact_index.json", original_index)
        evidence.validate_index(index_base)
        c.require(
            "nested/artifact_index.json" in original_index["files_sha256"],
            "Nested basename exclusion",
        )
        log("index_exact_closure_includes_nested_index")
        for name in (
            "empty",
            "omit_schema",
            "unknown_entry",
            "missing_file",
            "extra_file",
            "nested_tamper",
            "schema_tamper",
            "whitespace",
            "duplicate_key",
        ):
            target = directory / ("index-" + name)
            shutil.copytree(index_base, target)
            altered = copy.deepcopy(original_index)
            if name == "empty":
                altered["files_sha256"] = {}
            if name == "omit_schema":
                del altered["files_sha256"]["array_schema.json"]
            if name == "unknown_entry":
                altered["files_sha256"]["../unknown"] = "0" * 64
            if name == "missing_file":
                (target / "array_schema.json").unlink()
            if name == "extra_file":
                (target / "extra").write_text("x")
            if name == "nested_tamper":
                (target / "nested/artifact_index.json").write_text("changed")
            if name == "schema_tamper":
                altered["excludes_self"] = False
            payload = c.canonical(altered) + b"\n"
            if name == "whitespace":
                payload += b" "
            if name == "duplicate_key":
                payload = b'{"files_sha256":{},"files_sha256":{}}\n'
            (target / "artifact_index.json").write_bytes(payload)
            reject("index_" + name, lambda: evidence.validate_index(target))

        # Synthetic report binding fixture: no freeze function or measured study is invoked.
        bindings = directory / "report-binding-components"
        bindings.mkdir()
        (bindings / "arrays").mkdir()
        for name in evidence.BINDING_FILES:
            write_json(bindings / name, {"synthetic_component_fixture": name})
        for name in ("attempt.started.json", "worker.started.json"):
            (bindings / name).write_bytes(
                (directory / "live-ancestor-once" / name).read_bytes()
            )
        source_map = {"toy.py": "1" * 64}
        runtime = {"synthetic_runtime": True}
        frozen = {
            "manifest_sha256": "2" * 64,
            "runtime": runtime,
            "runtime_sha256": c.fingerprint(runtime),
            "source_files_sha256": source_map,
        }
        (bindings / "runtime.preparation.json").write_bytes(
            c.canonical(runtime) + b"\n"
        )
        (bindings / "sources.json").write_bytes(
            c.canonical({"files": source_map}) + b"\n"
        )
        (bindings / "saved_evidence_audit.json").write_bytes(
            c.canonical(evidence.audit_metadata(False, 6)) + b"\n"
        )
        for name in ("records.jsonl", "diagnostics.jsonl", "evidence.npz"):
            (bindings / name).write_bytes(b"synthetic component bytes\n")
        for name in arrays:
            (bindings / "arrays" / (name + ".npy")).write_bytes(
                b"synthetic array binding bytes\n"
            )
        with patch.object(c, "ROOT", bindings):
            report = {
                "attempt_binding": evidence.validate_attempt(bindings),
                "scientific_summary_sha256": c.fingerprint(c.load_json("summary.json")),
                "manifest_sha256": frozen["manifest_sha256"],
                "runtime_sha256": frozen["runtime_sha256"],
                "source_files_sha256": source_map,
                "array_files_sha256": {
                    name + ".npy": c.sha(bindings / "arrays" / (name + ".npy"))
                    for name in arrays
                },
                "records_sha256": c.sha(bindings / "records.jsonl"),
                "diagnostics_sha256": c.sha(bindings / "diagnostics.jsonl"),
                "evidence_sha256": c.sha(bindings / "evidence.npz"),
                "binding_files_sha256": evidence.binding_hashes(bindings),
                "baseline_gate": c.load_json("baseline_gate.json"),
                "preparation": c.load_json("preparation.json"),
                "saved_evidence_verification": c.load_json("saved_evidence_audit.json"),
            }
            evidence.validate_report_bindings(report, frozen, arrays)
            log("report_component_bindings_unaltered", component_fixture_only=True)
            original_attempt = evidence.validate_attempt(bindings)
            for index, (filename, key, value, synchronize) in enumerate(
                (
                    ("attempt.started.json", "schema", True, True),
                    ("attempt.started.json", "cpu_budget_seconds", 600.0, True),
                    ("attempt.started.json", "wall_budget_seconds", 901, True),
                    ("attempt.started.json", "automatic_retry", 0, True),
                    ("attempt.started.json", "route", "unknown", True),
                    ("attempt.started.json", "launcher_pid", True, True),
                    ("attempt.started.json", "launcher_starttime_ticks", "01", True),
                    ("attempt.started.json", "extra", "field", True),
                    ("worker.started.json", "schema", True, False),
                    ("worker.started.json", "status", "other", False),
                    ("worker.started.json", "worker_pid", True, False),
                    ("worker.started.json", "worker_starttime_ticks", 1, False),
                    ("worker.started.json", "worker_starttime_ticks", "١", False),
                    ("worker.started.json", "attempt", {}, False),
                    ("worker.started.json", "extra", "field", False),
                    (
                        "worker.started.json",
                        "worker_pid",
                        original_attempt["attempt.started.json"]["launcher_pid"],
                        False,
                    ),
                )
            ):
                target = directory / f"attempt-schema-{index}"
                target.mkdir()
                changed = copy.deepcopy(original_attempt)
                changed[filename][key] = value
                if synchronize:
                    changed["worker.started.json"]["attempt"] = changed[
                        "attempt.started.json"
                    ]
                for name, item in changed.items():
                    write_json(target / name, item)
                reject(
                    f"attempt_schema_{index}", lambda: evidence.validate_attempt(target)
                )
                write_json(target / "artifact_index.json", evidence.index_value(target))
                reject(
                    f"rehashed_index_attempt_schema_{index}",
                    lambda: evidence.validate_index(target),
                )
            for filename in original_attempt:
                target = directory / ("missing-" + filename)
                target.mkdir()
                for name, item in original_attempt.items():
                    if name != filename:
                        write_json(target / name, item)
                write_json(target / "artifact_index.json", evidence.index_value(target))
                reject("missing_" + filename, lambda: evidence.validate_index(target))
            direct = directory / "python-valid-6"
            evidence.validate_attempt(direct)
            log("persisted_direct_attempt_identity_validates_without_live_worker")
            direct_value = c.decode_json((direct / "worker.started.json").read_bytes())
            direct_value["worker_pid"] += 1
            target = directory / "direct-worker-mismatch"
            target.mkdir()
            shutil.copy2(
                direct / "attempt.started.json", target / "attempt.started.json"
            )
            write_json(target / "worker.started.json", direct_value)
            reject(
                "persisted_direct_worker_identity_mismatch",
                lambda: evidence.validate_attempt(target),
            )
            for name in report:
                changed = {**report, name: {}}
                reject(
                    "report_binding_" + name,
                    lambda: evidence.validate_report_bindings(changed, frozen, arrays),
                )
            target = directory / "reindexed-valid-receipt-stale-report"
            shutil.copytree(bindings, target)
            changed = c.decode_json((target / "worker.started.json").read_bytes())
            changed["worker_starttime_ticks"] = str(
                int(changed["worker_starttime_ticks"]) + 1
            )
            (target / "worker.started.json").write_bytes(c.canonical(changed) + b"\n")
            write_json(target / "artifact_index.json", evidence.index_value(target))
            evidence.validate_index(target)
            with patch.object(c, "ROOT", target):
                reject(
                    "reindexed_valid_receipt_stale_report",
                    lambda: evidence.validate_report_bindings(report, frozen, arrays),
                    "attempt_binding",
                )
            # Rehashing the mutable outer index cannot legitimize a stale report digest.
            (bindings / "records.jsonl").write_bytes(b"changed synthetic bytes\n")
            write_json(bindings / "artifact_index.json", evidence.index_value(bindings))
            evidence.validate_index(bindings)
            reject(
                "rehashed_index_stale_report_digest",
                lambda: evidence.validate_report_bindings(report, frozen, arrays),
                "records_sha256",
            )
        runtime = c.runtime_identity()
        again = c.runtime_identity()
        c.validate_runtime(runtime, again)
        for path in (getattr(np, "__cached__", None), sys.modules["json"].__cached__):
            if not isinstance(path, str):
                raise RuntimeError("Expected existing cache path missing")
            c.require(
                Path(path).is_file(),
                "Expected existing cache absent",
            )
            c.require(
                str(Path(path).resolve())
                in {m["resolved"] for m in runtime["file_metadata"].values()},
                "Consumed cache not hashed",
            )
        c.require(
            all(
                name in runtime["files_sha256"]
                for name in (
                    "shebang_env",
                    "shebang_bash",
                    "uv_launcher",
                    "python",
                    "wall_timeout_launcher",
                )
            ),
            "Launcher closure incomplete",
        )
        log(
            "live_runtime_repeat_identity_and_consumed_cache_coverage",
            pyc_entries=sum(
                ".pyc" in m["resolved"] for m in runtime["file_metadata"].values()
            ),
        )
        for name in ("environment", "files_sha256", "file_metadata", "versions"):
            changed = {**runtime, name: {}}
            reject(
                "runtime_identity_" + name,
                lambda: c.validate_runtime(changed, runtime),
                "Runtime changed",
            )
        import selftest

        with patch.object(
            selftest.regression,
            "run",
            side_effect=AssertionError("measurement called systems subprocess tests"),
        ):
            # Exercise the actual default path using only the unchanged 22 tiny/fixed checks.
            measurement_checks = selftest.run()
        c.require(
            measurement_checks["systems_regressions"] is None
            and len(measurement_checks["new_checks"]) == 22,
            "Measurement fixed checks include systems subprocesses",
        )
        log(
            "measurement_fixed_checks_exclude_systems_subprocesses",
            fixed_checks_only=True,
        )
        result = {
            "status": "pass",
            "scope": "six fixed non-primary synthetic inputs (18 extraction records), isolated CLI/deadline stub workers and structural artifact component fixtures only; production study.verify/fullgrid/freeze/execute NEVER invoked",
            "tests": tests,
        }
    except BaseException as exc:
        result = {
            "status": "failed",
            "tests": tests,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(directory / "result.json", result)
        raise
    result.update(
        cpu_seconds=time.process_time() - start_cpu,
        wall_seconds=time.perf_counter() - start_wall,
        history_directory=directory.relative_to(c.ROOT).as_posix(),
        subprocess_cpu_accounting="parent process CPU excludes isolated stub subprocess CPU; wall includes child waits",
    )
    write_json(directory / "result.json", result)
    return result


if __name__ == "__main__":
    result = run()
    print(
        json.dumps(
            {
                "status": result["status"],
                "tests": len(result["tests"]),
                "history_directory": result["history_directory"],
            }
        )
    )
