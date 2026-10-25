"""R3 single-use reservation, adapted; only sys imported before worker receipt."""

import sys


def process_identity(pid):
    with open("/proc/" + str(pid) + "/stat") as handle:
        stat = handle.read(8192)
    head, tail = stat.rsplit(") ", 1)
    fields = tail.split()
    return int(head.split(" ", 1)[0]), int(fields[1]), fields[19], fields[0]


def attempt_text(route, pid, starttime):
    return (
        '{"automatic_retry":false,"cpu_budget_seconds":1800,"launcher_pid":'
        + str(pid)
        + ',"launcher_starttime_ticks":"'
        + starttime
        + '","route":"'
        + route
        + '","schema":2,"status":"exclusive_attempt_consumed","wall_budget_seconds":3600}\n'
    )


def live_ancestor(pid, starttime, worker):
    cursor, seen = worker[1], {worker[0]}
    for _ in range(128):
        if cursor <= 0 or cursor in seen:
            break
        seen.add(cursor)
        current = process_identity(cursor)
        if current[3] in ("Z", "X", "x"):
            break
        if current[0] == pid:
            if current[2] == starttime:
                return
            break
        cursor = current[1]
    raise RuntimeError(
        "Reservation requires same live launcher ancestor/PID:startticks"
    )


def grammar(argv):
    if argv in (["prepare"], ["check-preparation"], ["selftest"]):
        return
    if (
        len(argv) == 5
        and argv[0] in ("freeze", "execute")
        and argv[1] == "--archive"
        and argv[3] == "--out-root"
    ):
        paths = (argv[2], argv[4])
    elif (
        len(argv) in (3, 5)
        and argv[:2] == ["verify", "--out-root"]
        and (len(argv) == 3 or argv[3] == "--archive")
    ):
        paths = (argv[2],) if len(argv) == 3 else (argv[2], argv[4])
    else:
        raise RuntimeError(
            "Stage first; prepare/check-preparation/selftest; freeze/execute --archive ABS --out-root ABS; verify --out-root ABS [--archive ABS]"
        )
    if not all(
        p.startswith("/")
        and "//" not in p
        and all(x not in (".", "..") for x in p.split("/"))
        for p in paths
    ):
        raise RuntimeError("Absolute normalized paths required")


def bootstrap(argv):
    grammar(argv)
    if argv[0] != "execute":
        return None
    root = argv[4]
    worker = process_identity("self")
    token = sys._xoptions.get("full_support_reserved")
    if token is None:
        text = attempt_text("direct", worker[0], worker[2])
        with open(root + "/attempt.started.json", "x") as handle:
            handle.write(text)
        launcher_pid, launcher_start = worker[0], worker[2]
    else:
        parts = token.split(":") if type(token) is str else []
        if len(parts) != 2 or not all(
            p.isascii() and p.isdecimal() and p[0] != "0" for p in parts
        ):
            raise RuntimeError("Invalid reservation token")
        launcher_pid, launcher_start = int(parts[0]), parts[1]
        text = attempt_text("shell", launcher_pid, launcher_start)
        with open(root + "/attempt.started.json") as handle:
            if handle.read() != text:
                raise RuntimeError("Reservation marker missing/mismatched")
        live_ancestor(launcher_pid, launcher_start, worker)
    with open(root + "/worker.started.json", "x") as handle:
        handle.write(
            '{"attempt":'
            + text.strip()
            + ',"schema":1,"status":"exclusive_worker_started","worker_pid":'
            + str(worker[0])
            + ',"worker_starttime_ticks":"'
            + worker[2]
            + '"}\n'
        )
    return launcher_pid, launcher_start


def deadlines(identity):
    # Runs immediately after receipt, before scientific imports, for direct too.
    import os
    import resource
    import signal

    if identity is not None:
        resource.setrlimit(resource.RLIMIT_CPU, (1799, 1800))
        with open("/proc/uptime") as handle:
            uptime = float(handle.read().split()[0])
        elapsed = uptime - int(identity[1]) / os.sysconf("SC_CLK_TCK")
        if elapsed >= 3600:
            raise RuntimeError(
                "Launch deadline already exhausted; attempt remains consumed"
            )
        signal.setitimer(signal.ITIMER_REAL, max(0.001, 3600 - elapsed))
