#!/usr/bin/env bash
set -euo pipefail
# Stage/option order must be rejected before reservation or Python startup.
if [[ $# == 1 && "$1" =~ ^(prepare|check-preparation|selftest)$ ]]; then
    :
elif [[ $# == 5 && "$1" =~ ^(freeze|execute)$ && "$2" == --archive && "$4" == --out-root ]]; then
    :
elif [[ ( $# == 3 || $# == 5 ) && "$1" == verify && "$2" == --out-root && ( $# == 3 || "$4" == --archive ) ]]; then
    :
else
    printf '%s\n' 'stage first: prepare|check-preparation|selftest; freeze|execute --archive ABS --out-root ABS; verify --out-root ABS [--archive ABS]' >&2
    exit 2
fi
for full_support_arg in "$@"; do
    if [[ "$full_support_arg" == /* && ( "$full_support_arg" == *'/../'* || "$full_support_arg" == *'/./'* || "$full_support_arg" == *'//'* || "$full_support_arg" == */.. || "$full_support_arg" == */. ) ]]; then
        exit 2
    fi
done
full_support_source="${BASH_SOURCE[0]}"
full_support_parent=.
if [[ "$full_support_source" == */* ]]; then full_support_parent="${full_support_source%/*}"; fi
full_support_dir="$(cd -- "$full_support_parent" && pwd)"
full_support_project="$(cd -- "$full_support_dir/../.." && pwd)"
cd -- "$full_support_project"
export VIRTUAL_ENV="$full_support_project/.venv"
export PATH=/home/sherlock/.local/bin:/home/sherlock/.bun/bin:/usr/local/bin:/usr/bin:/bin
export PYTHONPATH='' PYTHONHASHSEED=0
export UV_OFFLINE=1 UV_PYTHON_DOWNLOADS=never PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 OMP_THREAD_LIMIT=1
if [[ "$1" == execute ]]; then
    [[ "$3" == /* && "$5" == /* && "$5" != "$full_support_dir" && "$5" != "$full_support_dir"/* ]]
    set -o noclobber
    exec {full_support_reservation_fd}> "$5/attempt.started.json"
    set +o noclobber
    IFS= read -r full_support_stat < "/proc/$$/stat"
    read -r -a full_support_fields <<< "${full_support_stat##*) }"
    full_support_start="${full_support_fields[19]}"
    [[ "$full_support_start" =~ ^[1-9][0-9]*$ ]]
    printf '{"automatic_retry":false,"cpu_budget_seconds":1800,"launcher_pid":%d,"launcher_starttime_ticks":"%s","route":"shell","schema":2,"status":"exclusive_attempt_consumed","wall_budget_seconds":3600}\n' "$$" "$full_support_start" >&"$full_support_reservation_fd"
    exec {full_support_reservation_fd}>&-
    exec /usr/bin/timeout --signal=KILL 3600 /home/sherlock/.local/bin/uv run --offline --no-project --python "$VIRTUAL_ENV/bin/python" python -B -X "full_support_reserved=$$:$full_support_start" "$full_support_dir/driver.py" "$@"
fi
exec /home/sherlock/.local/bin/uv run --offline --no-project --python "$VIRTUAL_ENV/bin/python" python -B "$full_support_dir/driver.py" "$@"
