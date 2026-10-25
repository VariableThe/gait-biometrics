#!/usr/bin/env bash
set -euo pipefail
# Validate before path resolution, launchers, or Python startup. Same grammar as study.py.
if [[ $# == 1 && ( "$1" == -h || "$1" == --help ) ]]; then
    printf '%s\n' 'usage: run.sh {prepare|check-preparation|freeze|execute|selftest|verify [--replay-correlations]}'
    exit 0
fi
if ! { [[ $# == 1 && "$1" =~ ^(prepare|check-preparation|freeze|execute|selftest|verify)$ ]] || [[ $# == 2 && "$1" == verify && "$2" == --replay-correlations ]]; }; then
    printf '%s\n' 'stage must come first; --replay-correlations is accepted only after verify' >&2
    exit 2
fi
study_source="${BASH_SOURCE[0]}"
study_parent=.
if [[ "$study_source" == */* ]]; then study_parent="${study_source%/*}"; fi
study_dir="$(cd -- "$study_parent" && pwd)"
project_dir="$(cd -- "$study_dir/../.." && pwd)"
export VIRTUAL_ENV="$project_dir/.venv"
export UV_OFFLINE=1
export UV_PYTHON_DOWNLOADS=never
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 OMP_THREAD_LIMIT=1
uv_bin=/home/sherlock/.local/bin/uv
if [[ "${1:-}" == execute ]]; then
    # Bash builtins reserve even if Python source/import/runtime startup subsequently fails.
    set -o noclobber
    exec {joint_reservation_fd}> "$study_dir/attempt.started.json"
    set +o noclobber
    export JOINT_RUN_LAUNCH_EPOCH="$EPOCHREALTIME"
    IFS= read -r joint_launcher_stat < "/proc/$$/stat"
    read -r -a joint_launcher_fields <<< "${joint_launcher_stat##*) }"
    joint_launcher_start="${joint_launcher_fields[19]}"
    [[ "$joint_launcher_start" =~ ^[1-9][0-9]*$ ]]
    printf '{"automatic_retry":false,"cpu_budget_seconds":600,"launcher_pid":%d,"launcher_starttime_ticks":"%s","route":"shell","schema":2,"status":"exclusive_attempt_consumed","wall_budget_seconds":900}\n' "$$" "$joint_launcher_start" >&"$joint_reservation_fd"
    exec {joint_reservation_fd}>&-
fi
if [[ "${1:-}" == execute ]]; then
    exec /usr/bin/timeout --signal=KILL 900 "$uv_bin" run --no-project --python "$VIRTUAL_ENV/bin/python" python -B -X "joint_run_reserved=$$:$joint_launcher_start" "$study_dir/study.py" "$@"
fi
exec "$uv_bin" run --no-project --python "$VIRTUAL_ENV/bin/python" python -B "$study_dir/study.py" "$@"
