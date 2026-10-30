#!/usr/bin/env bash
# Run inside server tmux. Each invocation gets fresh logs; result ledgers resume.
set -euo pipefail
mode=${1:?single or batch required}
plan=${2:?absolute plan path required}
pin=${3:?plan SHA256 required}
logs=${4:?existing absolute log directory required}
shift 4
[[ "$mode" == single || "$mode" == batch ]] || exit 2
[[ "$plan" == /* && "$logs" == /* && -d "$logs" && "$pin" =~ ^[0-9a-f]{64}$ ]] || exit 2
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
python=${NURSE_OPS_PYTHON:-python3}
umask 077
attempt=$(mktemp -d "$logs/attempt.XXXXXXXX")
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$root/src:$root/third_party/hermes_runtime"
unset ARK_API_KEY ARK_API_KEY2
ulimit -v 8388608
module=nurse_rcl.v2.live.ops_execution
[[ "$mode" == single ]] || module=nurse_rcl.v2.live.ops_batch
printf 'LOG_DIRECTORY=%s\n' "$attempt"
set +e
"$python" -u -m "$module" --plan "$plan" --plan-sha256 "$pin" "$@" 2>&1 | tee "$attempt/run.log"
code=${PIPESTATUS[0]}
set -e
printf 'EXIT_CODE=%s\n' "$code" | tee "$attempt/runner.exit"
exit "$code"
