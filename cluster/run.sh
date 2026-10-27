#!/bin/bash
# cluster/run.sh <project> <python args...>   project: . | envs/scienceworld | serve | train | analysis
#   cluster/run.sh envs/scienceworld -m core.run --config runs/pilot_sw_long.yaml
# Sources cluster/site.env, runs the project's uv venv from the repository root, records LAUNCH_* for provenance.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(dirname "$here")"
[ -f "$here/site.env" ] && source "$here/site.env"
: "${DATA_ROOT:?set DATA_ROOT in cluster/site.env or the environment}"
project="${1:?first argument is the project folder whose .venv to use}"; shift
venv="$root/$project/.venv"
[ -x "$venv/bin/python" ] || { echo "no venv at $venv; run: cd $root/$project && uv sync" >&2; exit 1; }
export DATA_ROOT PYTHONUNBUFFERED=1 PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
export LAUNCH_USER="$USER" LAUNCH_HOST="$(hostname)" LAUNCH_COMMAND="$project python $*"
cd "$root"
exec "$venv/bin/python" "$@"
