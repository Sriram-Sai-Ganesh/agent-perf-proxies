#!/bin/bash
# Start a vLLM server on this job's GPUs, run each line of a command file against it, then stop the server.
#   sbatch --gres=gpu:2 [other resource flags] cluster/serve_and_run.sbatch <name> <base> <commands file> [server.py args...]
# The server writes $DATA_ROOT/serve/<name>-<job id>/endpoint.json; command lines reach it as $ENDPOINT, for example
#   cluster/run.sh envs/scienceworld -m core.run --config runs/pilot_sw_long.yaml -o endpoint.file=$ENDPOINT
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?run this with sbatch}"
source cluster/site.env
name="${1:?name for the server folder}"; base="${2:?hub id or local folder}"; commands="${3:?file of command lines}"; shift 3
out="$DATA_ROOT/serve/$name-$SLURM_JOB_ID"
cluster/run.sh serve serve/server.py --base "$base" --out "$out" "$@" &
server=$!
until [ -f "$out/endpoint.json" ]; do
  kill -0 "$server" 2>/dev/null || { echo "server exited before serving; see $out/vllm.log" >&2; exit 1; }
  sleep 10
done
export ENDPOINT="$out/endpoint.json"
while IFS= read -r line; do
  [ -n "$line" ] || continue
  echo "# $line"
  eval "$line"
done < "$commands"
kill "$server"; wait "$server" || true
