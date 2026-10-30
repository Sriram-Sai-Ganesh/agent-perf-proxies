#!/bin/bash
# Queue one agent on the pilot's 15 long episodes or its 252 probes as a Slurm batch job that serves the agent's
# base model and runs against it (cluster/serve_and_run.sbatch).
#   cluster/submit.sh <agent id> <episodes|probes> <run suffix> [server.py args...]
#   GPUS=2 cluster/submit.sh gemma-4-12b/none/react-think probes g412bthink_16k --dp 2 --reasoning-parser gemma4
# Run ids are pilot_sw_long_<suffix>, or pilot_recovery_<suffix> and pilot_compounding_<suffix>.
# Output budget: 16384 tokens for a thinking agent (decision D8.3), 256 otherwise.
set -euo pipefail
cd "$(dirname "$0")/.." && source cluster/site.env
agent="${1:?agent id from configs/agents.yaml}"; kind="${2:?episodes or probes}"; suffix="${3:?run id suffix}"; shift 3
read -r base thinking < <(.venv/bin/python - "$agent" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open("configs/agents.yaml"))
a = next(a for a in cfg["agents"] if a["id"] == sys.argv[1])
print(cfg["bases"][a["base"]]["hub_id"], a.get("thinking", False))
PY
)
[ "$thinking" = True ] && max_tokens=16384 || max_tokens=256
run="cluster/run.sh envs/scienceworld -m core.run"
commands="$DATA_ROOT/logs/${suffix}_$kind.txt"
if [ "$kind" = episodes ]; then
  hours=8
  cat > "$commands" <<CMD
$run --config runs/pilot_sw_long.yaml -o run_id=pilot_sw_long_$suffix -o agents='[$agent]' -o endpoint.file=\$ENDPOINT -o tasks=$DATA_ROOT/catalogs/scienceworld_pilot.parquet -o suite.cap_per_type=2 -o sampling.max_tokens=$max_tokens -o workers=16
CMD
else
  hours=3
  cat > "$commands" <<CMD
$run --config runs/pilot_recovery.yaml -o run_id=pilot_recovery_$suffix -o agents='[$agent]' -o endpoint.file=\$ENDPOINT -o sampling.max_tokens=$max_tokens -o workers=8
$run --config runs/pilot_recovery.yaml -o run_id=pilot_compounding_$suffix -o probes=$DATA_ROOT/probes/scienceworld/compounding.jsonl -o agents='[$agent]' -o endpoint.file=\$ENDPOINT -o sampling.max_tokens=$max_tokens -o workers=8
CMD
fi
sbatch --account "$SLURM_ACCOUNT" --partition "$SLURM_PARTITION" --gres="gpu:${GPUS:-1}" --cpus-per-task=16 --mem=96G \
  --time="$hours:00:00" --job-name "$suffix-$kind" --output "$DATA_ROOT/logs/slurm-%j.out" \
  cluster/serve_and_run.sbatch "${base##*/}" "$base" "$commands" "$@"
