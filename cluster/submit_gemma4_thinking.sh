#!/bin/bash
# Queue the two Gemma-4 thinking agents on the pilot's 15 long episodes and 252 probes: four 2-GPU jobs, one per
# (model, episodes or probes), each serving its own data-parallel copy of the model. Output budget 8192 (decision D8.3).
set -euo pipefail
cd "$(dirname "$0")/.." && source cluster/site.env
run="cluster/run.sh envs/scienceworld -m core.run"
for m in e2b 12b; do
  base="google/gemma-4-${m^^}-it"
  agent="[gemma-4-$m/none/react-think]"
  episodes="$DATA_ROOT/logs/gemma4_${m}_think_episodes.txt"
  probes="$DATA_ROOT/logs/gemma4_${m}_think_probes.txt"
  cat > "$episodes" <<CMD
$run --config runs/pilot_sw_long.yaml -o run_id=pilot_sw_long_g4${m}think_8k -o agents='$agent' -o endpoint.file=\$ENDPOINT -o tasks=$DATA_ROOT/catalogs/scienceworld_pilot.parquet -o suite.cap_per_type=2 -o sampling.max_tokens=8192 -o workers=16
CMD
  cat > "$probes" <<CMD
$run --config runs/pilot_recovery.yaml -o run_id=pilot_recovery_g4${m}think_8k -o agents='$agent' -o endpoint.file=\$ENDPOINT -o sampling.max_tokens=8192 -o workers=8
$run --config runs/pilot_recovery.yaml -o run_id=pilot_compounding_g4${m}think_8k -o probes=$DATA_ROOT/probes/scienceworld/compounding.jsonl -o agents='$agent' -o endpoint.file=\$ENDPOINT -o sampling.max_tokens=8192 -o workers=8
CMD
  for kind in episodes probes; do
    [ "$kind" = episodes ] && hours=8 || hours=3
    sbatch --account "$SLURM_ACCOUNT" --partition "$SLURM_PARTITION" --gres=gpu:2 --cpus-per-task=16 --mem=96G \
      --time="$hours:00:00" --job-name "g4$m-$kind" --output "$DATA_ROOT/logs/slurm-%j.out" \
      cluster/serve_and_run.sbatch "gemma-4-$m" "$base" "$DATA_ROOT/logs/gemma4_${m}_think_$kind.txt" \
      --dp 2 --max-model-len 16384 --reasoning-parser gemma4
  done
done
