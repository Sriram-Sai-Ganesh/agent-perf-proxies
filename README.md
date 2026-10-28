# agent-perf-proxies

Which cheap evaluation best predicts a language-model agent's long-horizon performance, at a matched token
budget? We run small open models (Granite, Gemma) under different scaffolds and LoRA fine-tunes on
ScienceWorld (and later ALFWorld), measure their full long-horizon scores, and compare cheap predictors:
random subsets of long tasks, short tasks, static questions, and single-decision capability probes
(compounding errors, state tracking, error recovery, deferred instructions). [DESIGN_SPEC.md](DESIGN_SPEC.md)
is the plan; the proposal is in `tex/` (not tracked).

## Layout

| Folder | Venv | Purpose |
| --- | --- | --- |
| `core/` | root | Row schemas, trajectory store, provenance, model client, scaffolds, episode and probe runner, probe builders, `core.run` entry point |
| `envs/scienceworld/` | own | ScienceWorld adapter, task catalog, smoke test. Ships its own JRE via `jdk4py` |
| `serve/` | own | vLLM OpenAI-compatible server for one base model and its LoRA adapters |
| `configs/` | | `agents.yaml`: the agent pool (base, adapter, scaffold) |
| `runs/` | | Run configs for `core.run` |
| `cluster/` | | `run.sh` (pick a venv, source `site.env`), `site.env.example` |
| `tests/` | root | CPU tests with a scripted environment and a scripted model |

Each folder with a `pyproject.toml` is its own `uv` project and installs `core` by path. Environments, vLLM
and training never share a venv.

## Quickstart

`cluster/run.sh` sources `cluster/site.env` itself, but a `$DATA_ROOT` typed on the command line is expanded by
your shell first, so source it in your shell too: `source cluster/site.env`.

Login node, once:

```bash
cp cluster/site.env.example cluster/site.env   # set DATA_ROOT, HF caches, Slurm account
source cluster/site.env
UV_PYTHON=3.12 uv sync && .venv/bin/python -m pytest        # core + tests
cd envs/scienceworld && UV_PYTHON=3.12 uv sync && .venv/bin/python -m env_scienceworld.smoke && cd ../..
cluster/run.sh envs/scienceworld -m env_scienceworld.catalog --out $DATA_ROOT/catalogs/scienceworld.parquet
```

GPU node (interactive or sbatch), once per base model:

```bash
cd serve && UV_PYTHON=3.12 uv sync && cd ..
cluster/run.sh serve serve/server.py --base ibm-granite/granite-4.1-8b --out $DATA_ROOT/serve/granite-4.1-8b &
# wait for "serving [...] at http://127.0.0.1:PORT/v1"
```

Then, on the same node, run the pilot and build probes:

```bash
cluster/run.sh envs/scienceworld -m core.run --config runs/pilot_sw_long.yaml
cluster/run.sh envs/scienceworld -m core.probes.build recovery --env scienceworld \
    --tasks $DATA_ROOT/catalogs/scienceworld.parquet --out $DATA_ROOT/probes/scienceworld/recovery.jsonl --max-tasks 3
cluster/run.sh envs/scienceworld -m core.run --config runs/pilot_recovery.yaml
```

As a batch job, `cluster/serve_and_run.sbatch` starts the server on the job's GPUs, runs the lines of a command file
against it and stops it; `cluster/submit_gemma4_thinking.sh` shows the pattern.

Any setting in a run config can be overridden: `-o agents='[granite-4.1-3b/none/react]' -o seeds='[0,1]'`.
Runs resume: rerunning a config skips `(agent, item, seed)` triples already in `steps.jsonl`.

## Outputs

Under `$DATA_ROOT`: `catalogs/<env>.parquet` (tasks with gold trajectories and splits),
`probes/<env>/<family>.jsonl` (probe items), `runs/<run_id>/steps.{jsonl,parquet}` (one row per agent action,
with the prompt sent, the completion, the parsed action, environment score, and token counts), and
`provenance.json` beside each run (commit, patch of uncommitted changes, config, package versions).

## Status

Working: ScienceWorld adapter and catalog, ReAct scaffold, episode and probe runner with resume, recovery and
compounding probe builders, vLLM launcher, CPU tests. Not yet: ALFWorld, Notepad and Verify scaffolds, state
and deferred probes, static items (B3), LoRA training, analysis.
