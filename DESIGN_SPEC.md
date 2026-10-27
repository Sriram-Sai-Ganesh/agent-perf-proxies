# Design spec: Short Tests, Long Tasks

Code for the proposal in `tex/proposal.tex`: compare cheap evaluations of language-model agents against their
long-horizon scores, at matched token budgets. This document is the plan we edit before implementation. Each
section ends with the decisions it still needs.

Status: draft, 2026-10-27. Nothing here is implemented. Decisions marked **decided** were settled on 2026-10-27;
the rest are open. Build order: environments, agent pool and ground truth first (section 8).

## 1. What the code must produce

The proposal's deliverables, in the order they are needed:

| Output | Used for | Proposal section |
| --- | --- | --- |
| Long-horizon ground truth `y(a)` per agent, 3 seeds, with task-length metadata | everything | Exp. design |
| Per-agent cheap-evaluation features at each budget `β` for B1, B2, B3 and the four probe families | predictors | Approach |
| Predictions and metrics under splits S1, S2, S3 vs. `β` and task length | Q1–Q4, H1–H4 | Exp. design |
| Validity-check table: which variant moves which probe | H2 sanity | Exp. design |
| The agent pool itself: 4 base models × 3 scaffolds, plus LoRA adapters for Granite-4.1 on 3 data mixes | ground truth | Approach |

Everything the analysis needs is a table of (agent, item, seed, outcome, tokens). The code is a way of filling
that table cheaply and reproducibly.

## 2. Principles

1. **One trajectory store, many readers.** Every episode and every probe answer is written as rows with the
   same schema (section 4). Probe generation, SFT data, budget accounting, and the meta-evaluation all read
   from it. No stage keeps private state.
2. **Separate virtual environments per concern.** Each environment package (ScienceWorld, ALFWorld) has its
   own `uv` project. So do model serving (vLLM), training (TRL + PEFT), and analysis (scikit-learn, pandas,
   matplotlib). They talk through files and HTTP, never imports. The one shared Python package, `core/`, has
   light dependencies and is installed into every venv.
3. **Models are served, not loaded.** Agents call an OpenAI-compatible endpoint. A vLLM server per node hosts
   one base model and all its LoRA adapters. Token counts come from the server's `usage` field, so budgets are
   measured, not estimated.
4. **Every run is resumable and records provenance.** Rows are appended to JSONL as they complete, then
   consolidated to Parquet. A run records the git commit, the config, the server's model and adapter
   revisions, and the package versions. A dirty tree is recorded as a patch; a missing commit stops the run.
5. **Scaffolds are the agent's property.** Probes replay a prefix through the same scaffold the agent uses in
   full episodes, so the scaffold must be able to rebuild its state (for example the notepad) from a prefix.
6. **Budgets are tokens.** `C_full(a)` is the total prompt plus completion tokens of one seed of the long
   suite. A cheap evaluation at budget `β` may spend at most `β · C_full(a)` tokens, counted the same way.
   Cached prefill is reported as a secondary accounting, never used to pick items.
7. **Small pool, so simple predictors.** With about 20 agents, predictors are means and ridge regressions on
   at most a few features. No model selection beyond what the proposal names.

## 3. Repository layout

```
agent-perf-proxies/
  DESIGN_SPEC.md            this file
  README.md                 setup and the fixture chain
  pyproject.toml            the `core` package: schemas, scaffolds, episode runner, probe builders, provenance
  core/
    schema.py               row schemas and validation for every table
    provenance.py           git commit, patch, config, environment, timestamps
    store.py                JSONL append, Parquet consolidate, resume, lock
    tokens.py               budget accounting from `usage` fields
    llm.py                  OpenAI-compatible client with retries and per-call token records
    env.py                  the Environment interface every adapter implements
    agent.py                AgentSpec (model, adapter, scaffold, thinking, sampling)
    scaffolds/
      base.py               Scaffold interface: prompt building, action parsing, state from prefix
      react.py
      notepad.py
      verify.py
    runner.py               episode loop and prefix replay; writes step rows
    probes/
      base.py               ProbeItem schema and the scorer
      compounding.py        family (i)
      state_tracking.py     family (ii)
      recovery.py           family (iii)
      deferred.py           family (iv)
    suites.py               task suites: long, short, B1 subsets, train/dev/test splits
  envs/
    scienceworld/           uv project; adapter + task catalog + gold trajectories
      pyproject.toml
      adapter.py
      catalog.py            enumerate tasks and variations, gold lengths, splits
    alfworld/               uv project; same shape
      pyproject.toml
      adapter.py
      catalog.py
  serve/                    uv project: vLLM server launcher, adapter registry
    pyproject.toml
    server.py
  train/                    uv project: SFT data builders and LoRA training
    pyproject.toml
    data/
      clean.py  recovery.py  state.py
    sft.py
    runs/*.yaml
  static/                   B3: non-agentic item sets and their scorer (own uv project if it needs lm-eval)
  analysis/                 uv project: features, predictors, splits, metrics, figures
    pyproject.toml
    features.py  predict.py  splits.py  metrics.py  report.py
    runs/*.yaml
  configs/
    agents.yaml             the agent pool registry
    envs.yaml               per-environment settings (L, step caps, simplifications)
    budgets.yaml            β grid, seeds, repeats
  cluster/
    site.env.example        DATA_ROOT, HF_HOME, JAVA_HOME, partitions
    run.sh                  activates the right venv, exports site.env, records LAUNCH_* for provenance
    array.sbatch            run line N of a command file as array task N
    jobs/                   generated command files, gitignored
  tests/                    CPU-only tests with a scripted fake environment and a scripted fake model
  tex/                      proposal (gitignored today; decide whether to track it)
```

Every `uv` project depends on `core` by path (`core = { path = "..", editable = true }`). `core` depends
only on `pyarrow`, `pyyaml`, `httpx`, and `pydantic` (or dataclasses plus hand validation, see decisions).

Why not one venv: ALFWorld pins TextWorld and a planner build that is fragile across Python versions;
vLLM moves fast and pins torch; TRL pins a different torch window. Keeping them apart means a breakage in one
does not block the others, and the environment venvs can stay on whatever Python ALFWorld builds on.

Decisions needed:
- D3.1 `pydantic` for schemas, or plain dataclasses with a `validate()` per table? (Recommendation: pydantic;
  the row schemas are the contract between five venvs, and clear errors at the boundary are worth the dependency.)
- D3.2 **decided**: `tex/` stays out of the repo; this repo is code only.

## 4. Data model

All tables are Parquet, one folder per run under `$DATA_ROOT`, with the raw JSONL kept beside them. Identifiers
are strings. `meta` columns are JSON strings, for fields that vary by environment or family.

### 4.1 Tasks (`tasks.parquet`, one per environment, built once)

| column | meaning |
| --- | --- |
| `task_id` | `{env}/{task_type}/{variation}` |
| `env` | `scienceworld` or `alfworld` |
| `task_type` | ScienceWorld task name; ALFWorld task class (pick, clean, heat, cool, examine, pick2) |
| `variation` | ScienceWorld variation index; ALFWorld game path |
| `split` | `train`, `dev`, `test` as the environment defines them |
| `gold_len` | number of actions in the gold trajectory |
| `gold_actions` | JSON list of gold actions |
| `description` | task text as the agent sees it at reset |
| `meta` | simplification string, ALFWorld scene id, anything else |

Suites (section 5.5) are filters over this table and are stored as lists of `task_id`.

### 4.2 Steps (`steps.parquet`, the trajectory store)

One row per agent action, for full episodes and for probe answers alike.

| column | meaning |
| --- | --- |
| `run_id` | the run that produced the row |
| `agent_id` | from the agent registry, e.g. `granite-4.1-8b/none/react` |
| `kind` | `episode` or `probe` |
| `item_id` | `task_id` for episodes; `probe_id` for probes |
| `seed` | sampling seed for this episode or probe draw |
| `step` | 0-based action index within the episode; 0 for a probe answer |
| `observation` | text the environment returned before this action |
| `prompt` | JSON list of `{role, content}` messages actually sent, after scaffold formatting |
| `completion` | the raw model output |
| `action` | the parsed action string sent to the environment (or scored, for probes) |
| `valid` | whether the environment accepted the action (episodes) |
| `reward`, `score` | per-step reward and cumulative environment score after this action (episodes) |
| `done` | episode ended after this action |
| `prompt_tokens`, `completion_tokens`, `cached_tokens` | from the server's `usage` |
| `latency_s` | wall time of the model call |
| `scaffold_state` | JSON: notepad contents, verify verdict, thinking text if any |
| `meta` | per-step extras |

Storing the full prompt per step is redundant but makes replay, auditing and failure labelling trivial. At
roughly 20 agents × 3 seeds × 400 episodes × 60 steps × 4 KB that is about 6 GB; acceptable on scratch.

### 4.3 Episodes (`episodes.parquet`, derived from steps)

| column | meaning |
| --- | --- |
| `run_id`, `agent_id`, `item_id`, `seed` | keys |
| `n_steps` | actions taken |
| `success` | environment's completion flag |
| `final_score` | ScienceWorld score in [0, 1]; ALFWorld 0/1 |
| `progress` | max score reached at any step (progress rate, Ma et al. 2024) |
| `tokens` | prompt plus completion tokens summed over steps |
| `gold_len` | copied from tasks, for grouping by length |
| `stop_reason` | `done`, `step_cap`, `invalid_streak`, `error` |

### 4.4 Probe items (`probes/{family}/items.parquet`)

| column | meaning |
| --- | --- |
| `probe_id` | `{family}/{source_task_id}/{index}` |
| `family` | `compounding`, `state`, `recovery`, `deferred` |
| `env`, `source_task_id` | the train-split task the prefix came from |
| `source_agent_id` | for `compounding`: whose errors the prefix contains; null for gold-only prefixes |
| `prefix` | JSON list of `(observation, action)` pairs to replay, plus the task description |
| `final_observation` | what the agent sees when asked for the next action |
| `gold_actions` | JSON list; any one counts as correct |
| `distractors` | JSON list of plausible wrong actions, for likelihood scoring and for making static variants |
| `meta` | family-specific: distance back for `state`, injected step for `recovery`, cue step for `deferred` |

### 4.5 Probe answers

Rows in `steps.parquet` with `kind = probe`, plus a derived `probe_results.parquet` with `probe_id`,
`agent_id`, `seed`, `correct`, `tokens`, and optionally `gold_logprob` and `distractor_logprobs`.

### 4.6 Features, predictions, metrics (analysis outputs)

- `features.parquet`: `agent_id`, `method` (`B1`, `B2`, `B3`, `probe:compounding`, …, `probes+B2`),
  `beta`, `draw` (which random subsample under the budget), `feature` (JSON vector), `tokens_spent`.
- `truth.parquet`: `agent_id`, `target` (`sw_long`, `alfworld`, `sw_long_bin_{k}` by gold-length bin),
  `seed`, `y`.
- `predictions.parquet`: `split` (`S1`, `S2`, `S3`), `fold`, `method`, `beta`, `draw`, `agent_id`, `y_hat`.
- `metrics.parquet`: `split`, `method`, `beta`, `target`, `metric`, `value`, `ci_low`, `ci_high`.

## 5. Components

### 5.1 Environments (`envs/`)

Interface in `core/env.py`:

```python
class Environment(Protocol):
    def catalog(self) -> list[TaskRow]            # every task with split and gold trajectory
    def reset(self, task_id: str) -> str           # returns the first observation (includes task text)
    def step(self, action: str) -> StepResult      # observation, reward, score, done, valid, meta
    def valid_actions(self) -> list[str] | None    # for probe distractors; None if not supported
    def replay(self, task_id: str, actions: list[str]) -> str  # reset then step through a prefix; returns the last observation
```

`replay` exists so probes can put the environment in the right state and so `recovery` probes can check
whether an action after the prefix actually recovers (not just matches gold).

**ScienceWorld** (`scienceworld` 1.2.3 on PyPI). Needs a JVM; `scienceworld` launches the bundled JAR through
py4j. There is no Java module or conda on this cluster, so the adapter venv installs `jdk4py`, which ships a
JRE as a wheel, and the launcher exports its path as `JAVA_HOME` and prepends it to `PATH`. One JVM per
worker process; workers are plain processes, one per concurrent episode (section 6).

Details to fix in `configs/envs.yaml`: `simplification` string (the proposal does not say; AgentGym uses
`easy`), `envStepLimit` (step cap, proposal says 100 for long tasks), score normalisation (ScienceWorld
reports 0–100, and −100 on failure; store `score/100` clipped to [0, 1] and `progress` as the running max).
Splits: `get_variations_train/dev/test()` per task type. Gold trajectory from `get_gold_action_sequence()`
with `generateGoldPath=True`.

**ALFWorld** (`alfworld` 0.4.2). Text-only `AlfredTWEnv`; data via `alfworld-download` into `$DATA_ROOT`.
Splits: `train` (3553 games), `eval_in_distribution` (140), `eval_out_of_distribution` (134, the proposal's
134). Gold trajectory from the handcoded expert (`extra.expert_plan` in infos, with `expert_type: handcoded`).
Score is binary; `progress` is the fraction of subgoals satisfied if the environment exposes it, else equal
to `success`. Valid actions from `admissible_commands`.

Both adapters expose the same `catalog()` so `tasks.parquet` is built by one script, `envs/build_catalog.py`,
run once per environment venv.

Decisions needed:
- D5.1 ScienceWorld `simplification`: `easy` (as AgentGym) or none? Affects gold length and difficulty; pilot both on one task type.
- D5.2 Whether `valid_actions` is shown to the agent (ALFWorld often shows admissible commands; ScienceWorld agents usually do not get them). Recommendation: do not show them for either, so the probe distractor set is not something the agent saw.

### 5.2 Model serving (`serve/`)

`serve/server.py` starts a vLLM OpenAI-compatible server for one base model with `--enable-lora` and every
adapter registered for that base in `configs/agents.yaml` as `--lora-modules`. It writes `endpoint.json`
(`base_url`, `model`, served adapter names, revisions, vLLM version) into the run folder; the runner reads it.
Reuse logic: if an `endpoint.json` on this host already serves the required model and adapters, use it.

Settings: `gpu_memory_utilization`, `max_model_len` (long episodes with ReAct histories need 16k–32k; the
scaffold truncates history before that, section 5.3), `max_lora_rank` (set to the largest rank trained),
`enable_prefix_caching` on (free for replay probes; the `cached_tokens` column records what it saved).

Thinking mode (Granite-4.2) is a chat-template switch passed per request through `extra_body`; the
`AgentSpec` carries `thinking: bool` and the client applies it.

Base models and where they come from:

| id | hub | cached locally |
| --- | --- | --- |
| granite-4.1-3b, granite-4.1-8b | ibm-granite | yes |
| granite-4.2-3b, granite-4.2-8b | ibm-granite | 8b only |
| gemma-4-E2B-it, gemma-4-12B-it | google | yes |

Decisions needed:
- D5.3 **decided**: Gemma-4 (E2B and 12B) replaces the proposal's Gemma-3; it is cached and newer. The proposal text should be updated to match.
- D5.4 Hosted fallback (OpenRouter) for a pilot before the cluster venvs work? `core/llm.py` is endpoint-agnostic, so this costs nothing to allow.

### 5.3 Scaffolds (`core/scaffolds/`)

```python
class Scaffold(Protocol):
    name: str
    def start(self, task_text: str) -> State
    def prompt(self, state: State, observation: str) -> list[Message]   # what to send
    def parse(self, state: State, completion: str) -> tuple[str, State] # action and updated state
    def from_prefix(self, task_text: str, prefix: list[tuple[str, str]]) -> State  # rebuild state for replay
```

- **ReAct**: system prompt with the action grammar; history as alternating observation/assistant turns;
  `Thought: … Action: …` format; history window of the last `k` turns plus the task text (k from config,
  default whole episode up to `max_model_len`, then oldest-first drop with a one-line summary stub).
- **ReAct+Notepad**: the assistant output also carries a `Notepad:` block the agent rewrites each turn; the
  prompt shows only the latest notepad and the last `k` turns. `from_prefix` regenerates the notepad by
  running the model over the prefix turn by turn (costly) **or** by replaying the prefix with the agent's own
  stored notepads when the prefix came from its own trajectory (free). Which one is a per-probe-family
  choice recorded in `meta`.
- **ReAct+Verify**: after each observation, a second call asks "did the last action succeed; if not, why",
  and a failed verdict is inserted as an extra observation line before the next action. Two calls per step,
  both counted in tokens.

Action parsing is shared: extract the `Action:` line, lower-case, strip punctuation; environment adapters do
any further canonicalisation (ScienceWorld is forgiving, ALFWorld needs exact admissible commands).

Decisions needed:
- D5.5 Notepad state from prefix: rerun the model, or store-and-replay only? Store-and-replay is free but means Notepad agents' `state` and `deferred` probes must use prefixes from their own trajectories. Recommendation: store-and-replay for own-trajectory prefixes, rerun for gold prefixes, and report which was used.

### 5.4 Episode runner (`core/runner.py`)

`run_episode(agent, env, task_id, seed, max_steps) -> list[StepRow]`. Loop: observation → scaffold prompt →
model call → parse → env step → append row. Stops on `done`, `max_steps`, `invalid_streak` (N invalid actions
in a row, default 5), or an unrecoverable server error (recorded, not retried beyond the client's retries).

`run_probe(agent, env, item, seed) -> StepRow`: `env.replay(prefix)`, `scaffold.from_prefix`, one prompt,
one model call, parse, score against `gold_actions`; for `recovery`, also step the environment once and record
whether the score moved.

Entry point `core/run.py --config runs/<name>.yaml` takes an agent list, a suite or probe set, seeds, and a
concurrency; it uses a process pool where each worker owns one environment instance and shares the HTTP
endpoint. Output: `$DATA_ROOT/runs/<run_id>/{steps.jsonl, steps.parquet, episodes.parquet, provenance.json}`.
Resume: on start, read existing JSONL, skip `(agent, item, seed)` triples already complete.

### 5.5 Task suites and splits (`core/suites.py`)

From `tasks.parquet`:

- `sw_long`: ScienceWorld `test` variations with `gold_len ≥ L`. `L` set after the pilot (section 8).
  Capped per task type so one type does not dominate (cap from config). Also grouped into gold-length bins for
  the length analysis.
- `sw_short`: ScienceWorld `dev` variations with `gold_len ≤ L_short` (default 10). B2 pool.
- `alf_long`: ALFWorld `eval_out_of_distribution`, all 134. `alf_short`: ALFWorld `train` games with the
  shortest expert plans, disjoint from anything used for SFT.
- B1 subsets: random subsets of the target suite, drawn by `analysis/features.py` from the ground-truth runs
  themselves (no extra episodes are run; B1 at budget β is a subsample of the full run, which is exactly what
  running fewer episodes would have produced, seed for seed).
- Probe sources and SFT data: `train` split only. The probe-source tasks and the SFT-source tasks are disjoint
  lists written once to `configs/splits/`.

### 5.6 Probes (`core/probes/`)

Each family is a builder `build(tasks, trajectories, env, config) -> list[ProbeItem]` and shares one scorer.
Items are built from gold trajectories (and, for `compounding`, from agents' own trajectories) on `train`
tasks, replayed via `env.replay`, so the environment state is real and `valid_actions` can supply distractors.

- **(i) Compounding errors.** A gold prefix of length `n`, then the agent acts freely for `d` steps (its own
  errors accumulate in its context at run time), then one scored decision. The matched control is the same item
  with `d = 0`. Scored by exact match against the gold actions still untaken and by whether the score moves.
  Items are built without a model, so every agent answers the same items. (Revised 2026-10-27: the original
  design took prefixes from the agent's own episodes, but untuned agents leave gold at step 0, so no such prefix
  exists at the floor.)
- **(ii) State tracking.** Gold prefix in which a fact needed for the next action (an object's location, a
  container's state) was established `k` steps ago, with filler gold steps in between. Items tagged with `k`;
  the feature is the slope of accuracy against `k`, plus the mean.
- **(iii) Error recovery.** Gold prefix where one action is replaced by a silently failing one (wrong object,
  wrong container) with its real observation from `replay`, followed by zero or more gold steps; the agent must
  either redo the failed step or otherwise produce an action that raises the score. Scored by stepping the
  environment: correct if the resulting score is ≥ what the gold continuation reaches in one step, or if the
  action is in the recomputed gold set. Tagged with `steps_since_failure`.
- **(iv) Deferred instructions.** Task text augmented with a conditional instruction ("when X happens, do Y")
  that is consistent with the task; the prefix runs until the cue appears in an observation; correct if the
  next action is Y. Cues and Y are templated per environment (ScienceWorld: state changes like boiling,
  opening; ALFWorld: arriving at a receptacle).

Scoring modes, both stored: `exact` (parsed action ∈ gold set) and, where the server returns logprobs,
`likelihood` (gold vs. distractor completion logprobs, which gives a continuous score and helps close pairs).

Item counts: target 200–400 items per family per environment, so that at `β = 1%` of a `C_full` of roughly
2–4M tokens (20k calls × 100–200 tokens) the budget buys 100+ items per family.

Decisions needed:
- D5.6 "Gold from an off-path state" for (i) and (iii): ScienceWorld's gold path is computed at load time, not from arbitrary states. Options: (a) only accept recovery if the score increases after stepping; (b) define gold as "the first gold action not yet satisfied" using the task's subgoal list; (c) hand-write per-task-type recovery rules. Recommendation: (a) for scoring plus (b) as the stored `gold_actions`, and treat ALFWorld the same way with its subgoal checker.
- D5.7 **decided**: ScienceWorld goes end to end (adapter, pool, probes, ground truth) before any ALFWorld work starts. ALFWorld stays in the design as the secondary target.

### 5.7 B3: non-agentic items (`static/`)

A fixed set of multiple-choice or short-answer questions scored by likelihood or exact match, run through the
plain model (no scaffold) at the same endpoint. **Decided**: B3 is a fixed random sample of about 100 items
from a public mix (MMLU-Pro, GSM8K, BBH subsets), not PACE's predictive item selection. PACE selection would
have to be refit inside every split fold and needs a far larger pool; the fixed sample is a weaker but honest
non-agentic baseline, and the paper should say so. Cost is tiny, so B3 is computed at every β as the full set.

Implementation: `static/items.parquet` with `item_id, source, prompt, choices, answer`; `static/run.py` with
the same `core/llm.py` client and the same step-row output with `kind = static`.

Decision needed:
- D5.8 Own likelihood scorer over our client, no `lm-eval-harness`. Exact pools and item counts still open.

### 5.8 Fine-tuning (`train/`)

Data builders write SFT conversations in the exact prompt format the ReAct scaffold produces (so the adapter
is trained on what it will see), from `train`-split tasks disjoint from probe sources:

- `clean`: gold trajectories as (prompt, gold action) pairs with a templated `Thought:`.
- `+recovery`: `clean` plus trajectories with an injected wrong-object failure and the recovery step, built with
  `env.replay`; failure types disjoint from those used in probe (iii) (recorded in `configs/splits/`).
- `+state`: `clean` plus auxiliary turns where the prompt asks a state question ("where is the X") and the
  answer comes from the environment's object tree.

`train/sft.py`: TRL `SFTTrainer` with PEFT LoRA on Granite-4.1 3B and 8B, one config per mix, fixed rank and
steps across mixes so the only difference is data. Exports an adapter folder plus `adapter.json` (base,
revision, rank, data mix, data hash). The adapter is then registered in `configs/agents.yaml`.

### 5.9 Budget accounting (`core/tokens.py`, `analysis/features.py`)

`C_full(a)` = tokens of the long-suite run, mean over seeds, per target environment (so the ScienceWorld and
ALFWorld budgets are separate; the combined target uses the sum). For each method and β, `features.py`
draws `R` random item subsets (default `R = 20`) whose summed measured token cost is ≤ `β · C_full(a)`,
computes the feature vector per draw, and writes one row per draw. For probes, item cost is the measured
tokens of that agent's answer, so an agent with long outputs gets fewer items at the same β, as it would in
practice. Features:

| method | feature vector |
| --- | --- |
| B1 | mean `final_score` of the subset of long episodes |
| B2 | mean score on the short-task subset |
| B3 | accuracy on the static set (always full) |
| probe:family | mean accuracy, plus the family's slope feature where defined |
| probes | the four family means (and slopes) |
| probes+B2 | the above plus B2 |

### 5.10 Meta-evaluation (`analysis/predict.py`, `splits.py`, `metrics.py`)

Predictors: B1 is the identity (the subset mean is the prediction). Every other method fits ridge regression
from features to `y` on the development agents of the fold; alpha chosen by leave-one-out inside the
development agents. Targets: `sw_long`, `alfworld`, and `sw_long` per gold-length bin.

Splits over the agent registry:

- **S1** leave-one-agent-out, restricted to agents whose base, scaffold and adapter each appear elsewhere.
- **S2** leave-one-base-model-out.
- **S3** variant pairs: all `(a, a')` with the same base model; for each pair, fit on all other agents and
  predict the sign and size of `y(a') − y(a)`.

Metrics per (split, method, β, target): MAE, Kendall τ over the held-out predictions, S3 better-agent
accuracy, and the smallest reliably ranked gap (the smallest `|y(a') − y(a)|` at which sign accuracy is ≥ 0.9
over draws). Bootstrap over agents (and over draws) for intervals. Run-to-run noise floor: the same metrics
computed between seeds of the ground truth.

Validity table: for each targeted variant pair (base vs. Notepad, base vs. Verify, clean vs. +recovery, clean
vs. +state, thinking off vs. on), the change in each probe family's mean, with its interval.

### 5.11 Reporting (`analysis/report.py`)

Writes `metrics.parquet` to Markdown tables and the figures the proposal names: metric vs. β per method,
metric vs. gold-length bin, the S3 pair plot, and the validity heatmap. Figures go to `tex/figures/` as PDF
and PNG, built from the Parquet tables only.

## 6. Cluster workflow

Nodes have one H100 (80 GB). One job = one base model = one vLLM server plus a pool of environment workers on
the same node, running every agent built on that base (its scaffolds and adapters) over a suite. This keeps
the server warm and amortises model load. Command files for `cluster/array.sbatch` are generated by
`cluster/plan.py` from `configs/agents.yaml` × suites × seeds.

Rough cost, from the proposal: ~20k model calls per agent-seed on the full targets; at 12–24 agents × 3 seeds
that is 1–1.5M calls. An 8B model on one H100 at 1–2k tokens per call does 50–100 calls/s with batching, so
a few GPU-hours per base model per seed. Concurrency is bounded by environment workers (ScienceWorld JVMs are
~1–2 GB each; 32 workers per node is a safe start).

`cluster/run.sh <venv> <script> …` sources `cluster/site.env`, activates `<venv>/.venv`, exports `JAVA_HOME`
from `jdk4py` when the venv is `envs/scienceworld`, and sets `LAUNCH_*` for provenance.

Decision needed:
- D6.1 Container or bare `uv` venvs on compute nodes? The sibling project supports both. Recommendation: bare `uv` venvs (`LOCAL=1` style only) unless the vLLM wheel fails on the node image.

## 7. Configuration

`configs/agents.yaml`:

```yaml
bases:
  granite-4.1-8b: {hub_id: ibm-granite/granite-4.1-8b, revision: null, thinking: false}
  granite-4.2-8b: {hub_id: ibm-granite/granite-4.2-8b, revision: null, thinking: false}
adapters:
  granite-4.1-8b/clean:     {path: $DATA_ROOT/adapters/granite-4.1-8b/clean, rank: 16}
  granite-4.1-8b/recovery:  {path: ..., rank: 16}
scaffolds: [react, notepad, verify]
agents:   # expanded by cluster/plan.py; listed explicitly so the pool is a reviewable artefact
  - {id: granite-4.1-8b/none/react, base: granite-4.1-8b, adapter: null, scaffold: react}
  - ...
```

Run configs (`runs/*.yaml`) name an agent subset, a suite or probe set, seeds, step cap, concurrency, and the
endpoint source (`serve` locally, or `base_url`). Overrides with `-o key=value` as in the sibling project.

## 8. Milestones against the proposal's dates

| date | milestone | what must exist |
| --- | --- | --- |
| Oct 29 | pilot | `core` schemas, runner, ReAct; ScienceWorld adapter + catalog; `serve`; probes (i) and (iii) on ScienceWorld; one Granite agent on a 50-episode long sample and ~100 probes; a table of score and progress by gold length |
| Nov 2 | pool infrastructure | ALFWorld adapter; Notepad and Verify; `cluster/plan.py`; resume and provenance tested; `L` fixed from the pilot |
| Nov 9 | probes and pool | families (ii) and (iv); all bases × scaffolds registered; B2 suites; B3 item pool |
| Nov 16 | SFT variants | data builders, three adapters per Granite-4.1 size, registered |
| Nov 23 | ground truth | long suites × 3 seeds for every agent; all probes and B2/B3 for every agent |
| Dec 1 | analysis | features, predictions, metrics, figures |

The pilot decides `L`, the ScienceWorld simplification, whether the floor is a problem, and whether the
probes have spread across the two or three agents tried. Those decisions are written back into this document.

### Pilot results, 2026-10-27 (granite-4.1-8b, ReAct, ScienceWorld `easy`, 15 test episodes with gold length ≥ 30, 60 recovery probes)

| measure | value |
| --- | --- |
| success / final score / progress rate | 0.00 / 0.02 / 0.07 (progress 0.13 in the 60–100 gold bin) |
| steps per episode, valid-action rate | 64, 0.90; stop reasons: 9 step cap, 5 done, 1 invalid streak |
| tokens per episode | 262k (prompt-dominated: the whole history is resent each step) |
| recovery probes: exact / score-moved | 0.03 / 0.10; 1.1k tokens per item; 60 items answered in 12 s |
| compounding probes (192 items, 8 train task types): exact by depth d=0 / 2 / 5 | 0.61 / 0.48 / 0.44 (n = 64 / 56 / 48); valid-action rate 0.83 / 0.79 / 0.71; 1.2k tokens per item; 32 s total |
| wall time | 15 episodes in 186 s with 8 workers on one H100 |

What this changes:
- **Floor confirmed.** Final score is uninformative for untuned 8B; `progress` (max score reached) is the
  working target. One episode reached 0.73 then hit ScienceWorld's `focus on` trap (wrong focus ends the task
  at score 0). That trap is part of the environment and stays.
- **`C_full` is larger than the proposal's estimate.** About 260k tokens per long episode means roughly 70M
  tokens per agent-seed for a 270-episode suite, counted without prefix caching. Open decision D8.1: whether
  budgets count cached prompt tokens at full price (conservative, what a hosted API would charge) or at the
  server's cached rate. The server now reports `cached_tokens`, so both can be computed from the same rows.
- **Single-step recovery scoring is too strict.** After a wrong-room failure the agent reasonably says
  `look around`; exact match against the one gold action scores that wrong. Open decision D8.2: score recovery
  over a short continuation (the agent gets up to `k` steps, correct if it reaches the gold state or the score
  moves) rather than one decision. This costs more tokens per item but is still far below an episode.
- **Compounding is measurable at the floor.** Even for an agent whose episodes score near zero, next-action
  accuracy from a gold prefix is well above zero and falls monotonically with the number of free steps. This is
  the kind of spread the probes need; whether it predicts anything is the study's question.
- **Throughput is fine.** 8 environment workers kept one H100 busy; probes are two orders of magnitude cheaper
  per item than episodes, as the design assumes.

## 9. Testing

`tests/` runs on CPU in the `core` venv with:

- a scripted fake environment (a tiny gridworld with gold paths, partial credit and a `replay`) and a scripted
  fake model (returns the gold action with a configurable error rate), so the runner, scaffolds, probe
  builders, budget accounting and the meta-evaluation run end to end on fixtures in seconds;
- schema round-trip tests for every table;
- resume tests: kill a run mid-way, restart, check no duplicate `(agent, item, seed)` rows;
- a smoke test per environment venv (`envs/*/smoke.py`) that resets one task, replays its gold path and checks
  the final score is full, run manually after installing.

Every analysis script is deterministic given its config and seed.

## 10. Open decisions, collected

| id | question | recommendation |
| --- | --- | --- |
| D3.1 | pydantic vs. dataclasses for schemas | pydantic |
| D3.2 | track `tex/` | decided: no |
| D5.1 | ScienceWorld simplification | pilot `easy` vs. none |
| D5.2 | show valid actions to the agent | no |
| D5.3 | Gemma-3 (proposal) vs. Gemma-4 (cached) | decided: Gemma-4 |
| D5.4 | allow hosted endpoints | yes, zero cost |
| D5.5 | Notepad state from prefix | store-and-replay for own prefixes, rerun for gold |
| D5.6 | gold action off the gold path | score by score-increase, store subgoal-derived gold |
| D5.7 | probe families on both environments | decided: ScienceWorld end to end first |
| D5.8 | B3 items and scorer | decided: fixed static sample; own scorer |
| D6.1 | containers | bare uv venvs |
| D8.1 | count cached prompt tokens at full or cached price | compute both; decide after the first full agent |
| D8.2 | recovery probe: one decision or a k-step continuation | k-step, k≈3; keep the one-decision score as a secondary column |

Not in scope unless added later: failure-type labelling of long trajectories (proposal's optional validity
check), thinking-mode variants (declined 2026-10-27), environments beyond ScienceWorld and ALFWorld.
