"""Row schemas for every table. One dataclass per table; list and dict fields are stored as JSON strings in Parquet."""

import json
from dataclasses import asdict, dataclass, field, fields
from typing import get_origin


@dataclass
class TaskRow:
    """One task variation of one environment, with its gold trajectory."""

    task_id: str  # {env}/{task_type}/{variation}
    env: str
    task_type: str
    variation: str
    split: str  # train | dev | test
    gold_len: int
    gold_actions: list[str]
    description: str  # the task text the agent is given
    meta: dict = field(default_factory=dict)


@dataclass
class StepRow:
    """One agent action: a step of an episode, a probe answer, or the answer to a plain test question."""

    run_id: str
    agent_id: str
    kind: str  # episode | probe | static
    item_id: str  # task_id for episodes, probe_id for probes, item_id for static
    seed: int
    step: int
    observation: str
    prompt: list  # [{role, content}] as sent
    completion: str
    action: str
    valid: bool
    reward: float
    score: float  # cumulative environment score in [0, 1] after this action
    done: bool
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    latency_s: float
    scaffold_state: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


@dataclass
class ProbeItem:
    """A replayed prefix and the next action that counts as correct."""

    probe_id: str  # {family}/{source_task_id}/{index}
    family: str  # compounding | state | recovery | deferred
    env: str
    source_task_id: str
    source_agent_id: str | None
    task_text: str
    prefix: list  # [[observation, action], ...] to replay, in order
    final_observation: str  # what the agent sees when asked for the next action
    gold_actions: list[str]  # any one counts as correct
    distractors: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


@dataclass
class StaticItem:
    """An ordinary test question with a known answer, asked with no scaffold and no environment."""

    item_id: str  # mmlu_pro/{question_id} | gsm8k/{row index} | bbh/{task}/{row index}
    source: str  # mmlu_pro | gsm8k | bbh
    subset: str  # MMLU-Pro category, BBH task, or "main" for gsm8k
    question: str
    choices: list[str]  # the answer options in order; empty unless the question is multiple choice
    answer: str  # a letter A-J, a number as a string, or the target text, by answer_type
    answer_type: str  # letter | number | text
    meta: dict = field(default_factory=dict)


JSON_TYPES = (list, dict)


def to_record(row) -> dict:
    """Flatten a row for JSONL or Parquet: list and dict fields become JSON strings."""
    out = {}
    for k, v in asdict(row).items():
        out[k] = json.dumps(v) if isinstance(v, JSON_TYPES) else v
    return out


def from_record(cls, record: dict):
    """Inverse of ``to_record`` for the given row class."""
    kwargs = {}
    for f in fields(cls):
        v = record[f.name]
        if isinstance(v, str) and (f.type in JSON_TYPES or get_origin(f.type) in JSON_TYPES):
            v = json.loads(v)
        kwargs[f.name] = v
    return cls(**kwargs)


def summarize_episode(steps: list[StepRow]) -> dict:
    """One episode row from its steps: success, final score, progress (max score), tokens, stop reason."""
    last = steps[-1]
    return {
        "run_id": last.run_id,
        "agent_id": last.agent_id,
        "item_id": last.item_id,
        "seed": last.seed,
        "n_steps": len(steps),
        "success": bool(last.done and last.score >= 1.0),
        "final_score": last.score,
        "progress": max(s.score for s in steps),
        "tokens": sum(s.prompt_tokens + s.completion_tokens for s in steps),
        "stop_reason": last.meta.get("stop_reason", "done" if last.done else "unknown"),
    }
