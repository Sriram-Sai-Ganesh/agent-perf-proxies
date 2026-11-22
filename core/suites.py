"""Task suites are filters over the task catalog, capped per task type so no type dominates."""

import json
import random
from collections import defaultdict


def select(
    tasks: list[dict],
    split: str,
    min_gold: int | None = None,
    max_gold: int | None = None,
    task_types: list[str] | None = None,
    cap_per_type: int | None = None,
    seed: int = 0,
    ref_step: int | None = None,
    min_ref_score: float | None = None,
) -> list[dict]:
    """Tasks in `split`, optionally kept by gold length (min_gold, max_gold) and task type (task_types).

    With ref_step and min_ref_score, tasks whose gold path scores below min_ref_score after ref_step steps
    (or after its last step, if it is shorter) are dropped. At most cap_per_type tasks per type are kept,
    chosen at random with `seed` after the other filters.
    """
    rng = random.Random(seed)
    by_type = defaultdict(list)
    for t in tasks:
        if t["split"] != split:
            continue
        if min_gold is not None and t["gold_len"] < min_gold:
            continue
        if max_gold is not None and t["gold_len"] > max_gold:
            continue
        if task_types and t["task_type"] not in task_types:
            continue
        if ref_step is not None and min_ref_score is not None and _gold_score_at(t, ref_step) < min_ref_score:
            continue
        by_type[t["task_type"]].append(t)
    out = []
    for task_type in sorted(by_type):
        rows = sorted(by_type[task_type], key=lambda r: r["task_id"])
        if cap_per_type is not None and len(rows) > cap_per_type:
            rows = rng.sample(rows, cap_per_type)
        out.extend(rows)
    return out


def _gold_score_at(task: dict, step: int) -> float:
    """The gold path's score after `step` steps, or after its last step if it is shorter."""
    meta = task["meta"]
    if isinstance(meta, str):
        meta = json.loads(meta)
    if "gold_scores" not in meta:
        raise SystemExit(
            "The task catalog predates gold_scores; rebuild it with env_scienceworld.catalog to filter by gold score."
        )
    scores = meta["gold_scores"]
    return float(scores[min(step, len(scores)) - 1]) if len(scores) else 0.0
