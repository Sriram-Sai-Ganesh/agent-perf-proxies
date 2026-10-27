"""Task suites are filters over the task catalog, capped per task type so no type dominates."""

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
) -> list[dict]:
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
        by_type[t["task_type"]].append(t)
    out = []
    for task_type in sorted(by_type):
        rows = sorted(by_type[task_type], key=lambda r: r["task_id"])
        if cap_per_type is not None and len(rows) > cap_per_type:
            rows = rng.sample(rows, cap_per_type)
        out.extend(rows)
    return out
