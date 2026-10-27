"""Family (i), compounding errors: a gold prefix, then the agent acts freely for `d` steps, then one scored decision.

The free steps let the agent's own errors accumulate in its context at run time. The matched control is the same
item with d = 0, so the feature is accuracy at depth d minus accuracy at depth 0. Built without a model, so every
agent answers the same items.
"""

import random
from collections import Counter

from core.env import Environment
from core.probes import walk
from core.schema import ProbeItem, TaskRow


def remaining_gold(gold: list[str], taken: list[str]) -> list[str]:
    """Gold actions not yet taken (multiset difference), in gold order."""
    # shortcut: "not yet taken" ignores that an action may need redoing after a failure; revisit with D5.6
    left = Counter(gold) - Counter(taken)
    out = []
    for a in gold:
        if left[a] > 0:
            out.append(a)
            left[a] -= 1
    return out


def build(
    env: Environment,
    tasks: list[TaskRow],
    per_task: int,
    depths: tuple[int, ...],
    rng: random.Random,
) -> list[ProbeItem]:
    items = []
    for task in tasks:
        gold = task.gold_actions
        max_d = max(depths)
        if len(gold) < max_d + 4:
            continue
        for n in rng.sample(range(2, len(gold) - max_d - 1), min(per_task, len(gold) - max_d - 3)):
            pairs, final_obs, score, valid = walk(env, task, gold[:n])
            for d in depths:
                items.append(
                    ProbeItem(
                        probe_id=f"compounding/{task.task_id}/{n}/d{d}",
                        family="compounding",
                        env=task.env,
                        source_task_id=task.task_id,
                        source_agent_id=None,
                        task_text=task.description,
                        prefix=pairs,
                        final_observation=final_obs,
                        gold_actions=gold[n:],  # the runner narrows this to what is still untaken after the free steps
                        distractors=rng.sample(valid, min(8, len(valid))) if valid else [],
                        meta={"prefix_len": n, "depth": d, "free_steps": d, "control": d == 0, "score_before": score},
                    )
                )
    return items
