"""Family (i), compounding errors: the agent's own prefix, continued d steps past its first deviation from gold.

Each item has a matched control: the gold prefix of the same length. The feature is the accuracy gap.
"""

import random
from collections import Counter

from core.env import Environment
from core.probes import walk
from core.schema import ProbeItem, TaskRow


def remaining_gold(gold: list[str], taken: list[str]) -> list[str]:
    """Gold actions not yet taken in the prefix (multiset difference), in gold order."""
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
    tasks: dict[str, TaskRow],
    episodes: dict[str, list[dict]],
    agent_id: str,
    depths: tuple[int, ...],
    rng: random.Random,
) -> list[ProbeItem]:
    """``episodes`` maps task_id to that agent's step records (one episode), in step order."""
    items = []
    for task_id, steps in episodes.items():
        task = tasks[task_id]
        gold = task.gold_actions
        own = [s["action"] for s in steps]
        first_error = next((t for t, a in enumerate(own) if t >= len(gold) or a != gold[t]), None)
        if first_error is None or first_error == 0:
            continue
        for d in depths:
            n = first_error + d + 1  # prefix length: everything up to and including d steps after the first error
            if n >= len(own) or n >= len(gold):
                continue
            pairs, final_obs, score, valid = walk(env, task, own[:n])
            gold_left = remaining_gold(gold, own[:n])
            if not gold_left:
                continue
            common = {"env": task.env, "source_task_id": task_id, "task_text": task.description}
            items.append(
                ProbeItem(
                    probe_id=f"compounding/{task_id}/{agent_id}/d{d}",
                    family="compounding",
                    source_agent_id=agent_id,
                    prefix=pairs,
                    final_observation=final_obs,
                    gold_actions=gold_left,
                    distractors=rng.sample(valid, min(8, len(valid))) if valid else [],
                    meta={"first_error": first_error, "depth": d, "control": False, "score_before": score},
                    **common,
                )
            )
            cpairs, cobs, cscore, cvalid = walk(env, task, gold[:n])
            items.append(
                ProbeItem(
                    probe_id=f"compounding/{task_id}/{agent_id}/d{d}/control",
                    family="compounding",
                    source_agent_id=agent_id,
                    prefix=cpairs,
                    final_observation=cobs,
                    gold_actions=[gold[n]],
                    distractors=rng.sample(cvalid, min(8, len(cvalid))) if cvalid else [],
                    meta={"first_error": first_error, "depth": d, "control": True, "score_before": cscore},
                    **common,
                )
            )
    return items
