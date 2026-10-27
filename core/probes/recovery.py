"""Family (iii), error recovery: a gold prefix with one step replaced by a failing action of the same verb."""

import random

from core.env import Environment
from core.probes import walk
from core.schema import ProbeItem, TaskRow

# shortcut: a failure is "silent" when the observation does not say the action failed; refine per task type if the
# pilot shows most candidates are rejected outright
REJECTED = ("no known action", "can't", "cannot", "not possible", "nothing happens")


def build(env: Environment, tasks: list[TaskRow], per_task: int, rng: random.Random, after: int = 0) -> list[ProbeItem]:
    items = []
    for task in tasks:
        gold = task.gold_actions
        if len(gold) < 3:
            continue
        positions = rng.sample(range(1, len(gold) - 1), min(per_task * 3, len(gold) - 2))  # extra, as some are skipped
        made = 0
        for t in positions:
            if made >= per_task:
                break
            _, _, _, valid = walk(env, task, gold[:t])
            verb = gold[t].split()[0]
            wrong = [v for v in (valid or []) if v.split()[0] == verb and v != gold[t]]
            if not wrong:
                continue
            failing = rng.choice(wrong)
            actions = gold[:t] + [failing] + gold[t + 1 : t + 1 + after]
            pairs, final_obs, score, valid_after = walk(env, task, actions)
            fail_obs = final_obs if after == 0 else pairs[t + 1][0]  # what the environment said to the failing action
            if any(k in fail_obs.lower() for k in REJECTED):
                continue
            items.append(
                ProbeItem(
                    probe_id=f"recovery/{task.task_id}/{t}",
                    family="recovery",
                    env=task.env,
                    source_task_id=task.task_id,
                    source_agent_id=None,
                    task_text=task.description,
                    prefix=pairs,
                    final_observation=final_obs,
                    gold_actions=[gold[t]],
                    distractors=rng.sample(valid_after, min(8, len(valid_after))) if valid_after else [],
                    meta={
                        "failed_step": t,
                        "failing_action": failing,
                        "steps_since_failure": after,
                        "score_before": score,
                    },
                )
            )
            made += 1
    return items
