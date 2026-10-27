"""Probe builders: each turns tasks (and, for compounding, an agent's own trajectories) into ProbeItems."""

from core.env import Environment
from core.schema import TaskRow


def walk(env: Environment, task: TaskRow, actions: list[str]) -> tuple[list[list[str]], str, float, list[str] | None]:
    """Step through actions from reset. Returns the prefix pairs, the final observation, the score, and valid actions."""
    obs = env.reset(task.task_id)
    pairs, score, valid = [], 0.0, None
    for a in actions:
        res = env.step(a)
        pairs.append([obs, a])
        obs, score, valid = res.observation, res.score, res.valid_actions
    return pairs, obs, score, valid
