"""The interface every environment adapter implements, and how one is loaded by name."""

import importlib
from dataclasses import dataclass
from typing import Protocol

from core.schema import TaskRow


@dataclass
class StepResult:
    observation: str
    reward: float
    score: float  # cumulative, in [0, 1]
    done: bool
    valid: bool  # the environment accepted the action
    valid_actions: list[str] | None = None  # for probe distractors; None if the environment cannot say


class Environment(Protocol):
    name: str
    task_text: str  # the task description of the task last reset

    def catalog(
        self, splits: tuple[str, ...], task_types: list[str] | None, max_per_split: int | None
    ) -> list[TaskRow]: ...
    def reset(self, task_id: str) -> str: ...
    def step(self, action: str) -> StepResult: ...
    def action_templates(self) -> list[str]: ...
    def close(self) -> None: ...


def replay(env: Environment, task_id: str, actions: list[str]) -> StepResult:
    """Reset then step through a prefix; returns the state after the last action (or the reset state)."""
    obs = env.reset(task_id)
    last = StepResult(obs, 0.0, 0.0, False, True, None)
    for a in actions:
        last = env.step(a)
    return last


def load_environment(name: str, settings: dict) -> Environment:
    """Import ``env_<name>.adapter`` from the environment's own venv and build it with its settings."""
    module = importlib.import_module(f"env_{name}.adapter")
    return module.make(**settings)
