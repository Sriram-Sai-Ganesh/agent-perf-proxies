"""Scaffolds turn an observation history into a prompt and a completion into an action."""

from typing import Protocol


class Scaffold(Protocol):
    name: str

    def start(self, task_text: str, templates: list[str]) -> dict: ...
    def prompt(self, state: dict, observation: str) -> list[dict]: ...
    def parse(self, state: dict, observation: str, completion: str) -> str: ...
    def from_prefix(self, task_text: str, templates: list[str], prefix: list) -> dict: ...


def load_scaffold(name: str, **settings) -> Scaffold:
    from core.scaffolds.react import ReAct

    scaffolds = {"react": ReAct}
    if name not in scaffolds:
        raise SystemExit(f"no scaffold {name!r}; have {sorted(scaffolds)}")
    return scaffolds[name](**settings)
