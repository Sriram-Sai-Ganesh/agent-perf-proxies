"""ReAct: a system prompt with the action grammar, the observation history as turns, `Thought: ... Action: ...` output."""

import re

SYSTEM = """You are an agent in a text-based environment. At each turn you receive an observation and reply with one
thought and one action, in exactly this format:

Thought: <one or two sentences of reasoning>
Action: <one action>

Actions follow these templates, where OBJ is an object name as it appears in observations:
{templates}

Reply with one action per turn and nothing after the Action line."""

ACTION_RE = re.compile(r"Action:\s*(.+)", re.IGNORECASE)


class ReAct:
    name = "react"

    def __init__(self, window: int = 40):
        self.window = window  # how many past turns the prompt keeps

    def start(self, task_text: str, templates: list[str]) -> dict:
        return {"system": SYSTEM.format(templates="\n".join(templates)), "task": task_text, "history": []}

    def prompt(self, state: dict, observation: str) -> list[dict]:
        history = state["history"]
        msgs = [{"role": "system", "content": state["system"]}]
        kept = history[-self.window :]
        first = f"Task: {state['task']}\n\nObservation: " + (kept[0][0] if kept else observation)
        if len(kept) < len(history):
            first += f"\n\n({len(history) - len(kept)} earlier steps omitted)"
        if not kept:
            return msgs + [{"role": "user", "content": first}]
        msgs.append({"role": "user", "content": first})
        for i, (obs, completion) in enumerate(kept):
            msgs.append({"role": "assistant", "content": completion})
            nxt = kept[i + 1][0] if i + 1 < len(kept) else observation
            msgs.append({"role": "user", "content": f"Observation: {nxt}"})
        return msgs

    def parse(self, state: dict, observation: str, completion: str) -> str:
        state["history"].append((observation, completion))
        return parse_action(completion)

    def from_prefix(self, task_text: str, templates: list[str], prefix: list) -> dict:
        state = self.start(task_text, templates)
        state["history"] = [(obs, f"Action: {action}") for obs, action in prefix]
        return state


def parse_action(completion: str) -> str:
    """The last `Action:` line, else the last non-empty line; lower-cased, stripped of trailing punctuation."""
    found = ACTION_RE.findall(completion)
    text = found[-1] if found else (completion.strip().splitlines() or [""])[-1]
    return text.strip().strip("`*\"'.").lower()
