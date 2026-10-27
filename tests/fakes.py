"""A scripted environment and a scripted model, so the runner, probes and store run on CPU in seconds."""

import random
import re

from core.env import StepResult
from core.llm import Reply, Sampling
from core.schema import TaskRow

STEP_RE = re.compile(r"step (\d+)")


class ChainEnv:
    """Task chain/k: the gold path is `step 1` .. `step k`, each correct next step adds 1/k to the score.

    Any valid-looking `step n` out of order is accepted but changes nothing; anything else is rejected.
    """

    name = "chain"

    def __init__(self):
        self.k = 0
        self.at = 0
        self.task_text = ""

    def catalog(self, splits=("train", "test"), task_types=None, max_per_split=None):
        rows = []
        for split, ks in (("train", (4, 6)), ("test", (5, 8))):
            if split not in splits:
                continue
            for k in ks:
                gold = [f"step {i}" for i in range(1, k + 1)]
                rows.append(
                    TaskRow(f"chain/chain/{k}", "chain", "chain", str(k), split, k, gold, f"take {k} steps in order")
                )
        return rows

    def reset(self, task_id):
        self.k = int(task_id.rsplit("/", 1)[1])
        self.at = 0
        self.task_text = f"take {self.k} steps in order"
        return f"You are at step 0 of {self.k}."

    def step(self, action):
        m = STEP_RE.fullmatch(action.strip())
        if not m:
            return StepResult("No known action matches that input.", 0.0, self.at / self.k, False, False, self._valid())
        n = int(m.group(1))
        reward = 0.0
        if n == self.at + 1:
            self.at += 1
            reward = 1 / self.k
        done = self.at == self.k
        return StepResult(
            f"You are at step {self.at} of {self.k}.", reward, self.at / self.k, done, True, self._valid()
        )

    def _valid(self):
        return [f"step {i}" for i in range(1, self.k + 1)]

    def action_templates(self):
        return ["step N"]

    def close(self):
        pass


class ScriptedClient:
    """Replies `Action: step n` for the step the prompt is at, wrong with probability `error_rate`."""

    def __init__(self, error_rate=0.0, seed=0):
        self.error_rate = error_rate
        self.rng = random.Random(seed)
        self.calls = 0

    def chat(self, messages, seed, sampling: Sampling, extra=None):
        self.calls += 1
        last = messages[-1]["content"]
        at = int(re.search(r"step (\d+) of", last).group(1))
        n = at + 1
        if self.rng.random() < self.error_rate:
            n += 2
        text = f"Thought: next.\nAction: step {n}"
        return Reply(
            text,
            prompt_tokens=sum(len(m["content"]) // 4 for m in messages),
            completion_tokens=8,
            cached_tokens=0,
            latency_s=0.0,
        )
