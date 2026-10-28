"""ScienceWorld adapter. One JVM per instance, launched with the JRE that jdk4py bundles."""

import os

import jdk4py

os.environ["PATH"] = str(jdk4py.JAVA.parent) + os.pathsep + os.environ["PATH"]  # scienceworld runs `java` from PATH
# no /tmp/hsperfdata_<user>/<pid> file: with many JVMs on a shared node a locked one makes the JVM print a warning
# on stdout, where py4j expects only the port number, and the environment fails to start
os.environ.setdefault("JAVA_TOOL_OPTIONS", "-XX:-UsePerfData")

from scienceworld import ScienceWorldEnv

from core.env import StepResult
from core.schema import TaskRow

REJECTED = "No known action matches that input."


class ScienceWorld:
    name = "scienceworld"

    def __init__(self, simplification: str = "easy", step_limit: int = 100):
        self.simplification = simplification
        self.env = ScienceWorldEnv("", envStepLimit=step_limit)
        self.task_text = ""
        self._templates = None

    def catalog(self, splits=("train", "dev", "test"), task_types=None, max_per_split=None) -> list[TaskRow]:
        rows = []
        for task_type in task_types or self.env.get_task_names():
            self.env.load(task_type, 0, self.simplification, generateGoldPath=False)
            for split in splits:
                variations = getattr(self.env, f"get_variations_{split}")()[:max_per_split]
                for v in variations:
                    self.env.load(task_type, v, self.simplification, generateGoldPath=True)
                    gold = list(self.env.get_gold_action_sequence())
                    rows.append(
                        TaskRow(
                            task_id=f"scienceworld/{task_type}/{v}",
                            env="scienceworld",
                            task_type=task_type,
                            variation=str(v),
                            split=split,
                            gold_len=len(gold),
                            gold_actions=gold,
                            description=self.env.get_task_description(),
                            meta={"simplification": self.simplification},
                        )
                    )
        return rows

    def reset(self, task_id: str) -> str:
        _, task_type, variation = task_id.split("/")
        self.env.load(task_type, int(variation), self.simplification, generateGoldPath=False)
        obs, info = self.env.reset()
        self.task_text = info["taskDesc"]
        return obs

    def step(self, action: str) -> StepResult:
        obs, reward, done, info = self.env.step(action)
        score = min(max(info["score"] / 100.0, 0.0), 1.0)  # -100 marks a failed task; it counts as 0 and done
        return StepResult(
            observation=obs,
            reward=float(reward) / 100.0,
            score=score,
            done=bool(done) or info["score"] < 0,
            valid=bool(action.strip()) and obs.strip() != REJECTED,  # ScienceWorld accepts an empty action
            valid_actions=list(info.get("valid") or []),
        )

    def action_templates(self) -> list[str]:
        if self._templates is None:
            self._templates = list(self.env.get_possible_actions())
        return self._templates

    def close(self) -> None:
        self.env.close()


def make(**settings) -> ScienceWorld:
    return ScienceWorld(**settings)
