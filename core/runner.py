"""Run one episode or answer one probe, producing step rows."""

from core.env import Environment, replay
from core.llm import Client, Sampling
from core.probes.compounding import remaining_gold
from core.scaffolds import Scaffold
from core.schema import ProbeItem, StepRow, TaskRow


def _call(client: Client, prompt: list[dict], seed: int, sampling: Sampling, thinking: bool):
    # always sent: Granite 4.2's template defaults enable_thinking to true; templates without the variable ignore it
    extra = {"chat_template_kwargs": {"enable_thinking": thinking}}
    return client.chat(prompt, seed=seed, sampling=sampling, extra=extra)


def run_episode(
    client: Client,
    agent_id: str,
    thinking: bool,
    scaffold: Scaffold,
    env: Environment,
    task: TaskRow,
    seed: int,
    sampling: Sampling,
    run_id: str,
    max_steps: int = 100,
    invalid_streak: int = 5,
) -> list[StepRow]:
    observation = env.reset(task.task_id)
    state = scaffold.start(task.description, env.action_templates())
    rows: list[StepRow] = []
    invalid = 0
    stop = "step_cap"
    for step in range(max_steps):
        prompt = scaffold.prompt(state, observation)
        reply = _call(client, prompt, seed * 100003 + step, sampling, thinking)
        action = scaffold.parse(state, observation, reply.text)
        result = env.step(action)
        rows.append(
            StepRow(
                run_id=run_id,
                agent_id=agent_id,
                kind="episode",
                item_id=task.task_id,
                seed=seed,
                step=step,
                observation=observation,
                prompt=prompt,
                completion=reply.text,
                action=action,
                valid=result.valid,
                reward=result.reward,
                score=result.score,
                done=result.done,
                prompt_tokens=reply.prompt_tokens,
                completion_tokens=reply.completion_tokens,
                cached_tokens=reply.cached_tokens,
                latency_s=reply.latency_s,
                scaffold_state={"reasoning": reply.reasoning} if reply.reasoning else {},
            )
        )
        invalid = 0 if result.valid else invalid + 1
        observation = result.observation
        if result.done:
            stop = "done"
            break
        if invalid >= invalid_streak:
            stop = "invalid_streak"
            break
    rows[-1].meta["stop_reason"] = stop
    return rows


def run_probe(
    client: Client,
    agent_id: str,
    thinking: bool,
    scaffold: Scaffold,
    env: Environment,
    item: ProbeItem,
    seed: int,
    sampling: Sampling,
    run_id: str,
) -> StepRow:
    """Replay the prefix, let the agent act for `meta.free_steps` steps (default 0), then score the next action.

    Only the last step is scored (`meta.scored`), by exact match against the gold actions still untaken and by
    whether the environment score moved. Returns one row per step taken.
    """
    taken = [a for _, a in item.prefix]
    before = replay(env, item.source_task_id, taken)
    state = scaffold.from_prefix(item.task_text, env.action_templates(), item.prefix)
    free = int(item.meta.get("free_steps", 0))
    rows: list[StepRow] = []
    observation = before.observation
    for step in range(free + 1):
        prompt = scaffold.prompt(state, observation)
        reply = _call(client, prompt, seed * 100003 + step, sampling, thinking)
        action = scaffold.parse(state, observation, reply.text)
        after = env.step(action)
        last = step == free
        gold_left = remaining_gold(item.gold_actions, taken[len(item.prefix) :]) if last else []
        rows.append(
            StepRow(
                run_id=run_id,
                agent_id=agent_id,
                kind="probe",
                item_id=item.probe_id,
                seed=seed,
                step=step,
                observation=observation,
                prompt=prompt,
                completion=reply.text,
                action=action,
                valid=after.valid,
                reward=after.reward,
                score=after.score,
                done=after.done,
                prompt_tokens=reply.prompt_tokens,
                completion_tokens=reply.completion_tokens,
                cached_tokens=reply.cached_tokens,
                latency_s=reply.latency_s,
                scaffold_state={"reasoning": reply.reasoning} if reply.reasoning else {},
                meta={
                    "family": item.family,
                    "scored": last,
                    "correct_exact": action in gold_left if last else None,
                    "correct_progress": after.score > before.score if last else None,
                    "score_before": before.score,
                    **item.meta,
                },
            )
        )
        taken.append(action)
        observation, before = after.observation, after
        if after.done:
            break
    return rows
