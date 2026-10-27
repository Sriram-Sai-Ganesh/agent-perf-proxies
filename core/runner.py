"""Run one episode or answer one probe, producing step rows."""

from core.env import Environment, replay
from core.llm import Client, Sampling
from core.scaffolds import Scaffold
from core.schema import ProbeItem, StepRow, TaskRow


def _call(client: Client, prompt: list[dict], seed: int, sampling: Sampling, thinking: bool):
    extra = {"chat_template_kwargs": {"thinking": thinking}} if thinking else None
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
    """Replay the prefix, ask for one action, score it by exact match and by whether the score moved."""
    before = replay(env, item.source_task_id, [a for _, a in item.prefix])
    state = scaffold.from_prefix(item.task_text, env.action_templates(), item.prefix)
    prompt = scaffold.prompt(state, before.observation)
    reply = _call(client, prompt, seed * 100003, sampling, thinking)
    action = scaffold.parse(state, before.observation, reply.text)
    after = env.step(action)
    return StepRow(
        run_id=run_id,
        agent_id=agent_id,
        kind="probe",
        item_id=item.probe_id,
        seed=seed,
        step=0,
        observation=before.observation,
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
            "correct_exact": action in item.gold_actions,
            "correct_progress": after.score > before.score,
            "score_before": before.score,
            **item.meta,
        },
    )
