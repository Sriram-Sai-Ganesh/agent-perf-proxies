import random

from fakes import ChainEnv, ScriptedClient

from core import store
from core.llm import Sampling
from core.probes import compounding, recovery
from core.runner import run_episode, run_probe
from core.scaffolds.react import ReAct, parse_action
from core.schema import ProbeItem, StepRow, from_record, summarize_episode, to_record
from core.suites import select

S = Sampling()


def test_parse_action():
    assert parse_action("Thought: hi\nAction: Open Door.") == "open door"
    assert parse_action("just go north") == "just go north"


def test_episode_reaches_full_score_and_summary():
    env, task = ChainEnv(), ChainEnv().catalog()[0]
    rows = run_episode(ScriptedClient(), "a", False, ReAct(), env, task, 0, S, "r", max_steps=20)
    assert [r.action for r in rows] == task.gold_actions
    ep = summarize_episode(rows)
    assert ep["success"] and ep["final_score"] == 1.0 and ep["stop_reason"] == "done"
    assert rows[-1].prompt[0]["role"] == "system" and rows[-1].prompt_tokens > 0


def test_episode_stops_on_invalid_streak_and_window():
    env, task = ChainEnv(), ChainEnv().catalog()[0]
    rows = run_episode(ScriptedClient(error_rate=1.0), "a", False, ReAct(window=2), env, task, 0, S, "r", max_steps=20)
    assert (
        rows[-1].meta["stop_reason"] == "step_cap" and rows[-1].score == 0.0
    )  # wrong-order steps are valid but useless
    assert "earlier steps omitted" in rows[-1].prompt[1]["content"]


def test_store_roundtrip_and_resume(tmp_path):
    env, task = ChainEnv(), ChainEnv().catalog()[0]
    rows = run_episode(ScriptedClient(), "a", False, ReAct(), env, task, 0, S, "r", max_steps=20)
    path = str(tmp_path / "steps.jsonl")
    store.append(path, rows)
    back = [from_record(StepRow, r) for r in store.read(path)]
    assert back == rows
    assert store.done_keys(path) == {("a", task.task_id, 0)}
    store.write_parquet(store.read(path), str(tmp_path / "steps.parquet"))
    assert len(store.read_parquet(str(tmp_path / "steps.parquet"))) == len(rows)


def test_select_caps_per_type():
    tasks = [to_record(t) for t in ChainEnv().catalog()]
    assert [t["task_id"] for t in select(tasks, "test", min_gold=6)] == ["chain/chain/8"]
    assert len(select(tasks, "train", cap_per_type=1)) == 1


def test_recovery_probe_build_and_score():
    env = ChainEnv()
    items = recovery.build(env, env.catalog(("train",)), per_task=2, rng=random.Random(0))
    assert items and all(it.family == "recovery" for it in items)
    it = items[0]
    assert it.gold_actions == [f"step {it.meta['failed_step'] + 1}"]
    # a perfect agent redoes the failed step: correct by exact match and by score moving
    (row,) = run_probe(ScriptedClient(), "a", False, ReAct(), env, it, 0, S, "r")
    assert row.meta["scored"] and row.meta["correct_exact"] and row.meta["correct_progress"]
    back = from_record(ProbeItem, to_record(it))
    assert back == it


def test_compounding_probe_free_steps():
    env = ChainEnv()
    items = compounding.build(env, env.catalog(("train",)), per_task=1, depths=(0, 2), rng=random.Random(0))
    assert {it.meta["depth"] for it in items} == {0, 2} and all(it.gold_actions for it in items)
    deep = next(it for it in items if it.meta["depth"] == 2)
    n = deep.meta["prefix_len"]
    # a perfect agent: two free gold steps, then the scored step is gold too
    rows = run_probe(ScriptedClient(), "a", False, ReAct(), env, deep, 0, S, "r")
    assert [r.action for r in rows] == [f"step {n + 1}", f"step {n + 2}", f"step {n + 3}"]
    assert [r.meta["scored"] for r in rows] == [False, False, True] and rows[-1].meta["correct_exact"]
    # an always-wrong agent: its free steps add no progress and its last step is wrong
    rows = run_probe(ScriptedClient(error_rate=1.0), "a", False, ReAct(), env, deep, 0, S, "r")
    assert rows[-1].meta["correct_exact"] is False and rows[-1].meta["correct_progress"] is False


def test_probe_continuation_scores_progress_after_the_scored_step():
    env = ChainEnv()
    (item,) = recovery.build(env, env.catalog(("train",))[:1], per_task=1, rng=random.Random(0))
    # wrong twice (the scored step and the first continuation step), then right: no progress at the scored step,
    # progress within the continuation
    rows = run_probe(ScriptedClient(error_rate=1.0), "a", False, ReAct(), env, item, 0, S, "r", continuation=2)
    assert rows[0].meta["scored"] and rows[0].meta["continuation"] == 2
    assert rows[0].meta["correct_progress"] is False and rows[0].meta["correct_progress_within"] is False
    rows = run_probe(ScriptedClient(), "a", False, ReAct(), env, item, 0, S, "r", continuation=2)
    assert len(rows) <= 3 and rows[0].meta["correct_progress"] and rows[0].meta["correct_progress_within"]
    assert all(not r.meta["scored"] for r in rows[1:])
