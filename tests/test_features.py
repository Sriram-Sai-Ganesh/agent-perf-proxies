import random

import pytest
from fakes import ChainEnv, ScriptedClient

from analysis.features import b2_features, cutoff_scores, draw_under_budget, full_cost, gold_reference
from core.llm import Sampling
from core.runner import run_episode
from core.scaffolds.react import ReAct
from core.schema import StepRow, TaskRow, to_record

GOLD = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def catalog_record(task_id: str, meta: dict) -> dict:
    return to_record(TaskRow(task_id, "chain", "chain", "0", "dev", 5, [], "take steps", meta))


def short_episode(task_id: str, max_steps: int, agent_id: str = "a", seed: int = 0) -> list[dict]:
    """Steps of a perfect agent on a chain task, cut off after ``max_steps``."""
    task = ChainEnv().catalog()[0]
    task.task_id = task_id
    rows = run_episode(ScriptedClient(), agent_id, False, ReAct(), ChainEnv(), task, seed, Sampling(), "r", max_steps)
    return [to_record(r) for r in rows]


def truth_step(agent_id: str, item_id: str, seed: int, tokens: int, kind: str = "episode") -> dict:
    row = StepRow("r", agent_id, kind, item_id, seed, 0, "", [], "", "", True, 0.0, 0.0, False, tokens, 0, 0, 0.0)
    return to_record(row)


def test_gold_reference_uses_last_score_for_short_paths():
    assert gold_reference({"gold_scores": GOLD}, 3) == 0.2 + 0.2
    assert gold_reference({"gold_scores": [0.5, 1.0]}, 15) == 1.0
    assert gold_reference({}, 15) is None
    assert gold_reference({"gold_scores": []}, 15) is None


def test_cutoff_score_is_progress_over_gold_and_capped():
    steps = short_episode("chain/chain/5", max_steps=3)  # progress 3/5
    tasks = [catalog_record("chain/chain/5", {"gold_scores": GOLD})]
    scored, skipped = cutoff_scores(steps, tasks, ref_step=5)  # gold 0.8
    assert scored["a"][0][0] == pytest.approx(0.6 / 0.8)
    assert skipped == {}
    scored, _ = cutoff_scores(steps, tasks, ref_step=3)  # gold 0.4, below progress: capped
    assert scored["a"][0][0] == 1.0
    scored, _ = cutoff_scores(steps, tasks, ref_step=15)  # past the end of the gold path: last value 1.0
    assert scored["a"][0][0] == pytest.approx(0.6)


def test_cutoff_scores_count_skipped_episodes():
    steps = short_episode("chain/chain/5", 3) + short_episode("chain/chain/6", 3) + short_episode("chain/chain/7", 3)
    steps += short_episode("chain/chain/8", 3) + short_episode("chain/chain/9", 3)
    tasks = [
        catalog_record("chain/chain/5", {"gold_scores": GOLD}),
        catalog_record("chain/chain/6", {}),
        catalog_record("chain/chain/7", {"gold_scores": [0.0, 0.0]}),
        catalog_record("chain/chain/8", {"gold_scores": [0.0, 0.0]}),
    ]
    scored, skipped = cutoff_scores(steps, tasks, ref_step=15)
    assert len(scored["a"]) == 1
    assert skipped == {
        "task not in catalog": 1,
        "task has no gold scores": 1,
        "gold score at the reference step is 0": 2,
    }


def test_full_cost_is_mean_tokens_per_seed_over_episode_rows():
    rows = [truth_step("a", "t1", 0, 100), truth_step("a", "t2", 0, 50), truth_step("a", "t1", 1, 150)]
    rows += [truth_step("a", "p1", 0, 999, kind="probe"), truth_step("b", "t1", 3, 40)]
    assert full_cost(rows) == {"a": 150.0, "b": 40.0}


def test_draw_never_passes_budget_and_stops_at_first_that_does_not_fit():
    episodes = [(1.0, 30), (0.0, 5)]
    outcomes = {draw_under_budget(episodes, 10, random.Random(i)) for i in range(30)}
    # when the big episode comes first the draw ends empty, it does not skip ahead to the small one
    assert outcomes == {(None, 0, 0), (0.0, 1, 5)}
    sizes = [(0.5, n) for n in (3, 8, 5, 12, 7, 1)]
    for i in range(30):
        _, n_taken, spent = draw_under_budget(sizes, 15, random.Random(i))
        assert spent <= 15 and n_taken <= len(sizes)


def test_draw_is_none_when_nothing_fits_and_repeats_for_the_same_seed():
    assert draw_under_budget([(1.0, 30)], 10, random.Random(0)) == (None, 0, 0)
    episodes = [(i / 10, 10 + i) for i in range(10)]
    assert draw_under_budget(episodes, 45, random.Random("0/a/0.1/3")) == draw_under_budget(
        episodes, 45, random.Random("0/a/0.1/3")
    )


def test_b2_features_one_row_per_agent_beta_and_draw_and_same_for_any_agent_order():
    scored = {"a": [(1.0, 10), (0.5, 10), (0.0, 10)], "b": [(0.2, 5), (0.4, 5)]}
    c_full = {"a": 300.0, "b": 100.0}
    rows = b2_features(scored, c_full, betas=[0.05, 0.1], repeats=3, seed=7)
    assert len(rows) == 2 * 2 * 3
    assert {r["method"] for r in rows} == {"B2"}
    assert {"agent_id", "beta", "draw", "feature", "n_items", "tokens_spent", "budget_tokens", "c_full"} <= set(rows[0])
    row = next(r for r in rows if r["agent_id"] == "b" and r["beta"] == 0.05)
    assert row["budget_tokens"] == 5.0 and row["feature"] in (None, 0.2, 0.4)
    assert all(r["tokens_spent"] <= r["budget_tokens"] for r in rows)
    flipped = b2_features({"b": scored["b"], "a": scored["a"]}, c_full, betas=[0.05, 0.1], repeats=3, seed=7)
    key = lambda r: (r["agent_id"], r["beta"], r["draw"])
    assert sorted(rows, key=key) == sorted(flipped, key=key)
