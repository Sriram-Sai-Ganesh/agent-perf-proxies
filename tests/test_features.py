import json
import random

import pytest
from fakes import ChainEnv, ScriptedClient

from analysis.features import (
    b1_features,
    b2_features,
    b3_random_features,
    cutoff_scores,
    draw_under_budget,
    full_cost,
    gold_reference,
    long_episodes_by_seed,
    random_question_ids,
    static_results,
)
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


def long_step(agent_id: str, item_id: str, seed: int, tokens: int, score: float) -> dict:
    """One-step long episode that reached ``score`` and cost ``tokens``."""
    row = StepRow("r", agent_id, "episode", item_id, seed, 0, "", [], "", "", True, 0.0, score, True, tokens, 0, 0, 0.0)
    return to_record(row)


def long_runs() -> list[dict]:
    """Agent a: seeds 0, 1 and 2, four episodes of 10 tokens each, every episode of seed s reaching progress s / 10."""
    return [long_step("a", f"t{i}", s, 10, s / 10) for s in (0, 1, 2) for i in range(4)]


def test_long_episodes_by_seed_gives_progress_and_tokens_per_agent_and_seed():
    rows = long_runs() + [long_step("b", "t0", 5, 7, 0.4), truth_step("a", "p1", 0, 999, kind="probe")]
    by_seed = long_episodes_by_seed(rows)
    assert by_seed["a"][1] == [(0.1, 10)] * 4
    assert by_seed["b"] == {5: [(0.4, 7)]}
    assert set(by_seed["a"]) == {0, 1, 2}


def test_b1_one_row_per_agent_beta_episode_seed_and_draw_never_over_budget():
    runs = long_runs()
    rows = b1_features(long_episodes_by_seed(runs), full_cost(runs), betas=[0.5, 1.0], repeats=3, seed=7)
    assert full_cost(runs) == {"a": 40.0}
    assert len(rows) == 2 * 3 * 3
    assert {r["method"] for r in rows} == {"B1"}
    assert {"agent_id", "beta", "episode_seed", "draw", "feature", "n_items", "tokens_spent", "budget_tokens"} <= set(
        rows[0]
    )
    assert {(r["beta"], r["episode_seed"], r["draw"]) for r in rows} == {
        (beta, s, d) for beta in (0.5, 1.0) for s in (0, 1, 2) for d in range(3)
    }
    assert all(r["tokens_spent"] <= r["budget_tokens"] for r in rows)
    assert {r["n_items"] for r in rows if r["beta"] == 0.5} == {2}
    assert {r["n_items"] for r in rows if r["beta"] == 1.0} == {4}


def test_b1_draws_only_use_episodes_of_their_own_seed_and_repeat_for_the_same_seed():
    runs = long_runs()
    args = (long_episodes_by_seed(runs), full_cost(runs), [0.5])
    rows = b1_features(*args, repeats=5, seed=3)
    assert all(r["feature"] == pytest.approx(r["episode_seed"] / 10) for r in rows)
    assert rows == b1_features(*args, repeats=5, seed=3)


def test_b1_draws_differ_between_episode_seeds_and_agents():
    by_seed = {"a": {s: [(i / 10, 10) for i in range(8)] for s in (0, 1)}}
    rows = b1_features(by_seed, {"a": 80.0}, betas=[0.25], repeats=20, seed=0)
    features = lambda s: [r["feature"] for r in rows if r["episode_seed"] == s]
    assert features(0) != features(1)
    assert len(set(features(0))) > 1


def test_b1_feature_is_none_when_nothing_fits():
    runs = long_runs()
    rows = b1_features(long_episodes_by_seed(runs), full_cost(runs), betas=[0.1], repeats=2, seed=0)  # budget 4 < 10
    assert all(r["feature"] is None and r["n_items"] == 0 and r["tokens_spent"] == 0 for r in rows)
    assert all(r["budget_tokens"] == pytest.approx(4.0) for r in rows)


def static_step(agent_id: str, item_id: str, correct: bool, tokens: int = 10, meta_as_text: bool = True) -> dict:
    """A static question row: ``tokens`` is split into 6 + the rest of prompt and completion tokens."""
    meta = {"source": "gsm8k", "subset": "main", "correct": correct}
    row = StepRow("r", agent_id, "static", item_id, 0, 0, "", [], "", "", True, 0.0, 0.0, True, 6, tokens - 6, 0, 0.0)
    record = to_record(row)
    record["meta"] = json.dumps(meta) if meta_as_text else meta
    return record


def test_static_results_one_row_per_agent_and_question_with_tokens_and_ignoring_other_kinds():
    rows = [static_step("a", "q1", True), static_step("a", "q2", False, tokens=20), static_step("b", "q1", False)]
    rows += [static_step("b", "q2", True, meta_as_text=False), truth_step("a", "t1", 0, 5)]
    results = static_results(rows)
    assert len(results) == 4
    assert results[1] == {
        "agent_id": "a",
        "item_id": "q2",
        "source": "gsm8k",
        "subset": "main",
        "correct": False,
        "tokens": 20,
    }
    assert [r["correct"] for r in results] == [True, False, False, True]


POOL = [f"q{i:02d}" for i in range(20)]


def answers(agent_id: str, right: set[str], tokens: int = 15) -> list[dict]:
    return [static_step(agent_id, q, q in right, tokens) for q in POOL]


def test_random_question_ids_are_the_same_for_the_same_pool_and_seed_in_any_order():
    ids = random_question_ids(POOL, 5, seed=0)
    assert len(set(ids)) == 5 and set(ids) <= set(POOL)
    assert ids == random_question_ids(list(reversed(POOL)), 5, seed=0)
    assert ids != random_question_ids(POOL, 5, seed=1)


def test_b3_random_control_uses_the_same_questions_for_every_agent():
    chosen = set(random_question_ids(POOL, 5, seed=0))
    results = static_results(
        answers("on_chosen", chosen) + answers("off_chosen", set(POOL) - chosen) + answers("all_right", set(POOL))
    )
    c_full = {"on_chosen": 1000.0, "off_chosen": 1000.0, "all_right": 1000.0}
    rows = b3_random_features(results, POOL, c_full, betas=[0.1], n=5, seed=0)
    assert {r["agent_id"]: r["feature"] for r in rows} == {"on_chosen": 1.0, "off_chosen": 0.0, "all_right": 1.0}
    assert {(r["method"], r["draw"], r["n_items"], r["tokens_spent"]) for r in rows} == {("B3-random", 0, 5, 75)}
    assert rows == b3_random_features(results, POOL, c_full, betas=[0.1], n=5, seed=0)


def test_b3_random_control_is_none_when_over_budget_and_one_row_per_agent_and_beta():
    results = static_results(answers("a", set(POOL)))
    rows = b3_random_features(results, POOL, {"a": 1000.0}, betas=[0.05, 0.075, 0.1], n=5, seed=0)  # costs 75 tokens
    assert [(r["beta"], r["feature"]) for r in rows] == [(0.05, None), (0.075, 1.0), (0.1, 1.0)]
    assert rows[0]["budget_tokens"] == 50.0 and rows[0]["tokens_spent"] == 75 and rows[0]["c_full"] == 1000.0


def test_b3_random_control_says_how_many_questions_are_missing_and_uses_the_answered_ones(capsys):
    chosen = random_question_ids(POOL, 5, seed=0)
    partial = [static_step("a", q, q == chosen[0]) for q in chosen[:2]]
    rows = b3_random_features(static_results(partial), POOL, {"a": 1000.0}, betas=[0.1], n=5, seed=0)
    assert "3 of the 5 random questions have no answer" in capsys.readouterr().out
    assert rows[0]["feature"] == 0.5 and rows[0]["n_items"] == 2 and rows[0]["tokens_spent"] == 20
