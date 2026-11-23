import random

import pytest

from analysis.pace import accuracy_on, choose_questions

AGENTS = [f"agent{i}" for i in range(8)]
SCORES = {agent: i / 10 for i, agent in enumerate(AGENTS)}


def table() -> dict[str, dict[str, bool]]:
    """Questions: follows_scores is right for the better half of the agents, the always_* ones never differ."""
    return {
        agent: {
            "follows_scores": i >= 4,
            "alternates": i % 2 == 0,
            "only_best_miss": i != 7,
            "always_right": True,
            "always_wrong": False,
        }
        for i, agent in enumerate(AGENTS)
    }


def random_table(seed: int, agents: list[str], questions: int = 30) -> dict[str, dict[str, bool]]:
    rng = random.Random(seed)
    return {agent: {f"q{k:02d}": rng.random() < 0.5 for k in range(questions)} for agent in agents}


def test_question_that_follows_the_scores_ranks_first():
    assert choose_questions(table(), SCORES, n=3)[0] == "follows_scores"


def test_questions_everyone_gets_right_or_wrong_are_never_chosen_first():
    ranked = choose_questions(table(), SCORES, n=10)
    assert len(ranked) == 5
    assert set(ranked[-2:]) == {"always_right", "always_wrong"}
    assert ranked[-2:] == ["always_right", "always_wrong"]  # tied at zero, broken by question id


def test_fewer_questions_than_asked_for_gives_all_of_them_and_n_limits_the_result():
    assert len(choose_questions(table(), SCORES, n=100)) == 5
    assert len(choose_questions(table(), SCORES, n=2)) == 2


def test_only_questions_every_scored_agent_answered_are_used():
    answers = table()
    del answers["agent3"]["alternates"]
    assert "alternates" not in choose_questions(answers, SCORES, n=10)


def test_result_does_not_depend_on_dict_order():
    answers = random_table(0, AGENTS)
    shuffled_agents = {a: dict(reversed(list(answers[a].items()))) for a in reversed(AGENTS)}
    scores = {a: SCORES[a] for a in reversed(AGENTS)}
    assert choose_questions(answers, SCORES, n=10) == choose_questions(shuffled_agents, scores, n=10)


def test_agents_without_a_score_do_not_change_the_result():
    answers = random_table(1, AGENTS)
    held_out = random_table(2, ["held_out", "other_held_out"])
    assert choose_questions({**answers, **held_out}, SCORES, n=10) == choose_questions(answers, SCORES, n=10)


def test_same_input_gives_the_same_output():
    answers = random_table(3, AGENTS)
    assert choose_questions(answers, SCORES, n=10) == choose_questions(answers, SCORES, n=10)


def test_tied_scores_get_average_ranks_and_a_constant_score_gives_nothing_to_follow():
    # a score tied between two agents still ranks the question that separates the tied groups first
    scores = {"a": 0.0, "b": 0.0, "c": 1.0, "d": 1.0}
    answers = {
        "a": {"q1": False, "q2": True},
        "b": {"q1": False, "q2": False},
        "c": {"q1": True, "q2": True},
        "d": {"q1": True, "q2": False},
    }
    assert choose_questions(answers, scores, n=2)[0] == "q1"
    # every question then has relevance 0, so the order is by question id
    assert choose_questions(answers, {a: 0.5 for a in scores}, n=2) == ["q1", "q2"]


def test_fewer_than_three_agents_is_an_error():
    answers = random_table(4, AGENTS[:2])
    with pytest.raises(ValueError, match="at least 3 agents"):
        choose_questions(answers, {a: SCORES[a] for a in AGENTS[:2]})


def test_a_scored_agent_without_answers_is_an_error_that_names_it():
    answers = table()
    del answers["agent5"]
    with pytest.raises(ValueError, match="agent5"):
        choose_questions(answers, SCORES)


def test_accuracy_on_counts_only_the_listed_questions_the_agent_answered():
    right = {"q1": True, "q2": False, "q3": True}
    assert accuracy_on(right, ["q1", "q2"]) == 0.5
    assert accuracy_on(right, ["q1", "q3", "q4"]) == 1.0
    assert accuracy_on(right, ["q4"]) is None
    assert accuracy_on(right, []) is None
