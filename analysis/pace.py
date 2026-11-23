"""Choice of non-agentic questions, following PACE (Song et al. 2026, "PACE: a proxy for agentic capability
evaluation", github.com/neulab/pace).

The prediction step calls ``choose_questions`` once per leave-one-out fold, with only the agents the predictor is fit
on, so the held-out agent's answers never affect which questions are chosen.

Only PACE's global ranking is used: a question ranks high when it has high leverage (it carries much of the
structure of the agents-by-questions table of right and wrong answers) and when its right and wrong answers follow
the agents' long-task scores (rank correlation). PACE's second, locally chosen set and its tuned blend of the two are
not used, because with about 14 models there is nothing to tune them on.
"""

import numpy as np

MIN_AGENTS = 3
RELATIVE_ZERO = 1e-10  # singular values below this share of the largest one count as zero
TIE_DECIMALS = 10  # rank values are rounded to this many decimals, so rounding noise does not break ties by order


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks starting at 1, with tied values sharing the average of the ranks they cover."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start
        while end + 1 < len(values) and values[order[end + 1]] == values[order[start]]:
            end += 1
        ranks[order[start : end + 1]] = (start + end) / 2 + 1
        start = end + 1
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation of two equally long arrays; 0 when either one is constant."""
    ranks_a, ranks_b = average_ranks(a), average_ranks(b)
    if np.ptp(ranks_a) == 0 or np.ptp(ranks_b) == 0:
        return 0.0
    return float(np.corrcoef(ranks_a, ranks_b)[0, 1])


def choose_questions(correct: dict[str, dict[str, bool]], scores: dict[str, float], n: int = 100) -> list[str]:
    """The ``n`` questions whose answers best follow the agents' long-task scores.

    Args:
        correct: Per agent, per question id, whether the agent answered it right.
        scores: Per agent, its long-task score. Only these agents are used.
        n: How many questions to choose (all of them if fewer qualify).

    Returns:
        Question ids, best first. Only questions that every agent in ``scores`` answered are considered.
        A question's rank value is its leverage times the absolute Spearman correlation between the agents'
        answers to it (0 or 1) and their scores. Ties are broken by question id.
    """
    agents = sorted(scores)
    if len(agents) < MIN_AGENTS:
        raise ValueError(f"choosing questions needs at least {MIN_AGENTS} agents with scores, got {len(agents)}")
    unanswered = [a for a in agents if a not in correct]
    if unanswered:
        raise ValueError(f"no answers to the questions for scored agents: {', '.join(unanswered)}")
    item_ids = sorted(set.intersection(*(set(correct[a]) for a in agents)))
    table = np.array([[float(correct[a][i]) for i in item_ids] for a in agents]).reshape(len(agents), len(item_ids))
    _, singular_values, components = np.linalg.svd(table, full_matrices=False)
    kept = singular_values > RELATIVE_ZERO * singular_values.max(initial=0.0)
    leverage = (components[kept] ** 2).sum(axis=0)
    agent_scores = np.array([scores[a] for a in agents], dtype=float)
    rank_values = [
        round(float(leverage[k]) * abs(spearman(table[:, k], agent_scores)), TIE_DECIMALS) for k in range(len(item_ids))
    ]
    ranked = sorted(range(len(item_ids)), key=lambda k: (-rank_values[k], item_ids[k]))
    return [item_ids[k] for k in ranked[:n]]


def accuracy_on(correct_for_agent: dict[str, bool], item_ids: list[str]) -> float | None:
    """The share of the questions in ``item_ids`` that the agent answered right; None if it answered none of them."""
    answered = [correct_for_agent[i] for i in item_ids if i in correct_for_agent]
    return sum(answered) / len(answered) if answered else None
