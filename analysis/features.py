"""Cheap-evaluation features per agent, to compare against long-task scores at matched token budgets.

python -m analysis.features b1 --truth-steps $DATA_ROOT/runs/pilot_sw_long/steps.jsonl \
    --out $DATA_ROOT/features/b1.parquet
python -m analysis.features b2 --b2-steps $DATA_ROOT/runs/b2_sw_cutoff/steps.parquet \
    --truth-steps $DATA_ROOT/runs/pilot_sw_long/steps.jsonl --tasks $DATA_ROOT/catalogs/scienceworld.parquet \
    --out $DATA_ROOT/features/b2.parquet
python -m analysis.features b3 --static-steps $DATA_ROOT/runs/b3_static/steps.jsonl \
    --truth-steps $DATA_ROOT/runs/pilot_sw_long/steps.jsonl --pool $DATA_ROOT/static/pool.jsonl \
    --out-items $DATA_ROOT/features/b3_items.parquet --out $DATA_ROOT/features/b3_random.parquet

B1: the agent's long episodes themselves are drawn at random, from one seed's episodes at a time, until a share (beta)
of the agent's full long-evaluation cost in tokens is spent; the feature is their mean progress (best score).

B2: each agent's short episodes (cut off after 20 steps, runs/b2_sw_cutoff.yaml) are scored as their best score over
the official solution's score at a reference step (15 by default), at most 1. Episodes are then drawn at random until
a share (beta) of the agent's full long-evaluation cost in tokens is spent.

B3: each agent's plain model answers non-agentic questions (steps of kind "static"). ``b3`` writes every agent's result
on every question (right or wrong, tokens spent); the prediction step chooses which questions to use (analysis/pace.py).
``b3`` also writes the random control, "B3-random": the agent's accuracy on one fixed random set of questions, the same
for every agent, left empty at a budget that the tokens of those questions pass.
"""

import argparse
import json
import os
import random
from collections import defaultdict
from statistics import mean

from analysis import pace
from core import provenance, store
from core.summarize import episodes

BETAS = [0.005, 0.01, 0.02, 0.05, 0.1]


def gold_reference(meta: dict, ref_step: int) -> float | None:
    """The official solution's score after ``ref_step`` steps (its last score if it is shorter); None if unknown."""
    scores = meta.get("gold_scores")
    if not scores:
        return None
    return scores[min(ref_step, len(scores)) - 1]


def cutoff_scores(
    step_records: list[dict], tasks: list[dict], ref_step: int
) -> tuple[dict[str, list[tuple[float, int]]], dict[str, int]]:
    """Per agent, (cut-off score, episode tokens) of each episode, and how many episodes were skipped and why.

    The cut-off score is the episode's progress over the gold score at ``ref_step``, at most 1.
    """
    metas = {t["task_id"]: json.loads(t["meta"]) for t in tasks}
    scored: dict[str, list[tuple[float, int]]] = {}
    skipped: dict[str, int] = defaultdict(int)
    for ep in episodes(step_records, None):
        agent_scores = scored.setdefault(ep["agent_id"], [])
        if ep["item_id"] not in metas:
            skipped["task not in catalog"] += 1
            continue
        gold = gold_reference(metas[ep["item_id"]], ref_step)
        if gold is None:
            skipped["task has no gold scores"] += 1
        elif gold <= 0:
            skipped["gold score at the reference step is 0"] += 1
        else:
            agent_scores.append((min(1.0, ep["progress"] / gold), ep["tokens"]))
    return scored, dict(skipped)


def full_cost(truth_records: list[dict]) -> dict[str, float]:
    """Per agent, the mean tokens (prompt and completion) per seed over its episodes in the long evaluation."""
    tokens: dict[str, int] = defaultdict(int)
    seeds: dict[str, set] = defaultdict(set)
    for r in truth_records:
        if r["kind"] == "episode":
            tokens[r["agent_id"]] += r["prompt_tokens"] + r["completion_tokens"]
            seeds[r["agent_id"]].add(r["seed"])
    return {agent: tokens[agent] / len(seeds[agent]) for agent in tokens}


def draw_under_budget(
    episode_list: list[tuple[float, int]], budget: float, rng: random.Random
) -> tuple[float | None, int, int]:
    """Take episodes in random order until the next one would pass the budget.

    Returns:
        The mean score of the episodes taken (None if none fit), how many were taken, and the tokens they cost.
    """
    order = list(episode_list)
    rng.shuffle(order)
    taken, spent = [], 0
    for score, tokens in order:
        if spent + tokens > budget:
            break
        taken.append(score)
        spent += tokens
    return (mean(taken) if taken else None), len(taken), spent


def long_episodes_by_seed(truth_records: list[dict]) -> dict[str, dict[int, list[tuple[float, int]]]]:
    """Per agent and episode seed, (progress, tokens) of each long episode."""
    by_seed: dict[str, dict[int, list[tuple[float, int]]]] = {}
    for ep in episodes(truth_records, None):
        by_seed.setdefault(ep["agent_id"], {}).setdefault(ep["seed"], []).append((ep["progress"], ep["tokens"]))
    return by_seed


def b1_features(
    by_seed: dict[str, dict[int, list[tuple[float, int]]]],
    c_full: dict[str, float],
    betas: list[float],
    repeats: int,
    seed: int,
) -> list[dict]:
    """One row per agent, beta, episode seed and draw; each draw uses only the episodes of its own seed."""
    rows = []
    for agent, seeds in by_seed.items():
        for beta in betas:
            budget = beta * c_full[agent]
            for episode_seed, episode_list in seeds.items():
                for draw in range(repeats):
                    rng = random.Random(f"{seed}/{agent}/{beta}/{episode_seed}/{draw}")
                    feature, n_items, spent = draw_under_budget(episode_list, budget, rng)
                    rows.append(
                        {
                            "agent_id": agent,
                            "method": "B1",
                            "beta": beta,
                            "episode_seed": episode_seed,
                            "draw": draw,
                            "feature": feature,
                            "n_items": n_items,
                            "tokens_spent": spent,
                            "budget_tokens": budget,
                            "c_full": c_full[agent],
                        }
                    )
    return rows


def b2_features(
    scored: dict[str, list[tuple[float, int]]], c_full: dict[str, float], betas: list[float], repeats: int, seed: int
) -> list[dict]:
    """One row per agent, beta and draw."""
    rows = []
    for agent, episode_list in scored.items():
        for beta in betas:
            budget = beta * c_full[agent]
            for draw in range(repeats):
                rng = random.Random(f"{seed}/{agent}/{beta}/{draw}")
                feature, n_items, spent = draw_under_budget(episode_list, budget, rng)
                rows.append(
                    {
                        "agent_id": agent,
                        "method": "B2",
                        "beta": beta,
                        "draw": draw,
                        "feature": feature,
                        "n_items": n_items,
                        "tokens_spent": spent,
                        "budget_tokens": budget,
                        "c_full": c_full[agent],
                    }
                )
    return rows


def static_results(step_records: list[dict]) -> list[dict]:
    """One row per agent and question answered without a scaffold: whether it was right, and its tokens."""
    rows = []
    for r in step_records:
        if r["kind"] != "static":
            continue
        meta = json.loads(r["meta"]) if isinstance(r["meta"], str) else r["meta"]
        rows.append(
            {
                "agent_id": r["agent_id"],
                "item_id": r["item_id"],
                "source": meta["source"],
                "subset": meta["subset"],
                "correct": bool(meta["correct"]),
                "tokens": r["prompt_tokens"] + r["completion_tokens"],
            }
        )
    return rows


def random_question_ids(pool_ids: list[str], n: int, seed: int) -> list[str]:
    """``n`` question ids drawn at random from the pool; the same ids and seed give the same draw in any order."""
    return random.Random(seed).sample(sorted(pool_ids), n)


def b3_random_features(
    results: list[dict], pool_ids: list[str], c_full: dict[str, float], betas: list[float], n: int, seed: int
) -> list[dict]:
    """One row per agent and beta: the agent's accuracy on one fixed random set of questions.

    The feature is None when the tokens of the questions the agent answered pass the budget. A message says how many
    of the ``n`` questions an agent did not answer; the feature is then over the ones it did answer.
    """
    chosen = random_question_ids(pool_ids, n, seed)
    correct: dict[str, dict[str, bool]] = defaultdict(dict)
    tokens: dict[str, dict[str, int]] = defaultdict(dict)
    for r in results:
        correct[r["agent_id"]][r["item_id"]] = r["correct"]
        tokens[r["agent_id"]][r["item_id"]] = r["tokens"]
    rows = []
    for agent in sorted(correct):
        answered = [i for i in chosen if i in correct[agent]]
        if len(answered) < n:
            print(f"agent {agent}: {n - len(answered)} of the {n} random questions have no answer")
        spent = sum(tokens[agent][i] for i in answered)
        feature = pace.accuracy_on(correct[agent], chosen)
        for beta in betas:
            budget = beta * c_full[agent]
            rows.append(
                {
                    "agent_id": agent,
                    "method": "B3-random",
                    "beta": beta,
                    "draw": 0,
                    "feature": feature if spent <= budget else None,
                    "n_items": len(answered),
                    "tokens_spent": spent,
                    "budget_tokens": budget,
                    "c_full": c_full[agent],
                }
            )
    return rows


def read_steps(paths: list[str]) -> list[dict]:
    """Step records of every file, JSONL or Parquet."""
    rows = []
    for path in paths:
        rows += store.read_parquet(path) if path.endswith(".parquet") else store.read(path)
    return rows


def print_summary(rows: list[dict]) -> None:
    print(f"{'agent':32} {'beta':>6} {'mean feature':>12} {'draws w/o fit':>13} {'mean items':>10} {'c_full':>10}")
    groups = defaultdict(list)
    for r in rows:
        groups[(r["agent_id"], r["beta"])].append(r)
    for (agent, beta), g in groups.items():
        values = [r["feature"] for r in g if r["feature"] is not None]
        feature = f"{mean(values):12.3f}" if values else f"{'-':>12}"
        print(
            f"{agent:32} {beta:6} {feature} {len(g) - len(values):13d} {mean(r['n_items'] for r in g):10.1f} "
            f"{g[0]['c_full']:10.0f}"
        )


def run_b1(args: argparse.Namespace) -> None:
    out_dir = os.path.dirname(args.out) or "."
    record = provenance.start(out_dir, vars(args), name="provenance_features_b1.json")
    truth = read_steps(args.truth_steps)
    rows = b1_features(long_episodes_by_seed(truth), full_cost(truth), args.betas, args.repeats, args.seed)
    store.write_parquet(rows, args.out)
    print_summary(rows)
    provenance.finish(out_dir, record)


def run_b2(args: argparse.Namespace) -> None:
    out_dir = os.path.dirname(args.out) or "."
    record = provenance.start(out_dir, vars(args), name="provenance_features_b2.json")
    scored, skipped = cutoff_scores(read_steps(args.b2_steps), store.read_parquet(args.tasks), args.ref_step)
    c_full = full_cost(read_steps(args.truth_steps))
    for agent in sorted(set(scored) - set(c_full)):
        print(f"skipped agent {agent}: in the B2 files but not in the truth files")
    rows = b2_features({a: e for a, e in scored.items() if a in c_full}, c_full, args.betas, args.repeats, args.seed)
    store.write_parquet(rows, args.out)
    print_summary(rows)
    print(f"skipped episodes: {skipped or 'none'}")
    provenance.finish(out_dir, record)


def run_b3(args: argparse.Namespace) -> None:
    out_dir = os.path.dirname(args.out) or "."
    record = provenance.start(out_dir, vars(args), name="provenance_features_b3.json")
    results = static_results(read_steps(args.static_steps))
    c_full = full_cost(read_steps(args.truth_steps))
    for agent in sorted({r["agent_id"] for r in results} - set(c_full)):
        print(f"skipped agent {agent}: in the static files but not in the truth files")
    results = [r for r in results if r["agent_id"] in c_full]
    pool_ids = [item["item_id"] for item in store.read(args.pool)]
    store.write_parquet(results, args.out_items)
    rows = b3_random_features(results, pool_ids, c_full, args.betas, args.n, args.seed)
    store.write_parquet(rows, args.out)
    print_summary(rows)
    provenance.finish(out_dir, record)


def parse_betas(text: str) -> list[float]:
    return [float(x) for x in text.split(",")]


def main() -> None:
    ap = argparse.ArgumentParser()
    methods = ap.add_subparsers(dest="method", required=True)
    b1 = methods.add_parser("b1", help="mean progress of long episodes drawn under a token budget, one seed at a time")
    b1.add_argument("--truth-steps", nargs="+", required=True, help="steps files of the long evaluation")
    b1.add_argument("--out", required=True, help="features Parquet file")
    b1.add_argument("--betas", type=parse_betas, default=BETAS, help="comma-separated shares of the full cost")
    b1.add_argument("--repeats", type=int, default=20)
    b1.add_argument("--seed", type=int, default=0)
    b2 = methods.add_parser("b2", help="mean cut-off score of short episodes drawn under a token budget")
    b2.add_argument("--b2-steps", nargs="+", required=True, help="steps files of the short episodes")
    b2.add_argument("--truth-steps", nargs="+", required=True, help="steps files of the long evaluation")
    b2.add_argument("--tasks", required=True, help="task catalog Parquet file")
    b2.add_argument("--out", required=True, help="features Parquet file")
    b2.add_argument("--betas", type=parse_betas, default=BETAS, help="comma-separated shares of the full cost")
    b2.add_argument("--repeats", type=int, default=20)
    b2.add_argument("--ref-step", type=int, default=15, help="step of the gold score the cut-off score is judged by")
    b2.add_argument("--seed", type=int, default=0)
    b3 = methods.add_parser("b3", help="each agent's result on every static question, and the random-question control")
    b3.add_argument("--static-steps", nargs="+", required=True, help="steps files of the static questions")
    b3.add_argument("--truth-steps", nargs="+", required=True, help="steps files of the long evaluation")
    b3.add_argument("--pool", required=True, help="JSONL file of the question pool, one item with an item_id per line")
    b3.add_argument("--out-items", required=True, help="Parquet file of every agent's result on every question")
    b3.add_argument("--out", required=True, help="random-control features Parquet file")
    b3.add_argument("--betas", type=parse_betas, default=BETAS, help="comma-separated shares of the full cost")
    b3.add_argument("--n", type=int, default=100, help="how many random questions")
    b3.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.method == "b1":
        run_b1(args)
    elif args.method == "b2":
        run_b2(args)
    elif args.method == "b3":
        run_b3(args)


if __name__ == "__main__":
    main()
