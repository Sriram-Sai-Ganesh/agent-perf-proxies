"""Cheap-evaluation features per agent, to compare against long-task scores at matched token budgets.

python -m analysis.features b2 --b2-steps $DATA_ROOT/runs/b2_sw_cutoff/steps.parquet \
    --truth-steps $DATA_ROOT/runs/pilot_sw_long/steps.jsonl --tasks $DATA_ROOT/catalogs/scienceworld.parquet \
    --out $DATA_ROOT/features/b2.parquet

B2: each agent's short episodes (cut off after 20 steps, runs/b2_sw_cutoff.yaml) are scored as their best score over
the official solution's score at a reference step (15 by default), at most 1. Episodes are then drawn at random until
a share (beta) of the agent's full long-evaluation cost in tokens is spent.
"""

import argparse
import json
import os
import random
from collections import defaultdict
from statistics import mean

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


def parse_betas(text: str) -> list[float]:
    return [float(x) for x in text.split(",")]


def main() -> None:
    ap = argparse.ArgumentParser()
    methods = ap.add_subparsers(dest="method", required=True)
    b2 = methods.add_parser("b2", help="mean cut-off score of short episodes drawn under a token budget")
    b2.add_argument("--b2-steps", nargs="+", required=True, help="steps files of the short episodes")
    b2.add_argument("--truth-steps", nargs="+", required=True, help="steps files of the long evaluation")
    b2.add_argument("--tasks", required=True, help="task catalog Parquet file")
    b2.add_argument("--out", required=True, help="features Parquet file")
    b2.add_argument("--betas", type=parse_betas, default=BETAS, help="comma-separated shares of the full cost")
    b2.add_argument("--repeats", type=int, default=20)
    b2.add_argument("--ref-step", type=int, default=15, help="step of the gold score the cut-off score is judged by")
    b2.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.method == "b2":
        run_b2(args)


if __name__ == "__main__":
    main()
