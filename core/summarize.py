"""Summarise a run: episodes by agent and gold-length bin, or probe accuracy by agent and family.

python -m core.summarize $DATA_ROOT/runs/pilot_sw_long/steps.jsonl --tasks $DATA_ROOT/catalogs/scienceworld.parquet
python -m core.summarize $DATA_ROOT/runs/pilot_recovery/steps.jsonl
"""

import argparse
import json
from collections import defaultdict
from statistics import mean

from core import store
from core.schema import StepRow, from_record, summarize_episode

BINS = ((0, 30), (30, 60), (60, 100), (100, 10**9))


def episodes(path: str, tasks_path: str | None) -> list[dict]:
    gold = {t["task_id"]: t["gold_len"] for t in store.read_parquet(tasks_path)} if tasks_path else {}
    by_key = defaultdict(list)
    for r in store.read(path):
        if r["kind"] == "episode":
            by_key[(r["agent_id"], r["item_id"], r["seed"])].append(from_record(StepRow, r))
    out = []
    for key, steps in by_key.items():
        ep = summarize_episode(sorted(steps, key=lambda s: s.step))
        ep["gold_len"] = gold.get(key[1], -1)
        ep["valid_rate"] = mean(s.valid for s in steps)
        out.append(ep)
    return out


def print_episodes(eps: list[dict]) -> None:
    print(
        f"{'agent':32} {'gold bin':9} {'n':>3} {'success':>7} {'final':>6} {'progress':>8} {'steps':>6} {'valid':>6} {'tokens/ep':>10}"
    )
    for agent in sorted({e["agent_id"] for e in eps}):
        rows = [e for e in eps if e["agent_id"] == agent]
        for lo, hi in BINS:
            b = [e for e in rows if lo <= e["gold_len"] < hi]
            if not b:
                continue
            label = f"[{lo},{hi})" if hi < 10**9 else f"[{lo},inf)"
            print(
                f"{agent:32} {label:9} {len(b):3d} {mean(e['success'] for e in b):7.2f} {mean(e['final_score'] for e in b):6.2f} "
                f"{mean(e['progress'] for e in b):8.2f} {mean(e['n_steps'] for e in b):6.1f} {mean(e['valid_rate'] for e in b):6.2f} "
                f"{mean(e['tokens'] for e in b):10.0f}"
            )
        print(
            f"{agent:32} {'all':9} {len(rows):3d} {mean(e['success'] for e in rows):7.2f} {mean(e['final_score'] for e in rows):6.2f} "
            f"{mean(e['progress'] for e in rows):8.2f} {mean(e['n_steps'] for e in rows):6.1f} {mean(e['valid_rate'] for e in rows):6.2f} "
            f"{mean(e['tokens'] for e in rows):10.0f}   total tokens {sum(e['tokens'] for e in rows)}"
        )
        stops = defaultdict(int)
        for e in rows:
            stops[e["stop_reason"]] += 1
        print(f"{'':32} stop reasons: {dict(stops)}")


def print_probes(path: str) -> None:
    rows = [r for r in store.read(path) if r["kind"] == "probe" and json.loads(r["meta"]).get("scored", True)]
    print(
        f"{'agent':32} {'family':12} {'control':8} {'n':>4} {'exact':>6} {'progress':>8} {'valid':>6} {'tokens/item':>11}"
    )
    groups = defaultdict(list)
    for r in rows:
        m = json.loads(r["meta"])
        groups[(r["agent_id"], m["family"], str(m.get("control", "-")))].append((m, r))
    for (agent, family, control), g in sorted(groups.items()):
        print(
            f"{agent:32} {family:12} {control:8} {len(g):4d} {mean(m['correct_exact'] for m, _ in g):6.2f} "
            f"{mean(m['correct_progress'] for m, _ in g):8.2f} {mean(r['valid'] for _, r in g):6.2f} "
            f"{mean(r['prompt_tokens'] + r['completion_tokens'] for _, r in g):11.0f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("steps")
    ap.add_argument("--tasks", default=None)
    args = ap.parse_args()
    eps = episodes(args.steps, args.tasks)
    if eps:
        print_episodes(eps)
    if any(r["kind"] == "probe" for r in store.read(args.steps)):
        print_probes(args.steps)


if __name__ == "__main__":
    main()
