"""Build probe items, inside an environment venv.

    python -m core.probes.build recovery --tasks $DATA_ROOT/catalogs/scienceworld.parquet --out items.jsonl \
        --env scienceworld --split train --per-task 3 --max-tasks 60
    python -m core.probes.build compounding --tasks ... --out items.jsonl --env scienceworld \
        --steps $DATA_ROOT/runs/<run>/steps.jsonl --agent granite-4.1-8b/none/react --depths 0,2,5
"""

import argparse
import json
import random
from collections import defaultdict

from core import store
from core.env import load_environment
from core.probes import compounding, recovery
from core.schema import TaskRow, from_record
from core.suites import select


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=["recovery", "compounding"])
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--env", required=True)
    ap.add_argument("--env-settings", default="{}", help="JSON passed to the adapter")
    ap.add_argument("--split", default="train")
    ap.add_argument("--min-gold", type=int, default=5)
    ap.add_argument("--max-tasks", type=int, default=None, help="cap per task type")
    ap.add_argument("--per-task", type=int, default=2)
    ap.add_argument("--after", type=int, default=0, help="recovery: gold steps after the failure")
    ap.add_argument("--steps", help="compounding: an agent's steps.jsonl on train tasks")
    ap.add_argument("--agent", help="compounding: which agent's rows to use")
    ap.add_argument("--depths", default="0,2,5")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    env = load_environment(args.env, json.loads(args.env_settings))
    catalog = store.read_parquet(args.tasks)
    tasks = [
        from_record(TaskRow, r)
        for r in select(catalog, args.split, min_gold=args.min_gold, cap_per_type=args.max_tasks, seed=args.seed)
    ]
    if args.family == "recovery":
        items = recovery.build(env, tasks, args.per_task, rng, after=args.after)
    else:
        by_task = defaultdict(list)
        for r in store.read(args.steps):
            if r["agent_id"] == args.agent and r["kind"] == "episode":
                by_task[r["item_id"]].append(r)
        episodes = {k: sorted(v, key=lambda r: (r["seed"], r["step"])) for k, v in by_task.items()}
        episodes = {k: [r for r in v if r["seed"] == v[0]["seed"]] for k, v in episodes.items()}  # one seed per task
        task_map = {t.task_id: t for t in tasks}
        episodes = {k: v for k, v in episodes.items() if k in task_map}
        items = compounding.build(
            env, task_map, episodes, args.agent, tuple(int(d) for d in args.depths.split(",")), rng
        )
    env.close()
    store.append(args.out, items)
    print(f"wrote {len(items)} {args.family} items to {args.out}")


if __name__ == "__main__":
    main()
