"""Build probe items, inside an environment venv.

    python -m core.probes.build recovery --tasks $DATA_ROOT/catalogs/scienceworld.parquet --out items.jsonl \
        --env scienceworld --split train --per-task 3 --max-tasks 60
    python -m core.probes.build compounding --tasks ... --out items.jsonl --env scienceworld --depths 0,2,5
"""

import argparse
import json
import random

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
        items = compounding.build(env, tasks, args.per_task, tuple(int(d) for d in args.depths.split(",")), rng)
    env.close()
    store.append(args.out, items)
    print(f"wrote {len(items)} {args.family} items to {args.out}")


if __name__ == "__main__":
    main()
