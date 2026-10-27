"""Build the ScienceWorld task catalog (tasks.parquet).

python -m env_scienceworld.catalog --out $DATA_ROOT/catalogs/scienceworld.parquet [--task-types boil,melt] [--splits test] [--max-per-split 20]
"""

import argparse

from core import store
from core.schema import to_record
from env_scienceworld.adapter import ScienceWorld


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--simplification", default="easy")
    ap.add_argument("--task-types", default=None, help="comma-separated; default all 30")
    ap.add_argument("--splits", default="train,dev,test")
    ap.add_argument("--max-per-split", type=int, default=None)
    args = ap.parse_args()
    env = ScienceWorld(simplification=args.simplification)
    rows = env.catalog(
        splits=tuple(args.splits.split(",")),
        task_types=args.task_types.split(",") if args.task_types else None,
        max_per_split=args.max_per_split,
    )
    env.close()
    store.write_parquet([to_record(r) for r in rows], args.out)
    print(f"wrote {len(rows)} tasks to {args.out}")


if __name__ == "__main__":
    main()
