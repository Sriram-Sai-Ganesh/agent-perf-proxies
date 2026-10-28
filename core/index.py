"""Summarise every shared run under one folder, the way core.summarize does for one run.

    python -m core.index $DATA_ROOT/shared [--tasks <catalog.parquet>]

Expects the layout cluster/publish.sh uploads: traces/<site>/<run_id>/steps.parquet and provenance.json. Runs named
in <folder>/excluded.txt (one run id per line) are skipped.
"""

import argparse
import glob
import json
import os

from core import store
from core.summarize import episodes, print_episodes, print_probes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--tasks", default=None)
    args = ap.parse_args()
    excluded_path = os.path.join(args.folder, "excluded.txt")
    excluded = set()
    if os.path.exists(excluded_path):
        with open(excluded_path) as f:
            excluded = set(f.read().split())
    for steps in sorted(glob.glob(os.path.join(args.folder, "traces", "*", "*", "steps.parquet"))):
        run_dir = os.path.dirname(steps)
        site, run_id = run_dir.split(os.sep)[-2:]
        if run_id in excluded:
            continue
        with open(os.path.join(run_dir, "provenance.json")) as f:
            p = json.load(f)
        print(f"== {site}/{run_id}  commit {p['commit'][:7]}  finished {p['finished']}")
        rows = store.read_parquet(steps)
        eps = episodes(rows, args.tasks)
        if eps:
            print_episodes(eps)
        if any(r["kind"] == "probe" for r in rows):
            print_probes(rows)


if __name__ == "__main__":
    main()
