"""Build the pool of ordinary test questions that each agent's plain language model answers, with no scaffold.

    python -m core.static.build --out $DATA_ROOT/static/pool.jsonl --mmlu-pro 200 --gsm8k 150 --bbh 150 --seed 0

Questions come from MMLU-Pro (test split), GSM8K (main configuration, test split) and BBH (one subset per task).
Each source is sampled evenly over its subsets: MMLU-Pro categories, BBH tasks, and GSM8K's single subset.
"""

import argparse
import os
import random
from collections import Counter, defaultdict

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from core import store
from core.schema import StaticItem

MMLU_PRO_REPO = "TIGER-Lab/MMLU-Pro"
MMLU_PRO_FILE = "data/test-00000-of-00001.parquet"
GSM8K_REPO = "openai/gsm8k"
GSM8K_FILE = "main/test-00000-of-00001.parquet"
BBH_REPO = "lukaemon/bbh"  # one folder per task, each holding test-00000-of-00001.parquet


def read_hub_parquet(repo: str, path: str) -> list[dict]:
    """Download one Parquet file of a Hugging Face dataset repository and read its rows."""
    local = hf_hub_download(repo, path, repo_type="dataset")
    return pq.read_table(local).to_pylist()


def bbh_task_files() -> dict[str, str]:
    """Each BBH task name with the path of its test file in the repository."""
    files = HfApi().list_repo_files(BBH_REPO, repo_type="dataset")
    return {f.split("/")[0]: f for f in sorted(files) if f.endswith("/test-00000-of-00001.parquet")}


def mmlu_pro_items(rows: list[dict]) -> list[StaticItem]:
    """MMLU-Pro rows (question, options, answer letter, category, question_id) as multiple-choice items."""
    return [
        StaticItem(
            item_id=f"mmlu_pro/{r['question_id']}",
            source="mmlu_pro",
            subset=r["category"],
            question=r["question"],
            choices=list(r["options"]),
            answer=r["answer"],
            answer_type="letter",
        )
        for r in rows
    ]


def gsm8k_final_number(answer: str) -> str:
    """The number after the last "####" of a GSM8K answer, without thousands commas."""
    return answer.rsplit("####", 1)[1].strip().replace(",", "")


def gsm8k_items(rows: list[dict]) -> list[StaticItem]:
    """GSM8K rows (question, answer text ending "#### <number>") as number items, numbered by row index."""
    return [
        StaticItem(
            item_id=f"gsm8k/{i}",
            source="gsm8k",
            subset="main",
            question=r["question"],
            choices=[],
            answer=gsm8k_final_number(r["answer"]),
            answer_type="number",
        )
        for i, r in enumerate(rows)
    ]


def bbh_items(rows_by_task: dict[str, list[dict]]) -> list[StaticItem]:
    """BBH rows (input, target) of each task as text items, numbered by row index within the task."""
    return [
        StaticItem(
            item_id=f"bbh/{task}/{i}",
            source="bbh",
            subset=task,
            question=r["input"],
            choices=[],
            answer=r["target"],
            answer_type="text",
        )
        for task, rows in sorted(rows_by_task.items())
        for i, r in enumerate(rows)
    ]


def sample_evenly(items: list[StaticItem], n: int, seed: int) -> list[StaticItem]:
    """n items spread evenly over the subsets: n // subsets from each, the remainder one each from random subsets.

    A subset with fewer items than its share gives all of them. Deterministic for a seed and independent of the
    order of `items`. Returns the chosen items sorted by item_id.
    """
    rng = random.Random(seed)
    by_subset = defaultdict(list)
    for it in sorted(items, key=lambda it: it.item_id):
        by_subset[it.subset].append(it)
    subsets = sorted(by_subset)
    share = {s: n // len(subsets) for s in subsets}
    for s in rng.sample(subsets, n % len(subsets)):
        share[s] += 1
    chosen = []
    for s in subsets:
        chosen.extend(rng.sample(by_subset[s], min(share[s], len(by_subset[s]))))
    return sorted(chosen, key=lambda it: it.item_id)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mmlu-pro", type=int, default=200)
    ap.add_argument("--gsm8k", type=int, default=150)
    ap.add_argument("--bbh", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if os.path.exists(args.out):
        raise SystemExit(f"{args.out} already exists; the pool is built once and shared, so it is not rebuilt over")

    pool = sample_evenly(mmlu_pro_items(read_hub_parquet(MMLU_PRO_REPO, MMLU_PRO_FILE)), args.mmlu_pro, args.seed)
    pool += sample_evenly(gsm8k_items(read_hub_parquet(GSM8K_REPO, GSM8K_FILE)), args.gsm8k, args.seed)
    bbh_rows = {task: read_hub_parquet(BBH_REPO, path) for task, path in bbh_task_files().items()}
    pool += sample_evenly(bbh_items(bbh_rows), args.bbh, args.seed)
    store.append(args.out, pool)
    for source, count in sorted(Counter(it.source for it in pool).items()):
        print(f"{source}: {count}")
    print(f"wrote {len(pool)} items to {args.out}")


if __name__ == "__main__":
    main()
