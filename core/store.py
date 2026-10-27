"""Append rows to JSONL as they complete, resume from what is there, and consolidate to Parquet."""

import json
import os

import pyarrow as pa
import pyarrow.parquet as pq

from core.schema import to_record

RESUME_KEYS = ("agent_id", "item_id", "seed")


def append(path: str, rows: list) -> None:
    """Append rows (dataclasses) to a JSONL file in one write, so a partial episode is never on disk."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    text = "".join(json.dumps(to_record(r)) + "\n" for r in rows)
    with open(path, "a") as f:
        f.write(text)


def read(path: str) -> list[dict]:
    """Read every record of a JSONL file; a missing file is empty."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def done_keys(path: str) -> set[tuple]:
    """The (agent_id, item_id, seed) triples already in the file; rows are written per complete item."""
    return {tuple(r[k] for k in RESUME_KEYS) for r in read(path)}


def write_parquet(records: list[dict], path: str) -> None:
    """Write flat records (JSON strings for nested fields) to Parquet."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records), path)


def read_parquet(path: str) -> list[dict]:
    """Read a Parquet table as flat records."""
    return pq.read_table(path).to_pylist()
