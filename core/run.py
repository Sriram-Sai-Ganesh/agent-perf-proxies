"""Run agents over a task suite, a probe set or a pool of plain test questions, writing step rows as each item completes.

    python -m core.run --config runs/pilot_sw_long.yaml [-o key=value ...]

The config names the items with `tasks` (and `suite`), `probes` or `static`. Each worker process owns one environment
instance and shares the model endpoint; a config without `env` (plain test questions) starts no environment and loads
no scaffolds. Resume skips (agent, item, seed) triples already in the output JSONL.
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time

import yaml

from core import provenance, store
from core.agent import AgentSpec, load_agents
from core.env import load_environment
from core.llm import Client, Sampling
from core.runner import run_episode, run_probe, run_static
from core.scaffolds import load_scaffold
from core.schema import ProbeItem, StaticItem, TaskRow, from_record
from core.suites import select

_worker: dict = {}


def expand(text: str) -> str:
    return os.path.expandvars(text)


def load_config(path: str, overrides: list[str]) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for o in overrides:
        key, _, value = o.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)
    return cfg


def endpoint_of(cfg: dict) -> tuple[str, str | None]:
    ep = cfg["endpoint"]
    if "file" in ep:
        with open(expand(ep["file"])) as f:
            return json.load(f)["base_url"], ep.get("api_key_env")
    return ep["base_url"], ep.get("api_key_env")


def _init(cfg: dict, base_url: str, api_key_env: str | None, agents: list[AgentSpec]) -> None:
    if "env" in cfg:
        _worker["env"] = load_environment(cfg["env"]["name"], cfg["env"].get("settings") or {})
        _worker["scaffolds"] = {a.id: load_scaffold(a.scaffold, **(cfg.get("scaffold") or {})) for a in agents}
    _worker["clients"] = {a.id: Client(base_url, a.served_model, api_key_env) for a in agents}
    _worker["agents"] = {a.id: a for a in agents}
    _worker["sampling"] = Sampling(**(cfg.get("sampling") or {}))
    _worker["cfg"] = cfg


def _job(args: tuple) -> list:
    agent_id, item, seed = args
    cfg = _worker["cfg"]
    agent = _worker["agents"][agent_id]
    common = {
        "client": _worker["clients"][agent_id],
        "agent_id": agent_id,
        "thinking": agent.thinking,
        "seed": seed,
        "sampling": _worker["sampling"],
        "run_id": cfg["run_id"],
    }
    try:
        if isinstance(item, StaticItem):
            return run_static(item=item, **common)
        common |= {"scaffold": _worker["scaffolds"][agent_id], "env": _worker["env"]}
        if isinstance(item, ProbeItem):
            return run_probe(item=item, continuation=int(cfg.get("probe_continuation", 0)), **common)
        return run_episode(task=item, max_steps=cfg.get("max_steps", 100), **common)
    except Exception as e:  # noqa: BLE001 - one failed item must not end the run; it is logged and retried on resume
        print(f"FAILED {agent_id} {item_id_of(item)} seed={seed}: {e!r}", file=sys.stderr)
        return []


def item_id_of(item) -> str:
    """The id that names an item in the output: task_id, probe_id or item_id."""
    return getattr(item, "task_id", None) or getattr(item, "probe_id", None) or item.item_id


def items_file(cfg: dict) -> str:
    """The file the config reads its items from: `static`, `probes` or `tasks`."""
    return expand(cfg.get("static") or cfg.get("probes") or cfg["tasks"])


def items_of(cfg: dict) -> list:
    if "static" in cfg:
        return [from_record(StaticItem, r) for r in store.read(expand(cfg["static"]))]
    if "probes" in cfg:
        return [from_record(ProbeItem, r) for r in store.read(expand(cfg["probes"]))]
    tasks = store.read_parquet(expand(cfg["tasks"]))
    return [from_record(TaskRow, r) for r in select(tasks, **cfg["suite"])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("-o", "--override", action="append", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.override)
    out_dir = expand(cfg.get("out", "$DATA_ROOT/runs/" + cfg["run_id"]))
    jsonl = os.path.join(out_dir, "steps.jsonl")
    with open(items_file(cfg), "rb") as f:
        cfg["items_sha256"] = hashlib.sha256(f.read()).hexdigest()  # proves two sites answered the same items
    record = provenance.start(out_dir, cfg)

    agents = load_agents(cfg.get("agents_file", "configs/agents.yaml"), cfg["agents"])
    items = items_of(cfg)
    done = store.done_keys(jsonl)
    jobs = [
        (a.id, it, s)
        for a in agents
        for it in items
        for s in cfg.get("seeds", [0])
        if (a.id, item_id_of(it), s) not in done
    ]
    print(
        f"{len(items)} items x {len(agents)} agents x {len(cfg.get('seeds', [0]))} seeds; {len(done)} done, {len(jobs)} to run"
    )

    base_url, api_key_env = endpoint_of(cfg)
    workers = int(cfg.get("workers", 8))
    t0, n_rows = time.time(), 0
    with mp.get_context("fork").Pool(workers, initializer=_init, initargs=(cfg, base_url, api_key_env, agents)) as pool:
        for i, rows in enumerate(pool.imap_unordered(_job, jobs), 1):
            if rows:
                store.append(jsonl, rows)
                n_rows += len(rows)
            if i % 10 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} items, {n_rows} rows, {time.time() - t0:.0f}s")
    store.write_parquet(store.read(jsonl), os.path.join(out_dir, "steps.parquet"))
    provenance.finish(out_dir, record)


if __name__ == "__main__":
    main()
