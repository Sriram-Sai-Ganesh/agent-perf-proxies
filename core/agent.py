"""The agent registry: an agent is a base model, an optional adapter, and a scaffold."""

from dataclasses import dataclass

import yaml


@dataclass
class AgentSpec:
    id: str  # {base}/{adapter or none}/{scaffold}
    base: str  # key into bases
    hub_id: str
    adapter: str | None  # key into adapters
    adapter_path: str | None
    scaffold: str
    thinking: bool

    @property
    def served_model(self) -> str:
        """The name to ask the server for: the adapter's name when there is one, else the base hub id."""
        return self.adapter if self.adapter else self.hub_id


def load_agents(path: str, ids: list[str] | None = None) -> list[AgentSpec]:
    """Read ``configs/agents.yaml``; ``ids`` selects a subset in that order."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    bases, adapters = cfg["bases"], cfg.get("adapters") or {}
    specs = {}
    for a in cfg["agents"]:
        base = bases[a["base"]]
        adapter = a.get("adapter")
        specs[a["id"]] = AgentSpec(
            id=a["id"],
            base=a["base"],
            hub_id=base["hub_id"],
            adapter=adapter,
            adapter_path=adapters[adapter]["path"] if adapter else None,
            scaffold=a["scaffold"],
            thinking=bool(a.get("thinking", base.get("thinking", False))),
        )
    if ids is None:
        return list(specs.values())
    missing = [i for i in ids if i not in specs]
    if missing:
        raise SystemExit(f"agents not in {path}: {missing}")
    return [specs[i] for i in ids]
