"""Resolve a numbered point before constructing its IC or evolving its fields."""
import json
from pathlib import Path


def resolve_point(root, index, catalog):
    default = json.loads((root / "point_config.json").read_text())
    if index is None:
        index = default["index"]
    path = Path(catalog) / f"{index}.json"
    if index == default["index"] and not path.is_file():
        return default
    if not path.is_file():
        raise ValueError(f"Unknown point {index}: expected {path}")
    cfg = json.loads(path.read_text())
    if cfg["index"] != index:
        raise ValueError(f"Point file {path} declares index {cfg['index']}")
    for group in ("astro", "cosmo", "flags", "user"):
        missing = default[group].keys() - cfg[group].keys()
        if missing:
            raise ValueError(f"Incomplete {group} for point {index}: missing {sorted(missing)}")
    if index == default["index"] and any(cfg[g] != default[g] for g in ("astro", "cosmo", "flags", "user")):
        raise ValueError(f"Conflicting definitions for default point {index}")
    return cfg
