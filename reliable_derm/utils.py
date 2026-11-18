from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import yaml


def _parse(val: str):
    """CLI override value -> python. "true"->True, "5"->int, "3e-4"->float (plain YAML 1.1 reads that as str)."""
    v = yaml.safe_load(val)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            pass
    return v


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    for item in overrides or []:
        key, _, val = item.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse(val)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
