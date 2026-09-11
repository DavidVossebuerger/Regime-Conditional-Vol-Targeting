"""Config loader for the crypto risk pipeline.

Reads YAML, flattens nested dicts to dot-access, applies env-var overrides.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
import os

import yaml


class Config:
    """Dot-access config with env-var fallback."""

    def __init__(self, data: dict):
        self._d = data

    def __getattr__(self, key):
        if key in self._d:
            v = self._d[key]
            if isinstance(v, dict):
                return Config(v)
            return v
        # Env-var fallback: PIPELINE_<UPPER>_<KEY>
        env = os.environ.get(f"PIPELINE_{key.upper()}")
        if env is not None:
            return env
        raise AttributeError(f"Config has no key '{key}'")

    def get(self, key, default=None):
        return self._d.get(key, default)

    def to_dict(self) -> dict:
        return self._d


def load_config(path: str | Path) -> Config:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with p.open() as f:
        data = yaml.safe_load(f)
    return Config(data)
