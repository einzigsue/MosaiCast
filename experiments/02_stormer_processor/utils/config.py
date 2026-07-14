"""Config loading and flattening utilities."""
from __future__ import annotations
from pathlib import Path
import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base; override wins on conflicts."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(exp_config: str, exp_name: str) -> dict:
    """Deep-merge base.yaml < tier yaml < experiment yaml (later wins)."""
    base_path = Path("/g/data/z00/yxs900/aurora/experiments/configs/base.yaml")
    exp_path = Path(exp_config)

    def _load(p: Path) -> dict:
        with open(p) as f:
            return yaml.safe_load(f) or {}

    cfg = _load(base_path)
    if exp_path.exists():
        cfg = _deep_merge(cfg, _load(exp_path))

    cfg.setdefault("exp_name", exp_name)
    return cfg


def flatten_config(d: dict, prefix: str = "") -> dict:
    """Recursively flatten a nested dict with dot-separated keys.

    Used to pass a resolved config to mlflow.log_params.
    """
    out: dict = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(flatten_config(v, key))
        else:
            out[key] = v
    return out
