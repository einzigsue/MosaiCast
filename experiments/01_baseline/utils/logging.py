"""MLflow + metrics.jsonl logging helpers (CLAUDE.md rule 6).

MLflow experiment name = tier ('t1' / 't2').
Run name = '<exp_id>__<timestamp>'.
Metric namespaces: train/..., val/<var>/..., alloc/..., sys/...
metrics.jsonl is written as a portable fallback that compare.py reads without
a live MLflow server.
"""
from __future__ import annotations
import json
from pathlib import Path
import mlflow


def get_or_create_experiment(tier: str) -> str:
    """Return MLflow experiment ID for 'tier', creating it if absent."""
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(tier)
    if exp is None:
        return client.create_experiment(tier)
    return exp.experiment_id


def log_metrics_jsonl(path: Path, step: int, metrics: dict) -> None:
    """Append a record to metrics.jsonl (portable fallback)."""
    with open(path, "a") as f:
        f.write(json.dumps({"step": step, **metrics}) + "\n")
