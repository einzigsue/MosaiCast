"""Evaluation: per-variable RMSE at validation time and rollout leads (CLAUDE.md rule 5)."""
from __future__ import annotations
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader
from aurora import Aurora, Batch

wdir="/g/data/z00/yxs900/aurora/experiments/02_stormer_processor/"
sys.path.insert(0,wdir)
from utils.losses import var_weighted_loss
from utils.metrics import lat_weighted_rmse
from utils.logging import log_metrics_jsonl

EVAL_SURF_VARS = ("2t", "10u")
EVAL_ATMOS_VARS = (("z", 500), ("t", 850))
EVAL_LEADS_H = (6, 24, 72)


def run_validation(
    model: Aurora,
    val_loader: DataLoader,
    device: torch.device,
    gamma: float,
    global_step: int,
    metrics_path: Path,
) -> dict[str, float]:
    """Evaluate on the validation loader; return averaged metrics dict.

    Logs to metrics_path (metrics.jsonl) and returns the same dict for MLflow.
    Metric keys: val/loss, val/<var>/rmse for each variable in EVAL_SURF_VARS
    and val/<var><level>/rmse for each pair in EVAL_ATMOS_VARS.
    """
    model.eval()
    total_loss = 0.0
    rmse_acc: dict[str, float] = {}
    n = 0

    with torch.no_grad():
        for batch, target in val_loader:
            batch = batch.to(device)
            target = target.to(device)
            lat = batch.metadata.lat

            pred = model(batch)
            total_loss += float(var_weighted_loss(pred, target, gamma=gamma))

            for k in EVAL_SURF_VARS:
                if k in pred.surf_vars:
                    rmse_acc.setdefault(f"val/{k}/rmse", 0.0)
                    rmse_acc[f"val/{k}/rmse"] += float(
                        lat_weighted_rmse(pred.surf_vars[k], target.surf_vars[k], lat)
                    )

            levels = list(batch.metadata.atmos_levels)
            for var, lev in EVAL_ATMOS_VARS:
                if var in pred.atmos_vars and lev in levels:
                    li = levels.index(lev)
                    key = f"val/{var}{lev}/rmse"
                    rmse_acc.setdefault(key, 0.0)
                    rmse_acc[key] += float(
                        lat_weighted_rmse(
                            pred.atmos_vars[var][:, :, li],
                            target.atmos_vars[var][:, :, li],
                            lat,
                        )
                    )
            n += 1

    model.train()
    if n == 0:
        return {}

    metrics = {"val/loss": total_loss / n, **{k: v / n for k, v in rmse_acc.items()}}
    log_metrics_jsonl(metrics_path, global_step, metrics)
    return metrics


def evaluate_rollout(
    model: Aurora,
    batch: Batch,
    lead_hours: int,
    lat: torch.Tensor,
) -> dict[str, float]:
    """Autoregressively roll out model to lead_hours and return RMSE per variable.

    Returns dict with keys like 'rmse/2t', 'rmse/z500', 'rmse/t850', 'rmse/10u'.
    """
    raise NotImplementedError
