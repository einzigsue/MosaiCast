"""Interval-combination inference (CLAUDE.md §5, M5).

Scientific thesis: averaging predictions from multiple δt paths at long lead times
reduces variance without systematic bias, yielding lower RMSE than any single-member
path.  The single-member path is retained for extremes/tracking because averaging
blurs individual event amplitudes (A5 risk).

Path enumeration
----------------
At a target lead time T, a "homogeneous path" uses one δt throughout:
    K steps × δt hours = T hours  →  (δt, K) pair
For T = 168h (day 7) and δt ∈ {6, 12, 24}: [(6,28), (12,14), (24,7)] — all valid.
For any headline lead time (multiple of 24h) all three δt are always valid.

Single-member default
---------------------
dt_hours=24 (fewest autoregressive steps, least compound error accumulation).
Expose as a parameter; use rank_intervals() on the validation set to find the
empirically best single δt before committing to a default.
"""
from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

import torch

from aurora import Batch

from mosaicast.dynamics.rollout import rollout
from mosaicast.metrics import lat_rmse
from mosaicast.patching.plan import PatchIndex

if TYPE_CHECKING:
    import torch.nn as nn

__all__ = [
    "dt_paths",
    "average_batches",
    "single_member_forecast",
    "homogeneous_ensemble",
    "best_m_of_n_ensemble",
    "rank_intervals",
]


def dt_paths(
    target_lead: timedelta,
    dt_support: tuple[int, ...] = (6, 12, 24),
) -> list[tuple[int, int]]:
    """List all (δt_hours, K) pairs that reach target_lead homogeneously.

    A pair is valid when K × δt == T and K ≥ 1.

    Args:
        target_lead: desired forecast horizon.
        dt_support:  supported δt values in hours (default (6, 12, 24)).

    Returns:
        List of (dt_hours, k) pairs in the order of dt_support.

    Example::

        dt_paths(timedelta(days=7))
        # → [(6, 28), (12, 14), (24, 7)]
    """
    T = int(round(target_lead.total_seconds() / 3600))
    return [(dt, T // dt) for dt in dt_support if T % dt == 0 and T // dt >= 1]


def average_batches(batches: list[Batch]) -> Batch:
    """Variable-wise arithmetic mean of a list of Batches.

    All batches must share the same variable names, levels, and target time.
    Metadata is taken from batches[0].

    Args:
        batches: non-empty list of Batch at the same target time.

    Returns:
        Batch whose tensors are the element-wise mean across the list.
    """
    if not batches:
        raise ValueError("average_batches requires at least one Batch")
    surf  = {
        v: torch.stack([b.surf_vars[v]  for b in batches]).mean(0)
        for v in batches[0].surf_vars
    }
    atmos = {
        v: torch.stack([b.atmos_vars[v] for b in batches]).mean(0)
        for v in batches[0].atmos_vars
    }
    return Batch(
        surf_vars=surf,
        static_vars=batches[0].static_vars,
        atmos_vars=atmos,
        metadata=batches[0].metadata,
    )


def single_member_forecast(
    model:       "nn.Module",
    inp:         Batch,
    index:       PatchIndex,
    target_lead: timedelta,
    dt_hours:    int = 24,
) -> Batch:
    """Single-δt autoregressive forecast to target_lead.

    Runs K = target_lead / dt_hours steps of rollout under no_grad.
    Retain this path for extremes/tracking (averaging blurs amplitudes).

    Args:
        model:       MosaicastModel with stats attached.
        inp:         Input Batch (T=2 history frames).
        index:       PatchIndex for the current grid.
        target_lead: Desired forecast horizon.
        dt_hours:    Per-step δt (default 24h — fewest steps, least compound error).

    Returns:
        Batch at t + target_lead.

    Raises:
        ValueError: if target_lead is not a positive integer multiple of dt_hours.
    """
    T = int(round(target_lead.total_seconds() / 3600))
    if T % dt_hours != 0 or T // dt_hours < 1:
        raise ValueError(
            f"target_lead ({T}h) is not a positive integer multiple of dt_hours ({dt_hours}h)"
        )
    k = T // dt_hours
    model.eval()
    with torch.no_grad():
        preds, _ = rollout(model, inp, index, timedelta(hours=dt_hours), k=k)
    return preds[-1]


def homogeneous_ensemble(
    model:       "nn.Module",
    inp:         Batch,
    index:       PatchIndex,
    target_lead: timedelta,
    dt_support:  tuple[int, ...] = (6, 12, 24),
) -> Batch:
    """Average all valid homogeneous δt paths to target_lead.

    Variance reduction: independent paths with different step sizes have partially
    uncorrelated errors that cancel in the mean.

    Args:
        model:       MosaicastModel with stats attached.
        inp:         Input Batch.
        index:       PatchIndex.
        target_lead: Desired forecast horizon.
        dt_support:  δt values to include (only those dividing target_lead are used).

    Returns:
        Averaged Batch at t + target_lead.

    Raises:
        ValueError: if no valid paths exist for this target_lead and dt_support.
    """
    paths = dt_paths(target_lead, dt_support)
    if not paths:
        raise ValueError(
            f"No valid δt paths for target_lead={target_lead} with dt_support={dt_support}"
        )
    members = [
        single_member_forecast(model, inp, index, target_lead, dt_hours=dt)
        for dt, _ in paths
    ]
    return average_batches(members)


def best_m_of_n_ensemble(
    model:       "nn.Module",
    inp:         Batch,
    index:       PatchIndex,
    target_lead: timedelta,
    m:           int,
    ranking:     list[int],
) -> Batch:
    """Average the top-m δt paths by short-lead validation skill.

    Args:
        model:       MosaicastModel with stats attached.
        inp:         Input Batch.
        index:       PatchIndex.
        target_lead: Desired forecast horizon.
        m:           Number of members to average.
        ranking:     δt values sorted best-first by short-lead lat-RMSE
                     (produced by rank_intervals()).

    Returns:
        Averaged Batch from the top-m valid members.

    Raises:
        ValueError: if fewer than m valid paths exist.
    """
    T      = int(round(target_lead.total_seconds() / 3600))
    chosen = [dt for dt in ranking if T % dt == 0 and T // dt >= 1][:m]
    if len(chosen) < m:
        raise ValueError(
            f"Only {len(chosen)} valid paths for target_lead={target_lead}, "
            f"but m={m} requested.  ranking={ranking}"
        )
    members = [
        single_member_forecast(model, inp, index, target_lead, dt_hours=dt)
        for dt in chosen
    ]
    return average_batches(members)


def rank_intervals(
    model:       "nn.Module",
    val_batches: list[tuple[Batch, Batch]],
    index:       PatchIndex,
    score_lead:  timedelta,
    lat:         "torch.Tensor",
    dt_support:  tuple[int, ...] = (6, 12, 24),
    surf_var:    str = "2t",
) -> list[int]:
    """Rank δt values by mean lat-RMSE on the validation set at score_lead.

    Args:
        model:       MosaicastModel with stats attached (will be set to eval).
        val_batches: List of (inp, tar) validation pairs.
        index:       PatchIndex for the validation grid.
        score_lead:  Lead time to score at (e.g. timedelta(days=2)).
        lat:         (H,) latitude tensor.
        dt_support:  δt values to rank (only those dividing score_lead are scored).
        surf_var:    Surface variable used for scoring (default "2t").

    Returns:
        δt values sorted ascending by mean lat-RMSE (best first).
    """
    T          = int(round(score_lead.total_seconds() / 3600))
    valid_dts  = [dt for dt in dt_support if T % dt == 0]
    if not valid_dts:
        raise ValueError(f"No valid δt for score_lead={score_lead} in {dt_support}")

    scores: dict[int, float] = {}
    for dt in valid_dts:
        rmse_vals = []
        for inp, tar in val_batches:
            pred = single_member_forecast(model, inp, index, score_lead, dt_hours=dt)
            rmse = lat_rmse(
                pred.surf_vars[surf_var],
                tar.surf_vars[surf_var][:, -1],
                lat,
            )
            rmse_vals.append(rmse.mean().item())
        scores[dt] = sum(rmse_vals) / len(rmse_vals)

    return sorted(valid_dts, key=lambda dt: scores[dt])
