"""DataLoader collate functions for AuroraDataset and MosaicastDataset."""
from __future__ import annotations

from typing import Callable

import torch
from aurora import Batch, Metadata

from mosaicast.patching.plans import uniform_plan
from mosaicast.patching.plan import PatchPlan


def _stack_batches(dicts: tuple, unsqueeze_time: bool = False) -> Batch:
    """Stack single-sample dicts into one aurora.Batch."""
    def _cat(key: str) -> dict:
        return {k: torch.cat([d[key][k] for d in dicts], dim=0) for k in dicts[0][key]}

    surf  = _cat("surf_vars")
    atmos = _cat("atmos_vars")

    if unsqueeze_time:
        surf  = {k: v.unsqueeze(1) for k, v in surf.items()}
        atmos = {k: v.unsqueeze(1) for k, v in atmos.items()}

    return Batch(
        surf_vars=surf,
        static_vars=dicts[0]["static_vars"],
        atmos_vars=atmos,
        metadata=Metadata(
            lat=dicts[0]["metadata"]["lat"],
            lon=dicts[0]["metadata"]["lon"],
            time=sum((d["metadata"]["time"] for d in dicts), ()),
            atmos_levels=tuple(float(l) for l in dicts[0]["metadata"]["atmos_levels"]),
        ),
    )


def aurora_collate_fn(samples: list) -> tuple[Batch, Batch]:
    """Return (input_batch, target_batch) as aurora.Batch objects."""
    inps, tars = zip(*samples)
    return _stack_batches(inps), _stack_batches(tars, unsqueeze_time=True)


def mosaicast_collate_fn(
    patch_size: int = 4,
    plan_fn: Callable | None = None,
):
    """Factory returning a collate function for MosaicastDataset.

    Each sample is already a plain ``(inp_dict, tar_dict)`` pair — the δt was
    selected by ``DtBatchSampler`` before ``__getitem__`` was called, so the
    collate has no randomness to manage and no targets to discard.

    The collate:
      1. Stacks all inp dicts into one ``aurora.Batch``.
      2. Stacks all tar dicts into one ``aurora.Batch`` (T=1 dim added).
      3. Builds a ``PatchPlan`` via plan_fn (default: uniform_plan).
      4. Returns ``(inp_batch, tar_batch, patch_plan)``.

    The chosen δt is implicit in the Batch metadata::

        dt_hours = int((tar.metadata.time[0] - inp.metadata.time[0])
                       .total_seconds() / 3600)

    Args:
        patch_size: canonical patch size ``p`` for uniform_plan (default 4).
                    Ignored when plan_fn is provided.
        plan_fn:    callable(inp_batch, lat, lon) → PatchPlan.
                    If None, defaults to uniform_plan(lat, lon, p=patch_size).
                    For content-adaptive plans, pass a partial that captures the
                    budget, criterion variable, and level.  Called per-batch so
                    content-adaptive plans re-compute per sample (A7 ablation).
    """
    def collate(samples: list) -> tuple[Batch, Batch, PatchPlan]:
        inps, tars = zip(*samples)

        inp_batch = _stack_batches(inps)
        tar_batch = _stack_batches(tars, unsqueeze_time=True)

        lat = inps[0]["metadata"]["lat"]
        lon = inps[0]["metadata"]["lon"]

        if plan_fn is not None:
            plan = plan_fn(inp_batch, lat, lon)
        else:
            plan = uniform_plan(lat, lon, p=patch_size)

        return inp_batch, tar_batch, plan

    return collate
