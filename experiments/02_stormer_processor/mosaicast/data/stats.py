"""Offline residual-stats estimation entrypoint (CLAUDE.md §1.5, §8.2).

ResidualStats live in dynamics/randomized.py; this module provides the
offline estimation loop that scans the training split to compute per-δt,
per-variable, per-level mean/std of Δ = X_{t+δt} − X_t (dynamics mode,
control) or X_{t+δt} directly (absolute mode, A1 ablation).
"""
from __future__ import annotations

from mosaicast.dynamics.randomized import ResidualStats


def estimate_residual_stats(
    dataset,
    dt_hours: list[int],
    out_path: str,
    max_samples: int | None = None,
    mode: str = "dynamics",
) -> ResidualStats:
    """Scan dataset, compute residual statistics, write to out_path (.npz).

    For each sample index i and each dt in dt_hours, calls
    ``dataset[(i, dt)]`` to load exactly one (inp, tar) pair — no wasted
    target reads.  Accumulates per-(dt, var, level) statistics.

    Note: inp is loaded once per (i, dt) call, so the two history frames are
    read N_dt times for each sample.  This is acceptable for an offline
    one-time computation; the training loop (DtBatchSampler) pays no such cost.

    Args:
        dataset:     MosaicastDataset.
        dt_hours:    δt values to estimate stats for.
        out_path:    Path for the output .npz file.
        max_samples: Optional cap on dataset items scanned.
        mode:        "dynamics" (default/control) — accumulate Δ = X_{t+δt} − X_t.
                     "absolute" — accumulate X_{t+δt} directly (A1 ablation).
                     The mode is stored in the .npz so the inference path can
                     verify it matches the training target_mode config.

    Returns:
        The finalized ResidualStats object (same data as saved to out_path).
    """
    if mode not in ("dynamics", "absolute"):
        raise ValueError(f"mode must be 'dynamics' or 'absolute', got {mode!r}")

    stats = ResidualStats(
        dt_hours=dt_hours,
        surf_vars=dataset.surf_vars,
        atmos_vars=dataset.atmos_vars,
        atmos_levels=dataset.atmos_levels,
    )

    n = min(len(dataset), max_samples) if max_samples is not None else len(dataset)

    for i in range(n):
        for dt in dt_hours:
            inp_dict, tar_dict = dataset[(i, dt)]

            # Current state = last history frame (index -1 on the T axis).
            # Option A: inp frames are [t−6h, t], so index -1 is always t.
            if mode == "dynamics":
                inp_surf  = {v: inp_dict["surf_vars"][v][0, -1]  for v in dataset.surf_vars}
                inp_atmos = {v: inp_dict["atmos_vars"][v][0, -1] for v in dataset.atmos_vars}
                surf_sample  = {v: tar_dict["surf_vars"][v][0]  - inp_surf[v]
                                for v in dataset.surf_vars}
                atmos_sample = {v: tar_dict["atmos_vars"][v][0] - inp_atmos[v]
                                for v in dataset.atmos_vars}
            else:
                # A1 ablation: stats of the absolute target state X_{t+δt}
                surf_sample  = {v: tar_dict["surf_vars"][v][0]  for v in dataset.surf_vars}
                atmos_sample = {v: tar_dict["atmos_vars"][v][0] for v in dataset.atmos_vars}

            stats.update(surf_sample, atmos_sample, dt)

    stats.finalize()
    stats.save(out_path)
    return stats
