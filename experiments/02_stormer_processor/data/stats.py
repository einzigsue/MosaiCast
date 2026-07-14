"""Per-gridpoint mean/std for thresholded RMSE (Aurora SI Eqs. F17-F20).

The thresholded RMSE needs, for each variable/level, the per-latitude-longitude
mean ``mu_ij`` and standard deviation ``sigma_ij`` of the *target* field over all
training years (Eq. F20: ``b_ij = mu_ij + g * sigma_ij``). These are distinct
from Aurora's normalisation statistics, which are single scalars per variable.

Computed once from the training-split NC files with a single streaming pass
(sum and sum-of-squares), then cached to an ``.npz``. Surface variables are read
by their own name; atmospheric variables are read per level from the flat
``{var}{level}`` fields, matching :class:`apa.data.aurora_dataset.AuroraDataset`.

Keys in the cache use the same convention as the metric report keys:
``"2t"`` for surface, ``"z500"`` for atmospheric var ``z`` at 500 hPa.
"""
from __future__ import annotations
import glob
from pathlib import Path

import numpy as np
import torch
from netCDF4 import Dataset as ncDataset

_EPS = 1e-8


def gridpoint_stat_key(var: str, level: int | None = None) -> str:
    """Cache key: ``"2t"`` (surface) or ``"z500"`` (atmospheric)."""
    return var if level is None else f"{var}{level}"


def build_gridpoint_stats(
    nc_glob: str,
    surf_vars: tuple[str, ...] = ("2t", "10u", "10v", "msl"),
    atmos_vars: tuple[str, ...] = ("z", "u", "v", "t", "q"),
    atmos_levels: tuple[int, ...] = (50, 250, 500, 600, 700, 850, 1000),
    out_path: str | Path | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """Stream per-gridpoint mean/std over all frames in the matched NC files.

    ``nc_glob`` should match only the **training** files (never val/test — the
    thresholds must be defined from training data, Eq. F20). Returns
    ``{key: {"mu": (H, W), "sigma": (H, W),"unit": str}}``; also writes ``out_path`` (npz)
    when given.
    """
    paths = sorted(glob.glob(nc_glob))
    if not paths:
        raise FileNotFoundError(f"No NC files matched: {nc_glob}")

    keys = [gridpoint_stat_key(v) for v in surf_vars] + [
        gridpoint_stat_key(v, lev) for v in atmos_vars for lev in atmos_levels
    ]
    field_names = {gridpoint_stat_key(v): v for v in surf_vars}
    field_names.update(
        {gridpoint_stat_key(v, lev): f"{v}{lev}" for v in atmos_vars for lev in atmos_levels}
    )

    ssum: dict[str, np.ndarray] = {}
    ssq: dict[str, np.ndarray] = {}
    units_dict: dict[str, str] = {}

    count = 0

    for path in paths:
        with ncDataset(path, "r") as ds:
            n_t = int(ds["time"].shape[0])
            for key in keys:

                if key not in units_dict:
                    units_dict[key] = getattr(ds[field_names[key]], "units", "unknown")

                arr = np.asarray(ds[field_names[key]][:], dtype=np.float64)  # (T, H, W)
                s = arr.sum(axis=0)
                q = (arr ** 2).sum(axis=0)

                if key not in ssum:
                    ssum[key] = s
                    ssq[key] = q
                else:
                    ssum[key] += s
                    ssq[key] += q
            count += n_t

    stats: dict[str, dict[str, torch.Tensor]] = {}
    for key in keys:
        mu = ssum[key] / count
        var = ssq[key] / count - mu ** 2
        sigma = np.sqrt(np.clip(var, _EPS, None))
        stats[key] = {
            "mu": torch.tensor(mu, dtype=torch.float32),
            "sigma": torch.tensor(sigma, dtype=torch.float32),
            "unit": units_dict.get(key, "unknown"),
        }

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        flat = {}
        for key in keys:
            flat[f"{key}__mu"] = stats[key]["mu"].numpy()
            flat[f"{key}__sigma"] = stats[key]["sigma"].numpy()
            flat[f"{key}__unit"] = np.array(stats[key]["unit"], dtype=str)
        np.savez(out_path, **flat)

    return stats


def load_gridpoint_stats(path: str | Path) -> dict[str, dict[str, torch.Tensor]]:
    """Load a cache written by :func:`build_gridpoint_stats`."""
    data = np.load(path)
    stats: dict[str, dict[str, torch.Tensor]] = {}
    for name in data.files:
        key, which = name.rsplit("__", 1)
        if which == "unit":
            stats.setdefault(key, {})[which] = str(data[name])
        else:
            stats.setdefault(key, {})[which] = torch.tensor(data[name], dtype=torch.float32)
    return stats
