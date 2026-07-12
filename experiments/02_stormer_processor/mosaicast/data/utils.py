"""Dataset construction and chronological splitting utilities."""
from __future__ import annotations

import os

from torch.utils.data import ConcatDataset, Dataset, Subset

from .aurora_dataset import AuroraDataset


def make_datasets(ds_cfg: dict) -> tuple[Dataset, Dataset, Dataset]:
    """Return (train_dataset, val_dataset, test_dataset).

    Split logic (CLAUDE.md — chronological, never random):
      - Train: full years from train_year_start up to (not including) train_year_end.
      - Val/Test: the "boundary year" (int(train_year_end)) is loaded once and split
        into val and test using torch.utils.data.Subset with fraction-based indices.

    Assumes train_year_end is an integer (no fractional training cutoff within a year).
    File naming convention: {root}/aurora_{year}.nc
    """
    root = os.path.expandvars(ds_cfg["root"])
    static_nc = ds_cfg.get("static_nc")
    # static_nc is a filename relative to root (same convention as nc_filename)
    if static_nc:
        static_nc = f"{root}/{static_nc}"

    common_kwargs = {
        'surf_vars': tuple(ds_cfg['surf_vars']),
        'atmos_vars': tuple(ds_cfg['atmos_vars']),
        'atmos_levels': tuple(ds_cfg['atmos_levels']),
        'static_nc': static_nc,
        'static_vars': tuple(ds_cfg['static_vars']),
        'dt': int(ds_cfg['lead_hours']),
    }

    nc_template = ds_cfg.get("nc_filename", "aurora_{year}.nc")
    def build_file_list(start_year, end_year):
        files = []
        for y in range(int(start_year), int(end_year) + 1):
            filename = nc_template.format(year=y)
            files.append(os.path.join(root, filename))
        return files

    def _ds(fpths: list[str]) -> AuroraDataset:
        return AuroraDataset(
            nc_file_paths=fpths,
            **common_kwargs
        )

    ds = []
    for ci in ["train", "val", "test"]:
      ds.append(_ds(build_file_list(ds_cfg[f"{ci}_year_start"], ds_cfg[f"{ci}_year_end"])))

    return ds 
