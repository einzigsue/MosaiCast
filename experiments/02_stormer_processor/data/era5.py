"""ERA5 dataset loader — xarray/zarr → aurora.Batch.

Chronological splits by year fraction; never random (CLAUDE.md rule).
"""
from __future__ import annotations
from pathlib import Path
import xarray as xr
import torch
from torch.utils.data import Dataset
from aurora import Batch, Metadata

from .synthetic import ATMOS_LEVELS, ATMOS_VARS, STATIC_VARS, SURF_VARS


class ERA5Dataset(Dataset):
    """ERA5 zarr/netcdf store → aurora.Batch with chronological splits.

    Args:
        root: path to the ERA5 store (zarr or netcdf directory).
        split: 'train', 'val', or 'test'.
        train_year_start: first year of training data (float, e.g. 2010.0).
        train_year_end: exclusive end of training split (e.g. 2013.0 = 3 years).
        val_year_end: exclusive end of validation split (e.g. 2013.5).
        test_year_end: exclusive end of test split (e.g. 2014.0).
        lead_hours: forecast lead time (6 = single step).
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        train_year_start: float,
        train_year_end: float,
        val_year_end: float,
        test_year_end: float,
        lead_hours: int = 6,
    ) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple[Batch, Batch]:
        """Return (input_batch, target_batch) pair."""
        raise NotImplementedError
