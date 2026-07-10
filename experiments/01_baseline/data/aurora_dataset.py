import glob
import bisect
import os
from datetime import datetime
from itertools import accumulate
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from netCDF4 import Dataset as ncDataset, num2date


class AuroraDataset(Dataset):
    """Fine-tuning dataset for Aurora over ERA5-derived NetCDF files.

    Each file (aurora_{year}.nc) contains hourly ERA5 data on a 180×360 grid.
    Surface variables are stored directly (e.g. ``2t``, ``10u``); pressure-level
    variables are stored as flattened 2-D fields named ``{var}{level}``
    (e.g. ``z500``, ``t850``).

    Each sample consists of three consecutive frames separated by ``dt`` index
    steps:
        prev  (t - dt)  ─┐
        curr  (t)        ├─► input_batch  (surf/atmos shape: 1, 2, H, W / 1, 2, L, H, W)
        next  (t + dt)  ─── target        (surf/atmos shape: 1, H, W / 1, L, H, W)

    Returns
    -------
    (input_batch, target)
        input_batch : aurora.Batch
        target      : dict with "surf_vars" and "atmos_vars" tensors
    """

    def __init__(
        self,
        nc_file_paths: list[str],
        surf_vars: tuple = ("2t", "10u", "10v", "msl"),
        atmos_vars: tuple = ("z", "u", "v", "t", "q"),
        atmos_levels: tuple = (200, 500, 700, 850),
        static_nc: Optional[str] = None,
        static_vars: tuple = ("lsm", "z", "slt"),
        dt: int = 6,
    ) -> None:
        """
        Parameters
        ----------
        nc_file_paths : list of paths to aurora_{year}.nc files to load, in any order
        surf_vars    : surface variable names expected by the encoder
        atmos_vars   : atmospheric variable names (stored as {var}{level} in NC)
        atmos_levels : pressure levels to load, must exist in the NC files
        static_nc    : path to a separate NC file containing static fields (required)
        static_vars  : variable names to read from static_nc
        dt           : index stride between frames (6 → 6 h for hourly ERA5)
        """
        self.surf_vars = surf_vars
        self.atmos_vars = atmos_vars
        self.atmos_levels = atmos_levels
        self.dt = dt

        self.file_paths = sorted(nc_file_paths)
        if not self.file_paths:
            raise FileNotFoundError(
                f"No NC files given: nc_file_paths was empty ({nc_file_paths!r}). "
                "Check the data root / year range used to build this list."
            )

        self._handles: list[Optional[ncDataset]] = [None] * len(self.file_paths)
        self._build_index()

        with ncDataset(self.file_paths[0], "r") as ds:
            self.lat = torch.tensor(ds["lat"][:].data.astype(np.float32))
            self.lon = torch.tensor(ds["lon"][:].data.astype(np.float32))

        if not static_nc:
            raise FileNotFoundError(
                "static_nc is required: no path was given for the static fields "
                f"NC file (requested static_vars={list(static_vars)!r})."
            )
        if not os.path.exists(static_nc):
            raise FileNotFoundError(
                f"static_nc path does not exist: {static_nc!r} "
                f"(requested static_vars={list(static_vars)!r})."
            )
        self.static = self._load_static(static_nc, static_vars)

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        n_steps = []
        for path in self.file_paths:
            with ncDataset(path, "r") as ds:
                n_steps.append(int(ds["time"].shape[0]))
        self.n_steps = n_steps
        # valid centre indices per file: [dt, n_steps - dt), giving n_steps - 2*dt samples
        self.valid_counts = [max(0, n - 2 * self.dt) for n in n_steps]
        self.offsets = [0] + list(accumulate(self.valid_counts))[:-1]

    def __len__(self) -> int:
        return sum(self.valid_counts)

    # ------------------------------------------------------------------
    # File access helpers
    # ------------------------------------------------------------------

    def _handle(self, file_idx: int) -> ncDataset:
        if self._handles[file_idx] is None:
            self._handles[file_idx] = ncDataset(self.file_paths[file_idx], "r")
        return self._handles[file_idx]

    def _load_static(self, path: str, vars: tuple) -> dict:
        with ncDataset(path, "r") as ds:
            return {v: torch.tensor(ds[v][:].data.astype(np.float32)) for v in vars}

    def _decode_time(self, ds: ncDataset, tidx: int) -> datetime:
        t_var = ds["time"]
        cal = getattr(t_var, "calendar", "standard")
        t = num2date(t_var[tidx], units=t_var.units, calendar=cal)
        return datetime(t.year, t.month, t.day, t.hour, t.minute)

    # ------------------------------------------------------------------
    # Variable readers
    # ------------------------------------------------------------------

    def _read_surf(self, ds: ncDataset, tidx: int) -> dict:
        return {
            v: torch.tensor(ds[v][tidx].data.astype(np.float32))
            for v in self.surf_vars
        }

    def _read_atmos(self, ds: ncDataset, tidx: int) -> dict:
        # pressure-level vars are stored flat: {var}{level} → stack into (L, H, W)
        return {
            v: torch.stack([
                torch.tensor(ds[f"{v}{lev}"][tidx].data.astype(np.float32))
                for lev in self.atmos_levels
            ])
            for v in self.atmos_vars
        }

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, global_idx: int):
        file_idx = bisect.bisect_right(self.offsets, global_idx) - 1
        local_valid = global_idx - self.offsets[file_idx]
        t_curr = local_valid + self.dt      # centre frame index within file
        t_prev = t_curr - self.dt
        t_next = t_curr + self.dt

        ds = self._handle(file_idx)

        s_prev = self._read_surf(ds, t_prev)
        s_curr = self._read_surf(ds, t_curr)
        s_next = self._read_surf(ds, t_next)

        a_prev = self._read_atmos(ds, t_prev)
        a_curr = self._read_atmos(ds, t_curr)
        a_next = self._read_atmos(ds, t_next)

        # input: stack [prev, curr] along new time axis → (1, 2, H, W) / (1, 2, L, H, W)
        surf_in  = {v: torch.stack([s_prev[v], s_curr[v]]).unsqueeze(0) for v in self.surf_vars}
        atmos_in = {v: torch.stack([a_prev[v], a_curr[v]]).unsqueeze(0) for v in self.atmos_vars}

        surf_tgt  = {v: s_next[v].unsqueeze(0) for v in self.surf_vars}                                                             
        atmos_tgt = {v: a_next[v].unsqueeze(0) for v in self.atmos_vars}

	# pytorch dataloader requires batch must contain tensors, numpy arrays, numbers, dicts or lists
        # class 'aurora.batch.Batch' cannot be used here
        # return two dict instead.

        inp = {"surf_vars": surf_in, "atmos_vars": atmos_in, "static_vars": self.static, "metadata":{'lat':self.lat, 'lon':self.lon, 'time':(self._decode_time(ds, t_curr),), 'atmos_levels':self.atmos_levels}}
        tar = {"surf_vars": surf_tgt, "atmos_vars": atmos_tgt, "static_vars": self.static, "metadata":{'lat':self.lat, 'lon':self.lon, 'time':(self._decode_time(ds, t_next),), 'atmos_levels':self.atmos_levels}}

        return inp, tar

    def close(self) -> None:
        for h in self._handles:
            if h is not None:
                h.close()
