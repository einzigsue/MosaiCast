"""WB2/ERA5 loaders returning (inp, tar) pairs (CLAUDE.md §4, M3).

History step convention — CLAUDE.md §1.9, "Option A" (Stormer-style):
  Input frames are always [t − HISTORY_DT, t] = [t − 6h, t], regardless of δt.
  δt only affects the target frame and the adaLN-zero conditioning signal.
  Do NOT change HISTORY_DT to match δt ("Option B", rejected — see §1.9).

__getitem__ interface:
  Accepts a (global_idx, dt_hours_int) tuple produced by DtBatchSampler.
  Loads exactly one inp (2 frames) and one tar (1 frame at t + dt).
  Returns (inp_dict, tar_dict) — no wasted target reads.
"""
from __future__ import annotations

import bisect
from itertools import accumulate

import torch

from .aurora_dataset import AuroraDataset


class MosaicastDataset(AuroraDataset):
    """ERA5 dataset keyed by ``(global_idx, dt_hours_int)``.

    ``__getitem__((idx, dt))`` loads exactly:
      - inp: frames ``[t − 6h, t]`` (Option A, fixed history step)
      - tar: single frame ``[t + dt]``

    Returns ``(inp_dict, tar_dict)``.

    The caller (``DtBatchSampler``) is responsible for choosing dt and
    ensuring all samples in a batch share the same value.

    Args:
        nc_file_paths: NC file paths (same convention as AuroraDataset).
        surf_vars, atmos_vars, atmos_levels, static_nc, static_vars:
            forwarded to AuroraDataset.
        dt_hours: supported δt values in hours — used ONLY to set the valid
            index range (shrunk so that t + max_dt stays within the file).
    """

    # Fixed 6 h history stride — CLAUDE.md §1.9.  Do not vary with δt.
    HISTORY_DT: int = 6

    def __init__(
        self,
        nc_file_paths: list[str],
        surf_vars: tuple = ("2t", "10u", "10v", "msl"),
        atmos_vars: tuple = ("z", "u", "v", "t", "q"),
        atmos_levels: tuple = (50, 250, 500, 600, 700, 850, 925),
        static_nc: str | None = None,
        static_vars: tuple = ("lsm", "z", "slt"),
        dt_hours: list[int] | None = None,
        frame_cache_size: int = 32,
    ) -> None:
        if not dt_hours:
            raise ValueError("dt_hours must be a non-empty list")
        # Set before super().__init__ so our _build_index override can use it.
        self.dt_hours_list: list[int] = sorted(dt_hours)
        super().__init__(
            nc_file_paths=nc_file_paths,
            surf_vars=surf_vars,
            atmos_vars=atmos_vars,
            atmos_levels=atmos_levels,
            static_nc=static_nc,
            static_vars=static_vars,
            dt=self.HISTORY_DT,
            frame_cache_size=frame_cache_size,
        )

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        from netCDF4 import Dataset as ncDataset

        n_steps = []
        for path in self.file_paths:
            with ncDataset(path, "r") as ds:
                n_steps.append(int(ds["time"].shape[0]))
        self.n_steps = n_steps

        max_dt = max(self.dt_hours_list) if hasattr(self, "dt_hours_list") else self.HISTORY_DT

        # Cumulative step counts — used to map a global time step to (file, local).
        # Cross-file reads are supported: t_prev and t_next may span file boundaries.
        self._file_offsets: list[int] = [0] + list(accumulate(n_steps[:-1]))

        n_total = sum(n_steps)
        n_valid = max(0, n_total - self.HISTORY_DT - max_dt)

        # Per-file valid counts (informational): samples where t_curr falls in each file.
        # Boundary files gain samples relative to the old intra-file design:
        #   first file loses HISTORY_DT samples at the start (t_prev must exist);
        #   last  file loses max_dt      samples at the end   (t_next must exist).
        file_valid: list[int] = []
        for i, n in enumerate(n_steps):
            f_start = self._file_offsets[i]
            t_lo = max(self.HISTORY_DT, f_start)
            t_hi = min(n_total - max_dt, f_start + n)
            file_valid.append(max(0, t_hi - t_lo))
        self.valid_counts = file_valid
        self.offsets = [0] + list(accumulate(file_valid[:-1]))  # kept for display compatibility
        self._n_valid = n_valid

    def __len__(self) -> int:
        return self._n_valid

    # ------------------------------------------------------------------
    # Cross-file lookup
    # ------------------------------------------------------------------

    def _global_t_to_file(self, global_t: int) -> tuple[int, int]:
        """Return ``(file_idx, local_step)`` for a global time step."""
        file_idx = bisect.bisect_right(self._file_offsets, global_t) - 1
        return file_idx, global_t - self._file_offsets[file_idx]

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, key: tuple[int, int]) -> tuple[dict, dict]:
        """Load one (inp, tar) pair for the given sample index and δt.

        Args:
            key: ``(global_idx, dt_hours_int)`` — produced by DtBatchSampler.
                 ``global_idx`` selects the sample; ``dt_hours_int`` selects
                 which target frame to load.  Exactly one target frame is read.

        Returns:
            ``(inp_dict, tar_dict)`` where:
              - ``inp_dict`` has surf/atmos frames ``[t−6h, t]`` on the T axis.
              - ``tar_dict`` has the single frame ``[t + dt]``.
              Both dicts include ``metadata`` with ``time`` set to ``t`` and
              ``t + dt`` respectively, so lead time is recoverable as
              ``tar.time[0] − inp.time[0]``.
        """
        global_idx, dt = key

        # Global time positions (span all files in chronological order)
        t_curr_g = global_idx + self.HISTORY_DT
        t_prev_g = t_curr_g - self.HISTORY_DT   # always HISTORY_DT back (Option A)
        t_next_g = t_curr_g + dt

        # Resolve (file_idx, local_step) — cross-file reads handled transparently
        f_p, l_p = self._global_t_to_file(t_prev_g)
        f_c, l_c = self._global_t_to_file(t_curr_g)
        f_n, l_n = self._global_t_to_file(t_next_g)

        # --- Input: two history frames [t−6h, t] ---
        s_prev, a_prev = self._read_frame_cached(f_p, l_p)
        s_curr, a_curr = self._read_frame_cached(f_c, l_c)

        surf_in  = {v: torch.stack([s_prev[v], s_curr[v]]).unsqueeze(0)
                    for v in self.surf_vars}
        atmos_in = {v: torch.stack([a_prev[v], a_curr[v]]).unsqueeze(0)
                    for v in self.atmos_vars}

        inp = {
            "surf_vars":   surf_in,
            "atmos_vars":  atmos_in,
            "static_vars": self.static,
            "metadata": {
                "lat":          self.lat,
                "lon":          self.lon,
                "time":         (self._decode_time(self._handle(f_c), l_c),),
                "atmos_levels": self.atmos_levels,
            },
        }

        # --- Target: single frame [t + dt] ---
        s_next, a_next = self._read_frame_cached(f_n, l_n)

        tar = {
            "surf_vars":   {v: s_next[v].unsqueeze(0) for v in self.surf_vars},
            "atmos_vars":  {v: a_next[v].unsqueeze(0) for v in self.atmos_vars},
            "static_vars": self.static,
            "metadata": {
                "lat":          self.lat,
                "lon":          self.lon,
                "time":         (self._decode_time(self._handle(f_n), l_n),),
                "atmos_levels": self.atmos_levels,
            },
        }

        return inp, tar
