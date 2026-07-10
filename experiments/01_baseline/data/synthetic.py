"""Synthetic data generator for unit tests and smoke runs."""
from __future__ import annotations
from datetime import datetime
import torch
from aurora import Batch, Metadata

SURF_VARS = ("2t", "10u", "10v", "msl")
STATIC_VARS = ("lsm", "z", "slt")
ATMOS_VARS = ("z", "u", "v", "t", "q")
ATMOS_LEVELS = (50, 250, 500, 600, 700, 850, 1000)


def make_synthetic_batch(
    H: int = 32,
    W: int = 64,
    B: int = 2,
    T: int = 2,
    device: torch.device | str = "cpu",
    seed: int = 0,
) -> Batch:
    """Random aurora.Batch satisfying all aurora metadata constraints.

    lat strictly decreasing in [-90, 90]; lon strictly increasing in [0, 360).
    """
    lat = torch.linspace(90, -90, H)
    lon = torch.linspace(0, 360, W + 1)[:-1]
    g = torch.Generator().manual_seed(seed)

    def rnd(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=g)

    return Batch(
        surf_vars={k: rnd(B, T, H, W) for k in SURF_VARS},
        static_vars={k: rnd(H, W) for k in STATIC_VARS},
        atmos_vars={k: rnd(B, T, len(ATMOS_LEVELS), H, W) for k in ATMOS_VARS},
        metadata=Metadata(
            lat=lat,
            lon=lon,
            time=tuple(datetime(2020, 1, 1, 6 * i) for i in range(B)),
            atmos_levels=ATMOS_LEVELS,
        ),
    ).to(device)
