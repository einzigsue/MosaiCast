"""DataLoader collate function for AuroraDataset."""
from __future__ import annotations

import torch
from aurora import Batch, Metadata


def _stack_batches(dicts: tuple, unsqueeze_time: bool = False) -> Batch:
    """Stack single-sample dicts from AuroraDataset into one aurora.Batch.

    surf/atmos: cat along dim 0 (each sample has B=1).
    static, lat, lon, atmos_levels: identical across samples — take the first.
    time: one datetime per sample — concatenate into a flat tuple.

    unsqueeze_time=True adds a T=1 dimension to surf/atmos (needed for the
    target, whose shapes are (B,H,W)/(B,L,H,W) but the loss expects (B,T,...)).
    """
    def _cat(key: str) -> dict:
        return {k: torch.cat([d[key][k] for d in dicts], dim=0) for k in dicts[0][key]}

    surf = _cat("surf_vars")
    atmos = _cat("atmos_vars")

    if unsqueeze_time:
        surf = {k: v.unsqueeze(1) for k, v in surf.items()}
        atmos = {k: v.unsqueeze(1) for k, v in atmos.items()}

    return Batch(
        surf_vars=surf,
        static_vars=dicts[0]["static_vars"],
        atmos_vars=atmos,
        metadata=Metadata(
            lat=dicts[0]["metadata"]["lat"],
            lon=dicts[0]["metadata"]["lon"],
            time=sum((d["metadata"]["time"] for d in dicts), ()),
            # Float levels: int levels become an int64 tensor in the encoder's
            # levels_expansion, where int64 logspace truncates wavelengths to 0
            # (2*pi/0 = inf -> sin(inf) = NaN). The dataset keeps ints for NC
            # variable names; the model must see floats.
            atmos_levels=tuple(float(l) for l in dicts[0]["metadata"]["atmos_levels"]),
        ),
    )


def aurora_collate_fn(samples: list) -> tuple[Batch, Batch]:
    """Return (input_batch, target_batch) as aurora.Batch objects."""
    inps, tars = zip(*samples)
    return _stack_batches(inps), _stack_batches(tars, unsqueeze_time=True)


