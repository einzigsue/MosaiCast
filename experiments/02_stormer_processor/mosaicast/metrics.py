"""Evaluation metrics (CLAUDE.md §7, M5/M6).

All functions operate on arbitrary leading dimensions (..., H, W) so they work
on a single field, a batch, or a batch of variables at once.
"""
from __future__ import annotations

import torch
from torch import Tensor


def lat_weights(lat: Tensor) -> Tensor:
    """Cosine latitude weights normalised to sum to 1, shape (H,)."""
    w = torch.cos(torch.deg2rad(lat.float()))
    return w / w.sum()


def lat_rmse(pred: Tensor, target: Tensor, lat: Tensor) -> Tensor:
    """Latitude-weighted RMSE.

    Args:
        pred:   (..., H, W)
        target: (..., H, W)
        lat:    (H,) latitude in degrees

    Returns:
        (...,) — one RMSE value per leading (batch/var/level) combination.
    """
    w        = lat_weights(lat).to(dtype=pred.dtype, device=pred.device)  # (H,)
    diff2    = (pred - target) ** 2                                        # (..., H, W)
    row_mean = diff2.mean(dim=-1)                                          # (..., H) lon mean
    mse      = (row_mean * w).sum(dim=-1)                                  # (...) lat-weighted
    return torch.sqrt(mse)


def acc(pred: Tensor, target: Tensor, clim: Tensor, lat: Tensor) -> Tensor:
    """Anomaly Correlation Coefficient (WB2 convention).

    Anomalies are relative to climatology; no additional mean removal.

    Args:
        pred:   (..., H, W)
        target: (..., H, W)
        clim:   (..., H, W) climatological mean (broadcastable)
        lat:    (H,) latitude in degrees

    Returns:
        (...,) ACC in [-1, 1].
    """
    H, W   = pred.shape[-2], pred.shape[-1]
    # Per-cell weight: cos(lat)/W — sums to 1 over the full H×W grid
    w_cell = lat_weights(lat).to(dtype=pred.dtype, device=pred.device) / W
    w_cell = w_cell.reshape((1,) * (pred.dim() - 2) + (H, 1))  # (..., H, 1)

    ap  = pred   - clim
    at  = target - clim
    num = (w_cell * ap * at).sum(dim=(-2, -1))
    den = torch.sqrt(
        (w_cell * ap * ap).sum(dim=(-2, -1)) *
        (w_cell * at * at).sum(dim=(-2, -1))
    )
    return num / den.clamp(min=1e-8)


def thresholded_rmse(
    pred: Tensor, target: Tensor, lat: Tensor, threshold: float
) -> Tensor:
    """Latitude-weighted RMSE restricted to cells where |target| > threshold.

    Returns NaN where no cell exceeds the threshold — aggregate with nanmean()
    to distinguish "no extremes present" from "perfect skill" (both return 0.0
    under a naïve implementation).

    Args:
        pred:      (..., H, W)
        target:    (..., H, W)
        lat:       (H,) latitude in degrees
        threshold: cells where |target| ≤ threshold are excluded

    Returns:
        (...,) RMSE over extreme cells, or NaN if no cell qualifies.
    """
    w_cos = torch.cos(torch.deg2rad(lat.float())).to(dtype=pred.dtype, device=pred.device)
    w_cos = w_cos.reshape((1,) * (pred.dim() - 2) + (-1, 1))  # (..., H, 1)

    mask = (target.abs() > threshold)
    num  = (w_cos * mask * (pred - target) ** 2).sum(dim=(-2, -1))
    den  = (w_cos * mask).sum(dim=(-2, -1))
    return torch.where(den > 0, torch.sqrt(num / den), torch.full_like(num, float("nan")))


def power_spectrum(field: Tensor) -> Tensor:
    """Zonal (longitude-axis) power spectrum averaged over latitude.

    Sharpness diagnostic: high-wavenumber tail reveals spectral energy loss
    from ensemble averaging (A5 ensemble-blur risk, CLAUDE.md §6).

    Args:
        field: (..., H, W)

    Returns:
        (..., W//2 + 1) raw linear power per wavenumber, averaged over latitude
        rows and any leading batch dims.  Take log10 before plotting.
    """
    W     = field.shape[-1]
    F     = torch.fft.rfft(field.float(), dim=-1)  # (..., H, W//2+1)
    power = (F.abs() ** 2) / (W ** 2)              # per-wavenumber power
    return power.mean(dim=-2)                       # average over lat rows
