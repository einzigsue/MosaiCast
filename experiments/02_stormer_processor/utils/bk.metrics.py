"""Latitude-weighted metrics (CLAUDE.md rule 5)."""
from __future__ import annotations
import torch


def lat_weights(lat: torch.Tensor) -> torch.Tensor:
    """Normalised cos(lat) weights, shape (H,)."""
    w = torch.cos(torch.deg2rad(lat)).clamp(min=0)
    return w / w.mean()


def lat_weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    lat: torch.Tensor,
) -> torch.Tensor:
    """(B, T, H, W) → scalar MSE weighted by cos(lat)."""
    w = lat_weights(lat).to(pred.device)[None, None, :, None]
    return ((pred - target) ** 2 * w).mean()


def lat_weighted_rmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    lat: torch.Tensor,
) -> torch.Tensor:
    return lat_weighted_mse(pred, target, lat).sqrt()
