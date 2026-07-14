"""Region-decomposed validation diagnostics (D6).

RMSE is logged separately for high-saliency (top 20%) and low-saliency (bottom
50%) regions. The reference saliency is ALWAYS grad-z500 (D1 formula) regardless
of which saliency the variant used — masks are identical across all runs.
"""
from __future__ import annotations
import torch
from aurora import Batch

from ..patching.saliency import grad_z500
from .metrics import lat_weights

def reference_saliency(batch: Batch, lat: torch.Tensor) -> torch.Tensor:
    """Compute the fixed D6 reference saliency (grad-z500, D1 formula)."""
    return grad_z500(batch, lat)


def hi_lo_masks(
    ref_sal: torch.Tensor,
    hi_percentile: float = 80.0,
    lo_percentile: float = 50.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (hi_mask, lo_mask) binary (H, W) tensors.

    hi: top (100 - hi_percentile)% of cells by saliency value.
    lo: bottom lo_percentile% of cells.
    """
    flat = ref_sal.flatten()
    hi_thresh = torch.quantile(flat, hi_percentile / 100.0)
    lo_thresh = torch.quantile(flat, lo_percentile / 100.0)
    return ref_sal >= hi_thresh, ref_sal <= lo_thresh


def region_rmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    lat: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Latitude-weighted RMSE within a boolean (H, W) spatial mask.

    Per-sample form (Eq. F15 restricted to ``mask``): for each ``(B, T)`` frame,
    take the mask-renormalised, latitude-weighted mean squared error, sqrt it,
    then average over frames. ``pred``/``target`` are ``(B, T, H, W)``; ``lat`` is
    ``(H,)``; ``mask`` is a boolean ``(H, W)``. The D6 masks are fixed across the
    val set so this normaliser is constant per run.
    """
    w = lat_weights(lat).to(device=pred.device, dtype=pred.dtype)[:, None]  # (H, 1)
    ww = mask.to(pred.dtype) * w  # (H, W)
    denom = ww.sum().clamp_min(_EPS)
    se = (pred - target) ** 2
    mse = (ww * se).sum(dim=(-2, -1)) / denom  # (B, T)
    return mse.clamp_min(0).sqrt().mean()
