from datetime import datetime
import torch
from aurora import Batch, Metadata
from aurora.normalisation import normalise_atmos_var, normalise_surf_var

def var_weighted_loss(pred: Batch, target: Batch, gamma: float = 2.5) -> torch.Tensor:

    """
    MAE loss from Aurora paper (Eq. D12) with pretraining variable weights.

    The MAE is computed on NORMALISED variables (Aurora's own per-variable,
    per-level statistics): in raw physical units z (~1e5) and msl (~1e5) would
    dwarf q (~1e-2) by 5+ orders of magnitude, making the Eq. D12 per-variable
    weights meaningless.

    pred / target: Batch objects with shapes
        surf_vars:  (B, T, H, W)
        atmos_vars: (B, T, C, H, W)   C = pressure levels
    gamma: dataset weight (ERA5=2.0, GFS-T0=1.5, else 1.0)
    """
    alpha = 0.25   # surface loss weight (1/4)
    beta  = 1.0    # atmospheric loss weight

    # --- Pretraining variable weights (paper Section D.1) ---
    surf_weights: dict[str, float] = {
        "msl": 1.5,
        "10u": 0.77,
        "10v": 0.66,
        "2t":  3.0,
    }
    atmos_weights: dict[str, float] = {
        "z": 2.8,
        "q": 0.78,
        "t": 1.7,
        "u": 0.87,
        "v": 0.6,
    }

    # ── Surface loss ──────────────────────────────────────────────────────────
    # L_surf = alpha * sum_k [ w_k^S * (1/H*W) * sum_{i,j} |S_hat - S| ]
    # The (1 / H*W) average is taken over the spatial dims (mean over i,j).
    # We also average over batch B and time T (standard practice).
    V_S = len(pred.surf_vars)      # number of surface variables
    V_A = len(pred.atmos_vars)     # number of atmospheric variables

    surf_loss = torch.tensor(0.0, device=next(iter(pred.surf_vars.values())).device)
    for k, p in pred.surf_vars.items():
        t = target.surf_vars[k]           # (B, T, H, W)
        w = surf_weights.get(k, 1.0)
        # mean over B, T then sum over H,W divided by H*W  →  equivalent to .mean()
        mae = (normalise_surf_var(p, k) - normalise_surf_var(t, k)).abs().mean()
        surf_loss = surf_loss + w * mae

    # ── Atmospheric loss ──────────────────────────────────────────────────────
    # L_atmos = beta * sum_k [ (1/C*H*W) * sum_{c,i,j} w_{k,c}^A * |A_hat - A| ]
    # w_{k,c}^A is constant across pressure levels c in pretraining,
    # so it factors out of the spatial / level mean.
    atmos_loss = torch.tensor(0.0, device=next(iter(pred.atmos_vars.values())).device)
    levels = pred.metadata.atmos_levels
    for k, p in pred.atmos_vars.items():
        t = target.atmos_vars[k]          # (B, T, C, H, W)
        w = atmos_weights.get(k, 1.0)
        mae = (normalise_atmos_var(p, k, levels) - normalise_atmos_var(t, k, levels)).abs().mean()
        atmos_loss = atmos_loss + w * mae

    # ── Combine (Eq. D12) ────────────────────────────────────────────────────
    total = (gamma / (V_S + V_A)) * (
        alpha * surf_loss + beta * atmos_loss
    )
    return total
