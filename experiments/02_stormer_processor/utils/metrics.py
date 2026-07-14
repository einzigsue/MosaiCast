"""Latitude-weighted verification metrics (CLAUDE.md rule 5; Aurora SI Section F).

All metrics use the latitude weighting of Eq. (F14),

    w(i) = cos(lat_i) / mean_i' cos(lat_i')   (normalised to unit mean).

Shapes: predictions/targets are ``(B, T, H, W)`` where ``T`` is the forecast-time
axis (callers slice a pressure level out of ``(B, T, L, H, W)`` first). ``lat`` is
``(H,)`` in degrees. Metric functions reduce spatial dims per sample and, by
default, return a scalar mean over ``(B, T)``; pass ``reduce=False`` to get the
per-sample vector (used for bootstrap confidence intervals, SI F.2).

Two RMSE forms coexist deliberately:

* :func:`lat_weighted_mse` / :func:`lat_weighted_rmse` — the *pooled* form
  (``sqrt(mean over B,T,H,W)``). Cheap, differentiable; used as the training loss.
* :func:`lat_weighted_rmse_f15` — the paper's Eq. (F15): ``sqrt`` is taken
  *per sample*, then averaged over samples. This is the reporting metric; it
  differs from the pooled form by a Jensen gap.
"""
from __future__ import annotations
import torch

_EPS = 1e-12


def lat_weights(lat: torch.Tensor) -> torch.Tensor:
    """Normalised cos(lat) weights, shape (H,) — Eq. (F14)."""
    w = torch.cos(torch.deg2rad(lat)).clamp(min=0)
    return w / w.mean()


def _w_hw(lat: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """(H, 1) latitude weights broadcastable over (..., H, W), on ref's device/dtype."""
    return lat_weights(lat).to(device=ref.device, dtype=ref.dtype)[:, None]


def _nanmean(x: torch.Tensor) -> torch.Tensor:
    """Mean over all elements, ignoring NaNs (empty-mask frames)."""
    mask = ~torch.isnan(x)
    n = mask.sum()
    if n == 0:
        return torch.full((), float("nan"), device=x.device, dtype=x.dtype)
    return torch.where(mask, x, torch.zeros_like(x)).sum() / n


# ---------------------------------------------------------------------------
# Pooled RMSE (training loss) — kept for backward compatibility.
# ---------------------------------------------------------------------------
def lat_weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    lat: torch.Tensor,
) -> torch.Tensor:
    """(B, T, H, W) → scalar MSE weighted by cos(lat), pooled over all dims."""
    w = _w_hw(lat, pred)[None, None]  # (1, 1, H, 1)
    return ((pred - target) ** 2 * w).mean()


def lat_weighted_rmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    lat: torch.Tensor,
) -> torch.Tensor:
    """Pooled latitude-weighted RMSE, ``sqrt(mean over B,T,H,W)``.

    NOT Eq. (F15) — see :func:`lat_weighted_rmse_f15` for the paper-faithful,
    per-sample-averaged reporting metric.
    """
    return lat_weighted_mse(pred, target, lat).sqrt()


# ---------------------------------------------------------------------------
# Eq. (F15) — latitude-weighted RMSE, per-sample sqrt then averaged.
# ---------------------------------------------------------------------------
def lat_weighted_rmse_f15(
    pred: torch.Tensor,
    target: torch.Tensor,
    lat: torch.Tensor,
    reduce: bool = True,
) -> torch.Tensor:
    """Eq. (F15): ``(1/T) Σ_t sqrt( (1/HW) Σ_ij w(i)(X̂-X)² )``.

    ``pred``/``target``: ``(B, T, H, W)``; ``lat``: ``(H,)``.
    Returns a scalar (mean over B, T) if ``reduce`` else the per-sample
    ``(B*T,)`` vector. Because ``w`` has unit mean, the weighted spatial mean is
    just ``mean(w * se)`` over H, W.
    """
    w = _w_hw(lat, pred)  # (H, 1)
    mse = ((pred - target) ** 2 * w).mean(dim=(-2, -1))  # (B, T)
    rmse = mse.clamp_min(0).sqrt()
    return rmse.mean() if reduce else rmse.flatten()


# ---------------------------------------------------------------------------
# Eq. (F16) — anomaly correlation coefficient.
# ---------------------------------------------------------------------------
def acc(
    pred: torch.Tensor,
    target: torch.Tensor,
    clim: torch.Tensor,
    lat: torch.Tensor,
    reduce: bool = True,
) -> torch.Tensor:
    """Eq. (F16): latitude-weighted anomaly correlation coefficient.

    ``clim`` is the daily climatology C for the same variable/level/valid-time,
    broadcastable to ``pred`` (e.g. ``(B, T, H, W)`` or ``(H, W)``). Anomalies are
    ``pred - clim`` and ``target - clim``. The ``1/HW`` in the paper cancels
    between numerator and denominator, so weighted sums are used directly.
    """
    w = _w_hw(lat, pred)  # (H, 1)
    a = pred - clim
    b = target - clim
    num = (w * a * b).sum(dim=(-2, -1))  # (B, T)
    den = torch.sqrt(
        (w * a * a).sum(dim=(-2, -1)) * (w * b * b).sum(dim=(-2, -1))
    ).clamp_min(_EPS)
    out = num / den  # (B, T)
    return out.mean() if reduce else out.flatten()


# ---------------------------------------------------------------------------
# Eqs. (F17)-(F20) — thresholded RMSE for extreme-weather verification.
# ---------------------------------------------------------------------------
def thresholded_rmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    lat: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    g: float,
    side: str,
    reduce: bool = True,
) -> torch.Tensor:
    """Eqs. (F17)-(F20): RMSE restricted to gridpoints past a per-cell threshold.

    Threshold ``b_ij = mu_ij + g * sigma_ij`` (Eq. F20), with ``mu``/``sigma`` the
    per-gridpoint mean/std of the target variable over the training years,
    shape ``(H, W)``.

    ``side='right'`` (Eq. F18) keeps points where ``target > b``; ``side='left'``
    (Eq. F19) keeps ``target < b``. The indicator is on the **target** field. The
    kept latitude weights are renormalised to sum to one *per frame*, so

        RMSE_g = (1/T) Σ_t sqrt( Σ_ij w̃_ij (X̂-X)² ).

    Frames with no points past the threshold contribute NaN and are dropped from
    the reduction. Returns a scalar (nan-mean over B, T) if ``reduce`` else the
    per-sample ``(B*T,)`` vector (for the SI F.2 bootstrap).

    As ``g → -∞`` the right-sided RMSE_g → full-domain Eq. (F15); as ``g → +∞``
    the left-sided RMSE_g → full-domain Eq. (F15).
    """
    if side not in ("right", "left"):
        raise ValueError(f"side must be 'right' or 'left', got {side!r}")
    w = _w_hw(lat, pred)  # (H, 1)
    b = mu.to(device=pred.device, dtype=pred.dtype) + g * sigma.to(
        device=pred.device, dtype=pred.dtype
    )  # (H, W)
    keep = (target > b) if side == "right" else (target < b)  # (B, T, H, W)
    ww = keep.to(pred.dtype) * w  # (B, T, H, W)
    denom = ww.sum(dim=(-2, -1))  # (B, T) per-frame normaliser
    se = (pred - target) ** 2
    num = (ww * se).sum(dim=(-2, -1))  # (B, T)
    # Empty frames (denom == 0) → NaN, excluded from the reduction.
    mse = torch.where(denom > 0, num / denom.clamp_min(_EPS), torch.full_like(denom, float("nan")))
    rmse = mse.clamp_min(0).sqrt()
    return _nanmean(rmse) if reduce else rmse.flatten()


def threshold_sweep(
    pred: torch.Tensor,
    target: torch.Tensor,
    lat: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    g_values,
) -> dict[float, torch.Tensor]:
    """RMSE_g over a sweep of thresholds (SI F.2 curve).

    For each ``g`` uses the right-sided form when ``g >= 0`` and the left-sided
    form when ``g < 0`` (matching the paper's convention: right thresholds on the
    right of the plot, left on the left). Returns ``{g: per_sample_rmse (B*T,)}``
    so callers can bootstrap 95% CIs by resampling frames with replacement.
    """
    out: dict[float, torch.Tensor] = {}
    for g in g_values:
        side = "right" if g >= 0 else "left"
        out[float(g)] = thresholded_rmse(
            pred, target, lat, mu, sigma, float(g), side, reduce=False
        )
    return out
