"""DtSampler, ResidualStats, hybrid loss (CLAUDE.md §1.4–1.7, §8).

Invariants enforced here:
  §8.1  δt is fixed across K rollout steps; randomized across examples.
  §8.2  ResidualStats keys are per-(δt, var, level) — never shared across δt.
  §8.7  Loss is computed on the grid against tar, latitude-weighted, after scatter.

A1 ablation: reconstruct_state(add_base=False) and residual_loss(subtract_base=False)
             support absolute-state prediction mode.
A3 ablation: residual_loss(loss_norm=, pressure_weighted=) support L1/L2/hybrid and
             pressure-level weighting.
"""
from __future__ import annotations

import math
from datetime import timedelta

import numpy as np
import torch
from torch import Tensor

from aurora import Batch
from aurora.batch import Metadata

__all__ = [
    "DtSampler",
    "ResidualStats",
    "RolloutDtFixer",
    "lat_weighted_loss",
    "residual_loss",
    "reconstruct_state",
]


# ---------------------------------------------------------------------------
# DtSampler
# ---------------------------------------------------------------------------

class DtSampler:
    """Sample δt uniformly from the supported discrete set (CLAUDE.md §1.6).

    One δt is sampled per training batch; all items in the batch use the same
    value so the model's conditioning and the stats lookup are consistent.

    Args:
        dt_hours: supported δt values in hours, e.g. [6, 12, 24].
        seed:     optional RNG seed for reproducibility in unit tests.
    """

    def __init__(self, dt_hours: list[int], seed: int | None = None) -> None:
        if not dt_hours:
            raise ValueError("dt_hours must be non-empty")
        self.dt_hours = sorted(dt_hours)
        self._rng = np.random.default_rng(seed)

    def sample(self) -> timedelta:
        """Return one δt timedelta for the current batch."""
        dt = int(self._rng.choice(self.dt_hours))
        return timedelta(hours=dt)

    def __repr__(self) -> str:
        return f"DtSampler(dt_hours={self.dt_hours})"


# ---------------------------------------------------------------------------
# ResidualStats
# ---------------------------------------------------------------------------

class ResidualStats:
    """Per-δt, per-variable, per-level mean/std of Δ = X_{t+δt} − X_t.

    Stats are indexed by a triple ``(dt_hours: int, var: str, level: int)``.
    Surface variables use ``level = ResidualStats.SURF_LEVEL`` (0).
    Atmospheric variables use the integer pressure level in hPa.

    Invariant (CLAUDE.md §8.2): every (dt, var, level) combination has its own
    independent accumulator — stats are NEVER shared across δt values.

    Typical usage::

        stats = ResidualStats(dt_hours=[6,12,24], surf_vars=..., ...)
        for inp, tar, dt in dataset:
            surf_resid  = {v: tar_surf[v]  - inp_surf_t[v]  for v in surf_vars}
            atmos_resid = {v: tar_atmos[v] - inp_atmos_t[v] for v in atmos_vars}
            stats.update(surf_resid, atmos_resid, dt_hours=dt)
        stats.finalize()
        Δ_norm = stats.normalize(Δ, var="2t", level=0, dt_hours=6)
    """

    SURF_LEVEL: int = 0  # sentinel level for surface variables

    def __init__(
        self,
        dt_hours: list[int],
        surf_vars: tuple[str, ...],
        atmos_vars: tuple[str, ...],
        atmos_levels: tuple[int | float, ...],
    ) -> None:
        self.dt_hours      = list(dt_hours)
        self.surf_vars     = tuple(surf_vars)
        self.atmos_vars    = tuple(atmos_vars)
        self.atmos_levels  = tuple(int(l) for l in atmos_levels)

        self._n:      dict[tuple, int]   = {}
        self._sum:    dict[tuple, float] = {}
        self._sum_sq: dict[tuple, float] = {}

        self._mean: dict[tuple, float] = {}
        self._std:  dict[tuple, float] = {}
        self._finalized = False

        for dt in self.dt_hours:
            for v in self.surf_vars:
                self._init((dt, v, self.SURF_LEVEL))
            for v in self.atmos_vars:
                for lev in self.atmos_levels:
                    self._init((dt, v, int(lev)))

    def _init(self, key: tuple) -> None:
        self._n[key]      = 0
        self._sum[key]    = 0.0
        self._sum_sq[key] = 0.0

    def update(
        self,
        surf_resid:  dict[str, Tensor],
        atmos_resid: dict[str, Tensor],
        dt_hours: int,
    ) -> None:
        """Accumulate residuals Δ = tar − inp_t for one sample.

        Args:
            surf_resid:  {var: (H, W)} surface residuals.
            atmos_resid: {var: (C_A, H, W)} atmos residuals, C_A = len(atmos_levels).
            dt_hours:    which δt these residuals correspond to (for key lookup).
        """
        if self._finalized:
            raise RuntimeError("Cannot call update() after finalize().")
        if dt_hours not in self.dt_hours:
            raise ValueError(
                f"dt_hours={dt_hours} not in supported set {self.dt_hours}"
            )

        for var, delta in surf_resid.items():
            key  = (dt_hours, var, self.SURF_LEVEL)
            vals = delta.detach().float().cpu().reshape(-1)
            self._n[key]      += vals.numel()
            self._sum[key]    += vals.sum().item()
            self._sum_sq[key] += vals.pow(2).sum().item()

        for var, delta in atmos_resid.items():
            for i, lev in enumerate(self.atmos_levels):
                key  = (dt_hours, var, int(lev))
                vals = delta[i].detach().float().cpu().reshape(-1)
                self._n[key]      += vals.numel()
                self._sum[key]    += vals.sum().item()
                self._sum_sq[key] += vals.pow(2).sum().item()

    def finalize(self) -> None:
        """Compute mean and std from accumulators. Must be called before normalize."""
        for key in self._n:
            n = self._n[key]
            if n == 0:
                self._mean[key] = 0.0
                self._std[key]  = 1.0
            else:
                mean  = self._sum[key] / n
                var   = max(0.0, self._sum_sq[key] / n - mean ** 2)
                self._mean[key] = mean
                self._std[key]  = math.sqrt(var) if var > 0 else 1.0
        self._finalized = True

    def _check(self) -> None:
        if not self._finalized:
            raise RuntimeError("Call finalize() before normalize/denormalize.")

    def normalize(self, x: Tensor, var: str, level: int, dt_hours: int) -> Tensor:
        """(x − mean) / std  for the given (dt, var, level) key."""
        self._check()
        key  = (dt_hours, var, int(level))
        mean = torch.tensor(self._mean[key], dtype=x.dtype, device=x.device)
        std  = torch.tensor(self._std[key],  dtype=x.dtype, device=x.device)
        return (x - mean) / std

    def denormalize(self, x: Tensor, var: str, level: int, dt_hours: int) -> Tensor:
        """x * std + mean  (inverse of normalize)."""
        self._check()
        key  = (dt_hours, var, int(level))
        mean = torch.tensor(self._mean[key], dtype=x.dtype, device=x.device)
        std  = torch.tensor(self._std[key],  dtype=x.dtype, device=x.device)
        return x * std + mean

    def save(self, path: str) -> None:
        """Serialize finalized stats to a .npz file."""
        self._check()
        arrays: dict[str, np.ndarray] = {}
        for key in self._mean:
            dt, var, lev = key
            tag = f"{dt}__{var}__{lev}"
            arrays[f"mean__{tag}"] = np.array(self._mean[key], dtype=np.float64)
            arrays[f"std__{tag}"]  = np.array(self._std[key],  dtype=np.float64)
        arrays["dt_hours"]     = np.array(self.dt_hours)
        arrays["surf_vars"]    = np.array(list(self.surf_vars))
        arrays["atmos_vars"]   = np.array(list(self.atmos_vars))
        arrays["atmos_levels"] = np.array(self.atmos_levels)
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str) -> "ResidualStats":
        """Deserialize from a .npz file written by save()."""
        data = np.load(path, allow_pickle=True)
        dt_hours     = data["dt_hours"].tolist()
        surf_vars    = tuple(str(v) for v in data["surf_vars"].tolist())
        atmos_vars   = tuple(str(v) for v in data["atmos_vars"].tolist())
        atmos_levels = tuple(int(l) for l in data["atmos_levels"].tolist())

        obj = cls(dt_hours, surf_vars, atmos_vars, atmos_levels)
        for key in obj._n:
            dt, var, lev = key
            tag = f"{dt}__{var}__{lev}"
            obj._mean[key] = float(data[f"mean__{tag}"])
            obj._std[key]  = float(data[f"std__{tag}"])
        obj._finalized = True
        return obj

    def __repr__(self) -> str:
        return (
            f"ResidualStats(dt_hours={self.dt_hours}, "
            f"surf={self.surf_vars}, atmos={self.atmos_vars}, "
            f"levels={self.atmos_levels}, finalized={self._finalized})"
        )


# ---------------------------------------------------------------------------
# Loss utilities
# ---------------------------------------------------------------------------

def _lat_weights(lat: Tensor) -> Tensor:
    """Cosine latitude weights (H,), normalised to sum to 1."""
    w = torch.cos(torch.deg2rad(lat.float()))
    return w / w.sum()


def _pressure_weights(atmos_levels: tuple[int, ...]) -> dict[int, float]:
    """Normalized pressure weights: w(p) = p / Σp (WB2/Stormer convention).

    Up-weights near-surface levels (higher p) relative to upper troposphere.
    Surface variables always get weight 1.0 (handled separately in residual_loss).
    """
    total = sum(atmos_levels)
    return {int(lev): lev / total for lev in atmos_levels}


def lat_weighted_loss(
    pred:         Tensor,
    target:       Tensor,
    lat:          Tensor,
    norm:         str = "l2",
    level_weight: float = 1.0,
) -> Tensor:
    """Latitude-weighted L1 or L2 loss (CLAUDE.md §1.7).

    Args:
        pred:         (..., H, W) predictions.
        target:       same shape as pred.
        lat:          (H,) latitude in degrees — used to derive cosine weights.
        norm:         ``"l1"`` or ``"l2"``.
        level_weight: scalar weight multiplied into the loss before returning.
                      Used by residual_loss for pressure-level weighting (A3).

    Returns:
        Scalar loss tensor (already multiplied by level_weight).
    """
    w   = _lat_weights(lat).to(device=pred.device, dtype=pred.dtype)  # (H,)
    err = (pred - target).abs() if norm == "l1" else (pred - target).pow(2)
    w_shape = (1,) * (err.dim() - 2) + (w.shape[0], 1)
    return level_weight * (err * w.reshape(w_shape)).mean()


def residual_loss(
    pred:             "Batch",
    inp:              "Batch",
    tar:              "Batch",
    stats:            ResidualStats,
    dt_hours:         int,
    loss_norms:       dict[str, str] | None = None,
    subtract_base:    bool = True,
    loss_norm:        str = "l2",
    pressure_weighted: bool = False,
) -> Tensor:
    """Latitude-weighted loss on normalized residuals (CLAUDE.md §1.7, §8.7).

    Dynamics mode (subtract_base=True, control):
        Δ_pred = pred   − inp_t
        Δ_tar  = tar    − inp_t
        loss   = Σ_{var,lev} lat_weighted_loss(normalize(Δ_pred), normalize(Δ_tar))

    Absolute mode (subtract_base=False, A1 ablation):
        loss   = Σ_{var,lev} lat_weighted_loss(normalize(pred), normalize(tar))
        (No base subtraction; stats were estimated on absolute values.)

    Loss norm precedence (A3 ablation):
        Per-variable loss_norms dict overrides the global loss_norm default.
        loss_norm="hybrid" defers to loss_norms (default "l2" if not listed).

    Pressure weighting (A3 ablation):
        When pressure_weighted=True, atmospheric level terms are multiplied by
        w(p) = p / Σp (higher pressure = near-surface = higher weight).
        Surface terms always have weight 1.0.  The weighted sum is renormalized
        by the sum of weights so the loss magnitude stays comparable.

    Args:
        pred:             Predicted Batch at t+δt (no T dim).
        inp:              Input Batch with T history frames.
        tar:              Target Batch with T=1.
        stats:            Finalized ResidualStats.
        dt_hours:         δt in hours (for stats key lookup).
        loss_norms:       Optional per-variable norm override, e.g. {"2t": "l1"}.
        subtract_base:    If False, compute loss on absolute values (A1 ablation).
        loss_norm:        Global norm: "l1", "l2", or "hybrid" (defer to loss_norms).
        pressure_weighted: Weight atmospheric levels by pressure (A3 ablation).

    Returns:
        Scalar loss tensor.
    """
    lat    = inp.metadata.lat
    norms  = loss_norms or {}
    device = next(iter(inp.surf_vars.values())).device
    total  = torch.tensor(0.0, dtype=torch.float32, device=device)
    weight_sum = 0.0

    # Effective per-variable norm resolver
    def _norm_for(var: str) -> str:
        if loss_norm != "hybrid":
            return loss_norm          # "l1" or "l2" forced globally
        return norms.get(var, "l2")   # hybrid: per-variable dict

    # Pressure weights for atmospheric levels (A3)
    p_weights = _pressure_weights(stats.atmos_levels) if pressure_weighted else {}

    # --- Surface variables ---
    for var in pred.surf_vars:
        if subtract_base:
            inp_t = inp.surf_vars[var][:, -1]           # (B, H, W)
            p_res = pred.surf_vars[var] - inp_t
            t_res = tar.surf_vars[var][:, -1] - inp_t
        else:
            p_res = pred.surf_vars[var]                 # (B, H, W)
            t_res = tar.surf_vars[var][:, -1]

        p_norm = stats.normalize(p_res, var, ResidualStats.SURF_LEVEL, dt_hours)
        t_norm = stats.normalize(t_res, var, ResidualStats.SURF_LEVEL, dt_hours)

        total      = total + lat_weighted_loss(p_norm, t_norm, lat, norm=_norm_for(var), level_weight=1.0)
        weight_sum += 1.0

    # --- Atmospheric variables ---
    for var in pred.atmos_vars:
        for i, lev in enumerate(inp.metadata.atmos_levels):
            lev_int = int(lev)
            lw = p_weights.get(lev_int, 1.0)

            if subtract_base:
                inp_t = inp.atmos_vars[var][:, -1, i]   # (B, H, W)
                p_res = pred.atmos_vars[var][:, i] - inp_t
                t_res = tar.atmos_vars[var][:, -1, i] - inp_t
            else:
                p_res = pred.atmos_vars[var][:, i]
                t_res = tar.atmos_vars[var][:, -1, i]

            p_norm = stats.normalize(p_res, var, lev_int, dt_hours)
            t_norm = stats.normalize(t_res, var, lev_int, dt_hours)

            total      = total + lat_weighted_loss(p_norm, t_norm, lat, norm=_norm_for(var), level_weight=lw)
            weight_sum += lw

    return total / max(weight_sum, 1e-8)


# ---------------------------------------------------------------------------
# reconstruct_state (M4-05 / M5 / A1)
# ---------------------------------------------------------------------------

def reconstruct_state(
    inp:       Batch,
    resid:     Batch,
    lead_time: timedelta,
    stats:     ResidualStats,
    add_base:  bool = True,
) -> Batch:
    """X̂_{t+δt} = X_t + denorm(Δ̂_norm)  [dynamics mode]
    or X̂_{t+δt} = denorm(Ŷ_norm)          [absolute mode, A1 ablation].

    In dynamics mode (add_base=True, control): resid is treated as a normalized
    residual; denormalization undoes the stats scaling, and X_t is added back.

    In absolute mode (add_base=False, A1 ablation): resid is treated as a
    normalized absolute prediction; no base is added. Stats must be estimated
    on absolute values (absolute_stats.npz), not residuals.

    The addition (dynamics mode) is performed in float32 to prevent bf16
    catastrophic cancellation when a small residual is added to a large
    absolute state (e.g. geopotential ~5e4 m²/s²). Result is cast back to
    the original input dtype.

    Args:
        inp:       Input Batch with T history frames.  X_t = inp[..., -1, ...].
        resid:     Decoder output Batch (no T dim).
        lead_time: Forecast interval δt.  Selects the per-δt normalization stats.
        stats:     Finalized ResidualStats.
        add_base:  If True (default), add X_t (dynamics mode).
                   If False, skip base addition (absolute mode, A1 ablation).

    Returns:
        Reconstructed Batch at t + lead_time.

    Tensor shapes::

        inp.surf_vars[v]:    (B, T, H, W)     → X_t = [:, -1]         (B, H, W)
        resid.surf_vars[v]:  (B, H, W)        Δ̂_norm or Ŷ_norm
        output surf_vars[v]: (B, H, W)        X̂

        inp.atmos_vars[v]:   (B, T, L, H, W)  → X_t = [:, -1, i]     (B, H, W)
        resid.atmos_vars[v]: (B, L, H, W)     Δ̂_norm or Ŷ_norm
        output atmos_vars[v]:(B, L, H, W)     X̂
    """
    dt_hours   = int(round(lead_time.total_seconds() / 3600))
    orig_dtype = next(iter(resid.surf_vars.values())).dtype

    # Surface variables
    surf_out: dict[str, Tensor] = {}
    for var in resid.surf_vars:
        d_norm = resid.surf_vars[var].float()   # (B, H, W)
        d_phys = stats.denormalize(d_norm, var, ResidualStats.SURF_LEVEL, dt_hours)
        if add_base:
            x_t = inp.surf_vars[var][:, -1].float()
            surf_out[var] = (x_t + d_phys).to(orig_dtype)
        else:
            surf_out[var] = d_phys.to(orig_dtype)

    # Atmospheric variables
    atmos_out: dict[str, Tensor] = {}
    for var in resid.atmos_vars:
        levels_out = []
        for i, lev in enumerate(inp.metadata.atmos_levels):
            d_norm = resid.atmos_vars[var][:, i].float()  # (B, H, W)
            d_phys = stats.denormalize(d_norm, var, int(lev), dt_hours)
            if add_base:
                x_t = inp.atmos_vars[var][:, -1, i].float()
                levels_out.append((x_t + d_phys).to(orig_dtype))
            else:
                levels_out.append(d_phys.to(orig_dtype))
        atmos_out[var] = torch.stack(levels_out, dim=1)   # (B, L, H, W)

    return Batch(
        surf_vars=surf_out,
        static_vars=resid.static_vars,
        atmos_vars=atmos_out,
        metadata=resid.metadata,
    )


# ---------------------------------------------------------------------------
# M4: RolloutDtFixer
# ---------------------------------------------------------------------------

class RolloutDtFixer:
    """Fix δt to one value across all K rollout steps (CLAUDE.md §8.1).

    Wraps a DtSampler and enforces that the same timedelta is used for every
    step within one rollout.  Call begin_rollout() once per rollout to draw a
    new δt; then use .current throughout that rollout.

    Usage::

        fixer = RolloutDtFixer(DtSampler([6, 12, 24]))
        lead_time = fixer.begin_rollout()      # sample once
        preds, _ = rollout(model, inp, index, fixer.current, k=4)

    Args:
        sampler: A DtSampler that produces timedelta samples.
    """

    def __init__(self, sampler: "DtSampler") -> None:
        self._sampler = sampler
        self._current: timedelta | None = None

    def begin_rollout(self) -> timedelta:
        """Sample a fresh δt for the upcoming rollout and return it."""
        self._current = self._sampler.sample()
        return self._current

    @property
    def current(self) -> timedelta:
        """The δt fixed for the current rollout."""
        if self._current is None:
            raise RuntimeError(
                "RolloutDtFixer.current accessed before begin_rollout() was called."
            )
        return self._current
