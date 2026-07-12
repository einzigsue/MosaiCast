"""Fixed-δt K-step rollout — pushforward gradient + replay buffer (CLAUDE.md §5, M4).

Design notes
------------
advance_batch
    Slides the 2-frame history window by one step.  Used both by the main rollout
    (δt=6h, step size = history_dt) and the auxiliary 6h sub-chain (see below).

rollout — §1.9 compliance (M4-04 fix)
    CLAUDE.md §1.9 Option A: the two history frames are always [t − 6h, t], regardless
    of the forecast δt.  For δt=6h this holds naturally — advance_batch gives
    [x_t, x_{t+6h}] which is 6h-spaced.  For δt > 6h (12h or 24h), after the main
    step predicts x_{t+δt} we need x_{t+δt−6h} to form the next history.  We obtain
    it by running n_aux = δt/6h − 1 auxiliary no-grad forward passes at 6h from the
    same starting state `cur`:
        cur=[x_{t−6h}, x_t]  →(6h)→ x_{t+6h} →(6h)→ … → x_{t+δt−6h}
    Cost: n_aux extra no-grad forward passes per main step (1 for δt=12h, 3 for 24h).

    Gradient invariant (pushforward):
        steps 0 … K-2: torch.no_grad() + detach  (main prediction; aux always no-grad)
        step  K-1:      full gradient

ReplayBuffer — matched-target fix (M4-02)
    Stores (inp, tar, dt_hours) triples so the replayed start always has a known
    matching ground-truth target.  push() receives the original (inp, tar) from the
    dataloader (not a mid-rollout state) so the target is always consistent with the
    start state.  maybe_replace() returns (inp_effective, tar_effective) — either
    both from the buffer or both from the caller.
"""
from __future__ import annotations

from collections import deque
from datetime import timedelta
from typing import TYPE_CHECKING

import numpy as np
import torch

from aurora import Batch
from aurora.batch import Metadata

from mosaicast.patching.plan import PatchIndex

if TYPE_CHECKING:
    import torch.nn as nn

__all__ = ["advance_batch", "rollout", "ReplayBuffer"]


# ---------------------------------------------------------------------------
# Window slide
# ---------------------------------------------------------------------------

def advance_batch(inp: Batch, pred: Batch, lead_time: timedelta) -> Batch:
    """Slide the 2-frame history window by one step.

    Args:
        inp:       Current Batch with T=2 history frames — last frame is x_t.
        pred:      Model prediction at x_{t+step} (no T dim).
        lead_time: Step size used for this prediction.

    Returns:
        New Batch with T=2 frames [x_t, x_{t+step}].

    Tensor shapes
        surf_vars:  (B, 2, H, W)
        atmos_vars: (B, 2, L, H, W)
    """
    new_surf = {
        v: torch.stack([inp.surf_vars[v][:, -1], pred.surf_vars[v]], dim=1)
        for v in inp.surf_vars
    }
    new_atmos = {
        v: torch.stack([inp.atmos_vars[v][:, -1], pred.atmos_vars[v]], dim=1)
        for v in inp.atmos_vars
    }
    return Batch(
        surf_vars=new_surf,
        static_vars=inp.static_vars,
        atmos_vars=new_atmos,
        metadata=Metadata(
            lat=inp.metadata.lat,
            lon=inp.metadata.lon,
            time=pred.metadata.time,
            atmos_levels=inp.metadata.atmos_levels,
            rollout_step=pred.metadata.rollout_step,
        ),
    )


# ---------------------------------------------------------------------------
# K-step rollout
# ---------------------------------------------------------------------------

def rollout(
    model: "nn.Module",
    inp: Batch,
    index: PatchIndex,
    lead_time: timedelta | list[timedelta],
    k: int,
    history_dt: timedelta = timedelta(hours=6),
) -> tuple[list[Batch], list[Batch]]:
    """K steps of autoregressive rollout with pushforward gradient.

    lead_time is the SAME for all K steps (invariant §8.1, control).

    A6 ablation — randomized-within-rollout (destabiliser, do NOT ship):
        Pass lead_time as a list of K timedeltas, one per step.  This violates
        §8.1 and is expected to destabilise training.  The caller (MosaicastRolloutModule
        with randomize_dt_within_rollout=True) is responsible for marking this path.
        # DEVIATION: A6 ablation only — violates CLAUDE §8.1. Do not ship.

    §1.9 Option A compliance (M4-04 fix): every model call receives history
    frames spaced exactly history_dt (default 6h) apart.  For lead_time > history_dt,
    auxiliary no-grad sub-steps at history_dt generate x_{t+lead_time−history_dt} so
    the next history window is [x_{t+lead_time−history_dt}, x_{t+lead_time}].

    Args:
        model:       MosaicastModel (or any nn.Module with the same forward signature).
        inp:         Initial Batch with T=2 history frames.
        index:       PatchIndex — reused across all steps (grid is constant).
        lead_time:   Fixed δt for every main step (timedelta), OR a list of K
                     timedeltas for the A6 within-rollout destabiliser ablation.
        k:           Number of rollout steps (K).
        history_dt:  Fixed history frame spacing (§1.9 Option A, default 6h).

    Returns:
        (preds, inp_states) where:
          preds[i]      — Batch predicted at t + (i+1)*lead_time   (no T dim)
          inp_states[i] — Batch used as input for step i            (T=2 history)
        preds has length K; inp_states has length K.

    Raises:
        ValueError: if k < 1 or lead_time is not a positive integer multiple of history_dt.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    # Resolve per-step lead times (A6: list) vs scalar (control)
    if isinstance(lead_time, list):
        # A6 ablation: per-step lead times; length must match k
        # DEVIATION: A6 ablation only — violates CLAUDE §8.1. Do not ship.
        if len(lead_time) != k:
            raise ValueError(f"lead_time list length {len(lead_time)} != k={k}")
        lead_times_per_step = lead_time
    else:
        lead_times_per_step = [lead_time] * k

    # Validate all per-step lead times against history_dt
    hist_h = int(round(history_dt.total_seconds() / 3600))
    if hist_h <= 0:
        raise ValueError(f"history_dt ({hist_h}h) must be positive")
    for lt in lead_times_per_step:
        lead_h = int(round(lt.total_seconds() / 3600))
        if lead_h % hist_h != 0:
            raise ValueError(
                f"lead_time ({lead_h}h) must be a positive integer multiple of "
                f"history_dt ({hist_h}h)"
            )

    preds: list[Batch] = []
    inp_states: list[Batch] = []
    cur = inp

    for step in range(k):
        inp_states.append(cur)
        is_last = (step == k - 1)
        step_lt = lead_times_per_step[step]
        lead_h  = int(round(step_lt.total_seconds() / 3600))
        # Number of auxiliary 6h sub-steps needed to get x_{t+lead_time−history_dt}.
        n_aux   = lead_h // hist_h - 1

        # Main prediction — gradient only on the last step (pushforward).
        if is_last:
            pred = model(cur, index, step_lt)
        else:
            with torch.no_grad():
                pred_raw = model(cur, index, step_lt)
            pred = Batch(
                surf_vars={v: t.detach() for v, t in pred_raw.surf_vars.items()},
                static_vars=pred_raw.static_vars,
                atmos_vars={v: t.detach() for v, t in pred_raw.atmos_vars.items()},
                metadata=pred_raw.metadata,
            )

        preds.append(pred)

        if step < k - 1:
            if n_aux == 0:
                cur = advance_batch(cur, pred, step_lt)
            else:
                aux = cur
                with torch.no_grad():
                    for _ in range(n_aux):
                        aux_pred = model(aux, index, history_dt)
                        aux = advance_batch(aux, aux_pred, history_dt)
                new_surf = {
                    v: torch.stack([aux.surf_vars[v][:, -1], pred.surf_vars[v]], dim=1)
                    for v in pred.surf_vars
                }
                new_atmos = {
                    v: torch.stack([aux.atmos_vars[v][:, -1], pred.atmos_vars[v]], dim=1)
                    for v in pred.atmos_vars
                }
                cur = Batch(
                    surf_vars=new_surf,
                    static_vars=inp.static_vars,
                    atmos_vars=new_atmos,
                    metadata=Metadata(
                        lat=inp.metadata.lat,
                        lon=inp.metadata.lon,
                        time=pred.metadata.time,
                        atmos_levels=inp.metadata.atmos_levels,
                        rollout_step=pred.metadata.rollout_step,
                    ),
                )

    return preds, inp_states


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Circular buffer of (inp, tar, dt_hours) triples for rollout finetuning (CLAUDE.md §5).

    Stores matched (start, target) pairs from past training steps on CPU.
    During training, with probability p_replay the DataLoader's (inp, tar) is
    replaced with a buffered pair, exposing the model to previously-seen
    distribution regions and avoiding the compound-error mismatch that arises
    when only the start is replaced while the target remains from a different
    trajectory (M4-02 fix).

    push() should be called with the ORIGINAL dataloader (inp, tar) — not a
    mid-rollout state — so the buffered target is always consistent with the
    buffered start.

    Args:
        max_size:  Maximum entries kept (FIFO eviction after max_size pushes).
        p_replay:  Probability of replacing (inp, tar) with a buffered entry.
        seed:      RNG seed (optional, for reproducibility).

    Usage::

        buf = ReplayBuffer(max_size=256, p_replay=0.5)

        # After each training step, push the ground-truth pair.
        buf.push(inp, tar, dt_hours=dt_per_step)

        # At the start of the next step, optionally replace both inp and tar.
        inp, tar = buf.maybe_replace(inp, tar, device=device)
    """

    def __init__(
        self,
        max_size: int = 256,
        p_replay: float = 0.5,
        seed: int | None = None,
    ) -> None:
        self._buf: deque[tuple[Batch, Batch, int]] = deque(maxlen=max_size)
        self.p_replay = p_replay
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._buf)

    def push(self, inp: Batch, tar: Batch, dt_hours: int) -> None:
        """Store a matched (inp, tar, dt_hours) triple on CPU for future replay.

        Args:
            inp:      Input Batch (T=2 history frames).
            tar:      Matching ground-truth target (the dataloader's tar for this inp).
            dt_hours: Forecast interval in hours for this (inp, tar) pair.
        """
        def _to_cpu(b: Batch) -> Batch:
            return Batch(
                surf_vars={v: t.detach().cpu() for v, t in b.surf_vars.items()},
                static_vars={
                    k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
                    for k, v in b.static_vars.items()
                },
                atmos_vars={v: t.detach().cpu() for v, t in b.atmos_vars.items()},
                metadata=b.metadata,
            )

        self._buf.append((_to_cpu(inp), _to_cpu(tar), dt_hours))

    def sample(self) -> tuple[Batch, Batch, int] | None:
        """Return a uniformly random buffered (inp, tar, dt_hours) triple, or None."""
        if not self._buf:
            return None
        idx = int(self._rng.integers(len(self._buf)))
        return self._buf[idx]

    def maybe_replace(
        self,
        inp: Batch,
        tar: Batch,
        device: "torch.device | str | None" = None,
    ) -> tuple[Batch, Batch]:
        """Return (inp, tar), or (with probability p_replay) a buffered pair.

        The buffered pair is moved to device before being returned.
        If the buffer is empty, (inp, tar) is returned unchanged.
        Both inp and tar are replaced together so the target always matches
        the start state (M4-02 fix).
        """
        if not self._buf or self._rng.random() >= self.p_replay:
            return inp, tar

        entry = self.sample()
        if entry is None:
            return inp, tar

        buf_inp, buf_tar, _ = entry

        if device is not None:
            def _to_dev(b: Batch) -> Batch:
                return Batch(
                    surf_vars={v: t.to(device) for v, t in b.surf_vars.items()},
                    static_vars={
                        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                        for k, v in b.static_vars.items()
                    },
                    atmos_vars={v: t.to(device) for v, t in b.atmos_vars.items()},
                    metadata=b.metadata,
                )
            buf_inp = _to_dev(buf_inp)
            buf_tar = _to_dev(buf_tar)

        return buf_inp, buf_tar
