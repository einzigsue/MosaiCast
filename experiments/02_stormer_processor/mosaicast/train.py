"""Lightning CLI training entrypoints (CLAUDE.md §4, M3/M4).

M3 — single-step randomized-δt pretraining:
    conda run -n mosaicast python -m mosaicast.train fit \
        --config configs/pretrain_randomized.yaml

M4 — K-step rollout finetuning (curriculum K=1→4→8):
    conda run -n mosaicast python -m mosaicast.train fit \
        --config configs/finetune_rollout.yaml

Ablation knobs (all default to control — no-op when unset):
    A1: target_mode       A3: loss_norm, pressure_weighted
    A2: processor_type    A4: dt_cond, qk_norm, condition_on_rollout_step
    A6: randomize_dt_within_rollout (rollout module only)
    A7: tokenizer, plan_type, budget, canonical, criterion_var, criterion_level
    A8: stabilise_level_agg, latent_levels
"""
from __future__ import annotations

import os
from datetime import timedelta

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader

from aurora import Batch

from mosaicast.dynamics.randomized import ResidualStats, residual_loss, DtSampler
from mosaicast.dynamics.rollout import rollout, ReplayBuffer
from mosaicast.model.model import MosaicastModel
from mosaicast.patching.plan import PatchPlan
from mosaicast.data.datasets import MosaicastDataset
from mosaicast.data.collate import mosaicast_collate_fn
from mosaicast.data.sampler import DtBatchSampler


class MosaicastLightningModule(L.LightningModule):
    """Training / validation wrapper around MosaicastModel.

    Receives batches of ``(inp_batch, tar_batch, patch_plan)`` from the
    DataLoader (produced by mosaicast_collate_fn).

    The chosen δt is recovered from the Batch metadata::

        dt_hours = int((tar.metadata.time[0] - inp.metadata.time[0])
                       .total_seconds() / 3600)

    The ``PatchPlan`` from the batch is used to obtain a ``PatchIndex`` via
    ``plan.index(lat, lon)``, which is memoised at module level in
    ``patching/plan.py`` keyed by ``(plan, lat_hash, lon_hash)`` — so the
    expensive coverage validation runs at most once per unique (plan, grid).

    Args:
        surf_vars, static_vars, atmos_vars: variable lists forwarded to MosaicastModel.
        patch_size:                 canonical patch size p (default 4).
        latent_levels:              number of latent pressure levels (default 4, A8).
        embed_dim:                  transformer embedding dimension (default 256).
        encoder_depth:              Perceiver cross-attention depth (default 4).
        processor_depth:            StormerProcessor block count (default 8).
        processor_heads:            attention heads in processor (default 8).
        processor_head_dim:         per-head key/value dimension (default 64).
        mlp_ratio:                  FFN hidden-dim / embed_dim ratio (default 4.0).
        max_history_size:           history frames T (default 2).
        stats_path:                 path to a .npz written by ResidualStats.save().
        lr:                         AdamW learning rate (default 3e-4).
        weight_decay:               AdamW weight decay (default 0.1).
        loss_norms:                 optional per-variable norm override {"2t": "l1"}.
        target_mode:                "dynamics" (control) or "absolute" (A1).
        processor_type:             "isotropic" (control), "hybrid", "swin" (A2).
        dt_cond:                    "adaln_zero" (control), "additive", "none" (A4).
        qk_norm:                    True (control), False (A4).
        condition_on_rollout_step:  False (control), True (A4).
        stabilise_level_agg:        False (control), True (A8).
        tokenizer:                  "resize" (control), "attention_pool" (A7).
        loss_norm:                  "l2" (control), "l1", "hybrid" (A3).
        pressure_weighted:          False (control), True (A3).
    """

    def __init__(
        self,
        surf_vars:                 tuple[str, ...] = ("2t", "10u", "10v", "msl"),
        static_vars:               tuple[str, ...] = ("lsm", "z", "slt"),
        atmos_vars:                tuple[str, ...] = ("z", "u", "v", "t", "q"),
        patch_size:                int = 4,
        latent_levels:             int = 4,
        embed_dim:                 int = 256,
        encoder_depth:             int = 4,
        processor_depth:           int = 8,
        processor_heads:           int = 8,
        processor_head_dim:        int = 64,
        mlp_ratio:                 float = 4.0,
        max_history_size:          int = 2,
        stats_path:                str | None = None,
        lr:                        float = 3e-4,
        weight_decay:              float = 0.1,
        loss_norms:                dict[str, str] | None = None,
        # Ablation knobs — defaults are control (no-op)
        target_mode:               str = "dynamics",
        processor_type:            str = "isotropic",
        dt_cond:                   str = "adaln_zero",
        qk_norm:                   bool = True,
        condition_on_rollout_step: bool = False,
        stabilise_level_agg:       bool = False,
        tokenizer:                 str = "resize",
        loss_norm:                 str = "l2",
        pressure_weighted:         bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = MosaicastModel(
            surf_vars=tuple(surf_vars),
            static_vars=tuple(static_vars),
            atmos_vars=tuple(atmos_vars),
            patch_size=patch_size,
            latent_levels=latent_levels,
            embed_dim=embed_dim,
            encoder_depth=encoder_depth,
            processor_depth=processor_depth,
            processor_heads=processor_heads,
            processor_head_dim=processor_head_dim,
            mlp_ratio=mlp_ratio,
            max_history_size=max_history_size,
            target_mode=target_mode,
            processor_type=processor_type,
            dt_cond=dt_cond,
            qk_norm=qk_norm,
            condition_on_rollout_step=condition_on_rollout_step,
            stabilise_level_agg=stabilise_level_agg,
            tokenizer=tokenizer,
        )

        self.loss_norms       = loss_norms
        self.loss_norm        = loss_norm
        self.pressure_weighted = pressure_weighted

        if stats_path is not None:
            self.stats: ResidualStats | None = ResidualStats.load(stats_path)
            self.model.set_stats(self.stats)
        else:
            self.stats = None

    def set_stats(self, stats: ResidualStats) -> None:
        """Attach finalized ResidualStats (call before fit if stats_path was None)."""
        self.stats = stats
        self.model.set_stats(stats)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dt_from_batch(inp: Batch, tar: Batch) -> int:
        delta = tar.metadata.time[0] - inp.metadata.time[0]
        return int(delta.total_seconds() / 3600)

    def _residual_loss(self, pred, inp, tar, dt_hours) -> torch.Tensor:
        """Unified loss call including A1/A3 ablation knobs."""
        return residual_loss(
            pred, inp, tar, self.stats, dt_hours,
            loss_norms=self.loss_norms,
            subtract_base=(self.hparams.target_mode == "dynamics"),
            loss_norm=self.loss_norm,
            pressure_weighted=self.pressure_weighted,
        )

    # ------------------------------------------------------------------
    # LightningModule API
    # ------------------------------------------------------------------

    def forward(self, inp: Batch, plan: PatchPlan, lead_time: timedelta) -> Batch:
        index = plan.index(inp.metadata.lat, inp.metadata.lon)
        return self.model(inp, index, lead_time)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        inp, tar, plan = batch
        assert self.stats is not None, (
            "ResidualStats not set — call set_stats() or pass stats_path to __init__."
        )
        dt    = self._dt_from_batch(inp, tar)
        index = plan.index(inp.metadata.lat, inp.metadata.lon)
        pred  = self.model(inp, index, lead_time=timedelta(hours=dt))
        loss  = self._residual_loss(pred, inp, tar, dt)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        inp, tar, plan = batch
        assert self.stats is not None
        dt    = self._dt_from_batch(inp, tar)
        index = plan.index(inp.metadata.lat, inp.metadata.lon)
        pred  = self.model(inp, index, lead_time=timedelta(hours=dt))
        loss  = self._residual_loss(pred, inp, tar, dt)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )


class MosaicastDataModule(L.LightningDataModule):
    """Lightning DataModule for mosaicast training (CLAUDE.md §4).

    Args:
        root, nc_filename, patch_size, surf_vars, atmos_vars, atmos_levels,
        static_nc, static_vars, dt_hours: forwarded to dataset / collate.
        train/val/test year ranges: chronological data splits.
        batch_size, num_workers: DataLoader settings.
        plan_type:       "uniform" (control), "lat_band", "content_adaptive" (A7).
        canonical:       canonical patch size; must equal model patch_size (A7 sweep).
        budget:          max patch count for content_adaptive plan (A7).
        criterion_var:   field variable for roughness criterion (A7, default "z").
        criterion_level: pressure level for criterion field (A7, default 500).
    """

    def __init__(
        self,
        root: str = "data",
        nc_filename: str = "aurora_{year}_5.625deg.nc",
        patch_size: int = 4,
        surf_vars: tuple[str, ...] = ("2t", "10u", "10v", "msl"),
        atmos_vars: tuple[str, ...] = ("z", "u", "v", "t", "q"),
        atmos_levels: tuple[int, ...] = (50, 250, 500, 600, 700, 850, 925),
        static_nc: str | None = None,
        static_vars: tuple[str, ...] = ("lsm", "z", "slt"),
        dt_hours: list[int] = (6, 12, 24),
        train_year_start: int = 1979,
        train_year_end: int = 2018,
        val_year_start: int = 2019,
        val_year_end: int = 2019,
        test_year_start: int = 2020,
        test_year_end: int = 2020,
        batch_size: int = 4,
        num_workers: int = 2,
        # A7 ablation knobs
        plan_type: str = "uniform",
        canonical: int | None = None,
        budget: int = 512,
        criterion_var: str = "z",
        criterion_level: int = 500,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def _nc_files(self, year_start: int, year_end: int) -> list[str]:
        root = os.path.expandvars(self.hparams.root)
        return [
            os.path.join(root, self.hparams.nc_filename.format(year=y))
            for y in range(year_start, year_end + 1)
        ]

    def _make_dataset(self, year_start: int, year_end: int) -> MosaicastDataset:
        h = self.hparams
        root = os.path.expandvars(h.root)
        static_nc = os.path.join(root, h.static_nc) if h.static_nc else None
        return MosaicastDataset(
            nc_file_paths=self._nc_files(year_start, year_end),
            surf_vars=tuple(h.surf_vars),
            atmos_vars=tuple(h.atmos_vars),
            atmos_levels=tuple(h.atmos_levels),
            static_nc=static_nc,
            static_vars=tuple(h.static_vars),
            dt_hours=list(h.dt_hours),
        )

    def _make_plan_fn(self):
        """Build the plan factory function for the collate fn (A7 ablation)."""
        h = self.hparams
        p = h.canonical if h.canonical is not None else h.patch_size
        plan_type = h.plan_type

        if plan_type == "uniform":
            return None  # collate_fn default

        if plan_type == "lat_band":
            from mosaicast.patching.plans import latitude_band_plan
            def _plan_fn(inp_batch, lat, lon):
                return latitude_band_plan(lat, lon, p_lat=p)
            return _plan_fn

        if plan_type == "content_adaptive":
            from mosaicast.patching.plans import content_adaptive_plan
            atmos_levels = tuple(h.atmos_levels)
            crit_level   = h.criterion_level
            crit_var     = h.criterion_var
            budget       = h.budget

            def _plan_fn(inp_batch, lat, lon):
                # Extract criterion field: mean over batch of Z at criterion_level
                if crit_var in inp_batch.atmos_vars:
                    level_idx = atmos_levels.index(crit_level) if crit_level in atmos_levels else 0
                    field = inp_batch.atmos_vars[crit_var][:, -1, level_idx].float().mean(0)
                elif crit_var in inp_batch.surf_vars:
                    field = inp_batch.surf_vars[crit_var][:, -1].float().mean(0)
                else:
                    # Fallback: zero field → uniform-ish split
                    H, W = len(lat), len(lon)
                    field = torch.zeros(H, W, device=lat.device)
                return content_adaptive_plan(field, lat, lon, budget=budget, p=p)

            return _plan_fn

        raise ValueError(f"plan_type must be 'uniform', 'lat_band', or 'content_adaptive', got {plan_type!r}")

    def _collate(self):
        h = self.hparams
        p = h.canonical if h.canonical is not None else h.patch_size
        plan_fn = self._make_plan_fn()
        return mosaicast_collate_fn(patch_size=p, plan_fn=plan_fn)

    def _sampler(self, dataset, shuffle: bool, drop_last: bool) -> DtBatchSampler:
        try:
            num_replicas = self.trainer.world_size
            rank = self.trainer.global_rank
        except RuntimeError:
            num_replicas, rank = 1, 0
        return DtBatchSampler(
            n=len(dataset),
            batch_size=self.hparams.batch_size,
            dt_hours=list(self.hparams.dt_hours),
            shuffle=shuffle,
            drop_last=drop_last,
            seed=42,
            num_replicas=num_replicas,
            rank=rank,
        )

    def setup(self, stage: str | None = None) -> None:
        h = self.hparams
        if stage in ("fit", None):
            self._train_ds = self._make_dataset(h.train_year_start, h.train_year_end)
            self._val_ds   = self._make_dataset(h.val_year_start,   h.val_year_end)
        if stage in ("test", None):
            self._test_ds  = self._make_dataset(h.test_year_start,  h.test_year_end)

    def train_dataloader(self) -> DataLoader:
        nw = self.hparams.num_workers
        return DataLoader(
            self._train_ds,
            batch_sampler=self._sampler(self._train_ds, shuffle=True,  drop_last=True),
            num_workers=nw,
            collate_fn=self._collate(),
            pin_memory=True,
            persistent_workers=(nw > 0),
            prefetch_factor=2 if nw > 0 else None,
        )

    def val_dataloader(self) -> DataLoader:
        nw = self.hparams.num_workers
        return DataLoader(
            self._val_ds,
            batch_sampler=self._sampler(self._val_ds, shuffle=False, drop_last=False),
            num_workers=nw,
            collate_fn=self._collate(),
            pin_memory=True,
            persistent_workers=(nw > 0),
            prefetch_factor=2 if nw > 0 else None,
        )

    def test_dataloader(self) -> DataLoader:
        nw = self.hparams.num_workers
        return DataLoader(
            self._test_ds,
            batch_sampler=self._sampler(self._test_ds, shuffle=False, drop_last=False),
            num_workers=nw,
            collate_fn=self._collate(),
            pin_memory=True,
            persistent_workers=(nw > 0),
            prefetch_factor=2 if nw > 0 else None,
        )


# ---------------------------------------------------------------------------
# M4: rollout finetuning
# ---------------------------------------------------------------------------

class MosaicastRolloutModule(MosaicastLightningModule):
    """K-step rollout finetuning with pushforward gradient (CLAUDE.md §5, M4).

    Extends MosaicastLightningModule with a fixed-K rollout training step.

    A6 ablation: randomize_dt_within_rollout — if True, draws a fresh δt per
        rollout step instead of using the fixed lead_time.  This violates §8.1
        and is expected to destabilise training.  Document, do not ship.
        # DEVIATION: A6 ablation only — violates CLAUDE §8.1.

    Args:
        rollout_k:                  Number of rollout steps K.
        replay_buf_size:            Capacity of the ReplayBuffer.
        replay_prob:                Probability of replacing inp with a buffered state.
        randomize_dt_within_rollout: A6 destabiliser ablation (default False).
        All other args:             forwarded to MosaicastLightningModule.
    """

    def __init__(
        self,
        surf_vars: tuple[str, ...] = ("2t", "10u", "10v", "msl"),
        static_vars: tuple[str, ...] = ("lsm", "z", "slt"),
        atmos_vars: tuple[str, ...] = ("z", "u", "v", "t", "q"),
        patch_size: int = 4,
        latent_levels: int = 4,
        embed_dim: int = 256,
        encoder_depth: int = 4,
        processor_depth: int = 8,
        processor_heads: int = 8,
        processor_head_dim: int = 64,
        mlp_ratio: float = 4.0,
        max_history_size: int = 2,
        stats_path: str | None = None,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        loss_norms: dict[str, str] | None = None,
        target_mode: str = "dynamics",
        processor_type: str = "isotropic",
        dt_cond: str = "adaln_zero",
        qk_norm: bool = True,
        condition_on_rollout_step: bool = False,
        stabilise_level_agg: bool = False,
        tokenizer: str = "resize",
        loss_norm: str = "l2",
        pressure_weighted: bool = False,
        rollout_k: int = 1,
        replay_buf_size: int = 256,
        replay_prob: float = 0.5,
        randomize_dt_within_rollout: bool = False,
    ) -> None:
        super().__init__(
            surf_vars=surf_vars,
            static_vars=static_vars,
            atmos_vars=atmos_vars,
            patch_size=patch_size,
            latent_levels=latent_levels,
            embed_dim=embed_dim,
            encoder_depth=encoder_depth,
            processor_depth=processor_depth,
            processor_heads=processor_heads,
            processor_head_dim=processor_head_dim,
            mlp_ratio=mlp_ratio,
            max_history_size=max_history_size,
            stats_path=stats_path,
            lr=lr,
            weight_decay=weight_decay,
            loss_norms=loss_norms,
            target_mode=target_mode,
            processor_type=processor_type,
            dt_cond=dt_cond,
            qk_norm=qk_norm,
            condition_on_rollout_step=condition_on_rollout_step,
            stabilise_level_agg=stabilise_level_agg,
            tokenizer=tokenizer,
            loss_norm=loss_norm,
            pressure_weighted=pressure_weighted,
        )
        self.rollout_k = rollout_k
        self.randomize_dt_within_rollout = randomize_dt_within_rollout
        self._replay = ReplayBuffer(max_size=replay_buf_size, p_replay=replay_prob)

    # ------------------------------------------------------------------
    # LightningModule API
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        inp, tar, plan = batch
        assert self.stats is not None, (
            "ResidualStats not set — pass stats_path or call set_stats() before fit."
        )

        dt_total = self._dt_from_batch(inp, tar)
        assert dt_total % self.rollout_k == 0, (
            f"dt_total ({dt_total}h) not divisible by rollout_k ({self.rollout_k}). "
            "DtBatchSampler must sample dt_hours = k * dt_per_step."
        )
        dt_per_step = dt_total // self.rollout_k
        assert dt_per_step in self.stats.dt_hours, (
            f"dt_per_step ({dt_per_step}h) not in stats.dt_hours {self.stats.dt_hours}."
        )

        lat    = inp.metadata.lat
        lon    = inp.metadata.lon
        index  = plan.index(lat, lon)
        device = next(iter(inp.surf_vars.values())).device

        inp_eff, tar_eff = self._replay.maybe_replace(inp, tar, device=device)

        # A6 ablation: randomized-within-rollout (destabiliser)
        # DEVIATION: A6 ablation only — violates CLAUDE §8.1. Do not ship.
        if self.randomize_dt_within_rollout:
            rng = np.random.default_rng()
            lead_times_k = [
                timedelta(hours=int(rng.choice(self.stats.dt_hours)))
                for _ in range(self.rollout_k)
            ]
        else:
            lead_times_k = timedelta(hours=dt_per_step)  # scalar: same for all steps

        preds, inp_states = rollout(
            self.model, inp_eff, index, lead_times_k, k=self.rollout_k
        )
        final_pred = preds[-1]

        loss = self._residual_loss(final_pred, inp_states[-1], tar_eff, dt_per_step)

        self.log("train/loss",      loss,                    on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/rollout_k", float(self.rollout_k),   on_step=True, prog_bar=False)

        self._replay.push(inp, tar, dt_per_step)
        return loss

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        inp, tar, plan = batch
        assert self.stats is not None
        dt_total = self._dt_from_batch(inp, tar)
        assert dt_total % self.rollout_k == 0, (
            f"val dt_total ({dt_total}h) not divisible by rollout_k ({self.rollout_k})."
        )
        dt_per_step = dt_total // self.rollout_k
        assert dt_per_step in self.stats.dt_hours, (
            f"val dt_per_step ({dt_per_step}h) not in stats.dt_hours {self.stats.dt_hours}."
        )
        lead_time = timedelta(hours=dt_per_step)
        index     = plan.index(inp.metadata.lat, inp.metadata.lon)

        with torch.no_grad():
            preds, inp_states = rollout(self.model, inp, index, lead_time, k=self.rollout_k)
        final_pred = preds[-1]

        loss = self._residual_loss(final_pred, inp_states[-1], tar, dt_per_step)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss


class MosaicastRolloutDataModule(MosaicastDataModule):
    """DataModule for M4 rollout finetuning.

    Extends MosaicastDataModule with a rollout_k parameter.  The
    DtBatchSampler samples dt_hours from ``[rollout_k * dt for dt in
    base_dt_hours]`` so that each loaded target is exactly K steps ahead.

    Args:
        rollout_k:      Number of rollout steps.  Multiplies base_dt_hours.
        base_dt_hours:  Per-step δt values, e.g. [6, 12, 24].
        All other args: forwarded to MosaicastDataModule.
    """

    def __init__(
        self,
        root: str = "data",
        nc_filename: str = "aurora_{year}_5.625deg.nc",
        patch_size: int = 4,
        surf_vars: tuple[str, ...] = ("2t", "10u", "10v", "msl"),
        atmos_vars: tuple[str, ...] = ("z", "u", "v", "t", "q"),
        atmos_levels: tuple[int, ...] = (50, 250, 500, 600, 700, 850, 925),
        static_nc: str | None = None,
        static_vars: tuple[str, ...] = ("lsm", "z", "slt"),
        base_dt_hours: list[int] = (6, 12, 24),
        rollout_k: int = 1,
        train_year_start: int = 1979,
        train_year_end: int = 2018,
        val_year_start: int = 2019,
        val_year_end: int = 2019,
        test_year_start: int = 2020,
        test_year_end: int = 2020,
        batch_size: int = 4,
        num_workers: int = 2,
        plan_type: str = "uniform",
        canonical: int | None = None,
        budget: int = 512,
        criterion_var: str = "z",
        criterion_level: int = 500,
    ) -> None:
        rollout_dt_hours = [rollout_k * dt for dt in base_dt_hours]
        super().__init__(
            root=root,
            nc_filename=nc_filename,
            patch_size=patch_size,
            surf_vars=surf_vars,
            atmos_vars=atmos_vars,
            atmos_levels=atmos_levels,
            static_nc=static_nc,
            static_vars=static_vars,
            dt_hours=rollout_dt_hours,
            train_year_start=train_year_start,
            train_year_end=train_year_end,
            val_year_start=val_year_start,
            val_year_end=val_year_end,
            test_year_start=test_year_start,
            test_year_end=test_year_end,
            batch_size=batch_size,
            num_workers=num_workers,
            plan_type=plan_type,
            canonical=canonical,
            budget=budget,
            criterion_var=criterion_var,
            criterion_level=criterion_level,
        )
