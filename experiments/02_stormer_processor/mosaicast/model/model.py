"""MosaicastModel — top-level forward (CLAUDE.md §3).

A1 ablation: target_mode in {"dynamics", "absolute"} — controls whether the decoder
             output is treated as a residual or an absolute state.
A2 ablation: processor_type in {"isotropic", "hybrid", "swin"} — processor variant.
A4 ablation: dt_cond, qk_norm, condition_on_rollout_step — conditioning variants.
A7 ablation: tokenizer in {"resize", "attention_pool"} — patch tokenizer variant.
A8 ablation: stabilise_level_agg, latent_levels sweep.
"""
from __future__ import annotations

from datetime import timedelta

import torch.nn as nn

from aurora.batch import Batch
from aurora.model.decoder import Perceiver3DDecoder
from aurora.model.encoder import Perceiver3DEncoder

from mosaicast.dynamics.randomized import ResidualStats, reconstruct_state
from mosaicast.patching.plan import PatchIndex

from .decoder import MosaicastDecoder
from .encoder import MosaicastEncoder
from .processor import HybridProcessor, StormerProcessor

__all__ = ["MosaicastModel"]


def _make_processor(
    processor_type: str,
    embed_dim: int,
    depth: int,
    num_heads: int,
    head_dim: int,
    mlp_ratio: float,
    qk_norm: bool,
    dt_cond: str,
    condition_on_rollout_step: bool,
    latent_levels: int,
) -> nn.Module:
    """Factory: build the processor specified by processor_type.

    "isotropic" → StormerProcessor (control).
    "hybrid"    → HybridProcessor with 1 merge stage (A2 ablation).
    "swin"      → Aurora Swin3DTransformerBackbone (A2 ablation).
                  Output is 2*embed_dim; a thin linear projects back to embed_dim.
                  # DEVIATION: swin emits 2*embed_dim due to skip-concat (CLAUDE §1.3).
                  This projection is ablation-only — never used in production.
    """
    if processor_type == "isotropic":
        return StormerProcessor(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            head_dim=head_dim,
            mlp_ratio=mlp_ratio,
            qk_norm=qk_norm,
            dt_cond=dt_cond,
            condition_on_rollout_step=condition_on_rollout_step,
        )

    if processor_type == "hybrid":
        return HybridProcessor(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            head_dim=head_dim,
            mlp_ratio=mlp_ratio,
            qk_norm=qk_norm,
            dt_cond=dt_cond,
            condition_on_rollout_step=condition_on_rollout_step,
            n_merge_stages=1,
            latent_levels=latent_levels,
        )

    if processor_type == "swin":
        from aurora.model.swin3d import Swin3DTransformerBackbone
        swin = Swin3DTransformerBackbone(embed_dim=embed_dim)
        # Thin projection for the 2*embed_dim skip-concat output — ablation only.
        # DEVIATION: ablation-only linear (CLAUDE §1.3). Do not ship.
        proj = nn.Linear(2 * embed_dim, embed_dim)

        class _SwinWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.swin = swin
                self.proj = proj

            def forward(self, x, lead_time, patch_res, rollout_step=0):
                out = self.swin(x, lead_time, rollout_step, patch_res)
                return self.proj(out)

        return _SwinWrapper()

    raise ValueError(
        f"processor_type must be 'isotropic', 'hybrid', or 'swin', got {processor_type!r}"
    )


class MosaicastModel(nn.Module):
    """Aurora encoder + processor + Aurora decoder with adaptive patching.

    Forward: inp → encoder(adaptive embed) → processor(global attn, δt cond)
             → decoder(Δ̂_norm or Ŷ_norm) → reconstruct_state → Batch at t+δt.

    Stats must be attached via set_stats() before calling forward().

    Ablation knobs (all default to control value; one at a time):
        target_mode:              "dynamics" (control) or "absolute" (A1).
        processor_type:           "isotropic" (control), "hybrid", "swin" (A2).
        dt_cond:                  "adaln_zero" (control), "additive", "none" (A4).
        qk_norm:                  True (control), False (A4).
        condition_on_rollout_step: False (control), True (A4).
        stabilise_level_agg:      False (control), True (A8).
        tokenizer:                "resize" (control), "attention_pool" (A7).
    """

    def __init__(
        self,
        surf_vars:                 tuple[str, ...],
        static_vars:               tuple[str, ...] | None,
        atmos_vars:                tuple[str, ...],
        patch_size:                int = 4,
        latent_levels:             int = 4,
        embed_dim:                 int = 512,
        encoder_depth:             int = 2,
        processor_depth:           int = 8,
        processor_heads:           int = 8,
        processor_head_dim:        int = 64,
        mlp_ratio:                 float = 4.0,
        max_history_size:          int = 2,
        # Ablation knobs (control defaults — must be no-ops when all at default)
        target_mode:               str = "dynamics",
        processor_type:            str = "isotropic",
        dt_cond:                   str = "adaln_zero",
        qk_norm:                   bool = True,
        condition_on_rollout_step: bool = False,
        stabilise_level_agg:       bool = False,
        tokenizer:                 str = "resize",
    ) -> None:
        """
        Args:
            surf_vars:       dynamic surface variable names.
            static_vars:     static variable names or None.
            atmos_vars:      atmospheric variable names.
            patch_size:      canonical patch size P; must equal PatchPlan.canonical.
            latent_levels:   number of latent levels (A8 sweep: 3, 4, 5).
            embed_dim:       token embedding dimension D.
                             CLAUDE.md §1.3: decoder uses embed_dim (NOT embed_dim*2).
            encoder_depth:   Perceiver cross-attention depth in encoder.
            processor_depth: number of processor blocks.
            processor_heads: attention heads in processor.
            processor_head_dim: head dimension in processor.
            mlp_ratio:       FFN hidden-dim / embed_dim ratio.
            max_history_size: history window T (Aurora uses 2).
            target_mode:     "dynamics" → predict residual Δ (control).
                             "absolute" → predict X_{t+δt} directly (A1 ablation).
            processor_type:  "isotropic" (control), "hybrid", "swin" (A2 ablation).
            dt_cond:         conditioning mode for the processor (A4 ablation).
            qk_norm:         QK-norm in processor for bf16 stability (A4 ablation).
            condition_on_rollout_step: add rollout-step embed to conditioning (A4).
            stabilise_level_agg: ln_k_q in Aurora level-aggregation Perceiver (A8).
            tokenizer:       "resize" (control, bilinear), "attention_pool" (A7).
        """
        super().__init__()
        if target_mode not in ("dynamics", "absolute"):
            raise ValueError(f"target_mode must be 'dynamics' or 'absolute', got {target_mode!r}")
        self.target_mode   = target_mode
        self.latent_levels = latent_levels
        self.stats: ResidualStats | None = None

        aurora_enc = Perceiver3DEncoder(
            surf_vars=surf_vars,
            static_vars=static_vars,
            atmos_vars=atmos_vars,
            patch_size=patch_size,
            latent_levels=latent_levels,
            embed_dim=embed_dim,
            depth=encoder_depth,
            max_history_size=max_history_size,
            stabilise_level_agg=stabilise_level_agg,  # A8
        )
        # CLAUDE.md §1.3: embed_dim (not *2) — no Swin skip-concat in Mosaicast
        aurora_dec = Perceiver3DDecoder(
            surf_vars=surf_vars,
            atmos_vars=atmos_vars,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )

        self.encoder = MosaicastEncoder(aurora_enc, tokenizer=tokenizer, embed_dim=embed_dim)
        self.processor = _make_processor(
            processor_type=processor_type,
            embed_dim=embed_dim,
            depth=processor_depth,
            num_heads=processor_heads,
            head_dim=processor_head_dim,
            mlp_ratio=mlp_ratio,
            qk_norm=qk_norm,
            dt_cond=dt_cond,
            condition_on_rollout_step=condition_on_rollout_step,
            latent_levels=latent_levels,
        )
        self.decoder = MosaicastDecoder(aurora_dec, latent_levels=latent_levels)

    def set_stats(self, stats: ResidualStats) -> None:
        """Attach finalized ResidualStats.  Must be called before forward()."""
        self.stats = stats

    def forward(self, inp: Batch, index: PatchIndex, lead_time: timedelta) -> Batch:
        """Run encoder → processor → decoder → reconstruct_state.

        In dynamics mode (target_mode="dynamics", control):
            decoder output = Δ̂_norm; reconstruct_state adds X_t.
        In absolute mode (target_mode="absolute", A1 ablation):
            decoder output = Ŷ_norm; reconstruct_state skips the base addition.

        Args:
            inp:       input aurora.Batch (history T frames).
            index:     PatchIndex — built once per (plan, grid) via pplan.index(lat, lon).
            lead_time: forecast interval δt as a timedelta.

        Returns:
            Reconstructed aurora.Batch at t + lead_time (absolute physical state).
        """
        assert self.stats is not None, (
            "MosaicastModel.stats not set — call set_stats() before forward()."
        )
        patch_res    = (self.latent_levels, len(index.plan), 1)
        rollout_step = inp.metadata.rollout_step

        tokens = self.encoder(inp, lead_time, index)
        tokens = self.processor(tokens, lead_time=lead_time, patch_res=patch_res,
                                rollout_step=rollout_step)
        resid  = self.decoder(tokens, inp, patch_res=patch_res, lead_time=lead_time, index=index)
        return reconstruct_state(
            inp, resid, lead_time, self.stats,
            add_base=(self.target_mode == "dynamics"),
        )
