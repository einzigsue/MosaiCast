"""MosaicastDecoder — Perceiver3DDecoder + AdaptivePatchReconstruct (CLAUDE.md §3).

Key constraint (CLAUDE.md §1.3): decoder is built with embed_dim=embed_dim (NOT *2).
"""
from __future__ import annotations

from datetime import timedelta

import torch
import torch.nn as nn
from torch import Tensor

from aurora.batch import Batch, Metadata
from aurora.model.decoder import Perceiver3DDecoder
from aurora.model.fourier import levels_expansion
from aurora.model.util import check_lat_lon_dtype

from mosaicast.patching.plan import PatchIndex
from mosaicast.patching.reconstruct import AdaptivePatchReconstruct, adaptive_unpatchify

__all__ = ["MosaicastDecoder"]


class MosaicastDecoder(nn.Module):
    """Wrap Perceiver3DDecoder, replacing unpatchify with adaptive scatter.

    All learnable parameters (surf/atmos linear heads, level Perceiver) live in the
    held Perceiver3DDecoder and are trained as usual.

    Constraint: the encoder produces C = latent_levels-1 aggregated atmos levels.
    The decoder de-aggregates those back to the physical C_A pressure levels.
    ``latent_levels`` must be stored here so the rearrange in forward is correct.
    """

    def __init__(
        self,
        aurora_decoder: Perceiver3DDecoder,
        latent_levels: int,
    ) -> None:
        super().__init__()
        self._dec = aurora_decoder
        self.latent_levels = latent_levels
        self._reconstruct = AdaptivePatchReconstruct()

    def forward(
        self,
        x: Tensor,
        batch: Batch,
        patch_res: tuple[int, int, int],
        lead_time: timedelta,
        index: PatchIndex,
    ) -> Batch:
        """Decode latent tokens to a full-resolution predicted Batch.

        Args:
            x:         (B, latent_levels * n_patches, D) from the processor.
            batch:     input Batch — supplies metadata (lat, lon, time, levels).
            patch_res: (latent_levels, n_patches, 1) — used to reshape x.
            lead_time: forecast interval added to batch.metadata.time.
            index:     validated PatchIndex for the target grid.

        Returns:
            aurora.Batch at time t + lead_time.
        """
        dec = self._dec
        B = x.size(0)

        surf_vars = tuple(batch.surf_vars.keys())
        atmos_vars = tuple(batch.atmos_vars.keys())
        atmos_levels = batch.metadata.atmos_levels
        C_A = len(atmos_levels)
        P = dec.patch_size

        # Modulation heads: add _mod-suffixed vars (empty by default)
        surf_vars_dec = surf_vars + tuple(f"{v}_mod" for v in surf_vars if v in dec.modulation_heads)
        atmos_vars_dec = atmos_vars + tuple(f"{v}_mod" for v in atmos_vars if v in dec.modulation_heads)

        lat, lon = batch.metadata.lat, batch.metadata.lon
        check_lat_lon_dtype(lat, lon)
        lat = lat.to(dtype=torch.float32)
        lon = lon.to(dtype=torch.float32)
        H, W = lat.shape[0], lon.shape[-1]

        # Rearrange: (B, L', D) → (B, n_patches, latent_levels, D)
        C_lat, n, W1 = patch_res
        x = x.reshape(B, C_lat, n, W1, dec.embed_dim)          # (B, C_lat, n, 1, D)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, n * W1, C_lat, dec.embed_dim)
        # x: (B, n_patches, latent_levels, D)

        # --- Surf predictions ---
        x_surf_tok = x[..., :1, :]  # (B, n, 1, D)  — surface latent level
        x_surf_per_var = torch.stack(
            [dec.surf_heads[name](x_surf_tok) for name in surf_vars_dec], dim=-1
        )  # (B, n, 1, P², V_S)
        x_surf_per_var = x_surf_per_var.reshape(B, n, 1, -1)  # (B, n, 1, V_S*P²)
        surf_preds = adaptive_unpatchify(x_surf_per_var, len(surf_vars_dec), index, P)
        surf_preds = surf_preds.squeeze(2)  # (B, V_S, H, W)

        # --- Atmos: de-aggregate latent → physical pressure levels ---
        atmos_levels_encode = levels_expansion(
            torch.tensor(atmos_levels, device=x.device), dec.embed_dim
        ).to(dtype=x.dtype)
        levels_emb = dec.atmos_levels_embed(atmos_levels_encode)       # (C_A, D)
        levels_emb_expanded = levels_emb.expand(B, n, -1, -1)          # (B, n, C_A, D)

        x_atmos = dec.deaggregate_levels(
            levels_emb_expanded,
            x[..., 1:, :],   # (B, n, latent_levels-1, D) — atmos latent context
            dec.level_decoder,
        )  # (B, n, C_A, D)

        x_atmos_per_var = torch.stack(
            [dec.atmos_heads[name](x_atmos) for name in atmos_vars_dec], dim=-1
        )  # (B, n, C_A, P², V_A)
        x_atmos_per_var = x_atmos_per_var.reshape(B, n, C_A, -1)  # (B, n, C_A, V_A*P²)
        atmos_preds = adaptive_unpatchify(x_atmos_per_var, len(atmos_vars_dec), index, P)
        # (B, V_A, C_A, H, W)

        return Batch(
            surf_vars={v: surf_preds[:, i] for i, v in enumerate(surf_vars_dec)},
            static_vars=batch.static_vars,
            atmos_vars={v: atmos_preds[:, i] for i, v in enumerate(atmos_vars_dec)},
            metadata=Metadata(
                lat=lat,
                lon=lon,
                time=tuple(t + lead_time for t in batch.metadata.time),
                atmos_levels=atmos_levels,
                rollout_step=batch.metadata.rollout_step + 1,
            ),
        )
