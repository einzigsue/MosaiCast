"""MosaicastEncoder — Perceiver3DEncoder + AdaptivePatchEmbed (CLAUDE.md §3)."""
from __future__ import annotations

from datetime import timedelta

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from aurora.batch import Batch
from aurora.model.encoder import Perceiver3DEncoder
from aurora.model.fourier import (
    absolute_time_expansion,
    lead_time_expansion,
    levels_expansion,
    pos_expansion,
    scale_expansion,
)
from aurora.model.util import check_lat_lon_dtype

from mosaicast.patching.embed import AdaptivePatchEmbed, AttentionPoolPatchEmbed
from mosaicast.patching.plan import PatchIndex

__all__ = ["MosaicastEncoder"]


class MosaicastEncoder(nn.Module):
    """Wrap Perceiver3DEncoder, replacing fixed-patch embed with adaptive patches.

    All learnable parameters (surf/atmos LevelPatchEmbed weights, level Perceiver,
    position/scale/time embedding linears) live in the held Perceiver3DEncoder and
    are trained as usual.

    Constraints / deviations from vanilla Aurora encoder:
      • dynamic_vars=True and atmos_static_vars=True are not yet supported (M2).
      • level_condition is not yet supported (M2).
      • Position and scale encodings are computed from PatchIndex centroids/areas
        instead of from the raw lat/lon grid.  This is the core adaptive-patch
        change; everything after the token aggregation is identical to Aurora.

    tokenizer: "resize" (default/control) uses AdaptivePatchEmbed (bilinear resize).
               "attention_pool" uses AttentionPoolPatchEmbed (Perceiver cross-attn, A7).
    """

    def __init__(
        self,
        aurora_encoder: Perceiver3DEncoder,
        tokenizer: str = "resize",
        embed_dim: int | None = None,
    ) -> None:
        super().__init__()
        enc = aurora_encoder

        if enc.dynamic_vars:
            raise NotImplementedError("dynamic_vars not yet supported in MosaicastEncoder (M2)")
        if enc.atmos_static_vars:
            raise NotImplementedError("atmos_static_vars not yet supported in MosaicastEncoder (M2)")
        if enc.level_condition:
            raise NotImplementedError("level_condition not yet supported in MosaicastEncoder (M2)")
        if tokenizer not in ("resize", "attention_pool"):
            raise ValueError(f"tokenizer must be 'resize' or 'attention_pool', got {tokenizer!r}")

        self._enc = enc
        _embed_dim = embed_dim if embed_dim is not None else enc.embed_dim

        if tokenizer == "resize":
            # Shared weights from Aurora's LevelPatchEmbed
            self.surf_embed  = AdaptivePatchEmbed(enc.surf_token_embeds)
            self.atmos_embed = AdaptivePatchEmbed(enc.atmos_token_embeds)
        else:
            # A7 ablation: attention-pool tokenizer (trained from scratch, no shared weights)
            self.surf_embed  = AttentionPoolPatchEmbed(embed_dim=_embed_dim)
            self.atmos_embed = AttentionPoolPatchEmbed(embed_dim=_embed_dim)

    def forward(self, batch: Batch, lead_time: timedelta, index: PatchIndex) -> Tensor:
        """Encode a batch with adaptive patching.

        Args:
            batch:     aurora.Batch (history T frames, surf + static + atmos vars).
            lead_time: forecast interval δt (for the lead-time encoding).
            index:     validated PatchIndex — coverage-checked, memoised per (plan, grid).

        Returns:
            (B, latent_levels * n_patches, D) flat latent token sequence.
        """
        enc = self._enc

        surf_vars = tuple(batch.surf_vars.keys())
        static_vars = tuple(batch.static_vars.keys()) if batch.static_vars else ()
        atmos_vars = tuple(batch.atmos_vars.keys())
        atmos_levels = batch.metadata.atmos_levels

        x_surf = torch.stack(tuple(batch.surf_vars.values()), dim=2)   # (B, T, V_S, H, W)
        x_atmos = torch.stack(tuple(batch.atmos_vars.values()), dim=2) # (B, T, V_A, C_A, H, W)
        B, T, _, C_A, H, W = x_atmos.shape

        # Static vars are (H, W) tensors — expand to (B, T, V_St, H, W) before concat.
        # (Aurora Batch.static_vars stores 2D fields; the encoder treats them as surface vars.)
        if static_vars:
            x_static = torch.stack(tuple(batch.static_vars.values()), dim=0)  # (V_St, H, W)
            x_static = x_static.unsqueeze(0).unsqueeze(0).expand(B, T, -1, H, W)
            x_surf = torch.cat((x_surf, x_static), dim=2)   # (B, T, V_S+V_St, H, W)
            surf_vars = surf_vars + static_vars

        lat, lon = batch.metadata.lat, batch.metadata.lon
        check_lat_lon_dtype(lat, lon)
        lat = lat.to(dtype=torch.float32)
        lon = lon.to(dtype=torch.float32)

        # --- Adaptive patch embedding ---
        x_surf = rearrange(x_surf, "b t v h w -> b v t h w")
        x_surf = self.surf_embed(x_surf, surf_vars, index)  # (B, n_patches, D)
        dtype = x_surf.dtype

        x_atmos = rearrange(x_atmos, "b t v c h w -> (b c) v t h w")
        x_atmos = self.atmos_embed(x_atmos, atmos_vars, index)  # (B*C_A, n_patches, D)
        x_atmos = rearrange(x_atmos, "(b c) l d -> b c l d", b=B, c=C_A)  # (B, C_A, n, D)

        # --- Level-specific encodings (same as vanilla Aurora) ---
        x_surf = x_surf + enc.surf_level_encoding[None, None, :].to(dtype=dtype)
        x_surf = x_surf + enc.surf_norm(enc.surf_mlp(x_surf))

        atmos_levels_tensor = torch.tensor(atmos_levels, device=x_atmos.device)
        atmos_levels_encode = levels_expansion(atmos_levels_tensor, enc.embed_dim).to(dtype=dtype)
        atmos_levels_embed = enc.atmos_levels_embed(atmos_levels_encode)[None, :, None, :]
        x_atmos = x_atmos + atmos_levels_embed  # (B, C_A, n, D)

        x_atmos = enc.aggregate_levels(x_atmos)  # (B, C, n, D) where C = latent_levels-1

        # --- Concat surface + aggregated atmos → (B, latent_levels, n, D) ---
        x = torch.cat((x_surf.unsqueeze(1), x_atmos), dim=1)

        # --- Position and scale encodings from PatchIndex ---
        # Use index centroids and sqrt(area) — matches Aurora's patch_root_area convention
        centroid_lat = index.centroid_lat.to(device=x.device, dtype=torch.float32)  # (n,)
        centroid_lon = index.centroid_lon.to(device=x.device, dtype=torch.float32)  # (n,)
        area_sqrt = torch.sqrt(index.area.to(device=x.device, dtype=torch.float32)) # (n,)

        lat_enc = pos_expansion(centroid_lat, enc.embed_dim // 2)   # (n, D/2)
        lon_enc = pos_expansion(centroid_lon, enc.embed_dim // 2)   # (n, D/2)
        pos_enc = torch.cat([lat_enc, lon_enc], dim=-1)             # (n, D)
        scl_enc = scale_expansion(area_sqrt, enc.embed_dim)         # (n, D)

        pos_enc = enc.pos_embed(pos_enc[None, None].to(dtype=dtype))    # (1, 1, n, D)
        scl_enc = enc.scale_embed(scl_enc[None, None].to(dtype=dtype))  # (1, 1, n, D)
        x = x + pos_enc + scl_enc  # (B, latent_levels, n, D)

        # --- Flatten to (B, latent_levels * n_patches, D) ---
        x = x.reshape(B, -1, enc.embed_dim)

        # --- Lead time embedding ---
        lead_hours = lead_time.total_seconds() / 3600
        lead_times = lead_hours * torch.ones(B, dtype=dtype, device=x.device)
        lead_time_encode = lead_time_expansion(lead_times, enc.embed_dim).to(dtype=dtype)
        lead_time_emb = enc.lead_time_embed(lead_time_encode)  # (B, D)
        x = x + lead_time_emb.unsqueeze(1)

        # --- Absolute time embedding ---
        abs_times = torch.tensor(
            [t.timestamp() / 3600 for t in batch.metadata.time],
            dtype=torch.float32, device=x.device,
        )
        abs_time_encode = absolute_time_expansion(abs_times, enc.embed_dim)
        abs_time_emb = enc.absolute_time_embed(abs_time_encode.to(dtype=dtype))  # (B, D)
        x = x + abs_time_emb.unsqueeze(1)

        x = enc.pos_drop(x)
        return x
