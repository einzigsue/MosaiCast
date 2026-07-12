"""AdaptivePatchEmbed variants (CLAUDE.md §2.3).

A7 ablation: AttentionPoolPatchEmbed — Perceiver attention-pool tokenizer that
handles variable cell counts natively without bilinear resizing.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from aurora.model.patchembed import LevelPatchEmbed
from aurora.model.perceiver import PerceiverResampler

from .plan import PatchIndex


class AdaptivePatchEmbed(nn.Module):
    """Adaptive replacement for Aurora's fixed P×P patch embed.

    For each patch k defined by the PatchIndex:
        crop cells (rows[k] × cols[k]) from the input field
        → bilinear resize to (canonical × canonical)
        → per-variable linear embed via the shared LevelPatchEmbed

    Output shape: (B, n_patches, D) — same downstream layout as Aurora's (B, L, D).
    Weights are NOT copied: the LevelPatchEmbed is held by reference so Aurora's
    pretrained weights are shared directly.

    Shapes:
        input  x: (B, V, T, H, W)
        output:   (B, n_patches, D)
    """

    def __init__(self, level_embed: LevelPatchEmbed) -> None:
        super().__init__()
        self.level_embed = level_embed

    def forward(self, x: Tensor, var_names: tuple[str, ...], index: PatchIndex) -> Tensor:
        """Embed one level's variables using adaptive patches.

        Args:
            x:         (B, V, T, H, W) rearranged field (variables before time).
            var_names: V variable names matching LevelPatchEmbed's weight dict.
            index:     validated PatchIndex for this grid.

        Returns:
            (B, n_patches, D) — one token per patch.
        """
        B, V, T, H, W = x.shape
        p = index.plan.canonical
        n = len(index.plan)

        patch_list: list[Tensor] = []
        for r, c in zip(index.rows, index.cols):
            crop = x[:, :, :, r.start:r.stop, :][:, :, :, :, c]  # (B, V, T, H_k, W_k)
            H_k, W_k = r.stop - r.start, c.shape[0]
            if H_k == p and W_k == p:
                patch_list.append(crop)
            else:
                flat = crop.reshape(B * V * T, 1, H_k, W_k)
                resized = F.interpolate(
                    flat.float(), size=(p, p), mode="bilinear", align_corners=False,
                )
                patch_list.append(resized.reshape(B, V, T, p, p).to(x.dtype))

        stacked = torch.stack(patch_list, dim=1).reshape(B * n, V, T, p, p)
        tokens = self.level_embed(stacked, var_names)
        return tokens.reshape(B, n, tokens.size(-1))   # (B, n, D)


class AttentionPoolPatchEmbed(nn.Module):
    """Attention-pool tokenizer for adaptive patches (A7 ablation).

    Instead of bilinear resizing to (canonical × canonical), each patch's
    variable-many grid cells are projected to D and then cross-attended to a
    single latent token via a 1-latent PerceiverResampler.  This handles
    variable cell counts natively without a fixed resize step.

    Comparison point for A7: accuracy vs. AdaptivePatchEmbed (resize) at equal
    patch budget.  Trained from scratch — weights differ from Aurora's LevelPatchEmbed.

    Shapes:
        input  x: (B, V, T, H, W)
        output:   (B, n_patches, D)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        mlp_ratio: float = 4.0,
    ) -> None:
        """
        Args:
            embed_dim: token dimension D (must match the rest of the model).
            num_heads: attention heads in the cross-attention resampler.
            head_dim:  per-head dimension.
            mlp_ratio: FFN hidden / embed_dim ratio in the resampler.
        """
        super().__init__()
        self.embed_dim = embed_dim
        # Cell projection: maps a single (V, T) feature vector to D
        self.cell_proj = nn.LazyLinear(embed_dim)
        # 1-latent Perceiver: cross-attends over the cell tokens → 1 token
        self.resampler = PerceiverResampler(
            latent_dim=embed_dim,
            context_dim=embed_dim,
            depth=1,
            num_heads=num_heads,
            head_dim=head_dim,
            mlp_ratio=mlp_ratio,
        )
        # Single learned latent query initialised to zeros (will be broadcast over B and n)
        self.latent = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(self, x: Tensor, var_names: tuple[str, ...], index: PatchIndex) -> Tensor:
        """Embed one level's variables via attention-pool patches.

        Args:
            x:         (B, V, T, H, W) rearranged field.
            var_names: variable names (used for ordering; not for weight lookup here).
            index:     validated PatchIndex.

        Returns:
            (B, n_patches, D) — one token per patch.
        """
        B, V, T, H, W = x.shape
        n = len(index.plan)
        device, dtype = x.device, x.dtype

        # Project each grid cell to D: flatten (V, T) → cell_proj → D
        # x: (B, V, T, H, W) → (B, H, W, V*T) → cell_proj → (B, H, W, D)
        x_flat = x.permute(0, 3, 4, 1, 2).reshape(B, H, W, V * T)
        x_cells = self.cell_proj(x_flat.float()).to(dtype)  # (B, H, W, D)

        # Process each patch with the resampler
        token_list: list[Tensor] = []
        latent_q = self.latent.expand(B, 1, self.embed_dim).to(device=device, dtype=dtype)

        for r, c in zip(index.rows, index.cols):
            # Crop cells for this patch: (B, H_k, W_k, D) → flatten → (B, H_k*W_k, D)
            cells = x_cells[:, r.start:r.stop, :, :][:, :, c, :]   # (B, H_k, W_k, D)
            H_k, W_k = cells.shape[1], cells.shape[2]
            cells = cells.reshape(B, H_k * W_k, self.embed_dim)     # (B, N_cell, D)

            # PerceiverResampler: (latent_q, context=cells) → (B, 1, D)
            tok = self.resampler(latent_q, cells)                    # (B, 1, D)
            token_list.append(tok[:, 0, :])                          # (B, D)

        return torch.stack(token_list, dim=1)   # (B, n_patches, D)
