"""AdaptivePatchReconstruct — scatter patch tokens to grid (CLAUDE.md §2.3)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .plan import PatchIndex


class AdaptivePatchReconstruct(nn.Module):
    """Scatter (B, n_patches, P²) projected token values back to an (H, W) grid.

    Inverse of the spatial step in AdaptivePatchEmbed:
        (canonical × canonical) tile → bilinear resize to patch cell dims → scatter.

    The plan is a partition (no overlaps), so scatter is a plain assignment.
    Each patch is resized independently; no averaging needed.
    """

    def forward(self, x: Tensor, index: PatchIndex) -> Tensor:
        """Scatter projected patch tokens to a full grid.

        Args:
            x:     (B, n_patches, P²) projected pixel values for one variable, one level.
            index: validated PatchIndex for the target grid.

        Returns:
            (B, H, W) reconstructed field.
        """
        B, n, P2 = x.shape
        p = index.plan.canonical
        H, W = index.grid_shape

        out = x.new_zeros(B, H, W)
        out_flat = out.view(B, H * W)  # view shares storage; writes propagate to out
        x_tiles = x.reshape(B, n, p, p)

        for k, (r, c) in enumerate(zip(index.rows, index.cols)):
            H_k = r.stop - r.start
            W_k = c.shape[0]
            tile = x_tiles[:, k]  # (B, p, p)

            if H_k != p or W_k != p:
                tile = F.interpolate(
                    tile.unsqueeze(1).float(), size=(H_k, W_k),
                    mode="bilinear", align_corners=False,
                ).squeeze(1).to(x.dtype)  # (B, H_k, W_k)

            # Flat grid positions for this patch
            ri = torch.arange(r.start, r.stop, device=x.device, dtype=torch.long)
            row_flat = ri.unsqueeze(1).expand(H_k, W_k).reshape(-1)         # (H_k*W_k,)
            col_flat = c.to(x.device).unsqueeze(0).expand(H_k, W_k).reshape(-1)
            grid_flat = row_flat * W + col_flat                               # (H_k*W_k,)

            out_flat[:, grid_flat] = tile.reshape(B, H_k * W_k)

        return out


def adaptive_unpatchify(x: Tensor, V: int, index: PatchIndex, P: int) -> Tensor:
    """Adaptive replacement for Aurora's `unpatchify`.

    Scatters a batch of per-patch, per-variable pixel values to a full spatial grid,
    handling variable-size patches via bilinear resize.

    Args:
        x:     (B, n_patches, C, V*P²) — decoder head outputs, all vars concatenated.
        V:     number of variables packed into the last dim.
        index: validated PatchIndex for the target grid.
        P:     canonical patch size (same as index.plan.canonical).

    Returns:
        (B, V, C, H, W) — unpatchified full-resolution output.
    """
    B, n, C, _ = x.shape
    H, W = index.grid_shape

    x = x.reshape(B, n, C, V, P, P)  # (B, n, C, V, P, P)
    out = x.new_zeros(B, V, C, H, W)
    out_flat = out.view(B, V, C, H * W)  # view; writes propagate to out

    for k, (r, c) in enumerate(zip(index.rows, index.cols)):
        H_k = r.stop - r.start
        W_k = c.shape[0]
        tile = x[:, k]  # (B, C, V, P, P)

        if H_k != P or W_k != P:
            tile_r = tile.reshape(B * C * V, 1, P, P)
            tile_r = F.interpolate(
                tile_r.float(), size=(H_k, W_k), mode="bilinear", align_corners=False,
            )
            tile = tile_r.reshape(B, C, V, H_k, W_k).to(x.dtype)
        # else tile is already (B, C, V, P, P) — treat as (B, C, V, H_k, W_k)

        # Permute to (B, V, C, H_k*W_k) for assignment
        tile = tile.permute(0, 2, 1, 3, 4).reshape(B, V, C, H_k * W_k)  # (B, V, C, H_k*W_k)

        ri = torch.arange(r.start, r.stop, device=x.device, dtype=torch.long)
        row_flat = ri.unsqueeze(1).expand(H_k, W_k).reshape(-1)
        col_flat = c.to(x.device).unsqueeze(0).expand(H_k, W_k).reshape(-1)
        grid_flat = row_flat * W + col_flat  # (H_k*W_k,)

        out_flat[:, :, :, grid_flat] = tile

    return out
