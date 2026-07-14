"""v1 scatter-grid — zero new parameters.

Every leaf is resized to p×p in pixel space (F.interpolate, never cv2),
embedded by the unchanged LevelPatchEmbed via the strip trick (from gvit
vit_unetr2d.py), then scattered (copied) to every fine-grid cell it covers.

Integration point: _pre_encoder_hook replaces the standard token extraction
with the leaf-strip path and then scatters back to the regular grid before
the backbone sees any tokens.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from aurora import Aurora, Batch

from ..patching.base import Leaf, PatchingStrategy
from ..patching.quadtree import FixedQuadTree
from ..patching.saliency import SALIENCY_FNS


class ScatterGridStrategy(PatchingStrategy):
    """Resize each leaf to p×p, scatter token to all covered fine-grid cells."""

    def __init__(self, patch_size: int, leaf_budget: int, leaf_sizes: set[int]) -> None:
        self.patch_size = patch_size
        self._tree_cfg = dict(budget=leaf_budget, leaf_sizes=leaf_sizes)

    def _make_tree(self, H: int, W: int, lon_roll: int = 0) -> FixedQuadTree:
        return FixedQuadTree(H=H, W=W, lon_roll=lon_roll, **self._tree_cfg)

    def build(self, saliency: torch.Tensor, lon_roll: int = 0) -> list[Leaf]:
        H, W = saliency.shape
        return self._make_tree(H, W, lon_roll).build(saliency)

    def serialize(
        self,
        fields: torch.Tensor,
        leaves: list[Leaf],
        patch_size: int,
    ) -> torch.Tensor:
        """Resize each leaf to (p, p) and stack into (L, C, p, p)."""
        C = fields.shape[0]
        patches = []
        for lf in leaves:
            region = fields[:, lf.row_start:lf.row_stop, lf.col_start:lf.col_stop]
            patch = F.interpolate(
                region.unsqueeze(0),
                size=(patch_size, patch_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            patches.append(patch)
        return torch.stack(patches)  # (L, C, p, p)

    def scatter(
        self,
        leaf_tokens: torch.Tensor,
        leaves: list[Leaf],
        patch_size: int,
        grid_h: int,
        grid_w: int,
    ) -> torch.Tensor:
        """Copy each leaf token to all (s//p)² fine-grid cells it covers."""
        D = leaf_tokens.shape[-1]
        grid = torch.zeros(grid_h, grid_w, D, device=leaf_tokens.device, dtype=leaf_tokens.dtype)
        for i, lf in enumerate(leaves):
            r0, r1 = lf.row_start // patch_size, lf.row_stop // patch_size
            c0, c1 = lf.col_start // patch_size, lf.col_stop // patch_size
            grid[r0:r1, c0:c1] = leaf_tokens[i]
        return grid.reshape(grid_h * grid_w, D)


class AuroraScatterGrid(Aurora):
    """Aurora v1: scatter-grid adaptive patching, zero new parameters."""

    def __init__(
        self,
        *args,
        patching: ScatterGridStrategy,
        saliency_fn: str = "grad-z500",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.patching = patching
        self._saliency_fn = SALIENCY_FNS[saliency_fn]

    def forward(self, batch: Batch) -> Batch:
        # TODO: hook into _pre_encoder_hook to inject leaf-strip serialization
        # and scatter back to regular grid before backbone call.
        raise NotImplementedError("AuroraScatterGrid.forward not yet implemented")
