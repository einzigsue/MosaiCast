"""v2 mixed-res — native-resolution per-size embedders.

One LevelPatchEmbed per leaf size in the tier's leaf_sizes set (kernel = stride
= s), each seeing native pixels — no pixel-space information loss. The resulting
token is deaggregated (replicated by default) to all (s//p)² fine-grid cells.

Parameter count: (len(leaf_sizes) - 1) extra LevelPatchEmbed sets beyond the
baseline size-p embedder. The size-p embedder shares weights with v0's embedder
(equivalence test constraint, CLAUDE.md).
"""
from __future__ import annotations
import torch
from aurora import Aurora, Batch

from ..patching.base import Leaf, PatchingStrategy
from ..patching.quadtree import FixedQuadTree
from ..patching.saliency import SALIENCY_FNS


class MixedResStrategy(PatchingStrategy):
    """Embed each leaf at its native size via a per-size LevelPatchEmbed."""

    def __init__(self, patch_size: int, leaf_budget: int, leaf_sizes: set[int]) -> None:
        self.patch_size = patch_size
        self.leaf_sizes = sorted(leaf_sizes)
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
        # Returns native-size crops grouped by leaf size for per-size embedding.
        raise NotImplementedError

    def scatter(
        self,
        leaf_tokens: torch.Tensor,
        leaves: list[Leaf],
        patch_size: int,
        grid_h: int,
        grid_w: int,
    ) -> torch.Tensor:
        # Replicate each leaf token to its (s//p)² fine-grid cells.
        raise NotImplementedError


class AuroraMixedRes(Aurora):
    """Aurora v2: mixed-resolution adaptive patching with per-size embedders."""

    def __init__(
        self,
        *args,
        patching: MixedResStrategy,
        saliency_fn: str = "grad-z500",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.patching = patching
        self._saliency_fn = SALIENCY_FNS[saliency_fn]
        # Per-size LevelPatchEmbed instances are registered in _build_size_embedders
        # after the parent __init__ has constructed the baseline embedder.
        self._build_size_embedders()

    def _build_size_embedders(self) -> None:
        """Register one LevelPatchEmbed per non-baseline leaf size."""
        # TODO: copy the baseline embedder for the base patch_size,
        # add new embedders for each larger size in patching.leaf_sizes.
        raise NotImplementedError

    def forward(self, batch: Batch) -> Batch:
        raise NotImplementedError("AuroraMixedRes.forward not yet implemented")
