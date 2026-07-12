"""StormerProcessor — isotropic adaLN-zero DiT stack (CLAUDE.md §1.1–1.2).

A2 ablation: HybridProcessor adds 1–2 U-Net merge/split stages along the patch axis.
A4 ablation: StormerProcessor gains dt_cond in {"adaln_zero","additive","none"} and
             optional rollout-step conditioning (condition_on_rollout_step).
"""
from __future__ import annotations

from datetime import timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from aurora.model.fourier import lead_time_expansion

__all__ = ["StormerProcessor", "HybridProcessor"]

_MAX_ROLLOUT_STEPS = 40  # embedding table size for rollout-step conditioning (A4)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class AdaLNZero(nn.Module):
    """6-factor adaLN-zero modulation from a conditioning vector.

    The final linear is zero-initialized so all modulation factors (scale, shift, gate)
    start at zero at init, making each block an identity function at the start of training.
    """

    def __init__(self, cond_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.act = nn.SiLU()
        self.linear = nn.Linear(cond_dim, 6 * embed_dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, c: Tensor) -> tuple[Tensor, ...]:
        """Return 6 modulation tensors of shape (B, embed_dim) each.

        Ordering: shift_attn, scale_attn, gate_attn, shift_ff, scale_ff, gate_ff.
        """
        return self.linear(self.act(c)).chunk(6, dim=-1)


class GlobalSelfAttention(nn.Module):
    """Global self-attention — seam for O(N) adaptive-patch attention (CLAUDE.md §8.9).

    Replace only the attention computation inside ``forward`` when swapping in a
    linear-complexity kernel; the surrounding block structure (norm, adaLN, residual)
    stays unchanged.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        qk_norm: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim
        self.to_q = nn.Linear(embed_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(embed_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(embed_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, embed_dim)
        # QK-norm for bf16 numerical stability (CLAUDE.md §1.2)
        if qk_norm:
            self.q_norm = nn.LayerNorm(head_dim)
            self.k_norm = nn.LayerNorm(head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """(B, L, D) → (B, L, D) global self-attention."""
        B, L, _ = x.shape
        h = self.num_heads
        d = self.head_dim

        q = self.to_q(x).reshape(B, L, h, d).transpose(1, 2)  # (B, h, L, d)
        k = self.to_k(x).reshape(B, L, h, d).transpose(1, 2)
        v = self.to_v(x).reshape(B, L, h, d).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        out = F.scaled_dot_product_attention(q, k, v)  # (B, h, L, d)
        out = out.transpose(1, 2).reshape(B, L, h * d)
        return self.to_out(out)


class StormerBlock(nn.Module):
    """One adaLN-zero DiT block: pre-norm attention + pre-norm FFN, both gated by δt.

    At init: all adaLN outputs are zero → gate=0 → block is identity. ✓
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        mlp_ratio: float,
        cond_dim: int,
        qk_norm: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.attn = GlobalSelfAttention(embed_dim, num_heads, head_dim, qk_norm=qk_norm)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, embed_dim),
        )
        self.adaLN = AdaLNZero(cond_dim, embed_dim)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        """
        Args:
            x: (B, L, D) token sequence.
            c: (B, cond_dim) conditioning vector.

        Returns:
            (B, L, D)
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(c)
        x_norm = (1 + scale_msa[:, None]) * self.norm1(x) + shift_msa[:, None]
        x = x + gate_msa[:, None] * self.attn(x_norm)
        x_norm = (1 + scale_mlp[:, None]) * self.norm2(x) + shift_mlp[:, None]
        x = x + gate_mlp[:, None] * self.ff(x_norm)
        return x


class StormerBlockNoCond(nn.Module):
    """Plain pre-norm attention + FFN block without conditioning.

    Used by dt_cond="additive" (δt injected once at entry) and dt_cond="none" (no δt).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        mlp_ratio: float,
        qk_norm: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = GlobalSelfAttention(embed_dim, num_heads, head_dim, qk_norm=qk_norm)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, embed_dim),
        )

    def forward(self, x: Tensor, c: Tensor | None = None) -> Tensor:  # c ignored
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Isotropic processor (control)
# ---------------------------------------------------------------------------

class StormerProcessor(nn.Module):
    """Isotropic DiT processor conditioned on forecast interval δt (CLAUDE.md §1.1–1.4).

    Replaces Aurora's Swin3DTransformerBackbone.  Same call signature:
        processor(x, lead_time=, patch_res=, rollout_step=)

    The processor is isotropic — global attention over the flat token sequence
    (B, latent_levels × n_patches, D).  patch_res is accepted for interface
    compatibility but is not needed (global attention is layout-agnostic).

    dt_cond controls how the forecast interval is injected:
        "adaln_zero" (control): adaLN-zero gates applied per block.
        "additive":             δt added to x once at entry; blocks have no conditioning.
        "none":                 no δt conditioning anywhere in the processor.

    condition_on_rollout_step: if True, add a learned rollout-step embedding to
        the conditioning vector (A4 ablation). Disabled by default (§1.4).
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        depth: int = 8,
        num_heads: int = 16,
        head_dim: int = 64,
        mlp_ratio: float = 4.0,
        qk_norm: bool = True,
        dt_cond: str = "adaln_zero",
        condition_on_rollout_step: bool = False,
    ) -> None:
        super().__init__()
        if dt_cond not in ("adaln_zero", "additive", "none"):
            raise ValueError(f"dt_cond must be 'adaln_zero', 'additive', or 'none', got {dt_cond!r}")
        self.embed_dim = embed_dim
        self.dt_cond = dt_cond
        self.condition_on_rollout_step = condition_on_rollout_step
        cond_dim = embed_dim

        if dt_cond in ("adaln_zero", "additive"):
            # Two-layer MLP to project Fourier δt encoding to conditioning vector
            self.dt_embed = nn.Sequential(
                nn.Linear(embed_dim, cond_dim),
                nn.SiLU(),
                nn.Linear(cond_dim, cond_dim),
            )

        if condition_on_rollout_step:
            self.rollout_embed = nn.Embedding(_MAX_ROLLOUT_STEPS, cond_dim)

        if dt_cond == "adaln_zero":
            self.blocks = nn.ModuleList([
                StormerBlock(embed_dim, num_heads, head_dim, mlp_ratio, cond_dim, qk_norm)
                for _ in range(depth)
            ])
        else:
            self.blocks = nn.ModuleList([
                StormerBlockNoCond(embed_dim, num_heads, head_dim, mlp_ratio, qk_norm)
                for _ in range(depth)
            ])

        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: Tensor,
        lead_time: timedelta,
        patch_res: tuple[int, int, int],
        rollout_step: int = 0,
    ) -> Tensor:
        """
        Args:
            x:           (B, L', D) flat latent sequence from encoder.
            lead_time:   forecast interval δt (conditions modulation).
            patch_res:   accepted for interface compatibility; ignored (attention is layout-agnostic).
            rollout_step: rollout step index (used when condition_on_rollout_step=True).

        Returns:
            (B, L', D) processed latent sequence.
        """
        B = x.size(0)
        dtype = x.dtype

        cond = None
        if self.dt_cond in ("adaln_zero", "additive"):
            lead_hours = lead_time.total_seconds() / 3600
            lead_times = lead_hours * torch.ones(B, dtype=dtype, device=x.device)
            cond = lead_time_expansion(lead_times, self.embed_dim).to(dtype=dtype)
            cond = self.dt_embed(cond)  # (B, cond_dim)

            if self.condition_on_rollout_step:
                step_t = torch.tensor(
                    rollout_step % _MAX_ROLLOUT_STEPS, device=x.device
                ).expand(B)
                cond = cond + self.rollout_embed(step_t)  # (B, cond_dim)

        elif self.condition_on_rollout_step:
            # dt_cond=="none" but rollout conditioning still active
            step_t = torch.tensor(
                rollout_step % _MAX_ROLLOUT_STEPS, device=x.device
            ).expand(B)
            cond = self.rollout_embed(step_t).to(dtype=dtype)

        if self.dt_cond == "additive" and cond is not None:
            # Inject δt once at entry; blocks receive no per-block conditioning
            x = x + cond[:, None, :]

        for block in self.blocks:
            x = block(x, cond)

        return self.final_norm(x)


# ---------------------------------------------------------------------------
# Hybrid processor (A2 ablation)
# ---------------------------------------------------------------------------

class HybridProcessor(nn.Module):
    """U-Net StormerProcessor with patch-axis merge/split stages (A2 ablation).

    Adds n_merge_stages of token-coarsening (merge adjacent patch pairs) and
    symmetric fine-level processing (split + skip), around a coarse-level block stack.

    IMPORTANT: token adjacency is only physically meaningful when patches are
    raster-ordered (uniform plan). Use plan_type=uniform with this processor.
    # DEVIATION: hybrid U-Net over the patch axis; CLAUDE.md §1.1 specifies isotropic only.

    Shapes (example, n_merge_stages=1):
        Entry: (B, L*n, D)   where L=latent_levels
        Merge: (B, L*n//2, D)
        Coarse blocks
        Split+skip: (B, L*n, D)
        Fine blocks
        Exit: (B, L*n, D)
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        depth: int = 8,
        num_heads: int = 16,
        head_dim: int = 64,
        mlp_ratio: float = 4.0,
        qk_norm: bool = True,
        dt_cond: str = "adaln_zero",
        condition_on_rollout_step: bool = False,
        n_merge_stages: int = 1,
        latent_levels: int = 4,
    ) -> None:
        super().__init__()
        if dt_cond not in ("adaln_zero", "additive", "none"):
            raise ValueError(f"dt_cond must be 'adaln_zero', 'additive', or 'none'")
        self.embed_dim = embed_dim
        self.dt_cond = dt_cond
        self.condition_on_rollout_step = condition_on_rollout_step
        self.n_merge_stages = n_merge_stages
        self.latent_levels = latent_levels
        cond_dim = embed_dim

        if dt_cond in ("adaln_zero", "additive"):
            self.dt_embed = nn.Sequential(
                nn.Linear(embed_dim, cond_dim),
                nn.SiLU(),
                nn.Linear(cond_dim, cond_dim),
            )
        if condition_on_rollout_step:
            self.rollout_embed = nn.Embedding(_MAX_ROLLOUT_STEPS, cond_dim)

        # Merge/split projections: one per stage
        self.merge_projs = nn.ModuleList([
            nn.Linear(2 * embed_dim, embed_dim) for _ in range(n_merge_stages)
        ])
        self.split_projs = nn.ModuleList([
            nn.Linear(embed_dim, 2 * embed_dim) for _ in range(n_merge_stages)
        ])

        def _make_block():
            if dt_cond == "adaln_zero":
                return StormerBlock(embed_dim, num_heads, head_dim, mlp_ratio, cond_dim, qk_norm)
            return StormerBlockNoCond(embed_dim, num_heads, head_dim, mlp_ratio, qk_norm)

        # Split depth into: fine_depth (before merge) + coarse_depth + fine_depth (after split)
        fine_depth  = depth // 4
        coarse_depth = depth - 2 * fine_depth

        self.fine_pre   = nn.ModuleList([_make_block() for _ in range(fine_depth)])
        self.coarse_mid = nn.ModuleList([_make_block() for _ in range(coarse_depth)])
        self.fine_post  = nn.ModuleList([_make_block() for _ in range(fine_depth)])
        self.final_norm = nn.LayerNorm(embed_dim)

    def _cond(self, x: Tensor, lead_time: timedelta, rollout_step: int) -> Tensor | None:
        B, _, dtype = x.size(0), x.size(1), x.dtype
        cond = None
        if self.dt_cond in ("adaln_zero", "additive"):
            lh = lead_time.total_seconds() / 3600
            lt = lh * torch.ones(B, dtype=dtype, device=x.device)
            cond = self.dt_embed(lead_time_expansion(lt, self.embed_dim).to(dtype))
        if self.condition_on_rollout_step:
            st = torch.tensor(rollout_step % _MAX_ROLLOUT_STEPS, device=x.device).expand(B)
            step_emb = self.rollout_embed(st).to(dtype)
            cond = (cond + step_emb) if cond is not None else step_emb
        return cond

    def _run_blocks(self, x: Tensor, blocks: nn.ModuleList, cond: Tensor | None) -> Tensor:
        if self.dt_cond == "additive" and cond is not None:
            x = x + cond[:, None, :]
            for blk in blocks:
                x = blk(x, None)
        else:
            for blk in blocks:
                x = blk(x, cond)
        return x

    def forward(
        self,
        x: Tensor,
        lead_time: timedelta,
        patch_res: tuple[int, int, int],
        rollout_step: int = 0,
    ) -> Tensor:
        """(B, L*n, D) → (B, L*n, D) with merge/split around coarse blocks."""
        B = x.size(0)
        L = self.latent_levels
        n = x.size(1) // L  # number of patches

        cond = self._cond(x, lead_time, rollout_step)

        # Fine pre-merge blocks
        x = self._run_blocks(x, self.fine_pre, cond)

        # Merge stages (coarsen along patch dim)
        skips: list[Tensor] = []
        cur_n = n
        for stage in range(self.n_merge_stages):
            if cur_n % 2 != 0:
                raise ValueError(
                    f"HybridProcessor merge stage {stage}: n_patches={cur_n} is not even. "
                    "Use a uniform plan with a patch count divisible by 2^n_merge_stages."
                )
            skips.append(x)
            # (B, L*cur_n, D) → (B, L, cur_n, D) → (B, L, cur_n//2, 2D) → (B, L, cur_n//2, D)
            x = x.reshape(B, L, cur_n, self.embed_dim)
            x = x.reshape(B, L, cur_n // 2, 2 * self.embed_dim)
            x = self.merge_projs[stage](x)            # (B, L, cur_n//2, D)
            cur_n = cur_n // 2
            x = x.reshape(B, L * cur_n, self.embed_dim)

        # Coarse blocks
        x = self._run_blocks(x, self.coarse_mid, cond)

        # Split stages (refine, reverse order of merge)
        for stage in reversed(range(self.n_merge_stages)):
            skip = skips[stage]
            x = x.reshape(B, L, cur_n, self.embed_dim)
            x = self.split_projs[stage](x)             # (B, L, cur_n, 2D)
            cur_n = cur_n * 2
            x = x.reshape(B, L, cur_n, self.embed_dim)
            x = x.reshape(B, L * cur_n, self.embed_dim)
            x = x + skip  # additive skip connection

        # Fine post-split blocks
        x = self._run_blocks(x, self.fine_post, cond)

        return self.final_norm(x)
