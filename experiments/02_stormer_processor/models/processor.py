"""StormerProcessor — isotropic adaLN-zero DiT stack.

Ported from the stormer codebase (https://github.com/tung-nd/stormer.git) and
stripped to the control configuration only:
  - dt_cond="adaln_zero" hardcoded (A4 "additive"/"none" variants removed)
  - qk_norm always on (A4 off-variant removed)
  - no rollout-step conditioning (A4/A6 removed)
  - HybridProcessor / swin variants (A2) removed

Interface matches aurora 1.8.0's Swin3DTransformerBackbone, which Aurora.forward
calls as:

    self.backbone(x, lead_time=self.timestep, patch_res=..., rollout_step=...)

i.e. ``lead_time`` is a datetime.timedelta (NOT a tensor — that is aurora>=2.0).
"""
from __future__ import annotations

from datetime import timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from aurora.model.fourier import lead_time_expansion

__all__ = ["StormerProcessor"]


class AdaLNZero(nn.Module):
    """6-factor adaLN-zero modulation from a conditioning vector.

    The final linear is zero-initialized so all modulation factors (scale, shift,
    gate) start at zero, making each block an identity function at init.
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
    """Global self-attention over the flat token sequence.

    Seam for a future O(N) adaptive-patch attention kernel: replace only the
    attention computation inside ``forward``; the surrounding block structure
    (norm, adaLN, residual) stays unchanged.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim
        self.to_q = nn.Linear(embed_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(embed_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(embed_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, embed_dim)
        # QK-norm for bf16 numerical stability (CLAUDE.md §1.2) — always on.
        self.q_norm = nn.LayerNorm(head_dim)
        self.k_norm = nn.LayerNorm(head_dim)

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

    At init: all adaLN outputs are zero → gate=0 → block is identity.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        mlp_ratio: float,
        cond_dim: int,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.attn = GlobalSelfAttention(embed_dim, num_heads, head_dim)
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


class StormerProcessor(nn.Module):
    """Isotropic DiT processor conditioned on forecast interval δt (adaLN-zero).

    Replaces Aurora's Swin3DTransformerBackbone. Global attention over the flat
    latent sequence (B, latent_levels × n_patches, D); ``patch_res`` is accepted
    for interface compatibility but ignored (attention is layout-agnostic).

    Under fixed-δt training (01-style, 6 h pairs) the conditioning input is a
    constant; the adaLN pathway is kept anyway so the same weights extend to
    randomized-δt training later without an architecture change.

    NOTE: output width is ``embed_dim`` — Aurora's decoder expects 2*embed_dim
    (Swin skip-concat width). models/stormer.py adds the bridging projection.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        depth: int = 8,
        num_heads: int = 8,
        head_dim: int = 64,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        cond_dim = embed_dim

        # Two-layer MLP projecting the Fourier δt encoding to the conditioning
        # vector — same convention as Swin3DTransformerBackbone.time_mlp.
        self.dt_embed = nn.Sequential(
            nn.Linear(embed_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        self.blocks = nn.ModuleList([
            StormerBlock(embed_dim, num_heads, head_dim, mlp_ratio, cond_dim)
            for _ in range(depth)
        ])

        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: Tensor,
        lead_time: timedelta,
        rollout_step: int = 0,
        patch_res: tuple[int, int, int] | None = None,
    ) -> Tensor:
        """
        Args:
            x:            (B, L, D) flat latent sequence from the Perceiver encoder.
            lead_time:    forecast interval δt as a timedelta (aurora 1.8.0 passes
                          ``lead_time=self.timestep``).
            rollout_step: accepted for interface compatibility; ignored.
            patch_res:    accepted for interface compatibility; ignored.

        Returns:
            (B, L, D) processed latent sequence.
        """
        B = x.size(0)
        dtype = x.dtype

        lead_hours = lead_time.total_seconds() / 3600
        lead_times = lead_hours * torch.ones(B, dtype=dtype, device=x.device)
        cond = lead_time_expansion(lead_times, self.embed_dim).to(dtype=dtype)
        cond = self.dt_embed(cond)  # (B, cond_dim)

        for block in self.blocks:
            x = block(x, cond)

        return self.final_norm(x)
