"""StormerSmall — AuroraSmall with the Swin3D backbone swapped for StormerProcessor.

Everything outside the backbone is identical to models/small.py's AuroraSmall:
same Perceiver3DEncoder (uniform patch_size=4 embed — no adaptive patching),
same Perceiver3DDecoder, same variables/levels, random init.
Any train/val metric difference vs AuroraSmall is therefore attributable to the
processor architecture (Swin3D U-Net vs isotropic δt-conditioned DiT).

Targets aurora 1.8.0 (module aurora/microsoft-1.8.0), whose Aurora.forward calls
    self.backbone(x, lead_time=<timedelta>, patch_res=..., rollout_step=...)
"""
from __future__ import annotations

from datetime import timedelta

import torch.nn as nn
from torch import Tensor
from aurora import Aurora

from .processor import StormerProcessor


class StormerBackbone(nn.Module):
    """StormerProcessor + linear bridge to the decoder width.

    # DEVIATION: Aurora's Perceiver3DDecoder is constructed with embed_dim * 2
    # because the Swin backbone's final skip-concat doubles the width. The
    # Stormer emits embed_dim, so a single linear projects up to 2*embed_dim.
    # (Mirror image of the projection the old 02 codebase used for its swin
    # ablation.) Alternative — rebuilding the decoder at embed_dim — would
    # require surgery on Aurora.__init__ internals; not worth it.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        depth: int = 8,
        num_heads: int = 8,
        head_dim: int = 64,
        mlp_ratio: float = 4.0,
        width: int | None = None,
    ) -> None:
        """width: internal processor width. None → embed_dim (control). Set wider
        (e.g. 768, with depth 10 / heads 12) to parameter-match AuroraSmall's
        Swin backbone (~109 M), whose channel count doubles per U-Net stage
        while the isotropic stack stays flat."""
        super().__init__()
        width = width or embed_dim
        self.in_proj = nn.Identity() if width == embed_dim else nn.Linear(embed_dim, width)
        self.processor = StormerProcessor(
            embed_dim=width,
            depth=depth,
            num_heads=num_heads,
            head_dim=head_dim,
            mlp_ratio=mlp_ratio,
        )
        self.out_proj = nn.Linear(width, 2 * embed_dim)

    def forward(
        self,
        x: Tensor,
        lead_time: timedelta,
        rollout_step: int = 0,
        patch_res: tuple[int, int, int] | None = None,
    ) -> Tensor:
        """(B, L, embed_dim) → (B, L, 2*embed_dim), matching Swin's output width."""
        in_dtype = x.dtype
        x = self.in_proj(x)
        x = self.processor(
            x, lead_time=lead_time, rollout_step=rollout_step, patch_res=patch_res
        )
        return self.out_proj(x).to(dtype=in_dtype)


def build_aurora_stormer(
    depth: int = 8,
    num_heads: int = 8,
    head_dim: int = 64,
    mlp_ratio: float = 4.0,
    width: int | None = None,
) -> Aurora:
    """Construct AuroraStormerSmall: AuroraSmall's encoder/decoder + Stormer processor.

    Aurora(...) args below are byte-identical to models/small.py — keep them in
    sync. encoder_depths / decoder_depths / window_size parameterise the Swin
    backbone that is built inside Aurora.__init__ and then discarded by the
    swap; they are retained only so the encoder/decoder construction matches
    AuroraSmall exactly.

    depth=8 is the CLAUDE.md control setting; tune it (see train.py's param
    count log) to match AuroraSmall's total parameters before the headline
    comparison run.
    """
    model = Aurora(
        surf_vars=("2t", "10u", "10v", "msl"),
        static_vars=("lsm", "z", "slt"),
        atmos_vars=("z", "u", "v", "t", "q"),
        encoder_depths=(2, 6, 2),
        encoder_num_heads=(4, 8, 16),
        decoder_depths=(2, 6, 2),
        decoder_num_heads=(16, 8, 4),
        embed_dim=256,
        num_heads=8,
        latent_levels=4,
        patch_size=4,
        window_size=(2, 6, 12),
        use_lora=False,
        autocast=True,
    )
    # Swap the backbone: Swin3D U-Net → isotropic Stormer DiT. The Swin weights
    # allocated in Aurora.__init__ are dropped here (never trained, random init
    # only), so the throwaway costs a moment of init time and nothing else.
    model.backbone = StormerBackbone(
        embed_dim=256,
        depth=depth,
        num_heads=num_heads,
        head_dim=head_dim,
        mlp_ratio=mlp_ratio,
        width=width,
    )
    return model
