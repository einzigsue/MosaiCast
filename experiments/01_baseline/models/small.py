"""AuroraSmall — T2 model config (HPC A100, bf16 autocast).

AuroraSmallPretrained-scale architecture (embed_dim 256, depths (2,6,2))
but constructed directly from the Aurora base class with random init —
no checkpoint is ever loaded (CLAUDE.md rule 9).
"""
from aurora import Aurora


def build_aurora_small() -> Aurora:
    """Construct AuroraSmall as specified in CLAUDE.md.

    Mirrors aurora.AuroraSmallPretrained's architecture hyperparameters.
    latent_levels=4 (1 surface + 3 atmospheric latents) with the default
    window_size=(2, 6, 12): 4 % 2 == 0 satisfies the swin3d assert.
    Token count: 4 × (128//4) × (256//4) = 8192.
    autocast=True per the T2 device rules (bf16 on A100, CLAUDE.md rule 8).
    """
    return Aurora(
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
