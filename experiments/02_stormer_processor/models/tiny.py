"""AuroraTiny — T1 model config (MBP M4, fp32, ~20–25 M params)."""
from aurora import Aurora


def build_aurora_tiny() -> Aurora:
    """Construct AuroraTiny as specified in CLAUDE.md.

    latent_levels=3 so latent_levels % window_size[0] == 0 (swin3d assert).
    Token count: 3 × (32//4) × (64//4) = 384.
    """
    return Aurora(
        surf_vars=("2t", "10u", "10v", "msl"),
        static_vars=("lsm", "z", "slt"),
        atmos_vars=("z", "u", "v", "t", "q"),
        encoder_depths=(2, 2, 2),
        encoder_num_heads=(4, 8, 16),
        decoder_depths=(2, 2, 2),
        decoder_num_heads=(16, 8, 4),
        embed_dim=128,
        num_heads=4,
        latent_levels=3,
        patch_size=4,
        window_size=(1, 4, 8),
        use_lora=False,
        autocast=False,
    )
