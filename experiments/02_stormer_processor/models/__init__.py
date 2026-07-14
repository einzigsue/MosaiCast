from aurora import Aurora

from .small import build_aurora_small
from .tiny import build_aurora_tiny
from .stormer import build_aurora_stormer
#from .v0_baseline import AuroraBaseline
#from .v1_scatter_grid import AuroraScatterGrid
#from .v2_mixed_res import AuroraMixedRes


def build_model(cfg: dict) -> Aurora:
    """Instantiate the model named in cfg['model']['name']."""
    name = cfg["model"]["name"]
    if name == "AuroraTiny":
        return build_aurora_tiny()
    if name == "AuroraSmall":
        return build_aurora_small()
    if name == "AuroraStormer":
        return build_aurora_stormer(
                depth=cfg["model"].get("processor_depth", 8),
                num_heads=cfg["model"].get("processor_heads", 8),
                head_dim=cfg["model"].get("processor_head_dim", 64),
                mlp_ratio=cfg["model"].get("processor_mlp_ratio", 4.0),
                width=cfg["model"].get("processor_width", None),
                )
    # v1/v2 dispatched here once their forward passes are implemented
    raise ValueError(f"Unknown model name: {name!r}")
