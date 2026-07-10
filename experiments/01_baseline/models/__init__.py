from aurora import Aurora

from .small import build_aurora_small
from .tiny import build_aurora_tiny


def build_model(cfg: dict) -> Aurora:
    """Instantiate the model named in cfg['model']['name']."""
    name = cfg["model"]["name"]
    if name == "AuroraTiny":
        return build_aurora_tiny()
    if name == "AuroraSmall":
        return build_aurora_small()
    raise ValueError(f"Unknown model name: {name!r}")
