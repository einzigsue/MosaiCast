"""v0 baseline — fixed patching, unmodified Aurora at the tier's patch_size."""
from aurora import Aurora


class AuroraBaseline(Aurora):
    """Drop-in Aurora with no adaptive patching.

    Exists so v1/v2 comparisons have a named class to instantiate rather than
    bare Aurora, and so all variants share the same import pattern.
    """
    pass
