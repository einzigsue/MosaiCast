"""Patch, PatchPlan, PatchIndex — geometry spec (CLAUDE.md §2.2).

Public helpers extract_patches / reconstruct_field implement the geometric
crop→resize→resize_back→scatter roundtrip used by AdaptivePatchEmbed /
AdaptivePatchReconstruct and by the M1 roundtrip test.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Module-level memoisation cache (keyed by (PatchPlan, lat_key, lon_key))
# ---------------------------------------------------------------------------

_index_cache: dict = {}


def _tensor_key(t: Tensor) -> tuple:
    """Hashable key for a small coordinate tensor."""
    return (tuple(t.shape), tuple(t.flatten().tolist()))


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _patch_area_km2(lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> float:
    """Spherical area of a lat/lon rectangle in km² (Aurora-consistent formula).

    Matches aurora.model.posencoding.patch_root_area which uses R² π |Δsin(φ)| |Δλ_rad|.
    The π factor keeps area encoding consistent with Aurora's scale_expansion range.
    """
    R = 6371.0
    lo = math.radians(min(lat_min, lat_max))
    hi = math.radians(max(lat_min, lat_max))
    dlon = math.radians(abs(lon_max - lon_min))
    return R * R * math.pi * abs(math.sin(hi) - math.sin(lo)) * dlon


def _boxes_overlap(a: "Patch", b: "Patch") -> bool:
    """True if two non-antimeridian boxes have a non-empty interior intersection."""
    lat_ov = a.lat_min < b.lat_max and b.lat_min < a.lat_max
    lon_ov = a.lon_min < b.lon_max and b.lon_min < a.lon_max
    return lat_ov and lon_ov


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Patch:
    """One lat/lon bounding box with a physical area estimate.

    area_size (km²) drives the Fourier area encoding — must vary per patch.
    lon_min < lon_max for normal patches; lon_min > lon_max signals antimeridian wrap.
    """
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    area_size: float  # km²


@dataclass(frozen=True)
class PatchPlan:
    """Ordered set of Patch boxes covering the globe.

    Geometry is validated once at construction (grid-agnostic).
    Coverage validation happens in index(), memoised per (plan, grid).
    """
    patches: tuple[Patch, ...]
    canonical: int = 4  # p — each patch resampled to (p × p) before embedding

    def __post_init__(self) -> None:
        self.validate_geometry()

    def __len__(self) -> int:
        return len(self.patches)

    def centroids(self) -> tuple[Tensor, Tensor]:
        """(n,) lat centroids and (n,) lon centroids."""
        lats = torch.tensor(
            [(p.lat_min + p.lat_max) / 2 for p in self.patches], dtype=torch.float32
        )
        lons = torch.tensor(
            [(p.lon_min + p.lon_max) / 2 for p in self.patches], dtype=torch.float32
        )
        return lats, lons

    def areas(self) -> Tensor:
        """(n,) area_size values in km²."""
        return torch.tensor([p.area_size for p in self.patches], dtype=torch.float32)

    def validate_geometry(self) -> None:
        """Raise ValueError for ill-formed or overlapping patches."""
        if self.canonical <= 0:
            raise ValueError(f"canonical must be > 0, got {self.canonical}")

        for i, p in enumerate(self.patches):
            if p.lat_min >= p.lat_max:
                raise ValueError(
                    f"Patch {i}: lat_min={p.lat_min} >= lat_max={p.lat_max}"
                )
            if p.area_size <= 0:
                raise ValueError(f"Patch {i}: area_size={p.area_size} <= 0")
            # lon_min > lon_max is allowed (antimeridian wrap) — no check here

        # Pairwise disjoint check (O(n²), grid-agnostic)
        for i in range(len(self.patches)):
            for j in range(i + 1, len(self.patches)):
                if _boxes_overlap(self.patches[i], self.patches[j]):
                    raise ValueError(
                        f"Patches {i} and {j} have overlapping interiors"
                    )

    def index(self, lat: Tensor, lon: Tensor) -> "PatchIndex":
        """Validate coverage, materialise row/col indices, memoize per (plan, grid)."""
        key = (self, _tensor_key(lat), _tensor_key(lon))
        if key not in _index_cache:
            _index_cache[key] = _build_index(self, lat, lon)
        return _index_cache[key]


@dataclass
class PatchIndex:
    """Materialised, coverage-validated view of a PatchPlan on a specific grid.

    Always produced by PatchPlan.index(lat, lon) — never constructed directly.
    The model forward consumes a PatchIndex, never a raw PatchPlan.
    """
    plan: PatchPlan
    rows: list[slice]        # contiguous lat-cell range per patch, shape (n,)
    cols: list[Tensor]       # lon-cell indices per patch; Tensor allows antimeridian wrap
    centroid_lat: Tensor     # (n,)
    centroid_lon: Tensor     # (n,)
    area: Tensor             # (n,) km²
    grid_shape: tuple[int, int]


# ---------------------------------------------------------------------------
# Index construction (called once per (plan, grid); result is cached)
# ---------------------------------------------------------------------------

def _build_index(plan: PatchPlan, lat: Tensor, lon: Tensor) -> PatchIndex:
    H, W = len(lat), len(lon)
    rows: list[slice] = []
    cols: list[Tensor] = []

    # Coverage counter — incremented via index_put_ with accumulate=True
    cell_count = torch.zeros(H, W, dtype=torch.int32)

    for k, patch in enumerate(plan.patches):
        # --- lat indices (closed interval) ---
        lat_mask = (lat >= patch.lat_min) & (lat <= patch.lat_max)
        lat_idx = torch.where(lat_mask)[0]
        if len(lat_idx) == 0:
            raise ValueError(
                f"Patch {k}: no lat cells in [{patch.lat_min}, {patch.lat_max}]. "
                f"Lat range of grid: [{lat.min().item():.3f}, {lat.max().item():.3f}]"
            )
        r_start = int(lat_idx[0].item())
        r_end = int(lat_idx[-1].item()) + 1
        expected = torch.arange(r_start, r_end, device=lat.device)
        if not torch.equal(lat_idx, expected):
            raise ValueError(
                f"Patch {k}: lat cells are not contiguous "
                f"(got indices {lat_idx.tolist()}, expected {expected.tolist()})"
            )
        rows.append(slice(r_start, r_end))

        # --- lon indices (antimeridian-aware) ---
        if patch.lon_min <= patch.lon_max:
            lon_mask = (lon >= patch.lon_min) & (lon <= patch.lon_max)
        else:
            # antimeridian wrap: e.g. lon_min=350, lon_max=10
            lon_mask = (lon >= patch.lon_min) | (lon <= patch.lon_max)
        col_idx = torch.where(lon_mask)[0]
        if len(col_idx) == 0:
            raise ValueError(f"Patch {k}: no lon cells in patch bbox")
        cols.append(col_idx)

        # --- update coverage counter ---
        H_k = r_end - r_start
        W_k = len(col_idx)
        ri = torch.arange(r_start, r_end, dtype=torch.long, device=lat.device)
        row_flat = ri.unsqueeze(1).expand(H_k, W_k).reshape(-1)
        col_flat = col_idx.unsqueeze(0).expand(H_k, W_k).reshape(-1)
        cell_count.index_put_(
            (row_flat, col_flat),
            torch.ones(H_k * W_k, dtype=torch.int32),
            accumulate=True,
        )

    # --- validate full coverage ---
    n_missed = int((cell_count == 0).sum().item())
    n_double = int((cell_count > 1).sum().item())
    if n_missed or n_double:
        raise ValueError(
            f"Coverage invalid: {n_missed} cells uncovered, "
            f"{n_double} cells covered more than once"
        )

    clat, clon = plan.centroids()
    return PatchIndex(
        plan=plan,
        rows=rows,
        cols=cols,
        centroid_lat=clat,
        centroid_lon=clon,
        area=plan.areas(),
        grid_shape=(H, W),
    )


# ---------------------------------------------------------------------------
# Geometric crop / scatter helpers (used by embed, reconstruct, and M1 tests)
# ---------------------------------------------------------------------------

def extract_patches(field: Tensor, index: PatchIndex) -> list[Tensor]:
    """Crop each patch from field and resize to (canonical × canonical).

    Args:
        field: (H, W) float tensor.
        index: materialised PatchIndex for this grid.

    Returns:
        List of n (canonical, canonical) tensors.
    """
    p = index.plan.canonical
    patches: list[Tensor] = []
    for r, c in zip(index.rows, index.cols):
        crop = field[r.start:r.stop, :][:, c]          # (H_k, W_k)
        resized = F.interpolate(
            crop.unsqueeze(0).unsqueeze(0).float(),
            size=(p, p),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)                          # (p, p)
        patches.append(resized)
    return patches


def reconstruct_field(patches: list[Tensor], index: PatchIndex) -> Tensor:
    """Resize each (canonical × canonical) patch back to its cell count and scatter.

    Args:
        patches: list of n (canonical, canonical) tensors.
        index:   materialised PatchIndex for this grid.

    Returns:
        (H, W) float tensor.
    """
    H, W = index.grid_shape
    out = torch.zeros(H, W, dtype=torch.float32)
    for patch, r, c in zip(patches, index.rows, index.cols):
        H_k = r.stop - r.start
        W_k = len(c)
        tile = F.interpolate(
            patch.unsqueeze(0).unsqueeze(0).float(),
            size=(H_k, W_k),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)                          # (H_k, W_k)
        ri = torch.arange(r.start, r.stop, dtype=torch.long)
        row_flat = ri.unsqueeze(1).expand(H_k, W_k).reshape(-1)
        col_flat = c.unsqueeze(0).expand(H_k, W_k).reshape(-1)
        out.index_put_((row_flat, col_flat), tile.reshape(-1), accumulate=False)
    return out
