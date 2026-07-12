"""Built-in PatchPlan factories (CLAUDE.md §2.7).

A7 ablation:
  content_adaptive_plan — budget-bounded quadtree refinement where grid roughness is high.
  Criterion: mean |∇field| (Sobel-style finite differences) computed per candidate box.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from .plan import Patch, PatchPlan, _patch_area_km2


def uniform_plan(lat: Tensor, lon: Tensor, p: int) -> PatchPlan:
    """Reproduce Aurora's fixed P×P patching — ablation control (CLAUDE.md §2.7).

    Divides the grid into non-overlapping p×p-cell patches.  Patch bboxes are
    set to midpoints between consecutive grid values so every cell falls in
    exactly one patch, regardless of lat/lon ordering.

    Args:
        lat: (H,) latitude values in degrees (any ordering).
        lon: (W,) longitude values in degrees (any ordering).
        p:   patch size in grid cells; also used as canonical resize target.

    Returns:
        PatchPlan with H//p × W//p patches, canonical=p.
    """
    H, W = len(lat), len(lon)
    if H % p != 0:
        raise ValueError(f"Lat grid size {H} is not divisible by p={p}")
    if W % p != 0:
        raise ValueError(f"Lon grid size {W} is not divisible by p={p}")

    lat_s, _ = torch.sort(lat, descending=True)   # north first
    lon_s, _ = torch.sort(lon, descending=False)  # west first

    lat_step = float(abs(lat_s[0] - lat_s[1]).item()) if H > 1 else 5.625
    lon_step = float(abs(lon_s[1] - lon_s[0]).item()) if W > 1 else 5.625
    half_lat = lat_step / 2
    half_lon = lon_step / 2

    n_lat, n_lon = H // p, W // p
    patches: list[Patch] = []

    for i in range(n_lat):
        band = lat_s[i * p : (i + 1) * p]
        lat_lo = float(band.min().item())
        lat_hi = float(band.max().item())
        lat_min = lat_lo - half_lat
        lat_max = lat_hi + half_lat

        for j in range(n_lon):
            seg = lon_s[j * p : (j + 1) * p]
            lon_lo = float(seg.min().item())
            lon_hi = float(seg.max().item())
            lon_min = lon_lo - half_lon
            lon_max = lon_hi + half_lon

            patches.append(Patch(
                lat_min=lat_min,
                lat_max=lat_max,
                lon_min=lon_min,
                lon_max=lon_max,
                area_size=_patch_area_km2(lat_min, lat_max, lon_min, lon_max),
            ))

    return PatchPlan(patches=tuple(patches), canonical=p)


def latitude_band_plan(lat: Tensor, lon: Tensor, *, p_lat: int = 4, polar_scale: int = 2) -> PatchPlan:
    """Coarser lon resolution near poles, finer at the equator (CLAUDE.md §2.7).

    Each latitude band has p_lat lat-cells.  The number of lon patches per band
    is halved every ``polar_scale`` bands from the equator, giving fewer (larger)
    patches at high latitudes.

    Args:
        lat:          (H,) latitude values in degrees.
        lon:          (W,) longitude values in degrees.
        p_lat:        number of lat cells per band (must divide H).
        polar_scale:  halve lon patches per this many bands from equator outward.

    Returns:
        PatchPlan with variable patch count, canonical=p_lat.
    """
    H, W = len(lat), len(lon)
    if H % p_lat != 0:
        raise ValueError(f"Lat grid size {H} is not divisible by p_lat={p_lat}")

    lat_s, _ = torch.sort(lat, descending=True)
    lon_s, _ = torch.sort(lon, descending=False)

    lat_step = float(abs(lat_s[0] - lat_s[1]).item()) if H > 1 else 5.625
    lon_step = float(abs(lon_s[1] - lon_s[0]).item()) if W > 1 else 5.625

    n_lat = H // p_lat
    equator_band = n_lat // 2

    patches: list[Patch] = []
    for i in range(n_lat):
        band = lat_s[i * p_lat : (i + 1) * p_lat]
        lat_lo = float(band.min().item())
        lat_hi = float(band.max().item())
        lat_min = lat_lo - lat_step / 2
        lat_max = lat_hi + lat_step / 2

        dist = abs(i - equator_band)
        n_lon_patches = max(1, W // (p_lat * (2 ** (dist // polar_scale))))
        while W % n_lon_patches != 0 and n_lon_patches > 1:
            n_lon_patches -= 1
        p_lon = W // n_lon_patches

        for j in range(n_lon_patches):
            seg = lon_s[j * p_lon : (j + 1) * p_lon]
            lon_lo = float(seg.min().item())
            lon_hi = float(seg.max().item())
            lon_min = lon_lo - lon_step / 2
            lon_max = lon_hi + lon_step / 2
            patches.append(Patch(
                lat_min=lat_min,
                lat_max=lat_max,
                lon_min=lon_min,
                lon_max=lon_max,
                area_size=_patch_area_km2(lat_min, lat_max, lon_min, lon_max),
            ))

    return PatchPlan(patches=tuple(patches), canonical=p_lat)


def content_adaptive_plan(
    field:  Tensor,
    lat:    Tensor,
    lon:    Tensor,
    budget: int,
    p:      int = 4,
) -> PatchPlan:
    """Budget-bounded quadtree refinement where gradient magnitude is high (A7 ablation).

    Algorithm:
        1. Start from a coarse uniform partition (p×p cells per patch).
        2. Compute per-box roughness = mean |∇field| (finite differences).
        3. Greedily split the highest-roughness box into 2×2 sub-boxes
           (each sub-box must be at least p×p cells to remain valid).
        4. Repeat until len(patches) >= budget or no box can split.
        5. Guarantee: coverage is exact (partition); areas > 0; no overlap.

    Because this produces a NEW PatchPlan per call (content changes per sample),
    PatchPlan.index() re-validates coverage on each call — explicitly expected
    for content-adaptive plans (CLAUDE.md §2.6).

    Args:
        field:  (H, W) 2-D scalar field for the roughness criterion (e.g. Z500).
        lat:    (H,) latitude in degrees (descending, north-first preferred).
        lon:    (W,) longitude in degrees (ascending, west-first preferred).
        budget: maximum number of patches.
        p:      minimum patch size in cells; patches are never smaller than p×p.
                Must equal PatchPlan.canonical (canonical resize target).

    Returns:
        PatchPlan with ≤ budget patches, canonical=p.
    """
    H, W = len(lat), len(lon)
    if H % p != 0:
        raise ValueError(f"Lat grid size {H} is not divisible by p={p}")
    if W % p != 0:
        raise ValueError(f"Lon grid size {W} is not divisible by p={p}")
    if budget < 1:
        raise ValueError(f"budget must be >= 1, got {budget}")

    lat_s, lat_si = torch.sort(lat, descending=True)
    lon_s, lon_si = torch.sort(lon, descending=False)

    lat_step = float(abs(lat_s[0] - lat_s[1]).item()) if H > 1 else 5.625
    lon_step = float(abs(lon_s[1] - lon_s[0]).item()) if W > 1 else 5.625

    # Re-order field to match sorted lat/lon
    field_sorted = field[lat_si][:, lon_si].float()  # (H, W)

    def _roughness(r0: int, r1: int, c0: int, c1: int) -> float:
        """Mean gradient magnitude of field_sorted[r0:r1, c0:c1]."""
        patch = field_sorted[r0:r1, c0:c1]
        dh = patch[1:, :] - patch[:-1, :]   # (≥0, W_k)
        dw = patch[:, 1:] - patch[:, :-1]   # (H_k, ≥0)
        return float((dh.abs().mean() + dw.abs().mean()) / 2)

    def _make_patch(r0: int, r1: int, c0: int, c1: int) -> Patch:
        band_lats = lat_s[r0:r1]
        band_lons = lon_s[c0:c1]
        lat_lo = float(band_lats.min())
        lat_hi = float(band_lats.max())
        lon_lo = float(band_lons.min())
        lon_hi = float(band_lons.max())
        return Patch(
            lat_min=lat_lo - lat_step / 2,
            lat_max=lat_hi + lat_step / 2,
            lon_min=lon_lo - lon_step / 2,
            lon_max=lon_hi + lon_step / 2,
            area_size=_patch_area_km2(
                lat_lo - lat_step / 2, lat_hi + lat_step / 2,
                lon_lo - lon_step / 2, lon_hi + lon_step / 2,
            ),
        )

    # Initial coarse partition (p×p cells per box)
    # Each entry: (roughness, r0, r1, c0, c1)
    import heapq
    heap: list[tuple[float, int, int, int, int]] = []
    for i in range(H // p):
        for j in range(W // p):
            r0, r1 = i * p, (i + 1) * p
            c0, c1 = j * p, (j + 1) * p
            rough = _roughness(r0, r1, c0, c1)
            heapq.heappush(heap, (-rough, r0, r1, c0, c1))  # max-heap via negation

    # Greedy split until budget reached or no box splittable
    while len(heap) < budget:
        if not heap:
            break
        neg_rough, r0, r1, c0, c1 = heapq.heappop(heap)
        h_k = r1 - r0
        w_k = c1 - c0
        # Can only split if each sub-box is >= p cells in both dims
        can_split_h = (h_k >= 2 * p)
        can_split_w = (w_k >= 2 * p)
        if not (can_split_h or can_split_w):
            # Already at minimum size; put back and stop (it won't split)
            heapq.heappush(heap, (neg_rough, r0, r1, c0, c1))
            break
        # Split on the longer axis when both are splittable; prefer lat
        if can_split_h and (not can_split_w or h_k >= w_k):
            r_mid = r0 + (h_k // (2 * p)) * p   # keep multiples of p
            for sub in [(r0, r_mid, c0, c1), (r_mid, r1, c0, c1)]:
                rough = _roughness(*sub)
                heapq.heappush(heap, (-rough, *sub))
        else:
            c_mid = c0 + (w_k // (2 * p)) * p
            for sub in [(r0, r1, c0, c_mid), (r0, r1, c_mid, c1)]:
                rough = _roughness(*sub)
                heapq.heappush(heap, (-rough, *sub))

    patches = tuple(_make_patch(r0, r1, c0, c1) for _, r0, r1, c0, c1 in heap)
    return PatchPlan(patches=patches, canonical=p)
