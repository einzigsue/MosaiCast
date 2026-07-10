"""WeatherBench2 daily climatology for the ACC (Aurora SI Eq. F16).

The ACC measures correlation of prediction and target *anomalies* relative to a
daily climatology ``C_ij^t`` — the mean of a variable for that day of year (and
hour) at each gridpoint. Per the researcher's decision we use the WeatherBench2
climatology (Rasp et al.), the same source the Aurora paper uses, rather than a
climatology estimated from our short local record.

Workflow (run once, offline):

    from apa.data.climatology import build_from_weatherbench2
    build_from_weatherbench2(
        wb2_path="gs://weatherbench2/datasets/era5-hourly-climatology/...zarr",
        lat=train_ds.lat, lon=train_ds.lon,           # our model grid
        atmos_levels=(50, 250, 500, 600, 700, 850, 1000),
        out_path="dataset/clim/wb2_5.625deg.npz",
    )

then at eval time:

    clim = Climatology.load("dataset/clim/wb2_5.625deg.npz")
    C = clim.clim_like("z500", target.metadata.time, ref=pred_z500)  # (B,1,H,W)

The WB2 climatology is regridded to our (lat, lon) with xarray ``interp`` and
cached to an ``.npz`` so eval never needs network or xarray. Cache keys follow the
metric-report convention: ``"2t"`` (surface), ``"z500"`` (atmospheric).
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import torch

# Short APA names → WeatherBench2 / ERA5 long variable names.
WB2_SURF_NAMES: dict[str, str] = {
    "2t": "2m_temperature",
    "10u": "10m_u_component_of_wind",
    "10v": "10m_v_component_of_wind",
    "msl": "mean_sea_level_pressure",
}
WB2_ATMOS_NAMES: dict[str, str] = {
    "z": "geopotential",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "t": "temperature",
    "q": "specific_humidity",
}


def clim_key(var: str, level: int | None = None) -> str:
    """Cache key: ``"2t"`` (surface) or ``"z500"`` (atmospheric)."""
    return var if level is None else f"{var}{level}"


def build_from_weatherbench2(
    wb2_path: str,
    lat: "torch.Tensor | np.ndarray",
    lon: "torch.Tensor | np.ndarray",
    surf_vars: tuple[str, ...] = ("2t", "10u", "10v", "msl"),
    atmos_vars: tuple[str, ...] = ("z", "u", "v", "t", "q"),
    atmos_levels: tuple[int, ...] = (50, 250, 500, 600, 700, 850, 1000),
    out_path: str | Path | None = None,
    method: str = "linear",
) -> "Climatology":
    """Open the WB2 climatology, regrid to (lat, lon), cache per-key arrays.

    Requires ``xarray`` (and ``zarr``/``gcsfs`` for a remote ``wb2_path``); this is
    an offline preprocessing step, so the import is local to keep eval dependencies
    light. The WB2 climatology carries ``dayofyear`` (1-366) and ``hour`` coords; we
    keep both so ACC can match each valid time's day *and* synoptic hour.
    """
    import xarray as xr

    lat_np = np.asarray(lat.cpu() if hasattr(lat, "cpu") else lat, dtype=np.float64)
    lon_np = np.asarray(lon.cpu() if hasattr(lon, "cpu") else lon, dtype=np.float64)

    ds = xr.open_zarr(wb2_path) if str(wb2_path).endswith(".zarr") else xr.open_dataset(wb2_path)

    # WB2 uses 'latitude'/'longitude'; be tolerant of 'lat'/'lon'.
    lat_name = "latitude" if "latitude" in ds.dims else "lat"
    lon_name = "longitude" if "longitude" in ds.dims else "lon"

    def _regrid(da):
        return da.interp({lat_name: lat_np, lon_name: lon_np}, method=method)

    doy = np.asarray(ds["dayofyear"], dtype=np.int64)
    hour = np.asarray(ds["hour"], dtype=np.int64) if "hour" in ds.coords or "hour" in ds.dims else np.array([0])

    values: dict[str, np.ndarray] = {}
    for v in surf_vars:
        da = _regrid(ds[WB2_SURF_NAMES[v]])
        values[clim_key(v)] = _to_doy_hour_hw(da, len(doy), len(hour))
    for v in atmos_vars:
        for lev in atmos_levels:
            da = _regrid(ds[WB2_ATMOS_NAMES[v]].sel(level=lev))
            values[clim_key(v, lev)] = _to_doy_hour_hw(da, len(doy), len(hour))

    clim = Climatology(values=values, dayofyear=doy, hour=hour,
                       lat=lat_np.astype(np.float32), lon=lon_np.astype(np.float32))
    if out_path is not None:
        clim.save(out_path)
    return clim


def _to_doy_hour_hw(da, n_doy: int, n_hour: int) -> np.ndarray:
    """Coerce a regridded DataArray to ``(dayofyear, hour, H, W)`` float32."""
    arr = np.asarray(da.transpose("dayofyear", ..., da.dims[-2], da.dims[-1]), dtype=np.float32)
    if arr.ndim == 3:  # no hour dim → add a singleton
        arr = arr[:, None]
    return arr.reshape(n_doy, n_hour, arr.shape[-2], arr.shape[-1])


class Climatology:
    """Cached WB2 daily climatology with per-valid-time gridpoint lookup."""

    def __init__(self, values: dict[str, np.ndarray], dayofyear: np.ndarray,
                 hour: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                 units: dict[str, str] = None) -> None:
        self.values = {k: torch.tensor(v, dtype=torch.float32) for k, v in values.items()}
        self.units = units or {k: "unknown" for k in self.values.keys()}
        self.dayofyear = np.asarray(dayofyear, dtype=np.int64)
        self.hour = np.asarray(hour, dtype=np.int64)
        self.lat = torch.tensor(lat, dtype=torch.float32)
        self.lon = torch.tensor(lon, dtype=torch.float32)
        self._doy_index = {int(d): i for i, d in enumerate(self.dayofyear)}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        flat = {f"val__{k}": v.numpy() for k, v in self.values.items()}
        flat_units = {f"unit__{k}": np.array(v, dtype=str) for k, v in self.units.items()}
        np.savez(path, dayofyear=self.dayofyear, hour=self.hour,
                 lat=self.lat.numpy(), lon=self.lon.numpy(), **flat, **flat_units)

    @classmethod
    def load(cls, path: str | Path) -> "Climatology":
        data = np.load(path)
        values = {name[len("val__"):]: data[name] for name in data.files if name.startswith("val__")}
        units = {name[len("unit__"):]: str(data[name]) for name in data.files if name.startswith("unit__")}
        return cls(values=values, dayofyear=data["dayofyear"], hour=data["hour"],
                   lat=data["lat"], lon=data["lon"], units=units)

    def _lookup_one(self, key: str, time) -> torch.Tensor:
        """(H, W) climatology for ``key`` at a single datetime's day/hour."""
        doy = time.timetuple().tm_yday
        # Day 366 folds to 365 when the cache has no leap day.
        i = self._doy_index.get(doy) or self._doy_index.get(min(doy, len(self.dayofyear)), 0)
        # Nearest synoptic hour.
        j = int(np.argmin(np.abs(self.hour - time.hour))) if self.hour.size > 1 else 0
        return self.values[key][i, j]

    def clim_like(self, key: str, times, ref: torch.Tensor) -> torch.Tensor:
        """Stack per-sample climatology into ``(B, 1, H, W)`` matching ``ref``.

        ``times`` is the batch's ``metadata.time`` (a flat tuple of B datetimes,
        the collated valid times). ``ref`` supplies device/dtype.
        """
        maps = [self._lookup_one(key, t) for t in times]
        out = torch.stack(maps, dim=0).unsqueeze(1)  # (B, 1, H, W)
        return out.to(device=ref.device, dtype=ref.dtype)
