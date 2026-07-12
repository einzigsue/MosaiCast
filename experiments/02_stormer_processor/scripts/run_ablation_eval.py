"""Ablation evaluation runner (CLAUDE.md §6, M6).

Loads a trained checkpoint + config, runs inference on the 2020 test split,
and writes a CSV with columns:
    ablation, lead_days, variable, lat_rmse, acc, thresholded_rmse_p95, mean_log_power_ratio

Usage:
    # Single checkpoint
    conda run -n mosaicast python scripts/run_ablation_eval.py \\
        --checkpoint lightning_logs/.../checkpoints/best.ckpt \\
        --config configs/ablations/_base_ablation.yaml \\
        --config configs/ablations/a1_target_dynamics_control.yaml \\
        --ablation_name a1_dynamics_control \\
        --data_dir data \\
        --out_csv results/a1_dynamics_control.csv

    # Ensemble mode (averages all valid δt paths)
    conda run -n mosaicast python scripts/run_ablation_eval.py \\
        --checkpoint ... \\
        --ablation_name a5_ensemble \\
        --ensemble_mode homogeneous \\
        --out_csv results/a5_ensemble.csv

    # Single-member mode
    conda run -n mosaicast python scripts/run_ablation_eval.py \\
        --checkpoint ... \\
        --ablation_name a5_single \\
        --ensemble_mode single \\
        --single_dt 24 \\
        --out_csv results/a5_single.csv

Output CSV columns (CLAUDE.md §7):
    ablation               string tag passed via --ablation_name
    lead_days              1, 2, 3, 5, 7, 10, 14
    variable               variable name (e.g. "2t", "z500")
    lat_rmse               latitude-weighted RMSE
    acc                    anomaly correlation coefficient (needs --clim_path)
    thresholded_rmse_p95   lat-RMSE restricted to |target| > 95th percentile threshold
    mean_log_power_ratio   mean(log10(pred_power / target_power)) over wavenumbers > 0
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from datetime import timedelta

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mosaicast.eval import dt_paths, homogeneous_ensemble, single_member_forecast
from mosaicast.metrics import acc, lat_rmse, power_spectrum, thresholded_rmse
from mosaicast.patching.plans import uniform_plan


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LEAD_DAYS = [1, 2, 3, 5, 7, 10, 14]
SURF_SCORE_VARS = ["2t", "10u", "10v", "msl"]
ATMOS_SCORE_VARS = {
    "z500":  ("z", 500),
    "t850":  ("t", 850),
    "u250":  ("u", 250),
    "v250":  ("v", 250),
    "q700":  ("q", 700),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_module(checkpoint_path: str, config_paths: list[str], device: str):
    """Load a MosaicastLightningModule from checkpoint + config."""
    from lightning.pytorch.cli import LightningCLI
    import yaml

    # Merge configs in order (later files override earlier)
    merged = {}
    for cfg_path in config_paths:
        with open(cfg_path) as f:
            partial = yaml.safe_load(f) or {}
        _deep_merge(merged, partial)

    module_cls_path = merged.get("model", {}).get("class_path", "mosaicast.train.MosaicastLightningModule")
    module_cls = _import_cls(module_cls_path)
    init_args = merged.get("model", {}).get("init_args", {})

    module = module_cls.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        **{k: v for k, v in init_args.items() if k not in ("stats_path",)},
    )
    module.eval()
    module.to(device)
    return module


def _deep_merge(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _import_cls(class_path: str):
    module_path, cls_name = class_path.rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)


def _extract_field(batch, var_name: str, atmos_levels: tuple) -> torch.Tensor:
    """Extract (B, H, W) field. Supports 'z500' shorthand or plain surf var."""
    if var_name in ATMOS_SCORE_VARS:
        var, lev = ATMOS_SCORE_VARS[var_name]
        if var not in batch.atmos_vars:
            return None
        idx = list(atmos_levels).index(lev) if lev in atmos_levels else None
        if idx is None:
            return None
        t = batch.atmos_vars[var]  # (B, C, H, W) or (B, H, W)
        if t.dim() == 4:
            return t[:, idx]
        return t
    else:
        if var_name not in batch.surf_vars:
            return None
        t = batch.surf_vars[var_name]  # (B, H, W) or (B, T, H, W)
        if t.dim() == 4:
            return t[:, -1]
        return t


def _clim_for(clim_data: dict | None, var_name: str) -> torch.Tensor | None:
    if clim_data is None:
        return None
    return clim_data.get(var_name)


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------

def run_eval(
    module,
    test_loader,
    lat: torch.Tensor,
    lon: torch.Tensor,
    lead_days: list[int],
    ensemble_mode: str,
    single_dt: int,
    dt_support: tuple[int, ...],
    ablation_name: str,
    clim_data: dict | None,
    device: str,
    thres_percentile: float = 95.0,
) -> list[dict]:
    """Run inference + scoring; return list of row dicts."""
    rows = []
    plan = None  # built on first batch from lat/lon

    # Accumulate predictions and targets per lead_day per variable
    preds_by_lead: dict[int, dict[str, list[torch.Tensor]]] = {ld: {} for ld in lead_days}
    tars_by_lead:  dict[int, dict[str, list[torch.Tensor]]] = {ld: {} for ld in lead_days}

    atmos_levels = None

    with torch.no_grad():
        for batch_idx, (inp, tar, _patch_plan) in enumerate(test_loader):
            inp = _to_device(inp, device)
            tar = _to_device(tar, device)

            if atmos_levels is None:
                atmos_levels = inp.metadata.atmos_levels

            if plan is None:
                plan = uniform_plan(lat.to(device), lon.to(device), p=4)
            index = plan.index(lat.to(device), lon.to(device))

            for ld in lead_days:
                target_lead = timedelta(days=ld)
                if ensemble_mode == "homogeneous":
                    pred = homogeneous_ensemble(
                        module.model, inp, index, target_lead, dt_support=dt_support
                    )
                else:
                    pred = single_member_forecast(
                        module.model, inp, index, target_lead, dt_hours=single_dt
                    )

                all_vars = list(SURF_SCORE_VARS) + list(ATMOS_SCORE_VARS.keys())
                for var_name in all_vars:
                    p_field = _extract_field(pred, var_name, atmos_levels)
                    t_field = _extract_field(tar,  var_name, atmos_levels)
                    if p_field is None or t_field is None:
                        continue

                    if var_name not in preds_by_lead[ld]:
                        preds_by_lead[ld][var_name] = []
                        tars_by_lead[ld][var_name]  = []

                    preds_by_lead[ld][var_name].append(p_field.cpu())
                    tars_by_lead[ld][var_name].append(t_field.cpu())

    lat_cpu = lat.cpu()

    for ld in lead_days:
        for var_name in preds_by_lead[ld]:
            p_all = torch.cat(preds_by_lead[ld][var_name], dim=0)  # (N, H, W)
            t_all = torch.cat(tars_by_lead[ld][var_name],  dim=0)  # (N, H, W)

            # lat RMSE (mean over samples)
            rmse = lat_rmse(p_all, t_all, lat_cpu).mean().item()

            # ACC (needs climatology)
            clim = _clim_for(clim_data, var_name)
            if clim is not None:
                if clim.dim() == 2:
                    clim = clim.unsqueeze(0).expand_as(p_all)
                acc_val = acc(p_all, t_all, clim, lat_cpu).mean().item()
            else:
                acc_val = float("nan")

            # Thresholded RMSE at p95
            threshold = float(torch.quantile(t_all.abs().float(), thres_percentile / 100.0))
            trmse = thresholded_rmse(p_all, t_all, lat_cpu, threshold)
            trmse_val = trmse.nanmean().item()

            # Power spectrum log ratio (mean over wavenumbers k>0)
            pred_ps = power_spectrum(p_all).mean(0)   # (W//2+1,)
            tar_ps  = power_spectrum(t_all).mean(0)
            # Skip DC (k=0); clamp denominator to avoid log(0)
            ratio = (pred_ps[1:] / tar_ps[1:].clamp(min=1e-30)).log10()
            mlpr  = ratio.mean().item()

            rows.append({
                "ablation":             ablation_name,
                "lead_days":            ld,
                "variable":             var_name,
                "lat_rmse":             rmse,
                "acc":                  acc_val,
                "thresholded_rmse_p95": trmse_val,
                "mean_log_power_ratio": mlpr,
            })

    return rows


def _to_device(batch, device: str):
    from aurora import Batch
    from aurora.batch import Metadata
    return Batch(
        surf_vars={k: v.to(device)  for k, v in batch.surf_vars.items()},
        static_vars={k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.static_vars.items()},
        atmos_vars={k: v.to(device) for k, v in batch.atmos_vars.items()},
        metadata=batch.metadata,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ablation evaluation and write results CSV."
    )
    parser.add_argument("--checkpoint",   required=True, help="Path to .ckpt file")
    parser.add_argument("--config",       action="append", default=[], dest="configs",
                        help="Config YAML(s) to load (stacked in order; later overrides earlier)")
    parser.add_argument("--ablation_name", required=True, help="Label for the 'ablation' column")
    parser.add_argument("--data_dir",     default="data")
    parser.add_argument("--nc_filename",  default="aurora_{year}_5.625deg.nc")
    parser.add_argument("--static_nc",    default=None)
    parser.add_argument("--surf_vars",    nargs="+", default=["2t", "10u", "10v", "msl"])
    parser.add_argument("--atmos_vars",   nargs="+", default=["z", "u", "v", "t", "q"])
    parser.add_argument("--atmos_levels", nargs="+", type=int,
                        default=[50, 250, 500, 600, 700, 850, 925])
    parser.add_argument("--static_vars",  nargs="+", default=["lsm", "z", "slt"])
    parser.add_argument("--dt_hours",     nargs="+", type=int, default=[6, 12, 24])
    parser.add_argument("--test_years",   nargs="+", type=int, default=[2020])
    parser.add_argument("--batch_size",   type=int, default=1)
    parser.add_argument("--num_workers",  type=int, default=2)
    parser.add_argument("--lead_days",    nargs="+", type=int, default=LEAD_DAYS)
    parser.add_argument("--ensemble_mode", default="homogeneous",
                        choices=["homogeneous", "single"],
                        help="'homogeneous' averages all valid δt paths; 'single' uses one δt")
    parser.add_argument("--single_dt",   type=int, default=24,
                        help="δt (hours) for single-member mode")
    parser.add_argument("--clim_path",   default=None,
                        help="Optional .npz with climatology fields (for ACC)")
    parser.add_argument("--out_csv",     default="results/ablation_eval.csv")
    parser.add_argument("--device",      default="cpu")
    args = parser.parse_args()

    # ----- Build dataset + loader -----
    from mosaicast.data.datasets import MosaicastDataset
    from mosaicast.data.collate import mosaicast_collate_fn
    from mosaicast.data.sampler import DtBatchSampler
    from torch.utils.data import DataLoader

    nc_files = [
        os.path.join(args.data_dir, args.nc_filename.format(year=y))
        for y in args.test_years
    ]
    static_nc = args.static_nc
    if static_nc and not os.path.isabs(static_nc):
        static_nc = os.path.join(args.data_dir, static_nc)

    ds = MosaicastDataset(
        nc_file_paths=nc_files,
        surf_vars=tuple(args.surf_vars),
        atmos_vars=tuple(args.atmos_vars),
        atmos_levels=tuple(args.atmos_levels),
        static_nc=static_nc,
        static_vars=tuple(args.static_vars),
        dt_hours=args.dt_hours,
    )
    sampler = DtBatchSampler(
        n=len(ds),
        batch_size=args.batch_size,
        dt_hours=args.dt_hours,
        shuffle=False,
        drop_last=False,
        seed=0,
    )
    loader = DataLoader(
        ds,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=mosaicast_collate_fn(patch_size=4),
        pin_memory=(args.device != "cpu"),
    )

    # Lat/lon from dataset
    sample_inp, _, _ = next(iter(loader))
    lat = sample_inp.metadata.lat.cpu()
    lon = sample_inp.metadata.lon.cpu()

    # ----- Load model -----
    module = _load_module(args.checkpoint, args.configs, args.device)

    # ----- Load climatology (optional) -----
    clim_data = None
    if args.clim_path is not None:
        raw = np.load(args.clim_path)
        clim_data = {k: torch.from_numpy(raw[k]).float() for k in raw.files}

    # ----- Run eval -----
    rows = run_eval(
        module=module,
        test_loader=loader,
        lat=lat,
        lon=lon,
        lead_days=args.lead_days,
        ensemble_mode=args.ensemble_mode,
        single_dt=args.single_dt,
        dt_support=tuple(args.dt_hours),
        ablation_name=args.ablation_name,
        clim_data=clim_data,
        device=args.device,
    )

    # ----- Write CSV -----
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    fieldnames = ["ablation", "lead_days", "variable", "lat_rmse", "acc",
                  "thresholded_rmse_p95", "mean_log_power_ratio"]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
