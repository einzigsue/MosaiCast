"""Offline residual-stats estimation CLI (CLAUDE.md §4, M3).

Scans the training split and writes per-δt, per-variable, per-level
mean/std of Δ = X_{t+δt} − X_t (dynamics mode, control) or X_{t+δt}
directly (absolute mode, A1 ablation) to a .npz file.

Usage (dynamics / control):
    conda run -n mosaicast python scripts/estimate_residual_stats.py \\
        --data_dir data \\
        --nc_filename "aurora_{year}_5.625deg.nc" \\
        --years 1991 1992 \\
        --surf_vars 2t 10u 10v msl \\
        --atmos_vars z u v t q \\
        --atmos_levels 50 250 500 600 700 850 925 \\
        --dt_hours 6 12 24 \\
        --out_path data/residual_stats.npz \\
        --max_samples 5000

Usage (A1 absolute ablation):
    conda run -n mosaicast python scripts/estimate_residual_stats.py \\
        ... \\
        --mode absolute \\
        --out_path data/absolute_stats.npz
"""
from __future__ import annotations

import argparse
import os
import sys

# Allow running as `python scripts/estimate_residual_stats.py` from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mosaicast.data.datasets import MosaicastDataset
from mosaicast.data.stats import estimate_residual_stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate per-δt residual statistics from ERA5 NC files."
    )
    parser.add_argument("--data_dir", default="data",
                        help="Directory containing aurora_{year}_5.625deg.nc files")
    parser.add_argument("--nc_filename", default="aurora_{year}_5.625deg.nc",
                        help="NC filename template (uses {year} placeholder)")
    parser.add_argument("--static_nc", default=None,
                        help="Path to static-fields NC file (optional)")
    parser.add_argument("--years", nargs="+", type=int, required=True,
                        help="Years to scan, e.g. --years 1991 1992")
    parser.add_argument("--surf_vars", nargs="+", default=["2t", "10u", "10v", "msl"])
    parser.add_argument("--atmos_vars", nargs="+", default=["z", "u", "v", "t", "q"])
    parser.add_argument("--atmos_levels", nargs="+", type=int,
                        default=[50, 250, 500, 600, 700, 850, 925])
    parser.add_argument("--static_vars", nargs="+", default=[])
    parser.add_argument("--dt_hours", nargs="+", type=int, default=[6, 12, 24],
                        help="δt values in hours to estimate stats for")
    parser.add_argument("--out_path", default="data/residual_stats.npz",
                        help="Output .npz path")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap on samples scanned (useful for quick smoke tests)")
    parser.add_argument("--mode", default="dynamics",
                        choices=["dynamics", "absolute"],
                        help="'dynamics' (Δ = tar−inp, control) or 'absolute' (tar, A1 ablation)")
    args = parser.parse_args()

    nc_files = [
        os.path.join(args.data_dir, args.nc_filename.format(year=y))
        for y in args.years
    ]
    static_nc = args.static_nc
    if static_nc and not os.path.isabs(static_nc):
        static_nc = os.path.join(args.data_dir, static_nc)

    print(f"Building dataset from {len(nc_files)} file(s): {nc_files}")
    dataset = MosaicastDataset(
        nc_file_paths=nc_files,
        surf_vars=tuple(args.surf_vars),
        atmos_vars=tuple(args.atmos_vars),
        atmos_levels=tuple(args.atmos_levels),
        static_nc=static_nc,
        static_vars=tuple(args.static_vars),
        dt_hours=args.dt_hours,
    )
    print(f"Dataset length: {len(dataset)}")

    n = args.max_samples if args.max_samples else len(dataset)
    print(f"Scanning {n} samples for dt_hours={args.dt_hours}, mode={args.mode!r} …")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)), exist_ok=True)
    stats = estimate_residual_stats(
        dataset, args.dt_hours, args.out_path, args.max_samples, mode=args.mode
    )

    print(f"Saved stats to {args.out_path}")
    print(repr(stats))


if __name__ == "__main__":
    main()
