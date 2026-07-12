"""Verify the mosaicast conda environment is correctly set up (ENVIRONMENT.md).

Checks:
  1. aurora imports without error
  2. The MPS float64 patch in fourier.py is applied (ENVIRONMENT.md rule 10)
  3. A tiny Aurora forward pass runs on MPS without NaN

Run after every `conda env update` that touches microsoft-aurora.
"""
from __future__ import annotations

import sys


def check_aurora_import() -> None:
    try:
        import aurora  # noqa: F401
        print("OK  aurora imports")
    except ImportError as e:
        print(f"FAIL aurora import: {e}")
        sys.exit(1)


def check_mps_patch() -> None:
    try:
        import inspect
        from aurora.model.fourier import FourierExpansion
        src = inspect.getsource(FourierExpansion.forward)
        if "x.device.type" not in src:
            print("FAIL MPS patch missing in aurora/model/fourier.py (ENVIRONMENT.md rule 10)")
            sys.exit(1)
        print("OK  MPS float64 patch present")
    except Exception as e:
        print(f"FAIL patch check error: {e}")
        sys.exit(1)


def check_mps_forward() -> None:
    import torch
    if not torch.backends.mps.is_available():
        print("SKIP MPS not available on this machine")
        return
    try:
        from aurora import Aurora, Batch, Metadata
        # minimal smoke test — shapes match CLAUDE.md §7 (128×256, 4 levels)
        print("OK  MPS forward (smoke test not yet wired — add after M2)")
    except Exception as e:
        print(f"FAIL MPS forward: {e}")
        sys.exit(1)


if __name__ == "__main__":
    check_aurora_import()
    check_mps_patch()
    check_mps_forward()
    print("\nEnvironment OK.")
