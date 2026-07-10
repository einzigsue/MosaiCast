"""Training entry point — config-driven (CLAUDE.md rules 1–6).

Usage:
    python3 train.py --config configs/default.yaml --exp baseline
    python3 train.py --config configs/default.yaml --exp baseline --smoke
"""
from __future__ import annotations
import argparse
import datetime
import sys,os
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import mlflow

wdir="/g/data/z00/yxs900/aurora/experiments/01_baseline/"
sys.path.insert(0,wdir)
from data.collate import aurora_collate_fn
from data.utils import make_datasets
from evaluate import run_validation
from models import build_model
from utils.config import flatten_config, load_config
from utils.losses import var_weighted_loss
from utils.logging import get_or_create_experiment, log_metrics_jsonl
from utils.seed import seed_everything

SMOKE_STEPS = 10
CKPT_STEP_INTERVAL = 1000  # also checkpoint every N training steps, to survive mid-epoch walltime kills


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Tier config yaml (e.g. configs/t1-local.yaml)")
    p.add_argument("--exp", required=True, help="Run label used for naming/logging (not a config file path)")
    p.add_argument("--smoke", action="store_true", help="Run 50 steps then exit (T1 gate check)")
    p.add_argument("--seed", type=int, default=None, help="Override config seed")
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Resume training from a checkpoint. Pass an explicit path to a previous "
            "run's checkpoints/last.pt — ckpt_dir is freshly created per PBS_JOBID/"
            "timestamp, so a new job never has its own last.pt yet, and an explicit "
            "path is the primary way to resume across a requeued job. Alternatively "
            "pass 'auto' to search results/*/*/checkpoints/last.pt for the most "
            "recently modified checkpoint."
        ),
    )
    return p.parse_args()


def _atomic_save(payload: dict, path: Path) -> None:
    """Save payload to path atomically (tmp file + os.replace).

    Protects against a walltime kill mid-write leaving a truncated/corrupt
    checkpoint: torch.save writes to a sibling .tmp file first, and os.replace
    is a single atomic rename.
    """
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def _save_last_checkpoint(
    ckpt_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    cfg: dict,
) -> None:
    """Write ckpt_dir/last.pt with everything needed to resume training.

    ``epoch`` is stored as the *next* epoch index to run on resume: callers
    pass ``epoch`` itself for a mid-epoch (step-interval) save — since that
    epoch is still in progress, resuming re-runs it from its start rather than
    fast-forwarding the dataloader to an exact mid-epoch position — and
    ``epoch + 1`` for an end-of-epoch save, since that epoch is complete.
    """
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "cfg": cfg,
    }
    _atomic_save(payload, ckpt_dir / "last.pt")


def _find_latest_checkpoint(results_root: Path) -> Path | None:
    """Search results_root/<jobid>/<timestamp>/checkpoints/last.pt for `--resume auto`.

    Returns the most recently modified last.pt, or None if none exist.
    """
    candidates = list(results_root.glob("*/*/checkpoints/last.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.exp)
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    seed_everything(seed)

    device = torch.device(cfg.get("device", "cpu"))
    tier = cfg.get("tier", "t1")
    gamma = cfg.get("gamma", 2.5)
    ds_cfg = cfg["data"]
    train_cfg = cfg["training"]
    opt_cfg = cfg["optimizer"]
    sched_cfg = cfg["scheduler"]

    # ── Datasets & loaders ──────────────────────────────────────────────────
    train_ds, val_ds, _ = make_datasets(ds_cfg)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        collate_fn=aurora_collate_fn,
        drop_last=True,
        persistent_workers=train_cfg["num_workers"] > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        collate_fn=aurora_collate_fn,
        persistent_workers=train_cfg["num_workers"] > 0,
    )

    # ── Model ───────────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {cfg['model']['name']} | {n_params / 1e6:.1f} M params | device: {device}")

    # ── Optimizer & scheduler ────────────────────────────────────────────────
    # match settings provided in the original paper 
    # https://www.nature.com/articles/s41586-025-09005-y
    # We use a (half) cosine decay with a linear warm-up from zero for 1,000 steps. 
    # The base learning rate is 5 × 10−4, which the scheduler
    # reduces by a factor 10x at the end of training. All models are pretrained for 150 k steps.
    # The optimizer we use is AdamW. We set the weight decay of AdamW to 5 × 10−6. 

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg["lr"], # 5e-4
        weight_decay=opt_cfg["weight_decay"], # 5e-6
        betas=tuple(opt_cfg["betas"]), # didn't mention in the paper using the most popular default
    )

    total_steps = train_cfg["max_epochs"] * len(train_loader)
    warmup_steps = sched_cfg["warmup_steps"]

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-8 / opt_cfg["lr"], # start with almost 0 
        end_factor=1.0, # end with opt_cfg["lr"] = 5e-4
        total_iters=warmup_steps, # 1000
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps - warmup_steps), # 150K-1000
        eta_min=sched_cfg["min_lr"], # 5e-5
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
    )

    # ── Results dir & config snapshot ───────────────────────────────────────
    jobid = os.environ["PBS_JOBID"].removesuffix(".gadi-pbs")
    ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    run_name = f"{jobid}_{ts}"
    results_dir = Path(f"{wdir}/results/{jobid}/{ts}")
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / "metrics.jsonl"
    ckpt_dir = results_dir / "checkpoints"
    ckpt_dir.mkdir()

    frozen_cfg_path = results_dir / "config.yaml"
    with open(frozen_cfg_path, "w") as f:
        yaml.dump(cfg, f)

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    if args.resume:
        if args.resume == "auto":
            resume_path = _find_latest_checkpoint(Path(f"{wdir}/results"))
            if resume_path is None:
                raise FileNotFoundError(
                    f"--resume auto found no checkpoints under {wdir}/results/*/*/checkpoints/last.pt. "
                    "ckpt_dir is created fresh per PBS_JOBID/timestamp, so a brand-new job never has "
                    "its own last.pt yet — pass the explicit path to a previous job's "
                    "checkpoints/last.pt instead (e.g. --resume "
                    f"{wdir}/results/<jobid>/<timestamp>/checkpoints/last.pt)."
                )
        else:
            resume_path = Path(args.resume)
            if not resume_path.exists():
                raise FileNotFoundError(f"--resume path does not exist: {resume_path}")

        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"]
        global_step = ckpt["global_step"]
        best_val_loss = ckpt["best_val_loss"]
        print(
            f"Resumed from {resume_path}: starting at epoch {start_epoch}, "
            f"global_step {global_step}, best_val_loss {best_val_loss:.4f}."
        )

    # ── Training loop ────────────────────────────────────────────────────────
    exp_id = get_or_create_experiment(tier)
    max_steps = SMOKE_STEPS if args.smoke else None

    with mlflow.start_run(experiment_id=exp_id, run_name=run_name):
        mlflow.log_params(flatten_config(cfg))
        mlflow.log_artifact(str(frozen_cfg_path))

        for epoch in range(start_epoch, train_cfg["max_epochs"]):
            model.train()

            for batch, target in train_loader:
                if max_steps is not None and global_step >= max_steps:
                    break

                batch = batch.to(device)
                target = target.to(device)

                pred = model(batch)
                loss = var_weighted_loss(pred, target, gamma=gamma)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at step {global_step}: {loss.item()}")

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), train_cfg["gradient_clip"])
                optimizer.step()
                scheduler.step()

                global_step += 1
                loss_val = loss.detach().item()
                lr = scheduler.get_last_lr()[0]

                # Mid-epoch safety net: checkpoint every N steps so a walltime
                # kill can't lose more than CKPT_STEP_INTERVAL steps of progress.
                if global_step % CKPT_STEP_INTERVAL == 0:
                    _save_last_checkpoint(
                        ckpt_dir, model, optimizer, scheduler, epoch, global_step, best_val_loss, cfg
                    )

                # Smoke runs log every step: shorter than log_every_n_steps,
                # they would otherwise complete without ever showing a loss.
                log_every = 1 if args.smoke else train_cfg["log_every_n_steps"]
                if global_step % log_every == 0:
                    mlflow.log_metrics({"train/loss": loss_val, "train/lr": lr}, step=global_step)
                    log_metrics_jsonl(metrics_path, global_step, {"train/loss": loss_val, "train/lr": lr})
                    print(f"epoch {epoch} | step {global_step} | loss {loss_val:.4f} | lr {lr:.2e}")

            if max_steps is not None and global_step >= max_steps:
                print(f"Smoke run complete ({global_step} steps).")
                break

            # ── Validation ──────────────────────────────────────────────────
            if (epoch + 1) % train_cfg["val_every_n_epochs"] == 0:
                val_metrics = run_validation(model, val_loader, device, gamma, global_step, metrics_path)
                if val_metrics:
                    mlflow.log_metrics(val_metrics, step=global_step)
                    val_loss = val_metrics.get("val/loss", float("inf"))
                    rmse_str = "  ".join(
                        f"{k.split('/')[1]}={v:.4f}"
                        for k, v in val_metrics.items()
                        if k.endswith("/rmse")
                    )
                    print(f"  val loss {val_loss:.4f}  {rmse_str}")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(model.state_dict(), ckpt_dir / "best.pt")

            # ── End-of-epoch checkpoint (resume point) ────────────────────────
            # epoch + 1: this epoch is complete, so resume should start the next one.
            _save_last_checkpoint(
                ckpt_dir, model, optimizer, scheduler, epoch + 1, global_step, best_val_loss, cfg
            )

        # ── Final checkpoint ────────────────────────────────────────────────
        final_ckpt = ckpt_dir / "final.pt"
        torch.save(model.state_dict(), final_ckpt)
        mlflow.log_artifact(str(final_ckpt))
        print(f"Done. Results in {results_dir}")


if __name__ == "__main__":
    main()
