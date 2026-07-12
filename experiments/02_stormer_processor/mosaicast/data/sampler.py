"""DtBatchSampler — batch-level δt selection (CLAUDE.md §1.9, M3).

Moves the "one δt per batch" decision out of the collate function and into
the sampler, so __getitem__ only ever loads the one target it actually needs.

Design rationale:
  The DataLoader calls __getitem__ independently per sample (in parallel workers
  if num_workers > 0).  A batch-level decision therefore cannot live inside
  __getitem__ without shared mutable state between workers, which is fragile.
  The BatchSampler runs in the main process after all indices for a batch are
  known, making it the natural and safe place for the per-batch δt draw.

DDP (B2 + M1 fix):
  Pass num_replicas=world_size and rank=global_rank so each GPU sees a disjoint
  shard of the dataset.  All ranks draw the same δt per batch (coordinated via a
  shared-seed dt_rng) while shuffling independently (per-rank shuffle_rng).
  Call set_epoch(epoch) each epoch so the shuffle varies across epochs.
  Lightning 2.x calls set_epoch on batch_sampler automatically.
"""
from __future__ import annotations

import numpy as np
import torch.utils.data


class DtBatchSampler(torch.utils.data.Sampler):
    """Yields batches of ``(global_idx, dt_hours_int)`` pairs.

    All indices within one batch share the same ``dt_hours_int``, drawn
    uniformly at the start of each batch.  Passing these tuples as the index
    to ``MosaicastDataset.__getitem__`` causes exactly one target frame to be
    loaded per sample — no wasted I/O.

    Args:
        n:            Total number of valid samples (``len(dataset)``).
        batch_size:   Samples per batch.
        dt_hours:     Supported δt values in hours, e.g. ``[6, 12, 24]``.
        shuffle:      Shuffle sample order each epoch (True for train).
        drop_last:    Drop the final incomplete batch (True for train).
        seed:         Base RNG seed.  Epoch is added at iteration time so the
                      shuffle changes every epoch.  Default 0.
        num_replicas: Number of DDP processes (B2 fix).  Default 1 (single-GPU).
        rank:         This process's rank (B2 fix).  Default 0.
    """

    def __init__(
        self,
        n: int,
        batch_size: int,
        dt_hours: list[int],
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 0,
        num_replicas: int = 1,
        rank: int = 0,
    ) -> None:
        if rank >= num_replicas:
            raise ValueError(f"rank ({rank}) must be < num_replicas ({num_replicas})")
        self.n            = n
        self.batch_size   = batch_size
        self.dt_hours     = list(dt_hours)
        self.shuffle      = shuffle
        self.drop_last    = drop_last
        self.seed         = seed
        self.num_replicas = num_replicas
        self.rank         = rank
        self._epoch       = 0

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for epoch-dependent shuffling (called by Lightning 2.x)."""
        self._epoch = epoch

    def __iter__(self):
        base = self.seed + self._epoch * 1000

        # Global shuffle RNG: same seed on all ranks so the shuffled order is
        # identical everywhere; strided shard then gives disjoint subsets (B2 fix).
        # δt RNG shares the same base so all ranks draw identical δt per batch (M1 fix).
        global_rng = np.random.default_rng(base)

        indices = np.arange(self.n)
        if self.shuffle:
            global_rng.shuffle(indices)

        # Truncate to a multiple of num_replicas so every rank has the same
        # number of batches — avoids AllReduce hangs in DDP (B2 fix).
        n_truncated = (self.n // self.num_replicas) * self.num_replicas
        indices = indices[:n_truncated]
        indices = indices[self.rank::self.num_replicas]

        # δt RNG: advance past the shuffle state so δt draws are independent of
        # the shuffle permutation; draw one dt per batch, same across all ranks.
        dt_rng = np.random.default_rng(base + 9999)

        batch: list[tuple[int, int]] = []
        dt = int(dt_rng.choice(self.dt_hours))

        for idx in indices:
            batch.append((int(idx), dt))
            if len(batch) == self.batch_size:
                yield batch
                batch = []
                dt = int(dt_rng.choice(self.dt_hours))

        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        n_per_rank = self.n // self.num_replicas
        if self.drop_last:
            return n_per_rank // self.batch_size
        return (n_per_rank + self.batch_size - 1) // self.batch_size
