"""Sliding-window dataset for next-step prediction (spec §10).

Windows are sliced **lazily** from the flat feature array (only ~100 MB total),
never materialized as a dense (N, W, F) tensor — important on an 8 GB machine.
Windows never cross trajectory boundaries, and the train/val/calibration split is
done at the **trajectory** level so no window straddles the split (no leakage).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

# A contiguous run of one trajectory in the flat feature array: (trajectory_id, lo, hi)
Range = tuple[int, int, int]


def trajectory_ranges(trajectory_id: np.ndarray) -> list[Range]:
    """Contiguous [lo, hi) ranges for each trajectory in a sorted id array."""
    tid = np.asarray(trajectory_id)
    if tid.size == 0:
        return []
    change = np.flatnonzero(np.diff(tid)) + 1
    bounds = np.concatenate(([0], change, [tid.size]))
    return [
        (int(tid[bounds[k]]), int(bounds[k]), int(bounds[k + 1]))
        for k in range(len(bounds) - 1)
    ]


def window_starts(ranges: list[Range], window: int) -> np.ndarray:
    """Global start indices i such that X[i:i+window] -> predict X[i+window],
    staying within a single trajectory."""
    parts = [
        np.arange(lo, hi - window)
        for _, lo, hi in ranges
        if (hi - lo) > window
    ]
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)


def split_ranges(
    ranges: list[Range],
    fracs: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> tuple[list[Range], list[Range], list[Range]]:
    """Shuffle and split trajectories into (train, val, calibration)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ranges))
    n = len(ranges)
    n_train = int(fracs[0] * n)
    n_val = int(fracs[1] * n)
    train = [ranges[i] for i in order[:n_train]]
    val = [ranges[i] for i in order[n_train : n_train + n_val]]
    calib = [ranges[i] for i in order[n_train + n_val :]]
    return train, val, calib


class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, starts: np.ndarray, window: int) -> None:
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.starts = torch.as_tensor(np.asarray(starts), dtype=torch.long)
        self.window = window

    def __len__(self) -> int:
        return int(self.starts.numel())

    def __getitem__(self, i: int):
        s = int(self.starts[i])
        return self.X[s : s + self.window], self.X[s + self.window]
