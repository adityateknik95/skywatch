"""Phase 4 tests: windowing (respecting trajectory boundaries + split), the LSTM
forward shape, and residual scoring / threshold calibration. Fast — no training."""

from __future__ import annotations

import numpy as np
import torch

from skywatch.model.dataset import (
    WindowDataset,
    split_ranges,
    trajectory_ranges,
    window_starts,
)
from skywatch.model.factory import build_model
from skywatch.model.lstm import LSTMConfig, LSTMPredictor
from skywatch.model.score import calibrate_threshold, residual_scores
from skywatch.model.transformer import TransformerConfig, TransformerPredictor
from torch.utils.data import DataLoader


def test_trajectory_ranges():
    tid = np.array([1, 1, 1, 2, 2, 3])
    assert trajectory_ranges(tid) == [(1, 0, 3), (2, 3, 5), (3, 5, 6)]


def test_trajectory_ranges_empty():
    assert trajectory_ranges(np.array([], dtype=int)) == []


def test_window_starts_within_one_trajectory():
    # traj A: rows 0..4 (len 5), traj B: rows 5..6 (len 2)
    tid = np.array([10, 10, 10, 10, 10, 20, 20])
    ranges = trajectory_ranges(tid)
    window = 2
    starts = window_starts(ranges, window)
    # A contributes starts 0,1,2 (target idx <=4); B (len 2 <= window) contributes none.
    assert starts.tolist() == [0, 1, 2]
    # No window crosses a trajectory boundary.
    for s in starts:
        assert tid[s] == tid[s + window]


def test_window_starts_skips_short_trajectories():
    tid = np.array([1, 1])  # length 2, window 5 -> no windows
    assert window_starts(trajectory_ranges(tid), window=5).size == 0


def test_split_ranges_disjoint_and_deterministic():
    ranges = [(i, i, i + 1) for i in range(100)]
    a = split_ranges(ranges, seed=7)
    b = split_ranges(ranges, seed=7)
    assert [r for r in a[0]] == [r for r in b[0]]  # deterministic
    train, val, calib = a
    ids = lambda rs: {r[0] for r in rs}
    assert ids(train) & ids(val) == set()
    assert ids(train) & ids(calib) == set()
    assert len(train) + len(val) + len(calib) == 100


def test_window_dataset_shapes_and_content():
    X = np.arange(6 * 3, dtype=np.float32).reshape(6, 3)  # one trajectory, 6 rows
    starts = window_starts([(1, 0, 6)], window=2)         # 0,1,2,3
    ds = WindowDataset(X, starts, window=2)
    assert len(ds) == 4
    w, y = ds[0]
    assert w.shape == (2, 3) and y.shape == (3,)
    assert torch.allclose(w, torch.tensor(X[0:2]))
    assert torch.allclose(y, torch.tensor(X[2]))


def test_lstm_forward_shape():
    cfg = LSTMConfig(n_features=4, hidden_size=8, num_layers=1)
    model = LSTMPredictor(cfg)
    out = model(torch.randn(5, 3, 4))  # (B=5, W=3, F=4)
    assert out.shape == (5, 4)


def test_residual_scores_and_threshold():
    X = np.random.default_rng(0).standard_normal((40, 3)).astype(np.float32)
    starts = window_starts([(1, 0, 40)], window=4)
    ds = WindowDataset(X, starts, window=4)
    dl = DataLoader(ds, batch_size=8)
    model = LSTMPredictor(LSTMConfig(n_features=3, hidden_size=8))
    scores = residual_scores(model, dl)
    assert scores.shape == (len(ds),)
    assert np.all(np.isfinite(scores)) and np.all(scores >= 0)


def test_calibrate_threshold_matches_percentile():
    scores = np.arange(1, 101, dtype=float)
    assert calibrate_threshold(scores, 99.0) == np.percentile(scores, 99.0)


def test_transformer_forward_shape():
    m = TransformerPredictor(
        TransformerConfig(n_features=6, d_model=16, nhead=2, num_layers=2, dim_feedforward=32)
    )
    out = m(torch.randn(4, 10, 6))  # (B, W, F)
    assert out.shape == (4, 6)


def test_build_model_factory_selects_by_arch():
    lstm = build_model({"arch": "lstm", "n_features": 5, "hidden_size": 8,
                        "num_layers": 1, "dropout": 0.0})
    assert isinstance(lstm, LSTMPredictor)
    tf = build_model({"arch": "transformer", "n_features": 5, "d_model": 16, "nhead": 2,
                      "num_layers": 1, "dim_feedforward": 32, "dropout": 0.0})
    assert isinstance(tf, TransformerPredictor)
    assert tf(torch.randn(2, 7, 5)).shape == (2, 5)
    # missing arch defaults to lstm (backward compatible)
    assert isinstance(
        build_model({"n_features": 5, "hidden_size": 8, "num_layers": 1, "dropout": 0.0}),
        LSTMPredictor,
    )
