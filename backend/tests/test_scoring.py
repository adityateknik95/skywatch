"""LiveScorer tests: warmup, normal stream, physics-caught teleport, gap reset,
and pruning. No DB / no trained model needed (the model is suppressed with a huge
threshold so we isolate the physics + state-machine behavior)."""

from __future__ import annotations

import numpy as np

from skywatch.api.scoring import LiveScorer
from skywatch.features import compute_trajectory_features, fit_scaler
from skywatch.model.lstm import LSTMConfig, LSTMPredictor

DT = 15
CRUISE = 230.0
STEP_DEG = CRUISE * DT / (111_320.0 * np.cos(np.radians(50.0)))  # east, per step


def _scorer(threshold=1e9, window=5):
    # Fit a scaler on realistic feature rows from a synthetic clean flight.
    n = 40
    lon = 8.0 + STEP_DEG * np.arange(n)
    X = compute_trajectory_features(
        np.full(n, 50.0), lon, np.full(n, 10000.0), np.full(n, 10050.0),
        np.full(n, CRUISE), np.full(n, 90.0), np.zeros(n), DT,
    )
    scaler = fit_scaler(X)
    model = LSTMPredictor(LSTMConfig(n_features=X.shape[1], hidden_size=8))
    return LiveScorer(model, scaler, threshold=threshold, window=window)


def _normal_point(t, i):
    return dict(icao24="abc123", t=t, lat=50.0, lon=8.0 + STEP_DEG * i,
               baro=10000.0, geo=10050.0, velocity=CRUISE, true_track=90.0,
               vertical_rate=0.0)


def test_warmup_first_point_returns_none():
    sc = _scorer()
    assert sc.score(**_normal_point(1000, 0)) is None


def test_normal_stream_not_flagged():
    sc = _scorer()
    res = None
    for i in range(10):
        res = sc.score(**_normal_point(1000 + i * DT, i))
    assert res is not None and res.is_anomaly is False and res.reason is None


def test_teleport_caught_by_physics():
    sc = _scorer()
    for i in range(6):
        sc.score(**_normal_point(1000 + i * DT, i))
    # next point jumps ~0.5 deg (~55 km) in one 15s step -> impossible speed
    t = 1000 + 6 * DT
    res = sc.score(icao24="abc123", t=t, lat=50.5, lon=8.0 + STEP_DEG * 6,
                   baro=10000.0, geo=10050.0, velocity=CRUISE, true_track=90.0,
                   vertical_rate=0.0)
    assert res is not None and res.is_anomaly and res.reason == "physics:speed"


def test_gap_resets_segment():
    sc = _scorer()
    sc.score(**_normal_point(1000, 0))
    sc.score(**_normal_point(1015, 1))
    # a 1-hour jump in time is a gap -> no feature, returns None
    assert sc.score(**_normal_point(1000 + 3600, 2)) is None


def test_prune_removes_stale():
    sc = _scorer()
    sc.score(**_normal_point(1000, 0))
    assert "abc123" in sc.states
    removed = sc.prune(now_t=1000 + 100_000, max_idle=900)
    assert removed == 1 and "abc123" not in sc.states
