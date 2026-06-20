"""Tests for the pure reconstruction helpers (segmentation, resampling, angular
interpolation, and the 'gaps are not bridged' property). No DB required."""

from __future__ import annotations

import numpy as np
import pandas as pd

from skywatch.reconstruct import (
    interp_angular,
    interp_linear,
    reconstruct_aircraft,
    resample_segment,
    segment_ids,
)

_POINT_COLS = ("lat", "lon", "baro_altitude", "geo_altitude",
               "velocity", "true_track", "vertical_rate")


def _aircraft_frame(times, **overrides) -> pd.DataFrame:
    n = len(times)
    data = {"t": np.asarray(times, dtype="int64")}
    # default smooth airborne values
    data["lat"] = np.linspace(50.0, 50.5, n)
    data["lon"] = np.linspace(8.0, 9.0, n)
    data["baro_altitude"] = np.full(n, 11000.0)
    data["geo_altitude"] = np.full(n, 11100.0)
    data["velocity"] = np.full(n, 230.0)
    data["true_track"] = np.full(n, 90.0)
    data["vertical_rate"] = np.zeros(n)
    data.update(overrides)
    return pd.DataFrame(data)


# --- segment_ids -----------------------------------------------------------
def test_segment_ids_splits_on_gap():
    t = np.array([0, 15, 30, 200, 215, 230])
    labels = segment_ids(t, gap_seconds=60)
    assert labels.tolist() == [0, 0, 0, 1, 1, 1]


def test_segment_ids_no_split_when_dense():
    t = np.array([0, 10, 20, 30, 40])
    assert segment_ids(t, gap_seconds=60).tolist() == [0, 0, 0, 0, 0]


def test_segment_ids_empty():
    assert segment_ids(np.array([]), 60).tolist() == []


# --- interp_linear ---------------------------------------------------------
def test_interp_linear_basic():
    t = np.array([0.0, 10.0])
    v = np.array([0.0, 100.0])
    out = interp_linear(np.array([0.0, 5.0, 10.0]), t, v)
    assert np.allclose(out, [0.0, 50.0, 100.0])


def test_interp_linear_all_nan_when_too_few_samples():
    t = np.array([0.0, 10.0])
    v = np.array([np.nan, 5.0])  # only 1 valid sample
    out = interp_linear(np.array([0.0, 5.0]), t, v)
    assert np.isnan(out).all()


# --- interp_angular (wraparound) ------------------------------------------
def test_interp_angular_handles_wraparound():
    # 350° -> 10° should pass through 0/360, not 180.
    t = np.array([0.0, 10.0])
    deg = np.array([350.0, 10.0])
    mid = interp_angular(np.array([5.0]), t, deg)[0]
    assert min(mid % 360.0, 360.0 - (mid % 360.0)) < 1.0  # ~0°, not ~180°


# --- resample_segment ------------------------------------------------------
def test_resample_segment_regular_grid():
    t = np.array([0, 10, 20, 30], dtype=float)
    cols = {c: np.full(4, 1.0) for c in _POINT_COLS}
    cols["lat"] = np.array([50.0, 50.1, 50.2, 50.3])
    seg = resample_segment(t, cols, dt=10)
    assert seg is not None
    assert seg.grid.tolist() == [0, 10, 20, 30]
    assert np.allclose(np.diff(seg.grid), 10)
    assert np.isclose(seg.columns["lat"][1], 50.1)


def test_resample_segment_too_short_returns_none():
    t = np.array([0.0, 5.0])
    cols = {c: np.full(2, 1.0) for c in _POINT_COLS}
    assert resample_segment(t, cols, dt=10) is None


# --- reconstruct_aircraft (end to end, in memory) --------------------------
def test_reconstruct_aircraft_splits_and_does_not_bridge_gaps():
    # Segment A: 0..200, then a 600s gap (> gap_seconds), Segment B: 800..1000.
    times = list(range(0, 201, 10)) + list(range(800, 1001, 10))
    df = _aircraft_frame(times)
    segs = list(reconstruct_aircraft(df, gap_seconds=60, dt=10, min_points=5))

    assert len(segs) == 2
    a, b = segs
    # No resampled point lands inside the real gap (200, 800): gaps stay gaps.
    assert a.grid.max() <= 200 and b.grid.min() >= 800
    for seg in segs:
        assert not np.any((seg.grid > 200) & (seg.grid < 800))


def test_reconstruct_aircraft_min_points_filter():
    times = list(range(0, 201, 10))  # 21 points spanning 200s
    df = _aircraft_frame(times)
    assert list(reconstruct_aircraft(df, gap_seconds=60, dt=10, min_points=5))
    # Require more points than exist -> nothing yielded.
    assert not list(reconstruct_aircraft(df, gap_seconds=60, dt=10, min_points=1000))


def test_reconstruct_aircraft_dedups_timestamps():
    times = [0, 0, 10, 10, 20, 30, 40]  # duplicate timestamps
    df = _aircraft_frame(times)
    segs = list(reconstruct_aircraft(df, gap_seconds=60, dt=10, min_points=2))
    assert len(segs) == 1
    # Grid is strictly increasing by dt (no duplicate-time artifacts).
    assert np.all(np.diff(segs[0].grid) == 10)
