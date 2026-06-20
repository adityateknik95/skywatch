"""Feature-math tests (spec §14): haversine distance, bearing, turn-rate wraparound,
and the consistency residuals. No DB required."""

from __future__ import annotations

import numpy as np

from skywatch.features import (
    FEATURE_NAMES,
    angular_diff_deg,
    compute_trajectory_features,
    encode_track,
    haversine_m,
    initial_bearing_deg,
)


# --- haversine -------------------------------------------------------------
def test_haversine_one_degree_latitude():
    # 1° of latitude is ~111.19 km on a sphere of radius 6371 km.
    d = haversine_m(0.0, 0.0, 1.0, 0.0)
    assert abs(d - 111195.0) < 50.0


def test_haversine_zero():
    assert haversine_m(50.0, 8.0, 50.0, 8.0) == 0.0


def test_haversine_symmetric_and_vectorized():
    a = haversine_m([50.0, 51.0], [8.0, 9.0], [50.1, 51.1], [8.1, 9.1])
    b = haversine_m([50.1, 51.1], [8.1, 9.1], [50.0, 51.0], [8.0, 9.0])
    assert np.allclose(a, b)
    assert a.shape == (2,)


# --- bearing ---------------------------------------------------------------
def test_bearing_north_and_east():
    assert abs(initial_bearing_deg(0.0, 0.0, 1.0, 0.0) - 0.0) < 1e-6     # due north
    assert abs(initial_bearing_deg(0.0, 0.0, 0.0, 1.0) - 90.0) < 1e-6    # due east


# --- angular diff / wraparound --------------------------------------------
def test_angular_diff_wraparound():
    assert abs(angular_diff_deg(10.0, 350.0) - 20.0) < 1e-9    # +20, not -340
    assert abs(angular_diff_deg(350.0, 10.0) + 20.0) < 1e-9    # -20
    assert abs(abs(angular_diff_deg(0.0, 180.0)) - 180.0) < 1e-9


def test_encode_track():
    s0, c0 = encode_track(0.0)
    assert abs(s0 - 0.0) < 1e-9 and abs(c0 - 1.0) < 1e-9
    s90, c90 = encode_track(90.0)
    assert abs(s90 - 1.0) < 1e-9 and abs(c90 - 0.0) < 1e-9


# --- turn rate (heading_change_rate) wraparound ---------------------------
def _features_of(**arrays):
    dt = arrays.pop("dt", 15.0)
    n = len(arrays["lat"])
    base = {
        "baro": np.zeros(n), "geo": np.zeros(n),
        "velocity": np.zeros(n), "vertical_rate": np.zeros(n),
        "true_track": np.zeros(n),
    }
    base.update(arrays)
    return compute_trajectory_features(
        base["lat"], base["lon"], base["baro"], base["geo"],
        base["velocity"], base["true_track"], base["vertical_rate"], dt,
    )


def test_turn_rate_uses_shortest_arc():
    idx = FEATURE_NAMES.index("heading_change_rate")
    # 350° -> 10° over dt=10s is a +20° turn, i.e. +2°/s (not -34°/s).
    feats = _features_of(
        lat=[50.0, 50.0], lon=[8.0, 8.0], true_track=[350.0, 10.0], dt=10.0
    )
    assert abs(feats[0, idx] - 2.0) < 1e-6


# --- consistency residuals on a self-consistent flight --------------------
def test_self_consistent_flight_has_small_residuals():
    """An eastbound flight whose reported velocity/track match its motion should
    have ~zero speed/heading residuals."""
    dt = 15.0
    lat = np.array([50.0, 50.0, 50.0])
    lon = np.array([8.0, 8.01, 8.02])
    # reported velocity = the actual groundspeed implied by the positions
    gs = haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:]) / dt
    velocity = np.array([gs[0], gs[0], gs[1]])
    true_track = np.array([90.0, 90.0, 90.0])  # due east, matches motion
    feats = compute_trajectory_features(
        lat, lon, np.zeros(3), np.zeros(3), velocity, true_track, np.zeros(3), dt
    )
    sr = FEATURE_NAMES.index("speed_residual")
    hr = FEATURE_NAMES.index("heading_residual")
    assert np.all(feats[:, sr] < 1.0)      # m/s
    assert np.all(feats[:, hr] < 1.0)      # degrees


def test_speed_residual_flags_inconsistency():
    """If reported velocity disagrees with position-derived speed, residual is large."""
    dt = 15.0
    lat = np.array([50.0, 50.0])
    lon = np.array([8.0, 8.05])  # a real ~3.5 km hop -> ~230 m/s
    velocity = np.array([0.0, 0.0])  # spoofer reports stationary
    feats = compute_trajectory_features(
        lat, lon, np.zeros(2), np.zeros(2), velocity, np.array([90.0, 90.0]),
        np.zeros(2), dt,
    )
    sr = FEATURE_NAMES.index("speed_residual")
    assert feats[0, sr] > 100.0


def test_vertical_rate_residual_and_alt_diff():
    dt = 15.0
    baro = np.array([10000.0, 10150.0])   # +150 m in 15s -> +10 m/s climb
    geo = np.array([10050.0, 10200.0])
    feats = compute_trajectory_features(
        np.array([50.0, 50.0]), np.array([8.0, 8.0]), baro, geo,
        np.zeros(2), np.zeros(2),
        vertical_rate=np.array([0.0, 0.0]),  # reports level flight -> inconsistent
        dt=dt,
    )
    cvr = FEATURE_NAMES.index("computed_vertical_rate")
    vrr = FEATURE_NAMES.index("vertical_rate_residual")
    diff = FEATURE_NAMES.index("baro_geo_alt_diff")
    assert abs(feats[0, cvr] - 10.0) < 1e-6
    assert abs(feats[0, vrr] - 10.0) < 1e-6
    assert abs(feats[0, diff] - (-50.0)) < 1e-6  # 10150 - 10200


def test_first_point_dropped():
    feats = compute_trajectory_features(
        np.array([50.0, 50.1, 50.2]), np.array([8.0, 8.0, 8.0]),
        np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), 15.0
    )
    assert feats.shape == (2, len(FEATURE_NAMES))  # 3 points -> 2 feature rows
