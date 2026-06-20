"""Injection tests (spec §14): generators are well-formed and labeled, the teleport
round-trip is recoverable, and the 'hard' attacks (drift/ghost) are — by design —
not caught by physics alone."""

from __future__ import annotations

import numpy as np

from skywatch.features import compute_trajectory_features
from skywatch.inject import ATTACKS, TRAJ_KEYS, teleport
from skywatch.physics import physics_flags

DT = 15.0


def _clean(n: int = 60):
    """A straight, self-consistent eastbound cruise."""
    cruise = 230.0
    step_deg = cruise * DT / (111_320.0 * np.cos(np.radians(50.0)))
    return {
        "lat": np.full(n, 50.0),
        "lon": 8.0 + step_deg * np.arange(n),
        "baro": np.full(n, 10_000.0),
        "geo": np.full(n, 10_050.0),
        "velocity": np.full(n, cruise),
        "true_track": np.full(n, 90.0),
        "vertical_rate": np.zeros(n),
    }


def _physics_on(traj):
    X = compute_trajectory_features(
        traj["lat"], traj["lon"], traj["baro"], traj["geo"],
        traj["velocity"], traj["true_track"], traj["vertical_rate"], DT,
    )
    return X, physics_flags(X)[0]


def test_clean_trajectory_has_no_physics_flags():
    _, flags = _physics_on(_clean())
    assert not flags.any()


def test_all_generators_well_formed():
    rng = np.random.default_rng(0)
    clean = _clean()
    for name, attack in ATTACKS.items():
        tr, lab = attack(clean, rng, DT)
        assert set(TRAJ_KEYS).issubset(tr.keys()), name
        n = len(clean["lat"])
        assert all(len(tr[k]) == n for k in TRAJ_KEYS), name
        assert lab.shape == (n,) and lab.any(), name


def test_teleport_roundtrip_is_recoverable():
    """The injected teleport point shows up as an impossible speed (physics:speed)."""
    rng = np.random.default_rng(3)
    tr, lab = teleport(_clean(), rng, DT, jump_deg=0.5)
    X, flags = _physics_on(tr)
    lab_feat = lab[1:]  # features drop the first point
    # the labeled jump point is flagged by physics
    assert flags[lab_feat].any()


def test_velocity_position_is_inconsistent():
    rng = np.random.default_rng(4)
    tr, lab = ATTACKS["velocity_position"](_clean(), rng, DT)
    X, _ = _physics_on(tr)
    from skywatch.features import FEATURE_NAMES
    sr = FEATURE_NAMES.index("speed_residual")
    lab_feat = lab[1:]
    assert np.nanmax(X[lab_feat, sr]) > 100.0  # reported speed disagrees with motion


def test_altitude_spoof_shows_baro_geo_mismatch():
    rng = np.random.default_rng(5)
    tr, lab = ATTACKS["altitude_spoof"](_clean(), rng, DT)
    X, _ = _physics_on(tr)
    from skywatch.features import FEATURE_NAMES
    d = FEATURE_NAMES.index("baro_geo_alt_diff")
    lab_feat = lab[1:]
    assert np.nanmax(np.abs(X[lab_feat, d])) > 1500.0


def test_ghost_motion_is_plausible():
    """A ghost flies a normal-looking path -> physics rarely fires (the hard case)."""
    rng = np.random.default_rng(6)
    tr, lab = ATTACKS["ghost"](_clean(), rng, DT)
    _, flags = _physics_on(tr)
    assert flags.mean() < 0.1


def test_gradual_drift_is_subtle():
    """Drift stays under the physics envelope -> not caught by hard rules."""
    rng = np.random.default_rng(7)
    tr, lab = ATTACKS["gradual_drift"](_clean(), rng, DT)
    _, flags = _physics_on(tr)
    assert flags.mean() < 0.1
