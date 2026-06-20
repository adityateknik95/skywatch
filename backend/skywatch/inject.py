"""Synthetic attack generators for evaluation (spec §8).

There is no labeled real spoofing data, so we *generate* attacks ourselves and check
the detector against them. Each generator takes a clean trajectory (dict of numpy
arrays: lat, lon, baro, geo, velocity, true_track, vertical_rate) and returns a
modified copy plus a per-point boolean label array (True = injected anomaly).

The set spans the difficulty spectrum on purpose:
  * kinematic / consistency attacks (teleport, impossible kinematics, velocity–position,
    altitude) — should be caught by physics and/or the model;
  * gradual drift — deliberately subtle, the documented weak spot;
  * ghost & replay — *plausible motion* by construction, so a motion-only model can't
    catch them (you'd need cross-receiver TDOA). Their poor scores are an honest result.
"""

from __future__ import annotations

import numpy as np

M_PER_DEG_LAT = 111_320.0
TRAJ_KEYS = ("lat", "lon", "baro", "geo", "velocity", "true_track", "vertical_rate")


def _copy(traj: dict) -> dict:
    return {k: np.asarray(traj[k], dtype=float).copy() for k in TRAJ_KEYS}


def _pick(rng: np.random.Generator, n: int, lo: float = 0.25, hi: float = 0.6) -> int:
    a = max(1, int(n * lo))
    b = max(a + 1, int(n * hi))
    return int(rng.integers(a, b))


def teleport(traj, rng, dt, jump_deg: float = 0.5):
    """Position jump: shift the rest of the flight ~55 km instantly."""
    tr = _copy(traj)
    n = len(tr["lat"])
    lab = np.zeros(n, dtype=bool)
    k = _pick(rng, n)
    tr["lat"][k:] += jump_deg
    lab[k] = True  # the discontinuity point
    return tr, lab


def impossible_kinematics(traj, rng, dt, climb_ms: float = 120.0, span: int = 5):
    """Impossible climb rate (well beyond any airframe)."""
    tr = _copy(traj)
    n = len(tr["lat"])
    lab = np.zeros(n, dtype=bool)
    span = min(span, max(1, n - 2))
    k = min(_pick(rng, n), n - span - 1)
    for j in range(span + 1):
        tr["baro"][k + j] = tr["baro"][k] + climb_ms * dt * j
    tr["vertical_rate"][k : k + span + 1] = climb_ms
    lab[k + 1 : k + span + 1] = True
    return tr, lab


def gradual_drift(traj, rng, dt, drift_ms: float = 2.5):
    """Slow positional drift away from the true track — subtle spoofing (hard case)."""
    tr = _copy(traj)
    n = len(tr["lat"])
    lab = np.zeros(n, dtype=bool)
    k = _pick(rng, n, lo=0.2, hi=0.4)
    per_step_deg = (drift_ms * dt) / M_PER_DEG_LAT
    for j, i in enumerate(range(k, n)):
        tr["lat"][i] += per_step_deg * (j + 1)
    lab[k:] = True
    return tr, lab


def velocity_position(traj, rng, dt, factor: float = 2.5, span: int = 8):
    """Reported velocity inconsistent with the position deltas."""
    tr = _copy(traj)
    n = len(tr["lat"])
    lab = np.zeros(n, dtype=bool)
    span = min(span, max(1, n - 2))
    k = min(_pick(rng, n), n - span - 1)
    tr["velocity"][k : k + span] *= factor
    lab[k : k + span] = True
    return tr, lab


def altitude_spoof(traj, rng, dt, offset_m: float = 3000.0, span: int = 8):
    """Baro altitude offset from geo altitude (altitude spoofing)."""
    tr = _copy(traj)
    n = len(tr["lat"])
    lab = np.zeros(n, dtype=bool)
    span = min(span, max(1, n - 2))
    k = min(_pick(rng, n), n - span - 1)
    tr["baro"][k : k + span] += offset_m
    lab[k : k + span] = True
    return tr, lab


def ghost(traj, rng, dt, cruise_ms: float = 235.0, alt_m: float = 10500.0):
    """Fabricated aircraft on a plausible straight cruise — normal motion by design."""
    n = len(traj["lat"])
    lat0 = rng.uniform(47.5, 54.5)
    lon0 = rng.uniform(5.5, 14.5)
    hdg = rng.uniform(0.0, 360.0)
    step_m = cruise_ms * dt
    dlat = step_m * np.cos(np.radians(hdg)) / M_PER_DEG_LAT
    dlon = step_m * np.sin(np.radians(hdg)) / (M_PER_DEG_LAT * np.cos(np.radians(lat0)))
    i = np.arange(n)
    noise = rng.normal(0.0, 2.0, n)
    tr = {
        "lat": lat0 + dlat * i,
        "lon": lon0 + dlon * i,
        "baro": np.full(n, alt_m) + noise,
        "geo": np.full(n, alt_m + 50.0) + noise,
        "velocity": np.full(n, cruise_ms),
        "true_track": np.full(n, hdg % 360.0),
        "vertical_rate": np.zeros(n),
    }
    return tr, np.ones(n, dtype=bool)


def replay(traj, rng, dt):
    """Clone a real trajectory to a new place/time — motion is identical (normal)."""
    tr = _copy(traj)
    n = len(tr["lat"])
    tr["lat"] += rng.uniform(-1.5, 1.5)
    tr["lon"] += rng.uniform(-2.5, 2.5)
    return tr, np.ones(n, dtype=bool)


ATTACKS: dict[str, callable] = {
    "teleport": teleport,
    "impossible_kinematics": impossible_kinematics,
    "gradual_drift": gradual_drift,
    "velocity_position": velocity_position,
    "altitude_spoof": altitude_spoof,
    "ghost": ghost,
    "replay": replay,
}
