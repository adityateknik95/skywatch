"""Deterministic physics / kinematic plausibility rules (spec §10).

Runs alongside the ML model. Anything it catches is flagged with
``reason=physics:<rule>`` regardless of what the model says — this guarantees
*impossible* kinematics are caught even before (or independent of) the model being
good. The model handles the subtle, learned-normal stuff; physics handles the
"no aircraft can do that" stuff.

Rules operate on the same per-point feature rows produced by
:func:`skywatch.features.compute_trajectory_features` (raw, unscaled), so they share
the computed-vs-reported quantities. NaNs (e.g. missing geo_altitude) never trip a
rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from skywatch.features import FEATURE_NAMES


@dataclass(frozen=True)
class PhysicsLimits:
    # Generous civil-aviation envelopes — meant to catch the impossible, not the merely unusual.
    max_groundspeed_ms: float = 340.0        # ~Mach 1 over ground
    max_vertical_rate_ms: float = 50.0       # ~10,000 ft/min
    max_turn_rate_dps: float = 12.0          # deg/s (standard rate is 3)
    max_accel_ms2: float = 10.0              # ~1 g sustained
    max_speed_residual_ms: float = 150.0     # reported speed vs position-implied
    max_baro_geo_diff_m: float = 1500.0      # baro vs geo altitude mismatch


DEFAULT_LIMITS = PhysicsLimits()

# Each rule: name -> (feature, test). Test takes the raw column, returns bool mask.
def _rules(limits: PhysicsLimits):
    return [
        ("physics:speed", "computed_groundspeed", lambda v: v > limits.max_groundspeed_ms),
        ("physics:climb", "computed_vertical_rate", lambda v: np.abs(v) > limits.max_vertical_rate_ms),
        ("physics:turn", "heading_change_rate", lambda v: np.abs(v) > limits.max_turn_rate_dps),
        ("physics:accel", "accel", lambda v: np.abs(v) > limits.max_accel_ms2),
        ("physics:velpos", "speed_residual", lambda v: v > limits.max_speed_residual_ms),
        ("physics:altdiff", "baro_geo_alt_diff", lambda v: np.abs(v) > limits.max_baro_geo_diff_m),
    ]


def physics_flags(
    X: np.ndarray,
    feature_names: list[str] = FEATURE_NAMES,
    limits: PhysicsLimits = DEFAULT_LIMITS,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate hard rules over raw feature rows.

    Args:
        X: (N, F) raw (unscaled) feature matrix in ``feature_names`` order.
    Returns:
        (flagged, reasons): ``flagged`` is a bool (N,) array; ``reasons`` is an
        object (N,) array with the first rule that fired (or "").
    """
    idx = {n: i for i, n in enumerate(feature_names)}
    n = X.shape[0]
    flagged = np.zeros(n, dtype=bool)
    reasons = np.full(n, "", dtype=object)
    for name, feature, test in _rules(limits):
        col = X[:, idx[feature]]
        with np.errstate(invalid="ignore"):
            hit = test(col)
        hit = np.asarray(hit, dtype=bool) & ~np.isnan(col)
        new = hit & ~flagged
        reasons[new] = name
        flagged |= hit
    return flagged, reasons


def check_row(row: dict, limits: PhysicsLimits = DEFAULT_LIMITS) -> str | None:
    """Convenience single-row check: returns the first rule fired, or None."""
    X = np.array([[row.get(n, np.nan) for n in FEATURE_NAMES]], dtype=float)
    flagged, reasons = physics_flags(X, limits=limits)
    return reasons[0] if flagged[0] else None
