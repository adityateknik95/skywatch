"""Physics rule tests (spec §14): each rule fires on a crafted violating point and
not on a normal one; NaNs never trip a rule."""

from __future__ import annotations

import numpy as np

from skywatch.features import FEATURE_NAMES
from skywatch.physics import check_row, physics_flags


def _row(**overrides) -> dict:
    """A self-consistent, well-within-envelope point."""
    base = {n: 0.0 for n in FEATURE_NAMES}
    base.update(
        latitude=50.0, longitude=8.0, baro_altitude=10000.0, geo_altitude=10300.0,
        velocity=230.0, track_cos=1.0,
        computed_groundspeed=230.0, speed_residual=3.0,
        heading_change_rate=1.0, heading_residual=2.0, accel=0.5,
        computed_vertical_rate=5.0, vertical_rate_residual=0.5,
        baro_geo_alt_diff=-300.0,
    )
    base.update(overrides)
    return base


def test_normal_point_not_flagged():
    assert check_row(_row()) is None


def test_each_rule_fires():
    assert check_row(_row(computed_groundspeed=400.0)) == "physics:speed"
    assert check_row(_row(computed_vertical_rate=120.0)) == "physics:climb"
    assert check_row(_row(heading_change_rate=20.0)) == "physics:turn"
    assert check_row(_row(accel=15.0)) == "physics:accel"
    assert check_row(_row(speed_residual=200.0)) == "physics:velpos"
    assert check_row(_row(baro_geo_alt_diff=3000.0)) == "physics:altdiff"


def test_negative_extremes_also_fire():
    assert check_row(_row(computed_vertical_rate=-120.0)) == "physics:climb"
    assert check_row(_row(heading_change_rate=-20.0)) == "physics:turn"
    assert check_row(_row(baro_geo_alt_diff=-3000.0)) == "physics:altdiff"


def test_nan_does_not_fire():
    assert check_row(_row(baro_geo_alt_diff=np.nan)) is None


def test_physics_flags_vectorized():
    X = np.array(
        [[_row()[n] for n in FEATURE_NAMES],
         [_row(computed_groundspeed=500.0)[n] for n in FEATURE_NAMES]],
        dtype=float,
    )
    flagged, reasons = physics_flags(X)
    assert flagged.tolist() == [False, True]
    assert reasons[1] == "physics:speed"
