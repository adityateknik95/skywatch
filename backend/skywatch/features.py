"""Feature engineering (Phase 3, spec §9).

Turns ``trajectory_points`` (fixed-``dt`` resampled flight segments) into a per-point
feature matrix. The *derived* features are where the spoofing signal lives: a
self-consistent real aircraft has computed-vs-reported quantities that agree;
spoofers usually get the self-consistency wrong.

Per point (current point = the one being described; deltas use the previous point):

  raw:      latitude, longitude, baro_altitude, geo_altitude, velocity,
            vertical_rate, track_sin, track_cos
  derived:  computed_groundspeed   = haversine(prev,cur)/dt
            speed_residual         = |computed_groundspeed - reported velocity|
            heading_change_rate    = Δtrue_track / dt        (turn rate, wrap-safe)
            heading_residual        = |bearing(prev,cur) - reported true_track|
            accel                  = Δvelocity / dt
            computed_vertical_rate = Δbaro_altitude / dt
            vertical_rate_residual = |computed_vertical_rate - reported vertical_rate|
            baro_geo_alt_diff      = baro_altitude - geo_altitude

The first point of each trajectory has no predecessor, so it is dropped. The scaler
(median-impute → standardize) is fit on this normal data and persisted alongside the
feature names so Phase 4 / serving use the exact same transform.

CLI::

    python -m skywatch.features            # build matrix from trajectory_points, fit+save scaler
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import select

from skywatch.db.models import Trajectory, TrajectoryPoint
from skywatch.db.session import dispose_engine, session_scope

log = logging.getLogger("skywatch.features")

EARTH_RADIUS_M = 6_371_000.0

# Feature order is part of the contract — persisted and reused everywhere.
FEATURE_NAMES: list[str] = [
    "latitude",
    "longitude",
    "baro_altitude",
    "geo_altitude",
    "velocity",
    "vertical_rate",
    "track_sin",
    "track_cos",
    "computed_groundspeed",
    "speed_residual",
    "heading_change_rate",
    "heading_residual",
    "accel",
    "computed_vertical_rate",
    "vertical_rate_residual",
    "baro_geo_alt_diff",
]

ARTIFACTS_DIR = Path(__file__).resolve().parent / "model" / "artifacts"
SCALER_PATH = ARTIFACTS_DIR / "scaler.joblib"
FEATURE_META_PATH = ARTIFACTS_DIR / "feature_meta.json"
FEATURE_MATRIX_PATH = ARTIFACTS_DIR / "features.npz"


# --------------------------------------------------------------------------- #
# Geo / angle math (unit-tested in tests/test_features.py)
# --------------------------------------------------------------------------- #
def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres between (lat,lon) points in degrees."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype=float))
                              for x in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def initial_bearing_deg(lat1, lon1, lat2, lon2):
    """Initial great-circle bearing from point 1 to point 2, degrees in [0,360)."""
    lat1, lat2 = np.radians(np.asarray(lat1, float)), np.radians(np.asarray(lat2, float))
    dlon = np.radians(np.asarray(lon2, float) - np.asarray(lon1, float))
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.mod(np.degrees(np.arctan2(x, y)), 360.0)


def angular_diff_deg(a, b):
    """Smallest signed difference a-b, wrapped to (-180, 180]."""
    return (np.asarray(a, float) - np.asarray(b, float) + 180.0) % 360.0 - 180.0


def encode_track(deg):
    """Heading in degrees -> (sin, cos) so 359°->1° isn't a discontinuity."""
    rad = np.radians(np.asarray(deg, float))
    return np.sin(rad), np.cos(rad)


# --------------------------------------------------------------------------- #
# Per-trajectory feature computation
# --------------------------------------------------------------------------- #
def compute_trajectory_features(
    lat, lon, baro, geo, velocity, true_track, vertical_rate, dt: float
) -> np.ndarray:
    """Return an (n-1, len(FEATURE_NAMES)) feature array for one trajectory.

    Arrays are the resampled per-point values ordered by time; the first point is
    dropped because its deltas are undefined. ``dt`` is the grid spacing in seconds.
    """
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    baro = np.asarray(baro, float)
    geo = np.asarray(geo, float)
    velocity = np.asarray(velocity, float)
    true_track = np.asarray(true_track, float)
    vertical_rate = np.asarray(vertical_rate, float)

    n = lat.shape[0]
    if n < 2:
        return np.empty((0, len(FEATURE_NAMES)), dtype=float)

    prev = slice(0, n - 1)
    cur = slice(1, n)

    gc = haversine_m(lat[prev], lon[prev], lat[cur], lon[cur])
    computed_gs = gc / dt
    speed_residual = np.abs(computed_gs - velocity[cur])

    bearing = initial_bearing_deg(lat[prev], lon[prev], lat[cur], lon[cur])
    heading_residual = np.abs(angular_diff_deg(bearing, true_track[cur]))
    heading_change_rate = angular_diff_deg(true_track[cur], true_track[prev]) / dt

    accel = (velocity[cur] - velocity[prev]) / dt
    computed_vr = (baro[cur] - baro[prev]) / dt
    vertical_rate_residual = np.abs(computed_vr - vertical_rate[cur])

    baro_geo_alt_diff = baro[cur] - geo[cur]
    track_sin, track_cos = encode_track(true_track[cur])

    return np.column_stack(
        [
            lat[cur],
            lon[cur],
            baro[cur],
            geo[cur],
            velocity[cur],
            vertical_rate[cur],
            track_sin,
            track_cos,
            computed_gs,
            speed_residual,
            heading_change_rate,
            heading_residual,
            accel,
            computed_vr,
            vertical_rate_residual,
            baro_geo_alt_diff,
        ]
    )


@dataclass
class FeatureMatrix:
    X: np.ndarray            # (N, F) unscaled features
    trajectory_id: np.ndarray  # (N,) which trajectory each row belongs to
    t: np.ndarray            # (N,) epoch second of each row
    feature_names: list[str]


# --------------------------------------------------------------------------- #
# Build from DB
# --------------------------------------------------------------------------- #
async def _load_points(session) -> pd.DataFrame:
    stmt = (
        select(
            TrajectoryPoint.trajectory_id,
            TrajectoryPoint.t,
            TrajectoryPoint.lat,
            TrajectoryPoint.lon,
            TrajectoryPoint.baro_altitude,
            TrajectoryPoint.geo_altitude,
            TrajectoryPoint.velocity,
            TrajectoryPoint.true_track,
            TrajectoryPoint.vertical_rate,
            Trajectory.dt_seconds,
        )
        .join(Trajectory, Trajectory.id == TrajectoryPoint.trajectory_id)
        .order_by(TrajectoryPoint.trajectory_id, TrajectoryPoint.t)
    )
    rows = (await session.execute(stmt)).all()
    cols = ["trajectory_id", "t", "lat", "lon", "baro_altitude", "geo_altitude",
            "velocity", "true_track", "vertical_rate", "dt_seconds"]
    return pd.DataFrame(rows, columns=cols)


async def build_feature_matrix() -> FeatureMatrix:
    async with session_scope() as session:
        df = await _load_points(session)

    if df.empty:
        return FeatureMatrix(
            np.empty((0, len(FEATURE_NAMES))), np.empty(0, np.int64),
            np.empty(0, np.int64), FEATURE_NAMES,
        )

    blocks, tids, ts = [], [], []
    for tid, g in df.groupby("trajectory_id", sort=False):
        dt = float(g["dt_seconds"].iloc[0])
        feats = compute_trajectory_features(
            g["lat"].to_numpy(), g["lon"].to_numpy(), g["baro_altitude"].to_numpy(),
            g["geo_altitude"].to_numpy(), g["velocity"].to_numpy(),
            g["true_track"].to_numpy(), g["vertical_rate"].to_numpy(), dt,
        )
        if feats.shape[0] == 0:
            continue
        blocks.append(feats)
        tids.append(np.full(feats.shape[0], tid, dtype=np.int64))
        ts.append(g["t"].to_numpy(dtype=np.int64)[1:])

    X = np.vstack(blocks).astype(np.float64)
    return FeatureMatrix(X, np.concatenate(tids), np.concatenate(ts), FEATURE_NAMES)


# --------------------------------------------------------------------------- #
# Scaler (median-impute + standardize), fit on normal data and persisted
# --------------------------------------------------------------------------- #
def fit_scaler(X: np.ndarray):
    """Fit a median-impute → standardize pipeline on normal feature data."""
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]
    )
    pipe.fit(X)
    return pipe


def save_artifacts(fm: FeatureMatrix, scaler) -> None:
    import joblib

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    np.savez_compressed(
        FEATURE_MATRIX_PATH,
        X=fm.X.astype(np.float32),
        trajectory_id=fm.trajectory_id,
        t=fm.t,
    )
    FEATURE_META_PATH.write_text(
        json.dumps(
            {
                "feature_names": fm.feature_names,
                "n_features": len(fm.feature_names),
                "n_rows": int(fm.X.shape[0]),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_scaler():
    import joblib

    return joblib.load(SCALER_PATH)


async def _amain() -> None:
    log.info("Building feature matrix from trajectory_points...")
    fm = await build_feature_matrix()
    if fm.X.shape[0] == 0:
        print("No features produced — run `python -m skywatch.reconstruct` first.")
        await dispose_engine()
        return

    log.info("Feature matrix: %d rows x %d features", fm.X.shape[0], fm.X.shape[1])
    scaler = fit_scaler(fm.X)
    save_artifacts(fm, scaler)

    # Quick sanity summary (on raw features).
    finite = np.where(np.isfinite(fm.X), fm.X, np.nan)
    means = np.nanmean(finite, axis=0)
    print(f"Feature matrix: {fm.X.shape[0]} rows x {fm.X.shape[1]} features "
          f"from {np.unique(fm.trajectory_id).size} trajectories")
    print(f"Saved scaler -> {SCALER_PATH}")
    print(f"Saved matrix -> {FEATURE_MATRIX_PATH}")
    print("\nPer-feature mean (raw):")
    for name, m in zip(fm.feature_names, means):
        print(f"  {name:24s} {m:12.3f}")
    await dispose_engine()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build feature matrix + fit scaler")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
