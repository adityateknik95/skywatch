"""Reconstruct per-aircraft flight trajectories from ``raw_states`` (Phase 2).

Pipeline (spec §13):
  1. Load airborne, positioned rows; time = coalesce(time_position, last_contact).
  2. Per aircraft: order by time, drop duplicate timestamps.
  3. Split into segments wherever the time gap exceeds ``gap_seconds`` — that gap is
     a receiver dropout / left-the-bbox boundary, and we never interpolate across it
     ("treat gaps as gaps, not jumps").
  4. Resample each segment onto a fixed ``dt`` grid (linear interp for
     position/altitude/speed; angular interp for heading so 359°->1° isn't a cliff).
  5. Drop fragments shorter than ``min_points``; write ``trajectories`` +
     ``trajectory_points``.

Reconstruction is derived data: by default it truncates and rebuilds both tables.

CLI::

    python -m skywatch.reconstruct                 # full rebuild with defaults
    python -m skywatch.reconstruct --dt 10 --gap-seconds 90 --min-points 20
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
from sqlalchemy import func, insert, or_, select, text

from skywatch.db.models import RawState, Trajectory, TrajectoryPoint
from skywatch.db.session import dispose_engine, session_scope

log = logging.getLogger("skywatch.reconstruct")

# Defaults (overridable via CLI). See module docstring for meaning.
DEFAULT_GAP_SECONDS = 60
DEFAULT_DT_SECONDS = 15
DEFAULT_MIN_POINTS = 10

# Numeric columns interpolated linearly; heading is handled separately (angular).
_LINEAR_COLS = ("lat", "lon", "baro_altitude", "geo_altitude", "velocity", "vertical_rate")
_POINT_COLS = (*_LINEAR_COLS, "true_track")


# --------------------------------------------------------------------------- #
# Pure helpers (no DB) — unit-tested in tests/test_reconstruct.py
# --------------------------------------------------------------------------- #
def segment_ids(times: np.ndarray, gap_seconds: float) -> np.ndarray:
    """Label points 0,1,2,... starting a new segment after a gap > ``gap_seconds``."""
    times = np.asarray(times, dtype=float)
    if times.size == 0:
        return np.zeros(0, dtype=int)
    breaks = np.diff(times) > gap_seconds
    return np.concatenate(([0], np.cumsum(breaks))).astype(int)


def interp_linear(grid: np.ndarray, t: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Linear interpolation onto ``grid`` using only the non-NaN samples.

    Returns all-NaN if fewer than 2 samples are available (column unreported).
    """
    values = np.asarray(values, dtype=float)
    mask = ~np.isnan(values)
    if mask.sum() < 2:
        return np.full(grid.shape, np.nan)
    return np.interp(grid, t[mask], values[mask])


def interp_angular(grid: np.ndarray, t: np.ndarray, degrees: np.ndarray) -> np.ndarray:
    """Interpolate a heading in degrees without the 359°->0° discontinuity."""
    degrees = np.asarray(degrees, dtype=float)
    mask = ~np.isnan(degrees)
    if mask.sum() < 2:
        return np.full(grid.shape, np.nan)
    unwrapped = np.unwrap(np.deg2rad(degrees[mask]))
    out = np.interp(grid, t[mask], unwrapped)
    return np.mod(np.rad2deg(out), 360.0)


@dataclass
class Segment:
    start_time: int
    end_time: int
    grid: np.ndarray                 # epoch seconds (int)
    columns: dict[str, np.ndarray]   # column name -> resampled values

    @property
    def point_count(self) -> int:
        return int(self.grid.size)


def resample_segment(
    t: np.ndarray, cols: dict[str, np.ndarray], dt: int
) -> Segment | None:
    """Resample one segment's observed points onto a fixed ``dt`` grid.

    Returns ``None`` if the segment spans less than one ``dt`` step.
    """
    t = np.asarray(t, dtype=float)
    t0, t1 = float(t[0]), float(t[-1])
    if t1 - t0 < dt:
        return None
    grid = np.arange(t0, t1 + 1e-6, dt)

    out: dict[str, np.ndarray] = {}
    for name in _LINEAR_COLS:
        out[name] = interp_linear(grid, t, cols[name])
    out["true_track"] = interp_angular(grid, t, cols["true_track"])

    grid_int = grid.astype(np.int64)
    return Segment(int(grid_int[0]), int(grid_int[-1]), grid_int, out)


def reconstruct_aircraft(
    df: pd.DataFrame, *, gap_seconds: int, dt: int, min_points: int
) -> Iterator[Segment]:
    """Yield resampled :class:`Segment`s for one aircraft's rows.

    ``df`` must contain column ``t`` plus the point columns, for a single icao24.
    """
    df = df.drop_duplicates(subset="t", keep="last").sort_values("t")
    if len(df) < 2:
        return
    t = df["t"].to_numpy(dtype=float)
    labels = segment_ids(t, gap_seconds)
    cols = {c: df[c].to_numpy(dtype=float) for c in _POINT_COLS}

    for seg_id in np.unique(labels):
        idx = labels == seg_id
        if idx.sum() < 2:
            continue
        seg_cols = {c: v[idx] for c, v in cols.items()}
        seg = resample_segment(t[idx], seg_cols, dt)
        if seg is not None and seg.point_count >= min_points:
            yield seg


# --------------------------------------------------------------------------- #
# DB orchestration
# --------------------------------------------------------------------------- #
async def _load_raw(session) -> pd.DataFrame:
    """Load airborne, positioned raw_states into a DataFrame ordered by aircraft+time."""
    t_expr = func.coalesce(RawState.time_position, RawState.last_contact)
    stmt = (
        select(
            RawState.icao24.label("icao24"),
            t_expr.label("t"),
            RawState.latitude.label("lat"),
            RawState.longitude.label("lon"),
            RawState.baro_altitude.label("baro_altitude"),
            RawState.geo_altitude.label("geo_altitude"),
            RawState.velocity.label("velocity"),
            RawState.true_track.label("true_track"),
            RawState.vertical_rate.label("vertical_rate"),
        )
        .where(
            RawState.latitude.is_not(None),
            RawState.longitude.is_not(None),
            t_expr.is_not(None),
            or_(RawState.on_ground.is_(None), RawState.on_ground.is_(False)),
        )
        .order_by(RawState.icao24, t_expr)
    )
    result = await session.execute(stmt)
    rows = result.all()
    cols = ["icao24", "t", "lat", "lon", "baro_altitude", "geo_altitude",
            "velocity", "true_track", "vertical_rate"]
    df = pd.DataFrame(rows, columns=cols)
    if not df.empty:
        df["t"] = df["t"].astype("int64")
    return df


async def reconstruct(
    *,
    gap_seconds: int = DEFAULT_GAP_SECONDS,
    dt: int = DEFAULT_DT_SECONDS,
    min_points: int = DEFAULT_MIN_POINTS,
    truncate: bool = True,
    batch_points: int = 5000,
) -> tuple[int, int]:
    """Rebuild trajectories from raw_states. Returns (n_trajectories, n_points)."""
    async with session_scope() as session:
        if truncate:
            await session.execute(
                text("TRUNCATE trajectory_points, trajectories RESTART IDENTITY CASCADE")
            )

        log.info("Loading raw_states...")
        df = await _load_raw(session)
        log.info("Loaded %d airborne positioned rows for %d aircraft",
                 len(df), df["icao24"].nunique() if not df.empty else 0)
        if df.empty:
            return 0, 0

        n_traj = 0
        n_points = 0
        pending: list[dict] = []

        for icao24, group in df.groupby("icao24", sort=False):
            for seg in reconstruct_aircraft(
                group, gap_seconds=gap_seconds, dt=dt, min_points=min_points
            ):
                tid = (
                    await session.execute(
                        insert(Trajectory)
                        .values(
                            icao24=icao24,
                            start_time=seg.start_time,
                            end_time=seg.end_time,
                            point_count=seg.point_count,
                            dt_seconds=dt,
                        )
                        .returning(Trajectory.id)
                    )
                ).scalar_one()
                n_traj += 1

                cols = seg.columns
                for i in range(seg.point_count):
                    pending.append(
                        {
                            "trajectory_id": tid,
                            "t": int(seg.grid[i]),
                            "lat": _f(cols["lat"][i]),
                            "lon": _f(cols["lon"][i]),
                            "baro_altitude": _f(cols["baro_altitude"][i]),
                            "geo_altitude": _f(cols["geo_altitude"][i]),
                            "velocity": _f(cols["velocity"][i]),
                            "true_track": _f(cols["true_track"][i]),
                            "vertical_rate": _f(cols["vertical_rate"][i]),
                        }
                    )
                n_points += seg.point_count

                if len(pending) >= batch_points:
                    await session.execute(insert(TrajectoryPoint), pending)
                    pending.clear()

        if pending:
            await session.execute(insert(TrajectoryPoint), pending)

    log.info("Wrote %d trajectories, %d trajectory_points", n_traj, n_points)
    return n_traj, n_points


def _f(x) -> float | None:
    """NaN -> None for nullable DB columns."""
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


async def _amain(args: argparse.Namespace) -> None:
    n_traj, n_points = await reconstruct(
        gap_seconds=args.gap_seconds,
        dt=args.dt,
        min_points=args.min_points,
        truncate=not args.no_truncate,
    )
    print(f"Reconstructed {n_traj} trajectories ({n_points} points).")
    await dispose_engine()


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild trajectories from raw_states")
    parser.add_argument("--gap-seconds", type=int, default=DEFAULT_GAP_SECONDS,
                        help="start a new segment after a gap longer than this")
    parser.add_argument("--dt", type=int, default=DEFAULT_DT_SECONDS,
                        help="resample grid spacing in seconds")
    parser.add_argument("--min-points", type=int, default=DEFAULT_MIN_POINTS,
                        help="discard segments with fewer resampled points than this")
    parser.add_argument("--no-truncate", action="store_true",
                        help="append instead of rebuilding (may duplicate)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
