"""Phase 2 EDA: plot a few reconstructed real flights to confirm they look right.

Produces, under ``--out`` (default ``eda_output/``):
  * ``overview_map.png`` — ground tracks of many trajectories on one lon/lat map
    (should look like real air routes through the bbox).
  * ``flight_<icao24>_<id>.png`` — for the longest few flights, a ground track plus
    an altitude profile, so you can eyeball that gaps weren't bridged and the path
    is smooth.

Usage::

    python -m skywatch.eda --top 6 --overview 250 --out eda_output
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, no display needed
import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy import select

from skywatch.db.models import Trajectory, TrajectoryPoint
from skywatch.db.session import dispose_engine, session_scope

log = logging.getLogger("skywatch.eda")


async def _top_trajectories(session, limit: int) -> list[Trajectory]:
    stmt = select(Trajectory).order_by(Trajectory.point_count.desc()).limit(limit)
    return list((await session.execute(stmt)).scalars())


async def _points(session, trajectory_id: int):
    stmt = (
        select(
            TrajectoryPoint.t,
            TrajectoryPoint.lat,
            TrajectoryPoint.lon,
            TrajectoryPoint.baro_altitude,
        )
        .where(TrajectoryPoint.trajectory_id == trajectory_id)
        .order_by(TrajectoryPoint.t)
    )
    rows = (await session.execute(stmt)).all()
    arr = np.array(rows, dtype=float)
    return arr  # columns: t, lat, lon, baro_altitude


def _plot_overview(tracks: list[np.ndarray], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    for arr in tracks:
        if arr.size:
            ax.plot(arr[:, 2], arr[:, 1], lw=0.5, alpha=0.5)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(f"Reconstructed ground tracks ({len(tracks)} flights)")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def _plot_flight(traj: Trajectory, arr: np.ndarray, out: Path) -> None:
    t, lat, lon, baro = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    mins = (t - t[0]) / 60.0
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    sc = ax1.scatter(lon, lat, c=mins, cmap="viridis", s=8)
    ax1.plot(lon, lat, lw=0.6, alpha=0.4, color="gray")
    ax1.set_xlabel("longitude")
    ax1.set_ylabel("latitude")
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.set_title(f"{traj.icao24} — ground track ({traj.point_count} pts)")
    fig.colorbar(sc, ax=ax1, label="minutes since start")

    ax2.plot(mins, baro, color="tab:blue")
    ax2.set_xlabel("minutes since start")
    ax2.set_ylabel("baro altitude (m)")
    ax2.set_title("Altitude profile")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


async def _amain(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    async with session_scope() as session:
        overview = await _top_trajectories(session, args.overview)
        tracks = [await _points(session, t.id) for t in overview]
        if not tracks:
            print("No trajectories found — run `python -m skywatch.reconstruct` first.")
            await dispose_engine()
            return
        _plot_overview(tracks, out_dir / "overview_map.png")
        log.info("Wrote %s", out_dir / "overview_map.png")

        for traj in overview[: args.top]:
            arr = await _points(session, traj.id)
            fname = out_dir / f"flight_{traj.icao24}_{traj.id}.png"
            _plot_flight(traj, arr, fname)
            log.info("Wrote %s", fname)

    print(f"Wrote {min(args.top, len(overview)) + 1} plot(s) to {out_dir.resolve()}")
    await dispose_engine()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot reconstructed trajectories (EDA)")
    parser.add_argument("--top", type=int, default=6, help="detailed per-flight plots")
    parser.add_argument("--overview", type=int, default=250,
                        help="number of tracks on the overview map")
    parser.add_argument("--out", default="eda_output", help="output directory")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
