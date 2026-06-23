"""Export a slice of ``raw_states`` into the bundled demo replay file.

Run once, with the local DB populated, to (re)build
``skywatch/demo_data/replay.json.gz`` — the self-contained data the hosted ``demo``
mode streams (no database needed at serve time).

    python -m skywatch.export_demo --minutes 20
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import sys
from pathlib import Path

from sqlalchemy import func, select

from skywatch.db.models import RawState
from skywatch.db.session import dispose_engine, session_scope

OUT_PATH = Path(__file__).resolve().parent / "demo_data" / "replay.json.gz"


async def export(minutes: int) -> None:
    async with session_scope() as session:
        max_t = (await session.execute(select(func.max(RawState.request_time)))).scalar()
        if max_t is None:
            print("No raw_states in the DB — nothing to export.")
            return
        since = max_t - minutes * 60
        rows = (await session.execute(
            select(
                RawState.request_time, RawState.icao24, RawState.callsign,
                RawState.latitude, RawState.longitude, RawState.baro_altitude,
                RawState.geo_altitude, RawState.velocity, RawState.true_track,
                RawState.vertical_rate, RawState.on_ground,
            )
            .where(RawState.request_time >= since)
            .order_by(RawState.request_time)
        )).all()

    snapshots: dict[int, list[dict]] = {}
    for r in rows:
        if r.on_ground:
            continue
        snapshots.setdefault(int(r.request_time), []).append({
            "icao24": r.icao24, "callsign": r.callsign,
            "lat": r.latitude, "lon": r.longitude,
            "baro_altitude": r.baro_altitude, "geo_altitude": r.geo_altitude,
            "velocity": r.velocity, "true_track": r.true_track,
            "vertical_rate": r.vertical_rate,
        })
    data = sorted(snapshots.items())

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUT_PATH, "wt", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
    n_rows = sum(len(a) for _, a in data)
    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f"Wrote {len(data)} snapshots, {n_rows} aircraft-points -> "
          f"{OUT_PATH} ({size_mb:.1f} MB)")
    await dispose_engine()


def main() -> None:
    p = argparse.ArgumentParser(description="Export the bundled demo replay slice")
    p.add_argument("--minutes", type=int, default=20)
    args = p.parse_args()
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(export(args.minutes))


if __name__ == "__main__":
    main()
