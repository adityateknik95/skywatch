"""Scoring service: the live loop that ties polling -> scoring -> persistence ->
WebSocket broadcast together (spec §7 SERVE, §11).

Two modes:
  * ``replay`` (default, no credentials/credits needed): streams recent ``raw_states``
    snapshots from the DB cycle-by-cycle, so the dashboard works offline.
  * ``live``: polls OpenSky each cycle, writes ``raw_states``, then scores.

Either way each cycle scores every aircraft (physics + model), writes
``anomaly_scores``, keeps the latest cycle in memory for the REST endpoints, and
broadcasts ``{time, aircraft:[...]}`` to all connected WebSocket clients.
"""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Any

import httpx
from sqlalchemy import func, insert, select, text

from skywatch.api.scoring import LiveScorer
from skywatch.config import Settings, get_settings
from skywatch.db.models import AnomalyScore, RawState
from skywatch.db.session import session_scope
from skywatch.opensky.client import OpenSkyClient
from skywatch.opensky.parser import parse_states_response
from skywatch.opensky.token_manager import TokenManager

log = logging.getLogger("skywatch.api.service")

_PUBLIC = ("icao24", "callsign", "lat", "lon", "baro_altitude",
           "geo_altitude", "velocity", "true_track", "vertical_rate")


class ConnectionManager:
    def __init__(self) -> None:
        self.active: set[Any] = set()

    async def connect(self, ws) -> None:
        await ws.accept()
        self.active.add(ws)

    def disconnect(self, ws) -> None:
        self.active.discard(ws)

    async def broadcast(self, message: dict) -> None:
        for ws in list(self.active):
            try:
                await ws.send_json(message)
            except Exception:
                self.active.discard(ws)


def _clean(v):
    """JSON-safe: drop NaN/inf."""
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


class ScoringService:
    def __init__(self, settings: Settings | None = None, *, mode: str = "replay",
                 replay_minutes: int = 20, replay_interval: float = 1.0) -> None:
        self.settings = settings or get_settings()
        self.mode = mode
        self.replay_minutes = replay_minutes
        self.replay_interval = replay_interval
        self.manager = ConnectionManager()
        self.scorer: LiveScorer | None = None
        self.latest: dict | None = None
        self._demo: dict | None = None  # live attack-injection demo state
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None

    def arm_demo(self, attack: str, icao24: str, cycles: int = 8) -> None:
        """Inject a synthetic attack into one live aircraft for the next N cycles."""
        self._demo = {"attack": attack, "icao24": icao24, "cycles": cycles}

    def _apply_demo(self, aircraft: list[dict]) -> None:
        d = self._demo
        if not d:
            return
        target = next((a for a in aircraft if a["icao24"] == d["icao24"]), None)
        if target is None:
            # target not in this snapshot — don't burn the budget, just wait for it
            d["misses"] = d.get("misses", 0) + 1
            if d["misses"] > 40:  # safety: give up if it never reappears
                self._demo = None
            return
        atk = d["attack"]
        if atk == "teleport":
            target["lat"] = (target.get("lat") or 50.0) + 0.5
        elif atk == "altitude":
            target["baro_altitude"] = (target.get("baro_altitude") or 10000.0) + 4000.0
        else:  # velocity-position inconsistency: reported speed far above motion
            # large absolute offset so it trips the rules for slow planes too
            target["velocity"] = (target.get("velocity") or 200.0) + 600.0
        d["cycles"] -= 1  # only count cycles where the attack was actually applied
        if d["cycles"] <= 0:
            self._demo = None

    # --- lifecycle -------------------------------------------------------- #
    def start(self) -> None:
        self.scorer = LiveScorer.from_artifacts()
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass

    # --- the loop --------------------------------------------------------- #
    async def _run(self) -> None:
        try:
            if self.mode == "live":
                await self._run_live()
            else:
                await self._run_replay()
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("Scoring loop crashed")

    async def _process_cycle(self, t: int, aircraft: list[dict]) -> None:
        self._apply_demo(aircraft)
        rows = []
        out = []
        for ac in aircraft:
            res = self.scorer.score(
                ac["icao24"], t, ac.get("lat"), ac.get("lon"),
                ac.get("baro_altitude"), ac.get("geo_altitude"),
                ac.get("velocity"), ac.get("true_track"), ac.get("vertical_rate"),
            )
            item = {k: _clean(ac.get(k)) for k in _PUBLIC}
            if res is not None:
                item.update(score=_clean(res.score), is_anomaly=res.is_anomaly,
                            reason=res.reason, threshold=res.threshold)
                rows.append({"icao24": res.icao24, "t": res.t, "score": res.score,
                             "threshold": res.threshold, "is_anomaly": res.is_anomaly,
                             "reason": res.reason})
            else:
                item.update(score=None, is_anomaly=False, reason=None,
                            threshold=self.scorer.threshold)
            out.append(item)

        if rows:
            async with session_scope() as session:
                await session.execute(insert(AnomalyScore), rows)

        self.latest = {"time": t, "aircraft": out}
        self.scorer.prune(t)
        await self.manager.broadcast(self.latest)

    async def _run_replay(self) -> None:
        snapshots = await self._load_replay_snapshots()
        if not snapshots:
            log.warning("Replay: no raw_states found — collect data first.")
            return
        log.info("Replay: streaming %d snapshots every %.1fs (loops)",
                 len(snapshots), self.replay_interval)
        async with session_scope() as session:
            await session.execute(text("TRUNCATE anomaly_scores RESTART IDENTITY"))
        while not self._stop.is_set():
            for t, aircraft in snapshots:
                if self._stop.is_set():
                    break
                await self._process_cycle(t, aircraft)
                await asyncio.sleep(self.replay_interval)
            # loop the recording; reset per-aircraft state for a clean restart
            self.scorer.states.clear()

    async def _load_replay_snapshots(self) -> list[tuple[int, list[dict]]]:
        async with session_scope() as session:
            max_t = (await session.execute(select(func.max(RawState.request_time)))).scalar()
            if max_t is None:
                return []
            since = max_t - self.replay_minutes * 60
            stmt = (
                select(
                    RawState.request_time, RawState.icao24, RawState.callsign,
                    RawState.latitude, RawState.longitude, RawState.baro_altitude,
                    RawState.geo_altitude, RawState.velocity, RawState.true_track,
                    RawState.vertical_rate, RawState.on_ground,
                )
                .where(RawState.request_time >= since)
                .order_by(RawState.request_time)
            )
            rows = (await session.execute(stmt)).all()

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
        return sorted(snapshots.items())

    async def _run_live(self) -> None:
        s = self.settings
        if not s.has_credentials:
            raise SystemExit("Live mode needs OpenSky credentials; use replay mode instead.")
        async with httpx.AsyncClient(timeout=30.0) as http:
            tokens = TokenManager(s.opensky_client_id, s.opensky_client_secret,
                                  s.opensky_token_url, http)
            client = OpenSkyClient(s, tokens, http)
            log.info("Live scoring: polling every %ds", s.poll_interval_seconds)
            while not self._stop.is_set():
                try:
                    payload = await client.fetch_states()
                    t, parsed = parse_states_response(payload)
                    async with session_scope() as session:
                        await session.execute(insert(RawState), parsed)
                    aircraft = [{
                        "icao24": p["icao24"], "callsign": p["callsign"],
                        "lat": p["latitude"], "lon": p["longitude"],
                        "baro_altitude": p["baro_altitude"], "geo_altitude": p["geo_altitude"],
                        "velocity": p["velocity"], "true_track": p["true_track"],
                        "vertical_rate": p["vertical_rate"],
                    } for p in parsed if not p["on_ground"]]
                    await self._process_cycle(t, aircraft)
                except Exception:
                    log.exception("Live cycle error")
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=s.poll_interval_seconds)
                except asyncio.TimeoutError:
                    pass
