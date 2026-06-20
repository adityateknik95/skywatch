"""FastAPI app: REST endpoints + live WebSocket over the scoring service (spec §11).

Run::

    python -m skywatch.api.main                      # or: uvicorn skywatch.api.main:app
    SCORING_MODE=live python -m skywatch.api.main     # live polling (needs credentials)
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select

from skywatch.api.service import ScoringService
from skywatch.config import get_settings
from skywatch.db.models import AnomalyScore, RawState
from skywatch.db.session import dispose_engine, session_scope

log = logging.getLogger("skywatch.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    service = ScoringService(
        settings,
        mode=settings.scoring_mode,
        replay_minutes=settings.replay_minutes,
        replay_interval=settings.replay_interval_seconds,
    )
    app.state.service = service
    try:
        service.start()  # loads artifacts + starts the loop
        log.info("Scoring service started (mode=%s)", service.mode)
    except Exception:
        log.exception("Scoring service failed to start; REST will still serve.")
    yield
    try:
        await service.stop()
    finally:
        await dispose_engine()


app = FastAPI(title="SkyWatch", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.get("/healthz")
async def healthz():
    s = app.state.service
    return {
        "status": "ok",
        "mode": s.mode,
        "model_loaded": s.scorer is not None,
        "ws_clients": len(s.manager.active),
        "latest_cycle": s.latest["time"] if s.latest else None,
        "tracked_aircraft": len(s.scorer.states) if s.scorer else 0,
    }


@app.get("/aircraft")
async def aircraft():
    """Current live states in the bbox with their latest score."""
    s = app.state.service
    return s.latest or {"time": None, "aircraft": []}


@app.get("/anomalies")
async def anomalies():
    """Currently flagged aircraft, ranked by score (highest first)."""
    latest = app.state.service.latest
    if not latest:
        return {"time": None, "anomalies": []}
    flagged = [a for a in latest["aircraft"] if a.get("is_anomaly")]
    flagged.sort(key=lambda a: (a.get("score") or 0.0), reverse=True)
    return {"time": latest["time"], "anomalies": flagged}


@app.get("/aircraft/{icao24}/track")
async def track(icao24: str, limit: int = 200):
    """Recent trajectory points + score series for one aircraft."""
    icao24 = icao24.lower()
    async with session_scope() as session:
        pts = (await session.execute(
            select(RawState.time_position, RawState.last_contact, RawState.latitude,
                   RawState.longitude, RawState.baro_altitude, RawState.velocity,
                   RawState.true_track)
            .where(RawState.icao24 == icao24, RawState.latitude.is_not(None))
            .order_by(RawState.request_time.desc()).limit(limit)
        )).all()
        scores = (await session.execute(
            select(AnomalyScore.t, AnomalyScore.score, AnomalyScore.threshold,
                   AnomalyScore.is_anomaly, AnomalyScore.reason)
            .where(AnomalyScore.icao24 == icao24)
            .order_by(AnomalyScore.t.desc()).limit(limit)
        )).all()

    # Access columns via ._mapping — `t` collides with the reserved Row.t attribute.
    track_pts = [{
        "t": m["time_position"] or m["last_contact"], "lat": m["latitude"], "lon": m["longitude"],
        "baro_altitude": m["baro_altitude"], "velocity": m["velocity"], "true_track": m["true_track"],
    } for m in (p._mapping for p in reversed(pts))]
    score_series = [{
        "t": m["t"], "score": m["score"], "threshold": m["threshold"],
        "is_anomaly": m["is_anomaly"], "reason": m["reason"],
    } for m in (r._mapping for r in reversed(scores))]
    return {"icao24": icao24, "track": track_pts, "scores": score_series}


@app.post("/eval/run")
async def eval_run(n_trajectories: int = 100, seed: int = 0):
    """Dev only: run the injection harness and return per-attack metrics (§11)."""
    from skywatch.evaluate import run_eval

    return await run_eval(n_trajectories, seed)


@app.post("/demo/inject")
async def demo_inject(attack: str = "velocity", cycles: int = 25):
    """Demo: inject a synthetic attack into a random live aircraft for N cycles,
    so the dashboard shows it get flagged in real time. attack: velocity|teleport|altitude."""
    import random

    s = app.state.service
    if not s.latest or not s.latest["aircraft"]:
        return {"armed": False, "error": "no live aircraft yet"}
    positioned = [a for a in s.latest["aircraft"] if a.get("lat") is not None]
    if not positioned:
        return {"armed": False, "error": "no positioned aircraft"}
    # Prefer an aircraft the model has already warmed up on (full feature window):
    # it flags immediately and the model residual can spike too.
    warmed = [
        a for a in positioned
        if s.scorer and (st := s.scorer.states.get(a["icao24"]))
        and len(st.feats) >= s.scorer.window
    ]
    target = random.choice(warmed or positioned)
    s.arm_demo(attack, target["icao24"], cycles)
    return {"armed": True, "attack": attack, "icao24": target["icao24"],
            "callsign": target.get("callsign"), "cycles": cycles}


@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    service = app.state.service
    await service.manager.connect(ws)
    try:
        if service.latest:
            await ws.send_json(service.latest)
        while True:
            await ws.receive_text()  # keep the socket open; client messages ignored
    except WebSocketDisconnect:
        service.manager.disconnect(ws)
    except Exception:
        service.manager.disconnect(ws)


def main() -> None:
    import sys
    import uvicorn

    logging.basicConfig(level="INFO",
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    if sys.platform == "win32":
        import asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
