"""FastAPI app: REST endpoints + live WebSocket over the scoring service (spec §11).

Run::

    python -m skywatch.api.main                       # uses settings.scoring_mode
    SCORING_MODE=demo  python -m skywatch.api.main      # self-contained (bundled data, no DB)
    SCORING_MODE=live  python -m skywatch.api.main      # live OpenSky polling (needs creds)
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from skywatch.api.service import ScoringService
from skywatch.config import get_settings

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
    await service.stop()
    if not service.demo:
        from skywatch.db.session import dispose_engine

        await dispose_engine()


_settings = get_settings()
_origins = (
    ["*"] if _settings.cors_origins.strip() == "*"
    else [o.strip() for o in _settings.cors_origins.split(",") if o.strip()]
)

app = FastAPI(title="SkyWatch", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=_origins, allow_methods=["*"], allow_headers=["*"]
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
    return app.state.service.latest or {"time": None, "aircraft": []}


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
    s = app.state.service

    if s.demo:  # serve from the in-memory history (no database)
        return {
            "icao24": icao24,
            "track": list(s.tracks_by_icao.get(icao24, []))[-limit:],
            "scores": list(s.scores_by_icao.get(icao24, []))[-limit:],
        }

    from sqlalchemy import select

    from skywatch.db.models import AnomalyScore, RawState
    from skywatch.db.session import session_scope

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
    """Dev only: run the injection harness and return per-attack metrics (§11).
    Disabled in the hosted demo (it needs the training artifacts + database)."""
    if app.state.service.demo:
        raise HTTPException(status_code=404, detail="eval is disabled in the demo")
    from skywatch.evaluate import run_eval

    return await run_eval(n_trajectories, seed)


@app.post("/demo/inject")
async def demo_inject(attack: str = "velocity", cycles: int = 25):
    """Inject a synthetic attack into a warmed-up live aircraft for N cycles, so the
    dashboard shows it get flagged in real time. attack: velocity|teleport|altitude."""
    import random

    s = app.state.service
    if not s.latest or not s.latest["aircraft"]:
        return {"armed": False, "error": "no live aircraft yet"}
    positioned = [a for a in s.latest["aircraft"] if a.get("lat") is not None]
    if not positioned:
        return {"armed": False, "error": "no positioned aircraft"}
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
    import os
    import sys

    import uvicorn

    logging.basicConfig(level="INFO",
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    if sys.platform == "win32":
        import asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
