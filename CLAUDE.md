# SkyWatch — conventions

Conventions + current status for SkyWatch. The project was built strictly phase by
phase (see **Current status** below); the original `§` references point to the
sections of the project brief this was built from. Don't scaffold a later phase
before the earlier one runs against real data.

## Hard rules (from the spec, repeated here so they aren't forgotten)
1. **Unsupervised only.** No labeled real spoofing data exists. The model trains
   on normal traffic; detection is validated against *synthetic* injected attacks.
   Never frame this as supervised classification.
2. **OpenSky OAuth2 client-credentials.** Basic auth is gone (removed March 2026).
   Exchange `client_id`/`client_secret` for a bearer token; **tokens expire ~30
   min — refresh proactively.**
3. **`/states/all` is on a daily credit budget**, cost scales with bbox area.
   Keep the bbox small, poll every 8–10s. Never hit the global feed.
4. **State vectors are arrays-of-arrays** (18 positional fields, parse by index).
   The DB schema and parser match this exactly (spec §6).
5. **Sensor artifacts look like anomalies.** Handle missing/None fields
   everywhere. Treat gaps as gaps, not jumps.

## Layout
- `backend/skywatch/` — Python package (config, opensky/, db/, collector, ...).
- `backend/alembic/` — migrations.
- `backend/tests/` — pytest.
- `frontend/` — React 18 + Vite + TS dashboard (MapLibre + deck.gl, Recharts).
- `docker-compose.yml` — Postgres 16 + TimescaleDB.

## Conventions
- Python 3.13 (spec asks 3.11; that minor isn't installed locally — 3.13 is the
  closest version with wheels for asyncpg/SQLAlchemy/PyTorch). Async everywhere
  in the I/O path (`httpx`, `asyncpg`, SQLAlchemy async).
- Settings come from `.env` via `skywatch.config.Settings` (pydantic-settings).
  Never read `os.environ` directly elsewhere.
- DB access through `skywatch.db.session`. Models in `skywatch.db.models`.
- Tests must not require network or real credentials: mock OpenSky, use a sample
  response fixture for the parser.

## Current status
Phases 0–3 are built and verified against real data:
- Phase 0 (scaffold/infra) and Phase 1 (OAuth2 + collector → `raw_states`).
- Phase 2 (reconstruction): `reconstruct.py` groups/segments/resamples `raw_states`
  into `trajectories` + `trajectory_points` (gap-split, fixed `dt`, no bridging of
  long gaps); `eda.py` plots flights for verification.
- Phase 3 (features): `features.py` builds the raw+derived feature matrix from
  `trajectory_points` (consistency residuals are the spoofing signal), fits a
  median-impute→standardize scaler on normal data, and persists scaler + matrix +
  metadata to `skywatch/model/artifacts/` (gitignored).
- Phase 4 (LSTM baseline): `model/` has the next-step predictor (`lstm.py`), lazy
  windowing + trajectory-level split (`dataset.py`), training with early stopping
  (`train.py`), and residual scoring + threshold calibration (`score.py`). Trains on
  CPU; persists `lstm.pt`, `model_config.json`, `threshold.json` (p99 of held-out
  normal residuals). Score = per-point MSE of the prediction residual.
- Phase 5 (injection + eval): `physics.py` (hard kinematic rules, `reason=physics:*`),
  `inject.py` (7 synthetic attacks, §8), `evaluate.py` (runs model+physics over clean
  + injected held-out data → precision/recall/ROC-AUC per attack). Result is honest:
  kinematic/consistency attacks caught; gradual drift weak; ghost/replay ~chance
  (plausible motion needs TDOA, not a motion model).
- Phase 6 (FastAPI scoring service): `api/scoring.py` (stateful per-aircraft
  `LiveScorer`: physics + LSTM over a rolling window), `api/service.py` (loop in
  `replay` mode from `raw_states` or `live` from OpenSky → writes `anomaly_scores`,
  broadcasts each cycle), `api/main.py` (REST `/healthz` `/aircraft`
  `/aircraft/{icao24}/track` `/anomalies` `/eval/run` + `WS /ws/live`). Default mode is
  `replay` so the dashboard works without credentials/credits.
- Phase 7 (React dashboard): `frontend/` — MapLibre (free demotiles, no token) base map
  with deck.gl as a synced `MapboxOverlay` (`FlightMap`), live `AnomalyPanel`, Recharts
  `ScoreTimeline`, `lib/ws.ts` (auto-reconnecting `WS /ws/live`). Flagged aircraft glow
  red; click selects → track + score-over-time. `npm run dev` (Vite, port 5173).
- Phase 8 (stretch): `model/transformer.py` + `model/factory.py` make the model a
  drop-in selected by `arch` in `model_config.json` (`train.py --arch transformer`);
  trained + evaluated — performs comparably to the LSTM and, honestly, doesn't crack
  drift/ghost/replay (the limit is data/approach, not capacity). Live demo:
  `POST /demo/inject` + the dashboard's "⚡ Inject attack" button fire a synthetic
  attack into a live aircraft so you watch it flag red. LSTM stays the default model.
