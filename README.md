# SkyWatch ✈️

**Real-time anomaly & spoofing detection for live aircraft.**

SkyWatch pulls live ADS-B data from the [OpenSky Network](https://opensky-network.org/),
learns what *normal* flight looks like with a sequence model, and flags the weird
stuff — impossible maneuvers, teleporting aircraft, spoofed "ghost" planes — on a
live map, as it happens.

It's a full end-to-end system: data collection → trajectory reconstruction → feature
engineering → an LSTM (with a Transformer drop-in) → a physics + ML scoring service →
a React dashboard you can actually watch.

---

## The honest part (and the whole point)

Most "AI detects spoofing!" projects skip the inconvenient truth: **there is no
labeled real-world spoofing data.** So SkyWatch is **unsupervised** — it only ever
trains on normal traffic, and it's validated against synthetic attacks I inject
myself.

The detector is a **hybrid**: hard physics rules catch the *impossible*, and an LSTM
catches the *subtle-but-wrong*. Here's how it does against 7 attack types, on
held-out flights at a 1% false-positive rate:

| Attack | ROC-AUC | Detected? |
|---|---|---|
| Teleport / position jump | **1.00** | ✅ reliably |
| Impossible kinematics | **1.00** | ✅ reliably |
| Altitude spoofing | **1.00** | ✅ reliably |
| Velocity–position mismatch | **0.98** | ✅ reliably |
| Gradual drift | 0.51 | ⚠️ barely — the genuinely hard case |
| Ghost aircraft | 0.67 | ❌ can't, by design |
| Replay / clone | 0.51 | ❌ can't, by design |

**The failures are the interesting part.** Ghost and replay attacks have perfectly
plausible motion — nothing about *how they move* is wrong — so a motion-based
detector fundamentally can't see them. Production systems catch these physically,
with **TDOA / multilateration** (comparing a signal's arrival time across many ground
receivers), which the free OpenSky API doesn't expose. Naming that limitation
honestly matters more than inflating a number.

> I also swapped the LSTM for a Transformer to be sure — it changes nothing on the
> hard cases. The limit is the **data**, not the model.

---

## See it run (no account needed)

The dashboard runs in **replay mode** off already-collected data, so you don't need
OpenSky credentials.

```bash
# 1. Database
docker compose up -d

# 2. Backend  (Python 3.13)
cd backend
pip install -e ".[dev]"
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build
alembic upgrade head
python -m skywatch.api.main            #  → http://127.0.0.1:8000

# 3. Dashboard  (Node 18+)
cd ../frontend
npm install
npm run dev                            #  → http://localhost:5173
```

Open **http://localhost:5173**, then hit **⚡ Inject attack** to spoof a live plane
and watch it flag red — with a plain-English reason for *why*.

---

## How it works

```
OpenSky API
    │   collector — OAuth2, polls a small bounding box every ~8s
    ▼
raw_states ─► reconstruct ─► trajectories ─► features + scaler
              split on gaps,                computed-vs-reported speed /
              resample to fixed dt          heading / climb = the spoof signal
    │
    ▼
LSTM next-step predictor   +   physics rules
   residual > threshold         impossible speed / turn / climb / altitude
    │
    ▼
FastAPI live scoring ─► WebSocket ─► React dashboard (MapLibre + deck.gl)
   writes anomaly_scores
```

Every box is a small, focused module under `backend/skywatch/`. The model trains
**only on normal data**; the anomaly score is its prediction error, and the threshold
is the 99th percentile of that error on held-out normal flights.

<details>
<summary><b>Build it from scratch — collect real data and train your own model</b></summary>

You'll need free OpenSky API credentials (Account → "API client") in `.env`
(`cp .env.example .env`). Keep the bounding box small — cost scales with its area.

```bash
cd backend
python -m skywatch.collector        # collect a few hours of normal traffic
python -m skywatch.reconstruct      # raw_states → trajectories + points
python -m skywatch.features         # → feature matrix + fitted scaler
python -m skywatch.model.train      # LSTM next-step predictor  (--arch transformer to swap)
python -m skywatch.evaluate         # inject 7 attacks → per-attack metrics + plot
python -m skywatch.eda              # plot a few real flights to sanity-check
```

**API endpoints** (replay or live): `GET /healthz`, `GET /aircraft`,
`GET /aircraft/{icao24}/track`, `GET /anomalies`, `WS /ws/live`,
`POST /demo/inject`. For live polling instead of replay: `SCORING_MODE=live`.
</details>

---

## Tech

Python 3.13 · PyTorch · SQLAlchemy 2 + asyncpg · FastAPI · PostgreSQL 16 +
TimescaleDB · React + Vite + TypeScript · deck.gl / MapLibre · Recharts.

## Tests

```bash
cd backend && pytest        # 65 tests — no network, no credentials (OpenSky is mocked)
```

<details>
<summary><b>Troubleshooting</b></summary>

- **Port 5432 already taken** (e.g. a local Postgres): set `DB_HOST_PORT` to a free
  port in `.env` and match it in `DATABASE_URL`. Compose publishes
  `${DB_HOST_PORT:-5432}:5432`.
- **`role "skywatch" does not exist`** right after `compose up`: the DB's first-init
  can report healthy a moment early — just re-run `alembic upgrade head`.
- **Docker Desktop crashes on start** with a unix-socket error (happens when the
  Windows username has a space): double-click **`Start-SkyWatch-Docker.bat`** instead
  of the Docker icon. It clears the leftover socket folders and brings everything up.

</details>

---

*Conventions and build notes in [`CLAUDE.md`](CLAUDE.md).*
