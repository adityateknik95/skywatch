# Deploying the SkyWatch demo

The live demo is **self-contained**: the backend container bakes in the trained model
and a ~20-minute slice of real replay data, so it runs with **no database and no
OpenSky credentials**. You deploy two pieces:

1. **Backend** (FastAPI + model + WebSocket) → a Docker host.
2. **Frontend** (static React build) → a static host, pointed at the backend.

The repo is already public on GitHub, so both platforms can build straight from it.

---

## 1. Backend → Hugging Face Spaces (recommended, free)

HF Spaces is free, ML-friendly, and has generous RAM (PyTorch fits easily).

1. https://huggingface.co → **New → Space** → SDK: **Docker** (blank).
2. In the Space settings, point it at this GitHub repo (or push the repo to the Space).
   It builds from the root [`Dockerfile`](Dockerfile) automatically.
3. Add this line to the Space's `README.md` front-matter so it serves on the right port:
   ```yaml
   app_port: 8000
   ```
4. Wait for the build; your backend is live at `https://<user>-<space>.hf.space`.
   Check `https://<user>-<space>.hf.space/healthz` → should show `"mode":"demo"`.

**Alternative — Render:** New → **Web Service** → from repo → Runtime **Docker**.
Render injects `$PORT` automatically (the container already honors it). Free tier
sleeps after ~15 min idle (first load ~30–60 s); ~$7/mo keeps it always-on.

> The container defaults to `SCORING_MODE=demo`. To lock CORS to your frontend,
> set the env var `CORS_ORIGINS=https://your-frontend.vercel.app`.

## 2. Frontend → Vercel (free)

1. https://vercel.com → **Add New → Project** → import the GitHub repo.
2. **Root Directory:** `frontend`. Framework preset: **Vite** (auto-detected).
3. **Environment Variables** (point the dashboard at your backend from step 1):
   ```
   VITE_API_URL = https://<your-backend-host>
   VITE_WS_URL  = wss://<your-backend-host>/ws/live
   ```
4. Deploy. Your dashboard is live at `https://<project>.vercel.app`.

(Netlify / Cloudflare Pages work the same way: root `frontend`, build `npm run build`,
output `dist`, same two env vars.)

## 3. Wire it into your portfolio

Point the portfolio's **Demo** button at the Vercel URL. Then hit **⚡ Inject attack**
in the dashboard to spoof a live aircraft and watch it flag red in real time.

---

## Local check before deploying

```bash
cd backend
SCORING_MODE=demo .venv/Scripts/python.exe -m skywatch.api.main   # no DB needed
# → http://127.0.0.1:8000/healthz  shows "mode":"demo"
```
Or build/run the container exactly as the host will:
```bash
docker build -t skywatch-demo .
docker run -p 8000:8000 skywatch-demo
```

## Refreshing the bundled demo data

To re-capture the replay slice from a freshly collected DB:
```bash
cd backend && python -m skywatch.export_demo --minutes 20   # rewrites demo_data/replay.json.gz
```
