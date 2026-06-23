# SkyWatch backend — self-contained demo container.
# Bakes in the trained model (model/artifacts/) and a bundled replay slice
# (demo_data/replay.json.gz), so it runs the live-scoring API + WebSocket with
# NO database and NO OpenSky credentials.  Deploy to any container host
# (Hugging Face Spaces, Render, Railway, Fly, ...). Frontend deploys separately.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SCORING_MODE=demo \
    HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app

# CPU build of PyTorch (smaller, no CUDA — there's no GPU on these hosts anyway).
RUN pip install --no-cache-dir "torch>=2.6" --index-url https://download.pytorch.org/whl/cpu

# Backend package + deps. Editable so the bundled model/data stay on disk at runtime.
COPY backend /app/backend
RUN pip install --no-cache-dir -e /app/backend

EXPOSE 8000
# Render/Railway inject $PORT; Hugging Face Spaces: set `app_port: 8000` in the Space.
CMD ["sh", "-c", "uvicorn skywatch.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
