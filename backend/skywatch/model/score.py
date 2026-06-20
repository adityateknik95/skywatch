"""Residual anomaly scoring + threshold calibration (spec §10).

The anomaly score for a window is the per-point prediction residual (mean squared
error over the feature vector). The threshold is a high percentile of those scores
on **held-out normal** data — anything above it is flagged. Persisted so the live
scoring service (Phase 6) uses the identical threshold.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from skywatch.features import ARTIFACTS_DIR

THRESHOLD_PATH = ARTIFACTS_DIR / "threshold.json"


@torch.no_grad()
def residual_scores(model, loader, device: str = "cpu") -> np.ndarray:
    """Per-window anomaly score = mean over features of (pred - target)**2."""
    model.eval()
    out: list[np.ndarray] = []
    for w, y in loader:
        w, y = w.to(device), y.to(device)
        pred = model(w)
        mse = ((pred - y) ** 2).mean(dim=1)
        out.append(mse.cpu().numpy())
    return np.concatenate(out) if out else np.empty(0, dtype=np.float64)


def calibrate_threshold(scores: np.ndarray, percentile: float) -> float:
    return float(np.percentile(scores, percentile))


def save_threshold(
    threshold: float, percentile: float, scores: np.ndarray, path: Path = THRESHOLD_PATH
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcts = {str(p): float(np.percentile(scores, p)) for p in (50, 90, 95, 99, 99.9)}
    path.write_text(
        json.dumps(
            {
                "score_type": "per_point_mse",
                "threshold": threshold,
                "percentile": percentile,
                "n_calibration": int(scores.size),
                "calibration_percentiles": pcts,
                "calibration_mean": float(scores.mean()) if scores.size else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_threshold(path: Path = THRESHOLD_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
