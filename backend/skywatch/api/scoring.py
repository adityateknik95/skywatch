"""Live, stateful per-aircraft anomaly scoring (spec §7 SERVE path).

Mirrors the offline pipeline incrementally: for each new point of an aircraft, compute
the feature vector from the previous point (with the real dt), run the physics rules,
and — once a full window of W features has accumulated — run the LSTM to get the
next-step prediction residual. A point is anomalous if the residual exceeds the
calibrated threshold OR a physics rule fires.

Per-aircraft state is tiny (previous raw point + a deque of the last W scaled
features), so thousands of aircraft fit comfortably in memory.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from skywatch.features import (
    FEATURE_NAMES,
    compute_trajectory_features,
    load_scaler,
)
from skywatch.model.score import load_threshold
from skywatch.physics import check_row

# raw point tuple layout: (t, lat, lon, baro, geo, velocity, true_track, vertical_rate)


@dataclass
class ScoreResult:
    icao24: str
    t: int
    score: float
    threshold: float
    is_anomaly: bool
    reason: str | None


class _State:
    __slots__ = ("prev", "feats", "last_t")

    def __init__(self, window: int) -> None:
        self.prev: tuple | None = None
        self.feats: deque = deque(maxlen=window)
        self.last_t: int | None = None


class LiveScorer:
    def __init__(self, model, scaler, threshold: float, window: int,
                 gap_seconds: int = 60) -> None:
        self.model = model.eval()
        self.scaler = scaler
        self.threshold = threshold
        self.window = window
        self.gap_seconds = gap_seconds
        self.states: dict[str, _State] = {}

    @classmethod
    def from_artifacts(cls, gap_seconds: int = 60) -> "LiveScorer":
        from skywatch.model.factory import load_active_model

        model, cfg = load_active_model()
        return cls(model, load_scaler(), load_threshold()["threshold"],
                   int(cfg["window"]), gap_seconds=gap_seconds)

    @torch.no_grad()
    def score(self, icao24: str, t: int, lat, lon, baro, geo, velocity,
              true_track, vertical_rate) -> ScoreResult | None:
        """Score one new point. Returns None during warmup / on a gap (no feature yet)."""
        st = self.states.get(icao24)
        if st is None:
            st = _State(self.window)
            self.states[icao24] = st
        st.last_t = t

        cur = (t, lat, lon, baro, geo, velocity, true_track, vertical_rate)
        if lat is None or lon is None:
            return None

        dt = None if st.prev is None else (t - st.prev[0])
        if dt is None or dt <= 0 or dt > self.gap_seconds:
            # first point or a gap -> start a fresh segment, can't form a feature yet
            st.prev = cur
            st.feats.clear()
            return None

        raw = self._pair_feature(st.prev, cur, float(dt))
        reason = check_row({n: raw[i] for i, n in enumerate(FEATURE_NAMES)})
        scaled = self.scaler.transform(raw.reshape(1, -1))[0].astype(np.float32)

        model_score: float | None = None
        if len(st.feats) >= self.window:
            win = np.stack(st.feats)[None, ...]  # (1, W, F)
            pred = self.model(torch.from_numpy(win)).numpy()[0]
            model_score = float(np.mean((pred - scaled) ** 2))

        st.feats.append(scaled)
        st.prev = cur

        model_anom = model_score is not None and model_score > self.threshold
        is_anomaly = (reason is not None) or model_anom
        out_reason = reason if reason is not None else ("ml" if model_anom else None)
        return ScoreResult(
            icao24=icao24, t=t,
            score=model_score if model_score is not None else 0.0,
            threshold=self.threshold, is_anomaly=is_anomaly, reason=out_reason,
        )

    @staticmethod
    def _pair_feature(prev: tuple, cur: tuple, dt: float) -> np.ndarray:
        col = lambda i: np.array([prev[i], cur[i]], dtype=float)  # None -> nan
        X = compute_trajectory_features(
            col(1), col(2), col(3), col(4), col(5), col(6), col(7), dt
        )
        return X[0]

    def prune(self, now_t: int, max_idle: int = 900) -> int:
        """Drop aircraft not seen for ``max_idle`` seconds. Returns count removed."""
        stale = [k for k, v in self.states.items()
                 if v.last_t is not None and now_t - v.last_t > max_idle]
        for k in stale:
            del self.states[k]
        return len(stale)
