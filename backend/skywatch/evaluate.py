"""Injection harness + evaluation (Phase 5, spec §13).

Takes held-out *normal* trajectories the model never trained on, injects each
synthetic attack (§8), runs the model + physics over clean and injected data, and
reports precision / recall / ROC-AUC per attack type.

Detector at the operating point: a point is flagged if the model residual exceeds
the calibrated threshold OR a physics hard-rule fires. ROC-AUC uses the continuous
model residual (physics is a hard backstop, not a score).

    python -m skywatch.evaluate                 # full table + plot
    python -m skywatch.evaluate --n-trajectories 200 --seed 1

The expected story (and the honest one): kinematic/consistency attacks are caught
well; gradual drift is weak; ghost & replay are ~chance because their motion is
plausible by construction — motion alone can't reveal them (you'd need TDOA).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sqlalchemy import bindparam, select

from skywatch.db.models import Trajectory, TrajectoryPoint
from skywatch.db.session import dispose_engine, session_scope
from skywatch.features import (
    FEATURE_MATRIX_PATH,
    compute_trajectory_features,
    load_scaler,
)
from skywatch.inject import ATTACKS, TRAJ_KEYS
from skywatch.model.dataset import split_ranges, trajectory_ranges, window_starts
from skywatch.model.score import load_threshold
from skywatch.physics import physics_flags

log = logging.getLogger("skywatch.evaluate")

EVAL_OUT = Path("eval_output")


# --------------------------------------------------------------------------- #
def _load_model():
    from skywatch.model.factory import load_active_model

    model, cfg = load_active_model()
    return model, int(cfg["window"]), cfg.get("arch", "lstm")


def _calib_trajectory_ids(seed: int = 42) -> list[int]:
    """Replicate the training split to recover the held-out calibration trajectories."""
    data = np.load(FEATURE_MATRIX_PATH)
    ranges = trajectory_ranges(data["trajectory_id"])
    _, _, calib = split_ranges(ranges, seed=seed)
    return [r[0] for r in calib]


async def _load_trajectories(ids: list[int]) -> dict[int, dict]:
    stmt = (
        select(
            TrajectoryPoint.trajectory_id, TrajectoryPoint.t,
            TrajectoryPoint.lat, TrajectoryPoint.lon,
            TrajectoryPoint.baro_altitude, TrajectoryPoint.geo_altitude,
            TrajectoryPoint.velocity, TrajectoryPoint.true_track,
            TrajectoryPoint.vertical_rate, Trajectory.dt_seconds,
        )
        .join(Trajectory, Trajectory.id == TrajectoryPoint.trajectory_id)
        .where(TrajectoryPoint.trajectory_id.in_(bindparam("ids", expanding=True)))
        .order_by(TrajectoryPoint.trajectory_id, TrajectoryPoint.t)
    )
    async with session_scope() as session:
        rows = (await session.execute(stmt, {"ids": ids})).all()

    cols = ["trajectory_id", "t", "lat", "lon", "baro", "geo",
            "velocity", "true_track", "vertical_rate", "dt"]
    import pandas as pd

    df = pd.DataFrame(rows, columns=cols)
    out: dict[int, dict] = {}
    for tid, g in df.groupby("trajectory_id", sort=False):
        out[int(tid)] = {
            **{k: g[k].to_numpy(dtype=float) for k in TRAJ_KEYS},
            "dt": float(g["dt"].iloc[0]),
        }
    return out


@torch.no_grad()
def _score_sequence(model, Xs: np.ndarray, window: int) -> np.ndarray:
    """Per-feature-row model residual (NaN for the first ``window`` rows)."""
    n = Xs.shape[0]
    scores = np.full(n, np.nan)
    if n > window:
        starts = np.arange(0, n - window)
        win = np.stack([Xs[s : s + window] for s in starts]).astype(np.float32)
        pred = model(torch.from_numpy(win)).numpy()
        tgt = Xs[starts + window]
        scores[starts + window] = ((pred - tgt) ** 2).mean(axis=1)
    return scores


def _eval_trajectory(traj, model, scaler, window, dt):
    """Return (model_scores, physics_flags) per feature row for one trajectory dict."""
    X = compute_trajectory_features(
        traj["lat"], traj["lon"], traj["baro"], traj["geo"],
        traj["velocity"], traj["true_track"], traj["vertical_rate"], dt,
    )
    if X.shape[0] == 0:
        return np.empty(0), np.empty(0, bool)
    Xs = scaler.transform(X).astype(np.float32)
    return _score_sequence(model, Xs, window), physics_flags(X)[0]


def _metrics(neg_score, neg_phys, pos_score, pos_phys, threshold):
    score = np.concatenate([neg_score, pos_score])
    phys = np.concatenate([neg_phys, pos_phys])
    label = np.concatenate([np.zeros(neg_score.size), np.ones(pos_score.size)])

    model_flag = np.where(np.isnan(score), False, score > threshold)
    combined = model_flag | phys

    valid = ~np.isnan(score)
    auc = (roc_auc_score(label[valid], score[valid])
           if valid.any() and len(np.unique(label[valid])) == 2 else float("nan"))

    pos = label == 1
    return {
        "n_pos": int(pos.sum()),
        "roc_auc": float(auc),
        "precision": float(precision_score(label, combined, zero_division=0)),
        "recall": float(recall_score(label, combined, zero_division=0)),
        "f1": float(f1_score(label, combined, zero_division=0)),
        "recall_physics": float(phys[pos].mean()) if pos.any() else 0.0,
        "recall_model": float(model_flag[pos].mean()) if pos.any() else 0.0,
    }


async def run_eval(n_trajectories: int, seed: int) -> dict:
    model, window, arch = _load_model()
    scaler = load_scaler()
    threshold = load_threshold()["threshold"]
    log.info("Evaluating active model: arch=%s", arch)

    ids = _calib_trajectory_ids()
    trajectories = await _load_trajectories(ids)
    usable = {tid: tr for tid, tr in trajectories.items() if len(tr["lat"]) > window + 5}
    rng = np.random.default_rng(seed)
    chosen = list(usable)
    rng.shuffle(chosen)
    chosen = chosen[:n_trajectories]
    log.info("Evaluating on %d held-out trajectories (window=%d, threshold=%.3f)",
             len(chosen), window, threshold)

    # Clean baseline (shared negatives).
    neg_score, neg_phys = [], []
    for tid in chosen:
        s, p = _eval_trajectory(usable[tid], model, scaler, window, usable[tid]["dt"])
        neg_score.append(s)
        neg_phys.append(p)
    neg_score = np.concatenate(neg_score)
    neg_phys = np.concatenate(neg_phys)
    fpr = float((np.where(np.isnan(neg_score), False, neg_score > threshold) | neg_phys).mean())
    log.info("Clean false-positive rate (combined detector): %.3f", fpr)

    results: dict[str, dict] = {}
    for name, attack in ATTACKS.items():
        pos_score, pos_phys = [], []
        arng = np.random.default_rng(seed + hash(name) % 10_000)
        for tid in chosen:
            inj, lab = attack(usable[tid], arng, usable[tid]["dt"])
            s, p = _eval_trajectory(inj, model, scaler, window, usable[tid]["dt"])
            lab_feat = lab[1:]  # features drop the first point
            m = lab_feat.astype(bool)
            pos_score.append(s[m])
            pos_phys.append(p[m])
        results[name] = _metrics(
            neg_score, neg_phys,
            np.concatenate(pos_score), np.concatenate(pos_phys), threshold,
        )
        log.info("  %-22s AUC=%.3f recall=%.3f precision=%.3f",
                 name, results[name]["roc_auc"], results[name]["recall"],
                 results[name]["precision"])

    await dispose_engine()
    return {"arch": arch, "clean_fpr": fpr, "threshold": threshold,
            "n_trajectories": len(chosen), "attacks": results}


def _print_table(summary: dict) -> None:
    print("\n=== Phase 5 detection metrics (per attack) ===")
    print(f"held-out trajectories: {summary['n_trajectories']}   "
          f"threshold(p99): {summary['threshold']:.3f}   "
          f"clean false-positive rate: {summary['clean_fpr']:.3f}\n")
    hdr = f"{'attack':24s} {'ROC-AUC':>8s} {'recall':>8s} {'precis.':>8s} " \
          f"{'F1':>6s} {'phys':>6s} {'model':>6s} {'n_pos':>7s}"
    print(hdr)
    print("-" * len(hdr))
    for name, m in summary["attacks"].items():
        print(f"{name:24s} {m['roc_auc']:8.3f} {m['recall']:8.3f} {m['precision']:8.3f} "
              f"{m['f1']:6.2f} {m['recall_physics']:6.2f} {m['recall_model']:6.2f} "
              f"{m['n_pos']:7d}")


def _plot(summary: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(summary["attacks"])
    recall = [summary["attacks"][n]["recall"] for n in names]
    auc = [summary["attacks"][n]["roc_auc"] for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - 0.2, recall, 0.4, label="recall (model OR physics)")
    ax.bar(x + 0.2, auc, 0.4, label="ROC-AUC (model score)")
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Detection performance by attack type")
    ax.legend()
    fig.tight_layout()
    EVAL_OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(EVAL_OUT / "attack_metrics.png", dpi=110)


def main() -> None:
    import asyncio
    import sys

    p = argparse.ArgumentParser(description="Run the injection + eval harness (Phase 5)")
    p.add_argument("--n-trajectories", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    summary = asyncio.run(run_eval(args.n_trajectories, args.seed))
    EVAL_OUT.mkdir(parents=True, exist_ok=True)
    (EVAL_OUT / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _print_table(summary)
    _plot(summary)
    print(f"\nSaved metrics -> {EVAL_OUT / 'metrics.json'} and plot -> "
          f"{EVAL_OUT / 'attack_metrics.png'}")


if __name__ == "__main__":
    main()
