"""Train the LSTM next-step predictor on normal data and calibrate the threshold.

Pipeline (spec §13, Phase 4):
  1. Load the Phase-3 feature matrix + scaler; standardize features.
  2. Split *trajectories* into train / val / calibration (no window crosses a split).
  3. Train the LSTM with MSE (Adam, early stopping on val loss).
  4. Score the held-out calibration set; threshold = high percentile of residuals.
  5. Persist model weights, architecture, and threshold to model/artifacts/.

CPU-friendly: windows are sliced lazily and the training set is capped
(``--max-windows``) so this finishes in well under an hour on a laptop CPU.

    python -m skywatch.model.train
    python -m skywatch.model.train --epochs 40 --hidden 64 --max-windows 400000
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from skywatch.config import get_settings
from skywatch.features import (
    FEATURE_MATRIX_PATH,
    FEATURE_META_PATH,
    load_scaler,
)
from skywatch.model.dataset import (
    WindowDataset,
    split_ranges,
    trajectory_ranges,
    window_starts,
)
from skywatch.model.factory import save_active_model, weights_path
from skywatch.model.lstm import LSTMConfig, LSTMPredictor
from skywatch.model.score import (
    calibrate_threshold,
    residual_scores,
    save_threshold,
)
from skywatch.features import ARTIFACTS_DIR

log = logging.getLogger("skywatch.model.train")

MODEL_CONFIG_PATH = ARTIFACTS_DIR / "model_config.json"


def _build_model(arch: str, n_features: int, args):
    if arch == "transformer":
        from skywatch.model.transformer import TransformerConfig, TransformerPredictor

        cfg = TransformerConfig(
            n_features=n_features, d_model=args.d_model, nhead=args.nhead,
            num_layers=args.tf_layers, dim_feedforward=args.ff, dropout=args.dropout,
        )
        return TransformerPredictor(cfg), cfg
    cfg = LSTMConfig(n_features=n_features, hidden_size=args.hidden, num_layers=args.layers)
    return LSTMPredictor(cfg), cfg


def _maybe_subsample(starts: np.ndarray, max_n: int, seed: int) -> np.ndarray:
    if max_n and starts.size > max_n:
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(starts, size=max_n, replace=False))
    return starts


def _run_epoch(model, loader, loss_fn, optimizer, device, train: bool) -> float:
    model.train(train)
    total, n = 0.0, 0
    for w, y in loader:
        w, y = w.to(device), y.to(device)
        if train:
            optimizer.zero_grad()
        pred = model(w)
        loss = loss_fn(pred, y)
        if train:
            loss.backward()
            optimizer.step()
        bs = w.size(0)
        total += loss.item() * bs
        n += bs
    return total / max(n, 1)


def train(args: argparse.Namespace) -> None:
    settings = get_settings()
    window = args.window or settings.model_window_size
    percentile = args.percentile or settings.anomaly_threshold_percentile
    device = "cpu"  # no CUDA on this machine; CPU is the supported path

    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)

    # 1) Load features + scaler, standardize.
    data = np.load(FEATURE_MATRIX_PATH)
    X = data["X"].astype(np.float64)
    trajectory_id = data["trajectory_id"]
    scaler = load_scaler()
    Xs = scaler.transform(X).astype(np.float32)
    n_features = Xs.shape[1]
    log.info("Loaded %d feature rows x %d features", Xs.shape[0], n_features)

    # 2) Trajectory-level split.
    ranges = trajectory_ranges(trajectory_id)
    train_r, val_r, calib_r = split_ranges(ranges, seed=args.seed)
    train_starts = _maybe_subsample(window_starts(train_r, window), args.max_windows, args.seed)
    val_starts = window_starts(val_r, window)
    calib_starts = window_starts(calib_r, window)
    log.info(
        "Windows  train=%d  val=%d  calib=%d  (W=%d)",
        train_starts.size, val_starts.size, calib_starts.size, window,
    )
    if train_starts.size == 0 or val_starts.size == 0:
        raise SystemExit("Not enough data to form windows — collect/reconstruct more.")

    train_dl = DataLoader(WindowDataset(Xs, train_starts, window),
                          batch_size=args.batch, shuffle=True, num_workers=0)
    val_dl = DataLoader(WindowDataset(Xs, val_starts, window),
                        batch_size=args.batch, shuffle=False, num_workers=0)

    # 3) Model + training with early stopping.
    model, cfg = _build_model(args.arch, n_features, args)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience = args.patience
    bad = 0
    log.info("Training %s (%d params, epochs<=%d, patience=%d)...",
             args.arch, n_params, args.epochs, patience)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = _run_epoch(model, train_dl, loss_fn, optimizer, device, train=True)
        with torch.no_grad():
            va = _run_epoch(model, val_dl, loss_fn, optimizer, device, train=False)
        dt = time.time() - t0
        improved = va < best_val - 1e-5
        log.info("epoch %2d  train=%.5f  val=%.5f  (%.1fs)%s",
                 epoch, tr, va, dt, "  *" if improved else "")
        if improved:
            best_val = va
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                log.info("Early stopping at epoch %d (best val=%.5f)", epoch, best_val)
                break

    model.load_state_dict(best_state)

    # 4) Threshold on held-out calibration data.
    if calib_starts.size:
        calib_dl = DataLoader(WindowDataset(Xs, calib_starts, window),
                              batch_size=args.batch, shuffle=False, num_workers=0)
        calib_scores = residual_scores(model, calib_dl, device)
        threshold = calibrate_threshold(calib_scores, percentile)
        save_threshold(threshold, percentile, calib_scores)
    else:
        calib_scores = np.empty(0)
        threshold = float("nan")
        log.warning("No calibration windows — threshold not computed.")

    # 5) Persist model + architecture (via the factory so arch is recorded).
    feature_names = json.loads(FEATURE_META_PATH.read_text())["feature_names"]
    cfg_dict = {**cfg.to_dict(), "window": window, "feature_names": feature_names,
                "best_val_mse": best_val, "n_params": n_params}
    save_active_model(best_state, cfg_dict)

    print(f"\n=== Training summary ({args.arch}, {n_params} params) ===")
    print(f"best val MSE (scaled): {best_val:.5f}")
    if calib_scores.size:
        for p in (50, 90, 99, 99.9):
            print(f"  calib residual p{p:<4}: {np.percentile(calib_scores, p):.5f}")
        print(f"threshold (p{percentile}): {threshold:.5f}")
    print(f"saved model  -> {weights_path(args.arch)}")
    print(f"saved config -> {MODEL_CONFIG_PATH}")


def main() -> None:
    p = argparse.ArgumentParser(description="Train the next-step predictor (Phase 4/8)")
    p.add_argument("--arch", choices=["lstm", "transformer"], default="lstm")
    p.add_argument("--window", type=int, default=0, help="0 = use settings.model_window_size")
    p.add_argument("--hidden", type=int, default=64, help="LSTM hidden size")
    p.add_argument("--layers", type=int, default=1, help="LSTM layers")
    p.add_argument("--d-model", type=int, default=64, help="Transformer d_model")
    p.add_argument("--nhead", type=int, default=4, help="Transformer attention heads")
    p.add_argument("--tf-layers", type=int, default=2, help="Transformer encoder layers")
    p.add_argument("--ff", type=int, default=128, help="Transformer feedforward dim")
    p.add_argument("--dropout", type=float, default=0.1, help="Transformer dropout")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--max-windows", type=int, default=500_000)
    p.add_argument("--percentile", type=float, default=0.0, help="0 = use settings")
    p.add_argument("--threads", type=int, default=0, help="torch CPU threads (0=default)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    train(args)


if __name__ == "__main__":
    main()
