"""Model registry: build / load / save the active model by ``arch`` (Phase 8).

The active model is described by ``model_config.json`` (which carries an ``arch``
field — ``lstm`` by default for backward compatibility) and its weights live in
``<arch>.pt``. Everything downstream (training, scoring, serving, eval) goes through
``load_active_model`` so swapping LSTM <-> Transformer is a one-flag change.
"""

from __future__ import annotations

import json

import torch
from torch import nn

from skywatch.features import ARTIFACTS_DIR
from skywatch.model.lstm import LSTMConfig, LSTMPredictor

CONFIG_PATH = ARTIFACTS_DIR / "model_config.json"


def weights_path(arch: str):
    return ARTIFACTS_DIR / f"{arch}.pt"


def build_model(cfg: dict) -> nn.Module:
    arch = cfg.get("arch", "lstm")
    if arch == "transformer":
        from skywatch.model.transformer import TransformerConfig, TransformerPredictor

        return TransformerPredictor(TransformerConfig.from_dict(cfg))
    return LSTMPredictor(LSTMConfig.from_dict(cfg))


def load_active_model() -> tuple[nn.Module, dict]:
    cfg = json.loads(CONFIG_PATH.read_text())
    arch = cfg.get("arch", "lstm")
    model = build_model(cfg)
    model.load_state_dict(torch.load(weights_path(arch), weights_only=True))
    model.eval()
    return model, cfg


def save_active_model(state_dict, cfg: dict) -> None:
    arch = cfg.get("arch", "lstm")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, weights_path(arch))
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
