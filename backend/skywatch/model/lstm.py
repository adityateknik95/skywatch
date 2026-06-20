"""LSTM next-step predictor (spec §10).

Given a sliding window of the last ``W`` feature vectors, predict the next feature
vector. Trained with MSE on normal data only; at serve time the prediction residual
is the anomaly score. Deliberately small (this is a low-dimensional, regular signal —
see the README's data-vs-capacity notes); a Transformer is the Phase 8 drop-in.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class LSTMConfig:
    n_features: int
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0

    def to_dict(self) -> dict:
        return {
            "arch": "lstm",
            "n_features": self.n_features,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LSTMConfig":
        return cls(**{k: d[k] for k in ("n_features", "hidden_size", "num_layers", "dropout")})


class LSTMPredictor(nn.Module):
    def __init__(self, cfg: LSTMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.lstm = nn.LSTM(
            input_size=cfg.n_features,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(cfg.hidden_size, cfg.n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, W, F) window -> (B, F) predicted next feature vector."""
        out, _ = self.lstm(x)        # (B, W, H)
        last = out[:, -1, :]         # final timestep's hidden state
        return self.head(last)       # (B, F)
