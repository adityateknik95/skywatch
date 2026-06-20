"""Transformer next-step predictor (Phase 8 stretch, spec §10).

A small Transformer encoder over the input window — a drop-in alternative to the
LSTM with the *same* interface ((B, W, F) -> (B, F)) and the same residual scoring,
so the rest of the system (training, scoring, serving) is unchanged. Attention can
relate the first and last points of the window directly, which is exactly what the
hardest attack (gradual drift) needs — the LSTM's memory of early points fades.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class TransformerConfig:
    n_features: int
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1
    arch: str = "transformer"

    def to_dict(self) -> dict:
        return {
            "arch": "transformer",
            "n_features": self.n_features,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TransformerConfig":
        keys = ("n_features", "d_model", "nhead", "num_layers", "dim_feedforward", "dropout")
        return cls(**{k: d[k] for k in keys})


def _sinusoidal_pe(length: int, d_model: int, device) -> torch.Tensor:
    pos = torch.arange(length, device=device).unsqueeze(1).float()
    idx = torch.arange(0, d_model, 2, device=device).float()
    div = torch.exp(-math.log(10000.0) * idx / d_model)
    pe = torch.zeros(length, d_model, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)  # (1, length, d_model)


class TransformerPredictor(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.input = nn.Linear(cfg.n_features, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
        self.head = nn.Linear(cfg.d_model, cfg.n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, W, F) window -> (B, F) predicted next feature vector."""
        h = self.input(x)
        h = h + _sinusoidal_pe(h.size(1), self.cfg.d_model, x.device)
        h = self.encoder(h)
        return self.head(h[:, -1, :])  # predict from the last position's encoding
