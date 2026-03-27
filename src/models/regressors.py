# src/models/regressors.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperatureSoftmax(nn.Module):
    def __init__(self, dim: int = 1, temperature: float = 1.5):
        super().__init__()
        self.dim = dim
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(x / self.temperature, dim=self.dim)


# -------------------------
# Attention Regressor
# -------------------------
@dataclass
class AttentionRegressorConfig:
    in_dim: int
    tau: float = 1.5
    scale_by_dim: bool = True
    dropout: float = 0.15


class AttentionRegressor(nn.Module):
    """
    Feature-wise attention over latent z_age: z -> w -> weighted_z -> age

    Input:  z_age (B, D)
    Output: age_pred (B,)
    """

    def __init__(self, in_dim: int, tau: float = 1.5, scale_by_dim: bool = True, dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.tau = tau
        self.scale_by_dim = scale_by_dim

        self.attn = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, in_dim),
        )
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(in_dim, 1),
        )

    def forward(self, z_age: torch.Tensor, return_weights: bool = False):
        """
        return_weights=True will also return attention weights (B, D)
        """
        if z_age.dim() != 2:
            raise ValueError(f"Expected z_age shape (B,D), got {z_age.shape}")

        d = z_age.size(1)
        logits = self.attn(z_age)  # (B, D)
        w = F.softmax(logits / self.tau, dim=1)  # (B, D)

        if self.scale_by_dim:
            w = w * d

        weighted = z_age * w
        pred = self.regressor(weighted).squeeze(1)  # (B,)

        if return_weights:
            return pred, w
        return pred


# -------------------------
# Conv Regressor with Gate + Multi-scale Conv1d
# -------------------------
@dataclass
class ConvAgeRegressorConfig:
    in_dim: int  # input feature dim (e.g., 256)
    hidden_channels: int = 1  # C
    length: int = 32  # L, must satisfy C*L == in_dim
    kernel_sizes: Tuple[int, ...] = (3, 5)
    dropout: float = 0.1
    gate_softmax_dim: int = 2  # softmax over channel dim by default
    residual_scale: float = 0.0
    bias_shift_init: float = 1.0
    tau: float = 2.5


class ConvAgeRegressor(nn.Module):
    """
    Turn (B, D) into (B, C, L), apply channel gate, multi-scale conv, then regress.

    Input:  z_age (B, D)
    Output: age_pred (B,)
    Extras: optionally return gate entropy for regularization
    """

    def __init__(self, cfg: ConvAgeRegressorConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.hidden_channels * cfg.length != cfg.in_dim:
            raise ValueError(
                f"ConvAgeRegressorConfig requires hidden_channels*length == in_dim, "
                f"got {cfg.hidden_channels}*{cfg.length} != {cfg.in_dim}"
            )

        self.ln = nn.LayerNorm(cfg.in_dim)

        # Gate: produces gate weights in same shape (B, C, L)
        self.gate_conv = nn.Sequential(
            nn.Conv1d(cfg.hidden_channels, cfg.hidden_channels, kernel_size=1, padding=0, bias=True),
            TemperatureSoftmax(dim=cfg.gate_softmax_dim, temperature=cfg.tau),
        )
        nn.init.normal_(self.gate_conv[0].weight, mean=0.0, std=0.02)
        nn.init.constant_(self.gate_conv[0].bias, 0.0)

        # Multi-scale conv blocks
        blocks = []
        for k in cfg.kernel_sizes:
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(cfg.hidden_channels, cfg.hidden_channels, kernel_size=1, padding=0),
                    nn.Dropout(cfg.dropout),
                    nn.GELU(),
                    nn.Conv1d(cfg.hidden_channels, cfg.hidden_channels, kernel_size=k, padding=k // 2, bias=True),
                    nn.GELU(),
                )
            )
        self.conv_blocks = nn.ModuleList(blocks)

        self.reg_head = nn.Sequential(
            nn.Flatten(),  # (B, C*L)
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_channels * cfg.length, 1),
        )

        self.bias_shift = nn.Parameter(torch.tensor(cfg.bias_shift_init, dtype=torch.float32))

    def forward(self, z_age: torch.Tensor):
        if z_age.dim() != 2 or z_age.size(1) != self.cfg.in_dim:
            raise ValueError(f"Expected z_age (B,{self.cfg.in_dim}), got {z_age.shape}")

        z = self.ln(z_age)
        x = z.view(-1, self.cfg.hidden_channels, self.cfg.length)  # (B, C, L)

        gate = self.gate_conv(x)  # (B, C, L)
        # True gating + residual
        x_gated = x * gate + x * self.cfg.residual_scale

        # Multi-scale fusion
        feats = [block(x_gated) for block in self.conv_blocks]  # list of (B,C,L)
        fused = torch.stack(feats, dim=0).mean(dim=0)  # (B,C,L)

        pred = self.reg_head(fused).squeeze(1) # (B,)

        return pred + self.bias_shift

# src/models/regressors.py
