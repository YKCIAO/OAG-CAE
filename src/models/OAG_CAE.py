# src/models/orthogonal_autoencoder.py
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

# External dependency: you already have this in Regressiondemo.py
from src.models.regressors import AttentionRegressor


@dataclass
class OAEConfig:
    input_size: int = 278            # FC matrix size: (278,278)
    z_age_dim: int = 16
    z_noise_dim: int = 112
    conv1_out: int = 16
    conv2_out: int = 32
    conv_kernel: int = 7
    conv_stride: int = 3
    dropout: float = 0.15
    n_age_groups: int = 7
    age_noise_sigma: float = 0.02    # noise added to z_age before reg/class
    tau: float = 1.5


class OrthogonalAutoEncoder(nn.Module):
    """
    Orthogonal AutoEncoder that splits latent into z_age and z_noise,
    reconstructs FC, predicts age (mu) from z_age, and classifies age group.

    Input:
      x: (B,278,278) or (B,1,278,278)

    Output:
      recon: (B,1,278,278)
      z_age: (B,z_age_dim)
      z_noise: (B,z_noise_dim)
      mu: (B,1) or (B,) depending on regressor
      logits: (B,n_age_groups)
    """

    def __init__(self, cfg: OAEConfig | None = None):
        super().__init__()
        self.cfg = cfg

        self.z_age_dim = self.cfg.z_age_dim
        self.z_noise_dim = self.cfg.z_noise_dim
        self.latent_dim = self.cfg.z_age_dim + self.cfg.z_noise_dim

        # Encoder conv: (1,278,278) -> (32,29,29) with your kernel/stride
        self.encoder = nn.Sequential(
            nn.Conv2d(1, self.cfg.conv1_out, kernel_size=self.cfg.conv_kernel, stride=self.cfg.conv_stride),
            nn.GELU(),
            nn.Conv2d(self.cfg.conv1_out, self.cfg.conv2_out, kernel_size=self.cfg.conv_kernel, stride=self.cfg.conv_stride),
        )
        self.flatten = nn.Flatten()

        # FC projection: 32*29*29 -> 256
        self.fc1 = nn.Sequential(
            nn.Linear(self.cfg.conv2_out * 29 * 29, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
        )

        # Latent heads
        self.to_z_age = nn.Linear(256, self.z_age_dim)
        self.to_z_noise = nn.Linear(256, self.z_noise_dim)

        # Decoder: latent -> (32,29,29) -> (1,278,278)
        self.fc3 = nn.Linear(self.latent_dim, 1024)
        self.fc4 = nn.Linear(1024, self.cfg.conv2_out * 29 * 29)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.cfg.conv2_out, self.cfg.conv1_out, kernel_size=self.cfg.conv_kernel, stride=self.cfg.conv_stride, output_padding=0),
            nn.GELU(),
            nn.ConvTranspose2d(self.cfg.conv1_out, 1, kernel_size=self.cfg.conv_kernel, stride=self.cfg.conv_stride, output_padding=1),
        )

        # Age regressor from z_age
        self.regressor = AttentionRegressor(in_dim=self.z_age_dim, tau=self.cfg.tau)

        # Age group classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.z_age_dim),
            nn.Dropout(self.cfg.dropout),
            nn.ReLU(),
            nn.Linear(self.z_age_dim, self.cfg.n_age_groups),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B,1,H,W)
        elif x.dim() == 4 and x.size(1) != 1:
            raise ValueError(f"Expected channel=1 for 4D input, got {x.shape}")

        x = self.encoder(x)
        x = self.flatten(x)
        x = self.fc1(x)
        z_age = self.to_z_age(x)
        z_noise = self.to_z_noise(x)
        return z_age, z_noise

    def decode(self, z_age: torch.Tensor, z_noise: torch.Tensor) -> torch.Tensor:
        z = torch.cat([z_age, z_noise], dim=1)
        x = self.fc3(z)
        x = self.fc4(x)
        x = x.view(x.size(0), self.cfg.conv2_out, 29, 29)
        y = self.decoder(x)
        # ✅ assert bugs
        assert y.dim() == 4, f"decoder output must be 4D, got {y.shape}"
        assert y.size(0) == z_age.size(0), f"batch mismatch: {y.size(0)} vs {z_age.size(0)}"
        assert y.size(1) == 1, f"channel must be 1, got {y.shape}"
        return y.view(x.size(0), 1, self.cfg.input_size, self.cfg.input_size)

    def forward(self, x: torch.Tensor):
        z_age, z_noise = self.encode(x)
        recon = self.decode(z_age, z_noise)

        # noisy z_age for auxiliary heads (keep your original intent)
        z_age_noisy = z_age + self.cfg.age_noise_sigma * torch.randn_like(z_age)

        mu = self.regressor(z_age_noisy)
        logits = self.classifier(z_age_noisy)

        return recon, z_age, z_noise, mu, logits
