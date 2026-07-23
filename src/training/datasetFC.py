"""
Datasets for FC-based brain-age modeling.

Notes:
- This code assumes FC matrices are square: (N, R, R).
- By default we keep the UPPER triangle (including diagonal) and mask zeros.
- Age labels: original code divides by 12.0. Here we make it configurable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


ArrayLike = Union[np.ndarray, torch.Tensor]


# -------------------------
# Utility functions
# -------------------------
def as_float32(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32, copy=False)


def triangular_mask_2d(size: int, k: int = 0, upper: bool = True) -> np.ndarray:
    """
    Create a boolean triangular mask.
    - upper=True: keep upper triangle (np.triu)
    - upper=False: keep lower triangle (np.tril)
    k behaves like numpy's k (diagonal offset).
    """
    if upper:
        return np.triu(np.ones((size, size), dtype=bool), k=k)
    return np.tril(np.ones((size, size), dtype=bool), k=k)


def apply_triangle_keep(matrix2d: np.ndarray, *, upper: bool = True, k: int = 0) -> np.ndarray:
    """
    Keep only one triangle of a 2D matrix, set the rest to 0.
    """
    assert matrix2d.ndim == 2, f"Expected 2D matrix, got shape {matrix2d.shape}"
    m = triangular_mask_2d(matrix2d.shape[0], k=k, upper=upper)
    out = np.zeros_like(matrix2d, dtype=np.float32)
    out[m] = matrix2d[m].astype(np.float32, copy=False)
    return out


def apply_triangle_keep_batch(mats3d: np.ndarray, *, upper: bool = True, k: int = 0) -> np.ndarray:
    """
    Apply triangle keep to a batch of matrices: (N, R, R) -> (N, R, R)
    """
    assert mats3d.ndim == 3, f"Expected (N,R,R), got {mats3d.shape}"
    n, r, c = mats3d.shape
    assert r == c, f"FC matrices must be square, got {r}x{c}"

    mask = triangular_mask_2d(r, k=k, upper=upper)  # (R,R)
    out = np.zeros_like(mats3d, dtype=np.float32)
    out[:, mask] = mats3d[:, mask].astype(np.float32, copy=False)
    return out


def add_gaussian_noise(x: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return x
    return x + np.random.normal(0.0, sigma, size=x.shape).astype(np.float32)


def make_mask_nonzero(x: np.ndarray) -> torch.Tensor:
    """
    Mask is 1 where x != 0, else 0. Returned as float32 torch tensor.
    """
    return torch.from_numpy((x != 0).astype(np.float32))


@dataclass
class LabelConfig:
    """
    label_unit:
      - "months": label is in months, convert to years by /12
      - "years": label already in years, no scaling
    """
    label_unit: str = "months"  # or "years"

    def to_years(self, label_value: float) -> float:
        if self.label_unit == "months":
            return float(label_value) / 12.0
        if self.label_unit == "years":
            return float(label_value)
        raise ValueError(f"Unknown label_unit: {self.label_unit}")


# -------------------------
# Base class
# -------------------------
class _BaseDataset(Dataset):
    def __init__(self, train: bool = True, augment: bool = False, noise_sigma: float = 0.02):
        self.train = train
        self.augment = augment
        self.noise_sigma = noise_sigma

    def _maybe_augment(self, x: np.ndarray) -> np.ndarray:
        if self.augment:
            return add_gaussian_noise(x, self.noise_sigma)
        return x


# -------------------------
# Datasets
# -------------------------
class fMRIDataset(_BaseDataset):
    """
    Returns:
      FCVBM (R,R) upper-triangle kept (float32 np array)
      ECmatrix (R,R) upper-triangle kept (float32 np array)
      label_years (float)
      FCmask (torch tensor, R,R)
      ECmask (torch tensor, R,R)
    """

    def __init__(
        self,
        FC_matrices: np.ndarray,   # (N,R,R)
        FC_labels: np.ndarray,     # (N,)
        train: bool = True,
        argument: bool = False,    # keep your original arg name for compatibility
        noise_sigma: float = 0.02,
        keep_upper: bool = True,
        triangle_k: int = 0,
        label_cfg: Optional[LabelConfig] = None,
    ):
        super().__init__(train=train, augment=argument, noise_sigma=noise_sigma)

        assert FC_matrices.shape[0] == len(FC_labels), "Mismatch between number of matrices and labels."
        assert FC_matrices.ndim == 3, f"FC_matrices should be (N,R,R), got {FC_matrices.shape}"

        self.FC_matrices = as_float32(FC_matrices)
        self.FC_labels = FC_labels

        self.keep_upper = keep_upper
        self.triangle_k = triangle_k
        self.label_cfg = label_cfg or LabelConfig(label_unit="months")  # default matches your original /12

    def __len__(self) -> int:
        return int(self.FC_matrices.shape[0])

    def __getitem__(self, index: int):
        fc = self.FC_matrices[index]  # (R,R)
        ec = self.FC_matrices[index]  # you used same array; keep behavior

        # keep triangle
        fc = apply_triangle_keep(fc, upper=self.keep_upper, k=self.triangle_k)
        ec = apply_triangle_keep(ec, upper=self.keep_upper, k=self.triangle_k)



        fc_mask = make_mask_nonzero(fc)
        ec_mask = make_mask_nonzero(ec)
        # optional augmentation on FC only (match your original; EC noise was commented out)
        fc = self._maybe_augment(fc)
        label_raw = self.FC_labels[index]
        label_years = self.label_cfg.to_years(label_raw)

        return fc, ec, label_years, fc_mask, ec_mask


class FCDataset(_BaseDataset):
    """
    Returns:
      FC (R,R) upper-triangle kept
      label_years (float)
      FCmask (torch tensor, R,R)
    """

    def __init__(
        self,
        FC_matrices: np.ndarray,
        FC_labels: np.ndarray,
        train: bool = True,
        argument: bool = False,
        noise_sigma: float = 0.02,
        keep_upper: bool = True,
        triangle_k: int = 0,
        label_cfg: Optional[LabelConfig] = None,
    ):
        super().__init__(train=train, augment=argument, noise_sigma=noise_sigma)
        assert FC_matrices.shape[0] == len(FC_labels), "Mismatch between number of matrices and labels."
        assert FC_matrices.ndim == 3, f"FC_matrices should be (N,R,R), got {FC_matrices.shape}"

        self.FC_matrices = as_float32(FC_matrices)
        self.FC_labels = FC_labels
        self.keep_upper = keep_upper
        self.triangle_k = triangle_k
        self.label_cfg = label_cfg or LabelConfig(label_unit="months")

    def __len__(self) -> int:
        return int(self.FC_matrices.shape[0])

    def __getitem__(self, index: int):
        fc = self.FC_matrices[index]
        fc = apply_triangle_keep(fc, upper=self.keep_upper, k=self.triangle_k)

        fc_mask = make_mask_nonzero(fc)
        fc = self._maybe_augment(fc)
        label_raw = self.FC_labels[index]
        label_years = self.label_cfg.to_years(label_raw)

        return fc, label_years, fc_mask


class AEDataset(_BaseDataset):
    """
    AE dataset for vectorized features (N,D).

    Returns:
      x (D,)
      label (float)  # no /12 here by default (matches your original)
    """

    def __init__(
        self,
        X: np.ndarray,            # (N,D)
        y: np.ndarray,            # (N,)
        train: bool = True,
        argument: bool = False,
        noise_sigma: float = 0.01,  # your original was (-0.01, 0.01) uniform-ish; we use gaussian sigma=0.01
        label_cfg: Optional[LabelConfig] = None,
        scale_labels: bool = False,
    ):
        super().__init__(train=train, augment=argument, noise_sigma=noise_sigma)
        assert X.shape[0] == len(y), "Mismatch between X and y."

        self.X = as_float32(X)
        self.y = y

        self.scale_labels = scale_labels
        self.label_cfg = label_cfg or LabelConfig(label_unit="years")  # your original did NOT /12 here

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, index: int):
        x = self.X[index].copy()  # 1D
        x = self._maybe_augment(x)

        label_raw = self.y[index]
        label_value = self.label_cfg.to_years(label_raw) if self.scale_labels else float(label_raw)

        return x, label_value
