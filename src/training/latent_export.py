from __future__ import annotations

from typing import Dict, List, Optional, Any
import numpy as np
import torch


def _unpack_batch(batch):
    """
    Compatible with current dataset formats:
    - len(batch) == 3: x, age_true, mask
    - len(batch) == 5: x, extra, age_true, mask, extra2
    - otherwise: x = batch[0], age_true = batch[1]
    """
    if len(batch) == 3:
        x, age_true, _ = batch
    elif len(batch) == 5:
        x, _, age_true, _, _ = batch
    else:
        x = batch[0]
        age_true = batch[1]

    return x, age_true


@torch.no_grad()
def extract_latent_spaces(
    encoder,
    loader,
    device,
    regressor=None,
    split_name: str = "test",
) -> Dict[str, Any]:
    """
    Extract z_age and z_noise from a trained encoder.

    If regressor is provided, also save age_pred based on z_age.
    """

    encoder.eval()

    if regressor is not None:
        regressor.eval()

    z_age_all: List[np.ndarray] = []
    z_noise_all: List[np.ndarray] = []
    age_true_all: List[np.ndarray] = []
    age_pred_all: List[np.ndarray] = []

    for batch in loader:
        x, age_true = _unpack_batch(batch)

        x = x.to(device)
        age_true = age_true.to(device)

        z_age, z_noise = encoder.encode(x)

        z_age_all.append(z_age.detach().cpu().numpy())
        z_noise_all.append(z_noise.detach().cpu().numpy())
        age_true_all.append(age_true.detach().cpu().numpy().reshape(-1))

        if regressor is not None:
            pred = regressor(z_age).view(-1)
            age_pred_all.append(pred.detach().cpu().numpy())

    out = {
        "split": split_name,
        "z_age": np.concatenate(z_age_all, axis=0),
        "z_noise": np.concatenate(z_noise_all, axis=0),
        "age_true": np.concatenate(age_true_all, axis=0),
    }

    if regressor is not None:
        out["age_pred"] = np.concatenate(age_pred_all, axis=0)

    return out