from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import shap


@dataclass
class KernelShapConfig:
    background_size: int = 20
    nsamples: int = 200
    random_seed: int = 42


def run_kernelshap_on_pca(
    model_wrapper,
    x_explain: np.ndarray,
    x_background: np.ndarray,
    cfg: KernelShapConfig
) -> Tuple[np.ndarray, float]:
    """
    Parameters
    ----------
    model_wrapper : callable
        Maps PCA scores -> predicted age
    x_explain : (N_test, K)
        PCA scores of samples to explain
    x_background : (N_train, K)
        Training-set PCA scores used as SHAP background
    cfg : KernelShapConfig

    Returns
    -------
    shap_values : (N_test, K)
    expected_value : float
    """
    rng = np.random.default_rng(cfg.random_seed)

    if x_background.shape[0] <= cfg.background_size:
        bg = x_background
    else:
        bg_idx = rng.choice(x_background.shape[0], size=cfg.background_size, replace=False)
        bg = x_background[bg_idx]

    explainer = shap.KernelExplainer(model_wrapper, bg)

    if x_explain.shape[0] < 20:
        ns = min(cfg.nsamples, max(10, x_explain.shape[0] * 10))
    else:
        ns = cfg.nsamples

    shap_values = explainer.shap_values(x_explain, nsamples=ns)

    # Regression: usually ndarray, but keep safe
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = float(np.array(expected_value).reshape(-1)[0])

    return np.asarray(shap_values), expected_value


def backproject_shap_to_fc(
    shap_values: np.ndarray,
    expected_value: float,
    pca,
    fc_shape: Tuple[int, int]
) -> Dict[str, np.ndarray]:
    """
    Convert PCA-space SHAP values back to FC-space contributions.

    shap_values: (N, K)
    PCA components_: (K, H*W)

    returns dict:
      shap_fc_per_sample: (N, H*W)
      shap_fc_mean_map: (H, W)
    """
    comps = pca.components_  # (K, H*W)
    shap_fc_per_sample = shap_values @ comps  # (N, H*W)
    shap_fc_mean_map = shap_fc_per_sample.mean(axis=0).reshape(fc_shape)

    return {
        "shap_fc_per_sample": shap_fc_per_sample,
        "shap_fc_mean_map": shap_fc_mean_map
    }