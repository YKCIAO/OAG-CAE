from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.training.utils import reset_seeds, z_score_normalize_fit, z_score_normalize_apply
from src.training.stage1_train import train_stage1
from src.training.stage2_train import train_stage2
from src.training.metrics import compute_metrics
from src.training.io_training import save_json, try_export_csv

# src import
from src.models.OAG_CAE import OrthogonalAutoEncoder,OAEConfig
from src.models.regressors import ConvAgeRegressor, ConvAgeRegressorConfig
from src.training.losses import orthogonal_guided_loss


@dataclass
class TrainConfig:
    seed: int = 1000
    device: str = "cuda"  # or "cpu"
    num_workers: int = os.cpu_count()-1
    batch_num: int = 2

    # stage1
    epochs_stage1: int = 2000
    lr_stage1: float = 2e-4
    wd_stage1: float = 2e-3
    warmup: int = 500
    tau_aux: float = 1.5



    # stage2
    epochs_stage2: int = 2000
    lr_stage2: float = 3e-2
    wd_stage2: float = 3e-3
    early_stop_patience: int = 10
    pct_start: float = 0.15
    anneal_strategy: str = "cos"
    div_factor: float = 10.0
    final_div_factor: float = 1000.0
    tau_regressor: float = 1.5
    hidden_channel: int = 1
    gate_softmax_dim: int = 2

    # loss weights
    w_recon: float = 0.1
    w_age: float = 0.3
    w_ortho: float = 0.2
    w_class: float = 0.1

    grad_clip: float = 5.0
    verbose: bool = True
    min_delta: float = 0.0

    # model dims
    input_dim: int = 278
    z_age_dim: int = 32
    z_noise_dim: int = 32

    # outputs
    out_dir: str = "outputs"
    # inputs
    input_dir: str = "inputs"

def _resolve_device(cfg: TrainConfig) -> torch.device:
    if cfg.device == "cuda" and torch.cuda.is_available():
        print(f"\nCUDA is available, using {torch.cuda.device_count()}")
        return torch.device("cuda")
    print(f"\nUsing cpu")
    return torch.device("cpu")


def _build_loaders(train_ds, val_ds, test_ds, cfg: TrainConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_ds, batch_size=int(len(train_ds)/TrainConfig.batch_num), shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, persistent_workers=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=int(len(val_ds)), shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, persistent_workers=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=int(len(test_ds)), shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, persistent_workers=True
    )
    return train_loader, val_loader, test_loader


def _build_models(cfg: TrainConfig, device: torch.device):
    # Stage1 OAG-CAE
    OAG_cfg =OAEConfig(input_size=cfg.input_dim, z_age_dim=cfg.z_age_dim, z_noise_dim=cfg.z_noise_dim, tau=cfg.tau_aux)
    encoder = OrthogonalAutoEncoder(OAG_cfg).to(device)

    # Stage2 regressor
    reg_cfg = ConvAgeRegressorConfig(in_dim=cfg.z_age_dim, hidden_channels=cfg.hidden_channel, length=cfg.z_age_dim, tau=cfg.tau_regressor,
                                     gate_softmax_dim=cfg.gate_softmax_dim)
    regressor = ConvAgeRegressor(reg_cfg).to(device)
    return encoder, regressor


def train_and_eval(
    folds: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    train_dataset_ctor,
    cfg: TrainConfig
) -> float:
    """
    folds: list of (train_x, train_y, val_x, val_y, test_x, test_y)
    train_dataset_ctor: callable -> (train_ds, val_ds, test_ds) from arrays

    """
    reset_seeds(cfg.seed)
    device = _resolve_device(cfg)

    fold_mae = []

    for i, (train_x, train_y, val_x, val_y, test_x, test_y) in enumerate(folds):
        print(f"\n===== Fold {i+1}/{len(folds)} =====")
        # debug on fold=2
        #if i!=2:
        #    continue
        # normalize based on the mean and std of train set
        mean, std = z_score_normalize_fit(train_x)
        train_x_n = z_score_normalize_apply(train_x, mean, std)
        val_x_n = z_score_normalize_apply(val_x, mean, std)
        test_x_n = z_score_normalize_apply(test_x, mean, std)

        train_ds, val_ds, test_ds = train_dataset_ctor(train_x_n, train_y, val_x_n, val_y, test_x_n, test_y)

        train_loader, val_loader, test_loader = _build_loaders(train_ds, val_ds, test_ds, cfg)

        encoder, regressor = _build_models(cfg, device)

        optimizer1 = torch.optim.AdamW(encoder.parameters(), lr=cfg.lr_stage1, weight_decay=cfg.wd_stage1)
        s1 = train_stage1(
            model=encoder,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            optimizer=optimizer1,
            loss_fn=orthogonal_guided_loss,
            device=device,
            cfg=cfg
        )
        encoder.load_state_dict(s1.best_state_dict)

        optimizer2 = torch.optim.AdamW(regressor.parameters(), lr=cfg.lr_stage2, weight_decay=cfg.wd_stage2)
        scheduler1 = torch.optim.lr_scheduler.OneCycleLR(
            optimizer2,
            max_lr=cfg.lr_stage2,
            epochs=cfg.epochs_stage2,
            steps_per_epoch=len(train_loader),
            pct_start=cfg.pct_start,
            anneal_strategy=cfg.anneal_strategy,
            div_factor=cfg.div_factor,
            final_div_factor=cfg.final_div_factor
        )
        s2 = train_stage2(
            encoder=encoder,
            regressor=regressor,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            optimizer=optimizer2,
            scheduler=scheduler1,
            device=device,
            cfg=cfg
        )
        #
        save_dir = f'../result/fold{i + 1}'
        os.makedirs(save_dir, exist_ok=True)

        #
        df = pd.DataFrame(s1.test_rows)
        save_path = os.path.join(save_dir, "stage1_test_predictions.xlsx")
        df.to_excel(save_path, index=False)
        #
        df = pd.DataFrame(s2.train_rows)
        save_path = os.path.join(save_dir, "stage2_train_predictions.xlsx")
        df.to_excel(save_path, index=False)
        #
        fold_mae.append(s2.best_val_mae)
        df = pd.DataFrame(s2.test_rows)
        save_path = os.path.join(save_dir, "stage2_test_predictions.xlsx")
        df.to_excel(save_path, index=False)
        #

        # printf the result on screen
        save_json(f"{cfg.out_dir}/fold{i+1}_summary.json", {
            "fold": i + 1,
            "stage1_best_val_loss": s1.best_val_loss,
            "stage2_best_val_mae": s2.best_val_mae,
            "stage1_test_loss": s1.test_loss,
            "stage2_test_mae": s2.test_mae
        })
        torch.save(s1.best_state_dict, f'../result/fold{i + 1}/fold{i + 1}_oag_cae_bestvalid.pth')
        torch.save(s2.best_state_dict, f'../result/fold{i + 1}/fold{i + 1}_regressor_bestvalid.pth')
    mean_mae = float(np.mean(fold_mae))
    save_json(f"{cfg.out_dir}/cv_summary.json", {"mean_mae": mean_mae, "fold_mae": fold_mae})
    print(f"\n===== CV DONE | mean MAE = {mean_mae:.4f} =====")
    return mean_mae
