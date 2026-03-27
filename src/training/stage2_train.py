from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

@dataclass
class Stage2Result:
    best_state_dict: dict
    best_val_mae: float
    test_mae: float = float("nan")
    test_rows: Optional[List[Dict[str, Any]]] = None
    train_rows: Optional[List[Dict[str, Any]]] = None
    best_epoch: int = -1


def predict_stage2_simple(encoder, regressor, loader, device) -> List[Dict[str, float]]:
    encoder.eval()
    regressor.eval()

    rows: List[Dict[str, float]] = []
    with torch.no_grad():
        for batch in loader:
            # This let the dateset can load different structure,
            # based on these structure you can add such VBM, freesurfer features etc.
            if len(batch) == 3:
                x, age_true, _ = batch
            elif len(batch) == 5:
                x, _, age_true, _, _ = batch
            else:
                x = batch[0]
                age_true = batch[1]

            x = x.to(device)
            age_true = age_true.to(device)

            z_age, _ = encoder.encode(x)
            pred = regressor(z_age).view(-1)

            pred = pred.detach().cpu()
            y = age_true.view(-1).detach().cpu()

            for i in range(len(y)):
                rows.append({
                    "age_true": float(y[i].item()),
                    "age_pred": float(pred[i].item()),
                })

    return rows


@torch.no_grad()
def _eval_stage2(encoder, regressor, loader, device) -> float:
    encoder.eval()
    regressor.eval()

    all_true = []
    all_pred = []

    for batch in loader:
        if len(batch) == 3:
            x, age_true, _ = batch
        elif len(batch) == 5:
            x, _, age_true, _, _ = batch
        else:
            x = batch[0]
            age_true = batch[1]

        x = x.to(device)
        age_true = age_true.to(device)

        z_age, z_noise = encoder.encode(x)
        pred = regressor(z_age)

        all_true.append(age_true.detach().cpu())
        all_pred.append(pred.detach().cpu())

    y_true = torch.cat(all_true).numpy().reshape(-1)
    y_pred = torch.cat(all_pred).numpy().reshape(-1)
    mae = float(abs(y_true - y_pred).mean())
    return mae


def train_stage2(
    encoder,
    regressor,
    train_loader,
    val_loader,
    test_loader,
    optimizer,
    scheduler,
    device,
    cfg
) -> Stage2Result:
    best_mae = float("inf")
    best_sd = None
    best_epoch = -1

    min_delta = getattr(cfg, "min_delta", 0.0)           # optional: require improvement by > min_delta
    patience = 0
    early_stop = cfg.early_stop_patience

    for epoch in range(cfg.epochs_stage2):
        encoder.eval()  # stage2 eval encoder
        regressor.train()

        for batch in train_loader:
            if len(batch) == 3:
                x, age_true, _ = batch
            elif len(batch) == 5:
                x, _, age_true, _, _ = batch
            else:
                x = batch[0]
                age_true = batch[1]

            x = x.to(device)
            age_true = age_true.to(device)

            with torch.no_grad():
                z_age, z_noise = encoder.encode(x)
                noise = torch.randn_like(z_age) * 0.02
                z_age = z_age + noise
                age_smooth = age_true + torch.randn_like(age_true) * 0.5
            pred = regressor(z_age)  # (B,) or (B,1)
            pred = pred.view(-1)
            loss = F.l1_loss(pred, age_smooth.float().view(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(regressor.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
        if epoch < cfg.warmup:
            # warm-up phase: no validation
            pass
        elif (epoch + 1) % 20 == 0:
            # --------------------
            # Validate (every epoch for early stop accuracy; print every 20)
            # --------------------
            # --- val metric for early stopping ---
            val_mae = _eval_stage2(encoder, regressor, val_loader, device)
            train_mae = _eval_stage2(encoder, regressor, train_loader, device)
            # --- report every 20 epochs ---
            if cfg.verbose:
                print(
                    f"[Stage2] epoch {epoch+1}/{cfg.epochs_stage2} "
                    f"train_mae={train_mae:.4f} val_mae={val_mae:.4f} best={best_mae:.4f} "
                    f"patience={patience}/{early_stop}"
                )
            improved = (best_mae - val_mae) > min_delta

            if improved:
                best_mae = val_mae
                best_epoch = epoch
                patience = 0
                best_sd = {k: v.detach().cpu().clone() for k, v in regressor.state_dict().items()}
            else:
                patience += 1
                if patience >= early_stop:
                    print(f"Early stopping at epoch {epoch + 1}. Best val loss={best_mae:.6f}")
                    break


    if best_sd is None:
        best_sd = {k: v.detach().cpu().clone() for k, v in regressor.state_dict().items()}

    # --- restore best regressor before final test ---
    regressor.load_state_dict(best_sd)
    regressor.to(device)

    # --- final test (ONLY ONCE) ---
    test_mae = _eval_stage2(encoder, regressor, test_loader, device)
    test_rows = predict_stage2_simple(encoder, regressor, test_loader, device)
    train_rows = predict_stage2_simple(encoder, regressor, train_loader, device)
    if cfg.verbose:
        print(f"[Stage2][TEST] mae={test_mae:.4f} n={len(test_rows)}")

    # Stage2Result is a struct constructed on the above
    return Stage2Result(
        best_state_dict=best_sd,
        best_val_mae=float(best_mae),

        test_mae=float(test_mae),
        test_rows=test_rows,
        train_rows=train_rows,
        best_epoch=int(best_epoch),
    )

