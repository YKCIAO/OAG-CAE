from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from typing import Optional, List, Dict, Any

@dataclass
class Stage1Result:
    best_state_dict: Dict[str, torch.Tensor]
    best_val_loss: float
    test_loss: float
    best_epoch: int
    last_log: Dict[str, float]
    test_rows: Optional[List[Dict[str, Any]]] = None


def age_to_group(age):
    start_age_year = 35
    group_interval_year = 10

    group = ((age - start_age_year) // group_interval_year).clamp(0, 6)
    return group.long()
def predict_stage1_simple(encoder, loader, device) -> List[Dict[str, float]]:
    encoder.eval()

    rows: List[Dict[str, float]] = []
    with torch.no_grad():
        for batch in loader:
            # 兼容你的 batch 格式
            if len(batch) == 3:
                x, age_true, _ = batch
            elif len(batch) == 5:
                x, _, age_true, _, _ = batch
            else:
                x = batch[0]
                age_true = batch[1]

            x = x.to(device)
            age_true = age_true.to(device)

            recon, z_age, z_noise, mu, logits = encoder(x)

            pred = mu.view(-1).detach().cpu()
            y = age_true.view(-1).detach().cpu()

            for i in range(len(y)):
                rows.append({
                    "age_true": float(y[i].item()),
                    "age_pred": float(pred[i].item()),
                })

    return rows
@torch.no_grad()
def _eval_stage1(model, loader, loss_fn, device, cfg) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total = 0.0
    n = 0
    log_sum = {}

    for batch in loader:
        if len(batch) == 3:
            x, age_true, mask = batch
            class_true = None
        elif len(batch) == 5:
            x, _, age_true, mask, _ = batch
            class_true = None
        else:
            x = batch[0]
            age_true = batch[1]
            mask = batch[-1]
            class_true = None

        x = x.to(device)
        age_true = age_true.to(device)

        recon, z_age, z_noise, mu, logits = model(x)

        if class_true is None:
            class_true = age_to_group(age_true)

        loss, log = loss_fn(
            recon=recon, x=x, mu=mu, age_true=age_true,
            class_pred_logits=logits, class_true=class_true,
            z_age=z_age, z_noise=z_noise, mask=mask.to(device),
            w_recon=cfg.w_recon, w_age=cfg.w_age, w_ortho=cfg.w_ortho, w_class=cfg.w_class
        )

        bs = x.size(0)
        total += loss.item() * bs
        n += bs
        for k, v in log.items():
            log_sum[k] = log_sum.get(k, 0.0) + float(v) * bs

    avg_loss = total / max(n, 1)
    avg_log = {k: v / max(n, 1) for k, v in log_sum.items()}
    return avg_loss, avg_log


from typing import Optional

def train_stage1(
    model,
    train_loader,
    val_loader,
    test_loader,            # NEW
    optimizer,
    loss_fn,
    device,
    cfg
) -> Stage1Result:
    """
    Stage 1 training with:
      - train/val/test split
      - early stopping on val
      - reporting every report_every epochs (default 20)
      - final test evaluation using the best checkpoint

    Assumes _eval_stage1(model, loader, loss_fn, device, cfg) exists and returns (loss, log_dict).
    """

    best_val = float("inf")
    best_sd = None
    best_epoch = -1
    last_log = {}

    # --- config defaults (safe) ---
    min_delta = getattr(cfg, "min_delta", 0.0)           # optional: require improvement by > min_delta
    patience = 0
    early_stop = cfg.early_stop_patience
    for epoch in range(cfg.epochs_stage1):
        # --------------------
        # Train
        # --------------------
        model.train()
        running = 0.0
        n = 0

        for batch in train_loader:
            if len(batch) == 3:
                x, age_true, mask = batch
                class_true = None
            elif len(batch) == 5:
                x, _, age_true, mask, _ = batch
                class_true = None
            else:
                x = batch[0]
                age_true = batch[1]
                mask = batch[-1]
                class_true = None

            x = x.to(device)
            age_true = age_true.to(device)
            mask = mask.to(device)

            recon, z_age, z_noise, mu, logits = model(x)

            if class_true is None:
                class_true = age_to_group(age_true)

            loss, log = loss_fn(
                recon=recon, x=x, mu=mu, age_true=age_true + torch.randn_like(age_true)*0.5,
                class_pred_logits=logits, class_true=class_true,
                z_age=z_age, z_noise=z_noise, mask=mask,
                w_recon=cfg.w_recon, w_age=cfg.w_age, w_ortho=cfg.w_ortho, w_class=cfg.w_class,
                epoch=epoch, max_epoch=cfg.epochs_stage1
            )

            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            bs = x.size(0)
            running += loss.item() * bs
            n += bs
            last_log = log

        train_loss = running / max(n, 1)

        if epoch < cfg.warmup:
            # warm-up phase: no validation
            pass
        elif (epoch + 1) % 20 == 0:
            # --------------------
            # Validate (every epoch for early stop accuracy; print every 20)
            # --------------------
            val_loss, val_log = _eval_stage1(model, val_loader, loss_fn, device, cfg)
            # --------------------
            # Reporting (every report_every epochs)
            # --------------------
            if cfg.verbose:
                print(
                    f"[Stage1] epoch {epoch + 1}/{cfg.epochs_stage1} "
                    f"train={train_loss:.4f} val={val_loss:.4f} "
                    f"best_val={best_val:.4f} patience={patience}/{early_stop} "
                    f"log={val_log}"
                )
            # --------------------
            # Early stopping on val
            # --------------------
            improved = (best_val - val_loss) > min_delta
            if improved:
                best_val = val_loss
                best_epoch = epoch
                best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= early_stop:
                    print(f"Early stopping at epoch {epoch + 1}. Best val loss={best_val:.6f}")
                    break


    # --------------------
    # Load best checkpoint before testing
    # --------------------
    if best_sd is None:
        best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_sd)  # restore best for final test
    model.to(device)

    # --------------------
    # Final test evaluation (ONLY ONCE)
    # --------------------
    test_loss, test_log = _eval_stage1(model, test_loader, loss_fn, device, cfg)
    test_rows = predict_stage1_simple(model, test_loader, device)
    if cfg.verbose:
        print(f"[Stage1][TEST] loss={test_loss:.4f} log={test_log}")

    #  Stage1Result is a struct constructed on the above
    return Stage1Result(
        best_state_dict=best_sd,
        best_val_loss=float(best_val),
        test_loss=float(test_loss),
        best_epoch=int(best_epoch),
        last_log=last_log,
        test_rows=test_rows
    )

