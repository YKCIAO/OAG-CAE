# src/training/losses.py
from __future__ import annotations

from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F


def orthogonal_loss(z_age: torch.Tensor, z_noise: torch.Tensor) -> torch.Tensor:
    """
    Encourage z_age and z_noise to be orthogonal (sample-wise cosine similarity -> 0).
    """
    z_age_centered = z_age - z_age.mean(0)
    z_noise_centered = z_noise - z_noise.mean(0)

    z_age_norm = F.normalize(z_age_centered, dim=1)
    z_noise_norm = F.normalize(z_noise_centered, dim=1)

    # abs cosine similarity averaged over batch
    return torch.abs(torch.sum(z_age_norm * z_noise_norm, dim=1)).mean()


def age_correlation_loss(z_age: torch.Tensor, age_true: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Negative correlation between z_age (all dims) and age_true.
    (Kept from your original version.)
    """
    z = (z_age - z_age.mean(0)) / (z_age.std(0) + eps)
    y = (age_true - age_true.mean(0)) / (age_true.std(0) + eps)
    y = y.view(-1, 1)
    cov = torch.mean(z * y) - torch.mean(z) * torch.mean(y)
    return -cov / (z.std() * y.std() + eps)


def decorrelation_loss(z1, z2, sim_w=25.0, var_w=25.0, cov_w=1.0, eps=1e-4):
    """
    VICReg-style: similarity + variance + covariance penalty.
    (Your original function, just cleaned formatting.)
    """
    sim = F.mse_loss(z1, z2)

    def std_loss(z):
        std = z.std(dim=0)
        return torch.mean(F.relu(1.0 - std))

    v = std_loss(z1) + std_loss(z2)

    def offdiag(m):
        return (m - torch.diag(torch.diag(m)))

    z1c = z1 - z1.mean(0)
    z2c = z2 - z2.mean(0)
    n = z1.size(0)
    cov1 = (z1c.T @ z1c) / (n - 1 + eps)
    cov2 = (z2c.T @ z2c) / (n - 1 + eps)
    c = offdiag(cov1).pow(2).mean() + offdiag(cov2).pow(2).mean()

    return sim_w * sim + var_w * v + cov_w * c


def orthogonal_guided_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    age_true: torch.Tensor,
    class_pred_logits: torch.Tensor,
    class_true: torch.Tensor,
    z_age: torch.Tensor,
    z_noise: torch.Tensor,
    mask: torch.Tensor,
    w_recon: float = 1.0,
    w_age: float = 0.3,
    w_ortho: float = 0.3,
    w_class: float = 0.01,
    # extra weights kept for future switches (you had them but commented out)
    w_corr: float = 0.01,
    w_maxz: float = 0.2,
    w_decor: float = 0.02,
    w_center: float = 0.001,
    epoch: Optional[int] = None,
    max_epoch: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Main composite loss (reconstruction + age regression + age-group classification + orthogonality).

    Notes:
    - recon loss uses masked MSE: mse((recon*mask), (x*mask))
    - age loss uses huber between mu and age_true
    - class loss uses cross_entropy(logits, class_true)
      (Your previous KL(onehot) is equivalent but less standard.)
    """
    # --- basic losses ---
    if recon.dim() == 4 and x.dim() == 3:
        x = x.unsqueeze(1)
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    recon_loss = F.mse_loss(recon * mask, x * mask)

    # Make shapes robust for huber
    age_true_f = age_true.float().view(-1)
    mu_f = mu.float().view(-1)
    age_loss = F.huber_loss(mu_f, age_true_f)

    # --- classification loss (standard) ---
    # class_true expected as integer class index tensor shape (B,)
    #class_true = class_true.long().view(-1)
    #class_loss = F.cross_entropy(class_pred_logits, class_true)
    # --- classification loss (KL) ---

    class_prob = F.log_softmax(class_pred_logits, dim=1)
    class_true = F.one_hot(class_true, num_classes=7).float()
    class_loss = F.kl_div(class_prob, F.softmax(class_true, dim=1), reduction="batchmean")
    '''
    K = 7
    sigma = 0.6 # Smaller means lower smoothing level, keep it higher than 0.6
    idx = torch.arange(K, device=class_true.device).float()

    target = torch.exp(-(idx[None, :] - class_true[:, None].float()) ** 2 / (2 * sigma ** 2))
    target = target / target.sum(dim=1, keepdim=True)

    logp = F.log_softmax(class_pred_logits, dim=1)
    class_loss = F.kl_div(logp, target, reduction="batchmean")
    '''
    # --- orthogonality ---
    ortho = orthogonal_loss(z_age, z_noise)

    # (optional extras: kept but off by default to match your current behavior)
    # corr = age_correlation_loss(z_age, age_true_f)
    # loss_max_z = torch.max(torch.abs(z_age), dim=1)[0].mean()

    # dynamic weights hook (kept; currently no real schedule)
    w_age_dyn = w_age
    w_corr_dyn = w_corr

    total = (
        w_recon * recon_loss
        + w_age_dyn * age_loss
        + w_class * class_loss
        + w_ortho * ortho
        # + w_corr_dyn * corr
        # + w_maxz * loss_max_z
    )

    logs = {
        "total": round(float(total.item()), 4),
        "recon": round(float(recon_loss.item()), 4),
        "auxilregss": round(float(age_loss.item()), 4),
        "auxilclass": round(float(class_loss.item()), 4),
        "ortho": round(float(ortho.item()), 4),
    }
    return total, logs


# -------------------------
# Extra helper losses you had (kept here)
# -------------------------
def drop_feature(z: torch.Tensor, drop_prob: float = 0.2) -> torch.Tensor:
    if not z.requires_grad:
        return z
    m = (torch.rand_like(z) > drop_prob).float()
    return z * m


def compute_age_relevant_mask(matrix: torch.Tensor, age_labels: torch.Tensor, lower=0.4, upper=0.8) -> torch.Tensor:
    x = matrix - matrix.mean(0)
    x = x / (matrix.std(0) + 1e-8)
    y = age_labels.unsqueeze(1) - age_labels.mean(0)
    y = y / (age_labels.std(0) + 1e-8)
    corr = torch.mean(x * y, dim=0)
    return (corr.abs() >= lower) & (corr.abs() <= upper)


def smooth_labels(age_group: torch.Tensor, num_classes: int, smoothing: float = 0.1) -> torch.Tensor:
    smoothed = torch.full(
        (age_group.size(0), num_classes),
        smoothing / (num_classes - 1),
        device=age_group.device,
    )
    smoothed.scatter_(1, age_group.unsqueeze(1), 1 - smoothing)
    return smoothed


def topk_corr_loss(z_age: torch.Tensor, age: torch.Tensor, k: int = 7, target_corr: float = 1.1, alpha: float = 1.0) -> torch.Tensor:
    b, d = z_age.shape
    losses = []
    for i in range(d):
        z = z_age[:, i]
        z_std = (z - z.mean()) / (z.std() + 1e-6)
        a_std = (age - age.mean()) / (age.std() + 1e-6)
        corr = torch.mean(z_std * a_std)
        losses.append(torch.relu(target_corr - torch.abs(corr)))
    losses = torch.stack(losses)
    return torch.topk(losses, k=k, largest=True)[0].mean() * alpha


def age_corrcoef_loss(z_age: torch.Tensor, age_true: torch.Tensor) -> torch.Tensor:
    age_true = age_true.float().view(-1)
    z_age = z_age.float()
    combined = torch.cat([z_age, age_true.unsqueeze(1)], dim=1)
    corr_matrix = torch.corrcoef(combined.T)
    corr_with_age = corr_matrix[:-1, -1]
    return -corr_with_age.mean()


def rank_loss(z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    diff = z.unsqueeze(1) - z.unsqueeze(0)
    label_diff = y.unsqueeze(1) - y.unsqueeze(0)
    sign_mismatch = (diff * label_diff < 0).float()
    return sign_mismatch.mean()
