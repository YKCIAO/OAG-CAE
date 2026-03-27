# scripts/training.py
import argparse
from pathlib import Path
from src.training.datasetFC import FCDataset
from src.training.train_pipeline import TrainConfig, train_and_eval
from src.training.utils import reset_seeds
from typing import Sequence, Tuple, List
import numpy as np


def dataset_ctor(train_x, train_y, val_x, val_y, test_x, test_y):
    train_ds = FCDataset(train_x, train_y, train=True, argument=True)
    val_ds   = FCDataset(val_x, val_y, train=False, argument=False)
    test_ds  = FCDataset(test_x, test_y, train=False, argument=False)
    return train_ds, val_ds, test_ds

def _filter_by_age(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_age: float = 95.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove subjects with age > max_age. Works for y shape [N] or [N,1]."""
    y_arr = np.asarray(y)
    y_flat = y_arr.reshape(-1)  # [N]
    keep = y_flat <= max_age

    x_f = x[keep]
    y_f = y_arr[keep]  # keep original shape

    return x_f, y_f


def build_nested_folds_from_group_paths(
    group_paths: Sequence[Tuple[str, str]],
    val_ratio: float = 0.1,
    seed: int = 111,
    *,
    max_age: float = 95.0 * 12,
) -> List[Tuple[np.ndarray, np.ndarray,
                np.ndarray, np.ndarray,
                np.ndarray, np.ndarray]]:
    """
    Build nested cross-validation splits from group-based folds.

    Outer loop:
        - One group is held out as TEST.
    Inner loop:
        - From the remaining groups (training pool),
          a subset is split off as VALIDATION.

    Returns:
        (train_x, train_y, val_x, val_y, test_x, test_y)
    """

    # ---------- load + filter each group ----------
    xs, ys = [], []
    for fc_path, label_path in group_paths:
        x = np.load(fc_path)
        y = np.load(label_path)
        x, y = _filter_by_age(x, y, max_age=max_age)  # <- delete old > 95 years
        xs.append(x)
        ys.append(y)

    k = len(xs)
    rng = np.random.RandomState(seed)

    folds = []

    for test_idx in range(k):
        # ---------- outer test ----------
        test_x, test_y = xs[test_idx], ys[test_idx]

        # ---------- training pool (other folds) ----------
        pool_x = np.concatenate([xs[j] for j in range(k) if j != test_idx], axis=0)
        pool_y = np.concatenate([ys[j] for j in range(k) if j != test_idx], axis=0)

        # ---------- nested train / val split ----------
        n = len(pool_y.reshape(-1))
        idx = np.arange(n)
        rng.shuffle(idx)

        n_val = max(1, int(round(n * val_ratio)))
        n_val = min(n_val, n - 1)  # ensure train not empty

        val_idx = idx[:n_val]
        train_idx = idx[n_val:]

        train_x, train_y = pool_x[train_idx], pool_y[train_idx]
        val_x, val_y     = pool_x[val_idx],   pool_y[val_idx]

        folds.append((train_x, train_y, val_x, val_y, test_x, test_y))

    return folds

def build_argparser():
    p = argparse.ArgumentParser()

    # basic
    p.add_argument("--seed", type=int, default=1024)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out_dir", type=str, default="/home/bio/PycharmProjects/OAG-/result/")
    p.add_argument("--input_dir", type=str, default="/nfsd/biopetmri4/Users/YikangCao/HCP_project/yk_autoencoder/crossvalid_dataset/")

    # wandb
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="Autoencoder-predictage")

    # loss weights (keep your current defaults)
    p.add_argument("--w_recon", type=float, default=0.1)
    p.add_argument("--w_age", type=float, default=0.4)
    p.add_argument("--w_ortho", type=float, default=0.2)
    p.add_argument("--w_class", type=float, default=0.75)

    return p


def main():
    args = build_argparser().parse_args()
    reset_seeds(args.seed)

    if args.use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args))

    # 你的 TrainConfig 如果字段名不是 out_dir（比如 out_root），这里改成对应字段
    cfg = TrainConfig(
        seed=args.seed,
        device=args.device,
        out_dir=str(Path(args.out_dir)),
        w_recon=args.w_recon,
        w_age=args.w_age,
        w_ortho=args.w_ortho,
        w_class=args.w_class,
    )

    input_dir = Path(args.input_dir)
    # BN278_FC_i = [Ni, 278, 278]
    # label_i = [Ni, age]
    group_paths = [
        (
            str(input_dir / f"BN278_FC_{i}.npy"),
            str(input_dir / f"label{i}.npy"),
        )
        for i in range(1, 6)
    ]

    folds = build_nested_folds_from_group_paths(group_paths, max_age=1140.0)

    mean_mae = train_and_eval(
        folds=folds,
        train_dataset_ctor=dataset_ctor,
        cfg=cfg
    )

    print("Average Validation MAE:", mean_mae)


if __name__ == "__main__":
    main()
