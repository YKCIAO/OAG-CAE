from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_npz(path: str, **arrays: np.ndarray) -> None:
    ensure_dir(os.path.dirname(path))
    np.savez(path, **arrays)


def save_json(path: str, obj: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def try_export_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    """
    Optional: export using pandas if available. If pandas isn't available, saves jsonl instead.
    """
    ensure_dir(os.path.dirname(path))
    try:
        import pandas as pd
        pd.DataFrame(rows).to_csv(path, index=False)
    except Exception:
        # fallback: jsonl
        jsonl_path = os.path.splitext(path)[0] + ".jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
def save_latent_outputs(prefix: str, latent_dict: Dict[str, Any]) -> None:
    """
    Save latent outputs as both .npz and .csv.

    prefix example:
        "../result/fold1/latent_test"
    This will save:
        latent_test.npz
        latent_test.csv
    """
    import os
    import numpy as np
    import pandas as pd

    ensure_dir(os.path.dirname(prefix))

    z_age = latent_dict["z_age"]
    z_noise = latent_dict["z_noise"]
    age_true = latent_dict["age_true"]
    age_pred = latent_dict.get("age_pred", None)
    split = latent_dict.get("split", "unknown")

    # ---------- save npz ----------
    npz_path = prefix + ".npz"

    if age_pred is not None:
        np.savez(
            npz_path,
            z_age=z_age,
            z_noise=z_noise,
            age_true=age_true,
            age_pred=age_pred,
        )
    else:
        np.savez(
            npz_path,
            z_age=z_age,
            z_noise=z_noise,
            age_true=age_true,
        )

    # ---------- save csv ----------
    df = pd.DataFrame()
    df["split"] = split
    df["sample_index"] = np.arange(z_age.shape[0])
    df["age_true"] = age_true

    if age_pred is not None:
        df["age_pred"] = age_pred

    z_age_df = pd.DataFrame(
        z_age,
        columns=[f"z_age_{i+1}" for i in range(z_age.shape[1])]
    )

    z_noise_df = pd.DataFrame(
        z_noise,
        columns=[f"z_noise_{i+1}" for i in range(z_noise.shape[1])]
    )

    df = pd.concat([df, z_age_df, z_noise_df], axis=1)

    csv_path = prefix + ".csv"
    df.to_csv(csv_path, index=False)