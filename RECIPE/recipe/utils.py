from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
from sklearn.metrics import r2_score


def set_seed(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def resolve_device(device_name: Optional[str] = None) -> torch.device:
    if device_name and device_name != "auto":
        return torch.device(device_name)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def ensure_parent_dir(path: Union[str, Path]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def safe_r2(y_true: Any, y_pred: Any) -> float:
    y_true_np = np.asarray(y_true).reshape(-1)
    y_pred_np = np.asarray(y_pred).reshape(-1)
    if y_true_np.size < 2:
        return float("nan")
    if np.allclose(y_true_np, y_true_np[0]):
        return float("nan")
    return float(r2_score(y_true_np, y_pred_np))


def save_json(path: Union[str, Path], payload: dict[str, Any]) -> None:
    output_path = ensure_parent_dir(path)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
