from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Mapping, Optional, Union

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


def json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_sanitize(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_sanitize(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def remap_legacy_rbulk_state_dict(state_dict: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    remapped: dict[str, Any] = {}
    renamed_keys: dict[str, str] = {}

    for key, value in state_dict.items():
        new_key = key
        if key.startswith("fc_paired."):
            new_key = f"fc_pause.{key[len('fc_paired.'):]}"
        elif key.startswith("encoder."):
            new_key = f"sequence_encoder.{key[len('encoder.'):]}"
        elif key.startswith("fc."):
            new_key = f"fusion.{key[len('fc.'):]}"
        elif key == "regressor.weight":
            new_key = "regressor.0.weight"
        elif key == "regressor.bias":
            new_key = "regressor.0.bias"

        remapped[new_key] = value
        if new_key != key:
            renamed_keys[key] = new_key

    return remapped, renamed_keys


def save_json(path: Union[str, Path], payload: dict[str, Any]) -> None:
    output_path = ensure_parent_dir(path)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(json_sanitize(payload), handle, indent=2, ensure_ascii=False, allow_nan=False)
