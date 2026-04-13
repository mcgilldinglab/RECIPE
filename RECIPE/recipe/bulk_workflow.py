from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .bulk_data import split_index_tensor
from .bulk_regression import (
    evaluate_graph_regression,
    load_bulk_dataframe,
    build_bulk_graph_from_dataframe,
    predict_bulk_outputs,
    train_single_graph_bulk,
)
from .config import BulkTaskConfig, get_bulk_task_config
from .models import RBULK
from .utils import resolve_device, save_json, set_seed


@dataclass(frozen=True)
class BulkSplitBundle:
    train_idx: torch.Tensor
    val_idx: torch.Tensor
    test_idx: torch.Tensor
    pool_idx: torch.Tensor


def build_bulk_graph_for_task(
    species: str,
    task: str,
    condition_name: str,
    scale_method: str = "log_median",
) -> tuple[BulkTaskConfig, pd.DataFrame, Any, dict[str, Any]]:
    config = get_bulk_task_config(task=task, species=species)
    condition = config.conditions[condition_name.upper()]
    bulk_df = load_bulk_dataframe(
        reference_csv_path=config.reference_csv,
        pause_csv_path=config.pause_csv,
        pause_col_name=condition.pause_col if config.pause_csv else None,
    )
    data, scaling_summary = build_bulk_graph_from_dataframe(
        bulk_df=bulk_df,
        condition=condition,
        sequence_npy_path=config.sequence_npy,
        ppi_csv_path=config.ppi_csv,
        scale_method=scale_method,
        reference_df=bulk_df,
        add_loops=True,
    )
    return config, bulk_df, data, scaling_summary


def build_labeled_splits(target_tensor: torch.Tensor, seed: int) -> BulkSplitBundle:
    target_values = target_tensor.detach().cpu().view(-1).numpy()
    labeled_idx = np.where(np.isfinite(target_values) & (~np.isclose(target_values, 0.0)))[0]
    unlabeled_idx = np.where(np.isclose(target_values, 0.0))[0]

    if labeled_idx.size < 3:
        raise ValueError("At least three labeled nodes are required for train/val/test splits.")

    train_idx, val_idx, test_idx = split_index_tensor(torch.tensor(labeled_idx, dtype=torch.long), seed=seed)
    pool_idx = torch.tensor(unlabeled_idx, dtype=torch.long)
    return BulkSplitBundle(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, pool_idx=pool_idx)


def load_model_state(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    payload = torch.load(checkpoint_path, map_location=device)
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if hasattr(payload, "state_dict"):
        payload = payload.state_dict()
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint payload type: {type(payload)!r}")
    clean_state = {}
    for key, value in payload.items():
        clean_key = key[7:] if key.startswith("module.") else key
        clean_state[clean_key] = value
    model.load_state_dict(clean_state, strict=False)
    return model


def make_bulk_model(data, device: torch.device) -> RBULK:
    return RBULK(sequence_dim=int(data.seq.shape[1])).to(device)


def save_bulk_outputs(
    output_dir: Path,
    bulk_df: pd.DataFrame,
    predictions: torch.Tensor,
    embeddings: torch.Tensor,
    splits: BulkSplitBundle,
    summary: dict[str, Any],
    checkpoint_path: Path | None = None,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_vector = predictions.view(-1).numpy()
    target_vector = bulk_df[summary["target_col"]].to_numpy(dtype=np.float32)
    transcript_ids = (
        bulk_df["transcript_id"].astype(str).tolist()
        if "transcript_id" in bulk_df.columns
        else [str(idx) for idx in range(len(bulk_df))]
    )

    split_labels = np.full(len(bulk_df), "unlabeled", dtype=object)
    split_labels[splits.train_idx.numpy()] = "train"
    split_labels[splits.val_idx.numpy()] = "val"
    split_labels[splits.test_idx.numpy()] = "test"

    prediction_df = pd.DataFrame(
        {
            "transcript_id": transcript_ids,
            "prediction": pred_vector,
            "target": target_vector,
            "split": split_labels,
            "is_labeled": split_labels != "unlabeled",
        }
    )
    prediction_csv = output_dir / "predictions.csv"
    embedding_npy = output_dir / "embeddings.npy"
    metrics_json = output_dir / "metrics.json"

    prediction_df.to_csv(prediction_csv, index=False)
    np.save(embedding_npy, embeddings.numpy())

    summary_payload = summary.copy()
    if checkpoint_path is not None:
        summary_payload["checkpoint"] = str(checkpoint_path)
    save_json(metrics_json, summary_payload)
    return {
        "prediction_csv": str(prediction_csv),
        "embedding_npy": str(embedding_npy),
        "metrics_json": str(metrics_json),
    }


def run_bulk_module(
    species: str,
    task: str,
    condition_name: str,
    output_dir: str | Path,
    seed: int = 12,
    device_name: str | None = None,
    train: bool = False,
    checkpoint_path: str | Path | None = None,
    learning_rate: float = 7e-2,
    max_epochs: int = 3000,
    patience: int = 200,
    log_every: int = 50,
    scale_method: str = "log_median",
) -> dict[str, Any]:
    set_seed(seed)
    device = resolve_device(device_name)
    output_dir = Path(output_dir)

    config, bulk_df, data, scaling_summary = build_bulk_graph_for_task(
        species=species,
        task=task,
        condition_name=condition_name,
        scale_method=scale_method,
    )
    splits = build_labeled_splits(data.y, seed=seed)
    model = make_bulk_model(data, device=device)
    data = data.to(device)

    checkpoint = Path(checkpoint_path) if checkpoint_path else config.default_checkpoint
    training_summary: dict[str, Any] = {"loaded_checkpoint": None}

    if train or checkpoint is None or not checkpoint.exists():
        model, training_summary = train_single_graph_bulk(
            model=model,
            data=data,
            train_idx=splits.train_idx.to(device),
            val_idx=splits.val_idx.to(device),
            test_idx=splits.test_idx.to(device),
            lr=learning_rate,
            max_epochs=max_epochs,
            patience=patience,
            log_every=log_every,
        )
        checkpoint = output_dir / "model.pth"
        torch.save(model.state_dict(), checkpoint)
    else:
        model = load_model_state(model, checkpoint, device=device)
        training_summary["loaded_checkpoint"] = str(checkpoint)

    predictions, embeddings = predict_bulk_outputs(model, data)
    train_metrics = evaluate_graph_regression(model, data, splits.train_idx.to(device))
    val_metrics = evaluate_graph_regression(model, data, splits.val_idx.to(device))
    test_metrics = evaluate_graph_regression(model, data, splits.test_idx.to(device))

    summary = {
        "species": species,
        "task": task,
        "condition": condition_name.upper(),
        "seed": seed,
        "device": str(device),
        "expression_col": scaling_summary["expression_col"],
        "target_col": scaling_summary["target_col"],
        "pause_col": scaling_summary["pause_col"],
        "node_count": int(data.num_nodes),
        "labeled_node_count": int(len(splits.train_idx) + len(splits.val_idx) + len(splits.test_idx)),
        "unlabeled_node_count": int(len(splits.pool_idx)),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "scaling": scaling_summary,
        "training": training_summary,
    }
    output_files = save_bulk_outputs(
        output_dir=output_dir,
        bulk_df=bulk_df,
        predictions=predictions,
        embeddings=embeddings,
        splits=splits,
        summary=summary,
        checkpoint_path=checkpoint,
    )
    summary["outputs"] = output_files
    return summary
