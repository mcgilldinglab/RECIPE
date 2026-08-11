from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .bulk_data import split_index_tensor
from .bulk_regression import (
    BulkConditionSpec,
    evaluate_graph_regression,
    load_bulk_dataframe,
    build_bulk_graph_from_dataframe,
    predict_bulk_outputs,
    train_single_graph_bulk,
)
from .config import BulkTaskConfig, get_bulk_task_config, with_bulk_input_paths
from .models import RBULK
from .utils import remap_legacy_rbulk_state_dict, resolve_device, save_json, set_seed


@dataclass(frozen=True)
class BulkSplitBundle:
    train_idx: torch.Tensor
    val_idx: torch.Tensor
    test_idx: torch.Tensor
    pool_idx: torch.Tensor


def _check_bulk_input_files(config: BulkTaskConfig) -> None:
    required_paths = {
        "reference_csv": config.reference_csv,
        "sequence_npy": config.sequence_npy,
        "ppi_csv": config.ppi_csv,
    }
    if config.pause_csv is not None:
        required_paths["pause_csv"] = config.pause_csv

    missing = [(name, path) for name, path in required_paths.items() if not Path(path).exists()]
    if not missing:
        return

    details = "; ".join(f"{name}={path}" for name, path in missing)
    extra = ""
    if config.species.lower() == "human" and config.task.lower() == "unknown":
        extra = (
            " The human unknown workflow requires the external PPI graph "
            "`data/networks/human_ppi_unknown.csv` (about 54 GB after extraction). "
            "Download instructions: https://github.com/mcgilldinglab/RECIPE/blob/main/RECIPE/docs/data.md"
        )
    raise FileNotFoundError(f"Missing RECIPE input file(s): {details}.{extra}")


def build_bulk_graph_for_task(
    species: str,
    task: str,
    condition_name: str,
    scale_method: str = "log_median",
    data_root: str | Path | None = None,
    model_root: str | Path | None = None,
    reference_csv: str | Path | None = None,
    sequence_npy: str | Path | None = None,
    ppi_csv: str | Path | None = None,
    pause_csv: str | Path | None = None,
    expression_col: str | None = None,
    target_col: str | None = None,
    pause_col: str | None = None,
    use_pause: bool = True,
) -> tuple[BulkTaskConfig, pd.DataFrame, Any, dict[str, Any]]:
    config = with_bulk_input_paths(
        get_bulk_task_config(task=task, species=species),
        data_root=data_root,
        model_root=model_root,
        reference_csv=reference_csv,
        sequence_npy=sequence_npy,
        ppi_csv=ppi_csv,
        pause_csv=pause_csv,
    )
    _check_bulk_input_files(config)
    condition = config.conditions[condition_name.upper()]
    condition = BulkConditionSpec(
        name=condition.name,
        expression_col=expression_col or condition.expression_col,
        target_col=target_col or condition.target_col,
        pause_col=(pause_col or condition.pause_col) if use_pause else None,
    )
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


def load_labeled_splits_from_csv(split_csv: str | Path, node_count: int) -> BulkSplitBundle:
    split_df = pd.read_csv(split_csv)
    if "split" not in split_df.columns:
        raise ValueError(f"Split CSV must include a 'split' column: {split_csv}")

    if "node_index" in split_df.columns:
        node_indices = split_df["node_index"].to_numpy(dtype=np.int64)
    else:
        node_indices = np.arange(len(split_df), dtype=np.int64)

    if len(node_indices) != node_count:
        raise ValueError(
            f"Split CSV row count ({len(node_indices)}) does not match graph node count ({node_count})."
        )
    if node_indices.min(initial=0) < 0 or node_indices.max(initial=-1) >= node_count:
        raise ValueError(f"Split CSV has node_index values outside 0..{node_count - 1}: {split_csv}")

    split_labels = split_df["split"].astype(str).str.lower().to_numpy()

    def _indices_for(label: str) -> torch.Tensor:
        return torch.tensor(node_indices[split_labels == label], dtype=torch.long)

    assigned_mask = np.isin(split_labels, ["train", "val", "test"])
    pool_idx = torch.tensor(node_indices[~assigned_mask], dtype=torch.long)
    return BulkSplitBundle(
        train_idx=_indices_for("train"),
        val_idx=_indices_for("val"),
        test_idx=_indices_for("test"),
        pool_idx=pool_idx,
    )


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
    clean_state, remapped_keys = remap_legacy_rbulk_state_dict(clean_state)
    incompatible = model.load_state_dict(clean_state, strict=False)
    load_report = {
        "path": str(checkpoint_path),
        "loaded_key_count": int(len(clean_state)),
        "remapped_legacy_key_count": int(len(remapped_keys)),
        "remapped_legacy_keys": remapped_keys,
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }
    setattr(model, "_recipe_load_report", load_report)
    if load_report["missing_keys"] or load_report["unexpected_keys"]:
        warnings.warn(
            f"Checkpoint loaded with key mismatch for {checkpoint_path}: "
            f"{len(load_report['missing_keys'])} missing, {len(load_report['unexpected_keys'])} unexpected.",
            RuntimeWarning,
            stacklevel=2,
        )
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
    data_root: str | Path | None = None,
    model_root: str | Path | None = None,
    reference_csv: str | Path | None = None,
    sequence_npy: str | Path | None = None,
    ppi_csv: str | Path | None = None,
    pause_csv: str | Path | None = None,
    split_csv: str | Path | None = None,
    expression_col: str | None = None,
    target_col: str | None = None,
    pause_col: str | None = None,
    use_pause: bool = True,
) -> dict[str, Any]:
    set_seed(seed)
    device = resolve_device(device_name)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config, bulk_df, data, scaling_summary = build_bulk_graph_for_task(
        species=species,
        task=task,
        condition_name=condition_name,
        scale_method=scale_method,
        data_root=data_root,
        model_root=model_root,
        reference_csv=reference_csv,
        sequence_npy=sequence_npy,
        ppi_csv=ppi_csv,
        pause_csv=pause_csv,
        expression_col=expression_col,
        target_col=target_col,
        pause_col=pause_col,
        use_pause=use_pause,
    )
    split_csv_path = Path(split_csv).expanduser().resolve() if split_csv else None
    splits = (
        load_labeled_splits_from_csv(split_csv_path, node_count=int(data.num_nodes))
        if split_csv_path
        else build_labeled_splits(data.y, seed=seed)
    )
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
        training_summary["checkpoint_load"] = getattr(model, "_recipe_load_report", None)

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
        "inputs": {
            "reference_csv": str(config.reference_csv),
            "sequence_npy": str(config.sequence_npy),
            "ppi_csv": str(config.ppi_csv),
            "pause_csv": str(config.pause_csv) if config.pause_csv else None,
            "split_csv": str(split_csv_path) if split_csv_path else None,
        },
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
