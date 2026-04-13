from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix

from .bulk_data import (
    strip_version,
    build_masks_from_target_values,
    load_fraction_average_expression,
    load_bulk_reference_table,
    load_ordered_cds_table,
    load_ppi_graph,
    merge_single_pause_file,
)
from .bulk_regression import build_bulk_graph_from_dataframe, predict_bulk_outputs, train_single_graph_bulk
from .bulk_regression import BulkConditionSpec
from .config import SINGLE_CELL_TRANSFER_CONFIG, SingleCellTransferConfig
from .models import RBULK, RSCHead
from .self_learning import run_self_learning
from .single_cell import (
    export_bulk_embeddings_for_cells,
    load_expression_matrix,
    load_metadata,
    load_pause_matrix,
)
from .utils import resolve_device, safe_r2, save_json, set_seed

NOTEBOOK_PROJECT_ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK_PAUSING_ROOT = Path("/mnt/md0/luying/ribo/308code/pausing")


def _load_model_state(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    payload = torch.load(checkpoint_path, map_location=device)
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if hasattr(payload, "state_dict"):
        payload = payload.state_dict()
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint payload type: {type(payload)!r}")
    cleaned = {key[7:] if key.startswith("module.") else key: value for key, value in payload.items()}
    model.load_state_dict(cleaned, strict=False)
    return model


def _phase0_ordered_table(config: SingleCellTransferConfig) -> pd.DataFrame:
    ordered_df = load_ordered_cds_table(config.cds_csv, config.transcript_order_csv)
    ordered_df = merge_single_pause_file(
        ordered_df,
        config.phase0_pause_csv,
        config.phase0_pause_col,
        merge_on="transcript_id",
    )
    return ordered_df


def _phase1_ordered_table(config: SingleCellTransferConfig) -> pd.DataFrame:
    ordered_df = load_ordered_cds_table(config.cds_csv, config.transcript_order_csv)
    ordered_df = merge_single_pause_file(
        ordered_df,
        config.phase0_pause_csv.parent / "fraction_rich_pause.csv",
        "phase1_pause",
        merge_on="transcript_id",
    )
    return ordered_df


def _build_notebook_phase0_data(seed: int) -> tuple[Data, dict[str, Any], pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw_cds_csv = NOTEBOOK_PAUSING_ROOT / "cds_df38510.csv"
    raw_order_csv = NOTEBOOK_PAUSING_ROOT / "data" / "sc11619genes422cell_normalized.csv"
    raw_bulk_reference_csv = NOTEBOOK_PROJECT_ROOT / "data" / "24077132kdncmergedf.csv"
    raw_pause_nc1_csv = NOTEBOOK_PAUSING_ROOT / "pause_scorescdsallnewnohupNC1_38510FINAL.csv"
    raw_sequence_npy = NOTEBOOK_PROJECT_ROOT / "data" / "all_sequence_outputsnewbulk11619.npy"
    raw_ppi_csv = NOTEBOOK_PROJECT_ROOT / "data" / "ppi_ebi_string_ppi3ensp_lr_IntAct_corummatrix4p_pbulk11619.csv"

    cds_df = pd.read_csv(raw_cds_csv).iloc[:, 1:9].copy()
    cds_df["transcript_id"] = strip_version(cds_df["transcript_id_x"])
    order_df = pd.read_csv(raw_order_csv)
    merged_df = cds_df.set_index("transcript_id").reindex(order_df["Unnamed: 0"]).reset_index(drop=True)
    merged_df.insert(0, "Unnamed: 0", order_df["Unnamed: 0"].astype(str).to_numpy())
    merged_df.fillna(0, inplace=True)
    merged_df["transcript_id"] = strip_version(merged_df["transcript_id_x"])

    merged_df3 = load_bulk_reference_table(raw_bulk_reference_csv)
    merged_df3["transcript_id"] = strip_version(merged_df3["transcript_id"])

    pausing = pd.read_csv(raw_pause_nc1_csv).copy()
    pausing.columns = ["protein_id", "High_Pause_Countsnc1", "transcript_id_x"]
    merged_df2 = pd.merge(merged_df, pausing, on="transcript_id_x", how="left")
    merged_df2["High_Pause_Countsnc1"] = merged_df2["High_Pause_Countsnc1"].fillna(0.0)
    merged_df2["transcript_id"] = strip_version(merged_df2["transcript_id"])

    set1 = set(merged_df2["Unnamed: 0"].astype(str))
    set2 = set(merged_df3["transcript_id"].astype(str))
    intersection = set1 & set2
    id_to_idx = {tid: idx for idx, tid in enumerate(merged_df2["Unnamed: 0"].astype(str))}
    labeled_idx = np.asarray([id_to_idx[tid] for tid in intersection], dtype=np.int64)

    train_idx, temp_idx = train_test_split(labeled_idx, test_size=0.25, random_state=seed)
    test_idx, val_idx = train_test_split(temp_idx, test_size=0.5, random_state=seed)

    x_values = np.log2((merged_df2[["rNC2"]].values / np.median(merged_df3[["rNC2"]].values)) + 1.0)
    y_values = np.log2((merged_df2[["NC3"]].values / np.median(merged_df3[["NC3"]].values)) + 1.0)
    sequence_embedding = np.load(raw_sequence_npy)
    ppi_matrix = pd.read_csv(raw_ppi_csv)
    edge_index, edge_weight = from_scipy_sparse_matrix(sp.coo_matrix(ppi_matrix).astype("float32"))

    data = Data(
        x=torch.tensor(x_values, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_weight,
        y=torch.tensor(y_values, dtype=torch.float32),
    )
    data.pause = torch.tensor(
        np.asarray(merged_df2["High_Pause_Countsnc1"], dtype=np.float32).reshape(-1, 1),
        dtype=torch.float32,
    )
    data.seq = torch.tensor(sequence_embedding, dtype=torch.float32)

    is_labeled = torch.zeros(data.y.size(0), dtype=torch.bool)
    is_labeled[torch.tensor(labeled_idx, dtype=torch.long)] = True
    pool_idx = torch.where(~is_labeled)[0].cpu().numpy()

    summary = {
        "style": "notebook_phase0",
        "bulk_reference_csv": str(raw_bulk_reference_csv),
        "cds_csv": str(raw_cds_csv),
        "order_csv": str(raw_order_csv),
        "pause_csv": str(raw_pause_nc1_csv),
        "sequence_npy": str(raw_sequence_npy),
        "ppi_csv": str(raw_ppi_csv),
        "intersection_count": int(len(intersection)),
        "labeled_gene_count": int(len(labeled_idx)),
        "unlabeled_gene_count": int(len(pool_idx)),
        "x_reference_median": float(np.median(merged_df3[["rNC2"]].values)),
        "y_reference_median": float(np.median(merged_df3[["NC3"]].values)),
    }
    return data, summary, merged_df2, merged_df3, train_idx, val_idx, test_idx, pool_idx


def _notebook_seed_everything(seed: int) -> None:
    random_state = int(seed)
    import random

    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_state)
        torch.cuda.manual_seed_all(random_state)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class _NotebookEarlyStopping:
    def __init__(self, patience: int = 50):
        self.patience = patience
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def step(self, loss: float) -> None:
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def _notebook_train_model(
    model: RBULK,
    data: Data,
    train_idx: torch.Tensor,
    y_true: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    patience: int = 50,
    max_epochs: int = 10000,
) -> dict[str, float]:
    early_stopping = _NotebookEarlyStopping(patience=patience)
    final_loss = float("nan")
    stop_epoch = max_epochs

    model.train()
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        out, _ = model(data)
        loss = criterion(out[train_idx], y_true[train_idx])
        loss.backward()
        optimizer.step()

        final_loss = float(loss.item())
        early_stopping.step(final_loss)
        if early_stopping.early_stop:
            stop_epoch = epoch
            print(f"Early Stopping at Epoch {epoch} with Loss {final_loss:.4f}")
            break

    return {"loss": final_loss, "stop_epoch": float(stop_epoch)}


def _notebook_evaluate_model(
    model: RBULK,
    data: Data,
    idx: torch.Tensor,
    y_true: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        out, _ = model(data)
        pred = out[idx]
        target = y_true[idx]
        loss = float(nn.MSELoss()(pred, target).item())
        r2 = safe_r2(target.detach().cpu().numpy(), pred.detach().cpu().numpy())
    return {"loss": loss, "r2": r2}


def _run_notebook_self_learning(
    data: Data,
    y: torch.Tensor,
    model: RBULK,
    device: torch.device,
    seed: int,
    initial_labeled_idx: torch.Tensor,
    val_idx: torch.Tensor,
    pool_idx: torch.Tensor,
    batch_size: int = 300,
    max_rounds: int = 100,
    learning_rate: float = 7e-2,
    train_patience: int = 50,
    train_max_epochs: int = 10000,
) -> tuple[RBULK, torch.Tensor, dict[str, Any]]:
    criterion = nn.MSELoss()
    working_target = y.clone()
    train_idx = initial_labeled_idx.clone()
    current_pool = pool_idx.clone()

    print(f"初始训练集大小: {len(train_idx)}")
    _notebook_seed_everything(seed)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    initial_fit = _notebook_train_model(
        model=model,
        data=data,
        train_idx=train_idx,
        y_true=working_target,
        optimizer=optimizer,
        criterion=criterion,
        patience=train_patience,
        max_epochs=train_max_epochs,
    )

    rounds: list[dict[str, float]] = [
        {
            "round": 0.0,
            "train_size": float(len(train_idx)),
            "loss": initial_fit["loss"],
            "stop_epoch": initial_fit["stop_epoch"],
        }
    ]

    for round_num in range(max_rounds):
        if len(current_pool) == 0:
            print("没有更多未标记样本，结束Self-Learning。")
            break

        model.eval()
        with torch.no_grad():
            outputs, _ = model(data)

        select_size = min(batch_size, len(current_pool))
        selected_idx = current_pool[:select_size]
        pseudo_labels = outputs[selected_idx].detach()
        working_target[selected_idx] = pseudo_labels
        train_idx = torch.cat([train_idx, selected_idx], dim=0)
        current_pool = current_pool[select_size:]

        print(f"第{round_num+1}轮: 添加{select_size}个伪标签样本，总训练集大小{len(train_idx)}")
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        fit_result = _notebook_train_model(
            model=model,
            data=data,
            train_idx=train_idx,
            y_true=working_target,
            optimizer=optimizer,
            criterion=criterion,
            patience=train_patience,
            max_epochs=train_max_epochs,
        )
        rounds.append(
            {
                "round": float(round_num + 1),
                "new_pseudo_labels": float(select_size),
                "train_size": float(len(train_idx)),
                "loss": fit_result["loss"],
                "stop_epoch": fit_result["stop_epoch"],
            }
        )

    val_metrics = _notebook_evaluate_model(model, data, val_idx, working_target)
    print(f"最终验证集Loss: {val_metrics['loss']:.4f}, 验证集R²: {val_metrics['r2']:.4f}")
    return model, working_target, {
        "style": "notebook_self_learning_process",
        "val_metrics": val_metrics,
        "train_size": int(len(train_idx)),
        "remaining_pool_size": int(len(current_pool)),
        "rounds": rounds,
    }


def _notebook_style_phase1_masks(target_values: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw_target = torch.tensor(np.asarray(target_values), dtype=torch.float32).view(-1)
    valid_mask = (~torch.isnan(raw_target)) & (raw_target != 0)
    valid_indices = valid_mask.nonzero(as_tuple=True)[0].cpu().numpy()

    train_val_idx, test_idx = train_test_split(valid_indices, test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.25, random_state=42)

    train_mask = torch.zeros(raw_target.numel(), dtype=torch.bool)
    val_mask = torch.zeros(raw_target.numel(), dtype=torch.bool)
    test_mask = torch.zeros(raw_target.numel(), dtype=torch.bool)
    train_mask[torch.tensor(train_idx, dtype=torch.long)] = True
    val_mask[torch.tensor(val_idx, dtype=torch.long)] = True
    test_mask[torch.tensor(test_idx, dtype=torch.long)] = True
    return train_mask, val_mask, test_mask


def _build_notebook_phase1_data(config: SingleCellTransferConfig) -> tuple[Data, dict[str, Any], int, pd.DataFrame]:
    raw_expression_csv = NOTEBOOK_PROJECT_ROOT / "data" / "sc11619genes422cell.csv"
    raw_meta_csv = NOTEBOOK_PROJECT_ROOT / "brforepridictmeta_dataall.csv"
    raw_pause_base_csv = NOTEBOOK_PAUSING_ROOT / "data" / "250429scribonew11619_422.csv"
    raw_pause_rich_csv = NOTEBOOK_PAUSING_ROOT / "pause_scorescdsallscribo293Rich_dedup3sball.csv"
    raw_cds_csv = NOTEBOOK_PAUSING_ROOT / "cds_df38510.csv"
    raw_order_csv = NOTEBOOK_PAUSING_ROOT / "data" / "sc11619genes422cell_normalized.csv"
    raw_bulk_reference_csv = NOTEBOOK_PROJECT_ROOT / "data" / "24077132kdncmergedf.csv"
    raw_sequence_npy = NOTEBOOK_PROJECT_ROOT / "data" / "all_sequence_outputsnewbulk11619.npy"
    raw_ppi_csv = NOTEBOOK_PROJECT_ROOT / "data" / "ppi_ebi_string_ppi3ensp_lr_IntAct_corummatrix4p_pbulk11619.csv"

    ordered_df = load_ordered_cds_table(raw_cds_csv, raw_order_csv)
    reference_df = load_bulk_reference_table(raw_bulk_reference_csv)

    fraction_avg_df, selected_cells = load_fraction_average_expression(
        expression_csv_path=raw_expression_csv,
        meta_csv_path=raw_meta_csv,
        fraction="Rich",
        feature_name="scribo",
    )
    fraction_avg_df = fraction_avg_df.rename(columns={fraction_avg_df.columns[0]: "transcript_id"})
    fraction_avg_df["transcript_id"] = strip_version(fraction_avg_df["transcript_id"])
    ordered_df["scribo"] = (
        fraction_avg_df.set_index("transcript_id")
        .reindex(ordered_df["ordered_transcript_id"])["scribo"]
        .to_numpy()
    )

    pause_base_df = pd.read_csv(raw_pause_base_csv)
    pause_base_df["transcript_id"] = strip_version(pause_base_df["transcript_id"])
    pause_rich_df = pd.read_csv(raw_pause_rich_csv)
    pause_rich_df.columns = ["protein_id", "High_Pause_Countsscrich", "transcript_id"]
    pause_rich_df["transcript_id"] = strip_version(pause_rich_df["transcript_id"])
    pause_lookup = (
        pause_base_df[["transcript_id"]]
        .merge(pause_rich_df[["transcript_id", "High_Pause_Countsscrich"]], on="transcript_id", how="left")
        .drop_duplicates(subset=["transcript_id"])
        .set_index("transcript_id")
    )
    ordered_df["phase1_pause"] = (
        pause_lookup.reindex(ordered_df["ordered_transcript_id"])["High_Pause_Countsscrich"]
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )

    phase1_condition = BulkConditionSpec(
        name="RichPseudoBulkNotebook",
        expression_col="scribo",
        target_col=config.phase0_target_col,
        pause_col="phase1_pause",
    )
    phase1_reference_df = pd.DataFrame(
        {
            "scribo": ordered_df["scribo"].to_numpy(dtype=np.float32),
            config.phase0_target_col: np.full(
                len(ordered_df),
                float(np.median(reference_df[[config.phase0_target_col]].values)),
                dtype=np.float32,
            ),
        }
    )

    data, scaling_summary = build_bulk_graph_from_dataframe(
        bulk_df=ordered_df,
        condition=phase1_condition,
        sequence_npy_path=raw_sequence_npy,
        ppi_csv_path=raw_ppi_csv,
        scale_method="log_median",
        reference_df=phase1_reference_df,
        add_loops=True,
    )
    train_mask, val_mask, test_mask = _notebook_style_phase1_masks(ordered_df[config.phase0_target_col].to_numpy())
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data, scaling_summary, len(selected_cells), ordered_df


def _valid_gene_splits(label_values: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid_gene_ids = np.where(np.isfinite(label_values) & (~np.isclose(label_values, 0.0)))[0]
    if valid_gene_ids.size < 3:
        raise ValueError("At least three genes with non-zero labels are required.")
    train_ids, temp_ids = train_test_split(valid_gene_ids, test_size=0.25, random_state=seed)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=seed)
    return np.asarray(train_ids), np.asarray(val_ids), np.asarray(test_ids)


def _iterate_gene_batches(gene_ids: np.ndarray, batch_size: int, shuffle: bool) -> list[np.ndarray]:
    ids = np.array(gene_ids, copy=True)
    if shuffle:
        np.random.shuffle(ids)
    return [ids[start : start + batch_size] for start in range(0, len(ids), batch_size)]


def build_cell_graph_edge_index(cell_by_gene: pd.DataFrame, n_neighbors: int, n_pcs: int, seed: int) -> tuple[torch.Tensor, dict[str, Any]]:
    values = np.nan_to_num(cell_by_gene.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    n_pcs = max(1, min(n_pcs, values.shape[0], values.shape[1]))
    reduced = PCA(n_components=n_pcs, random_state=seed).fit_transform(values)

    knn = NearestNeighbors(n_neighbors=min(n_neighbors + 1, len(cell_by_gene)), metric="euclidean")
    knn.fit(reduced)
    _, indices = knn.kneighbors(reduced)

    rows: list[int] = []
    cols: list[int] = []
    for cell_idx in range(len(cell_by_gene)):
        for neighbor_idx in indices[cell_idx, 1:]:
            rows.append(cell_idx)
            cols.append(int(neighbor_idx))
            rows.append(int(neighbor_idx))
            cols.append(cell_idx)

    edge_index = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
    return edge_index, {
        "num_cells": int(len(cell_by_gene)),
        "n_neighbors": int(n_neighbors),
        "n_pcs": int(n_pcs),
        "num_directed_edges": int(edge_index.shape[1]),
        "mean_degree": float(edge_index.shape[1] / max(len(cell_by_gene), 1)),
    }


def _evaluate_rsc(
    model: RSCHead,
    z_array: np.ndarray,
    edge_index: torch.Tensor,
    label_tensor: torch.Tensor,
    rich_mask: torch.Tensor,
    gene_ids: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[float, float]:
    model.eval()
    predictions = []
    labels = []
    total_loss = 0.0
    total_genes = 0

    with torch.no_grad():
        for batch_gene_ids in _iterate_gene_batches(gene_ids, batch_size=batch_size, shuffle=False):
            batch_tensor = torch.from_numpy(np.transpose(z_array[:, batch_gene_ids, :], (1, 0, 2))).to(device)
            batch_labels = label_tensor[batch_gene_ids]
            batch_predictions = model(batch_tensor, edge_index)
            batch_mean_predictions = batch_predictions[:, rich_mask].mean(dim=1)
            finite_mask = torch.isfinite(batch_mean_predictions) & torch.isfinite(batch_labels)
            if int(finite_mask.sum().item()) == 0:
                continue

            filtered_predictions = batch_mean_predictions[finite_mask]
            filtered_labels = batch_labels[finite_mask]
            loss = F.mse_loss(filtered_predictions, filtered_labels, reduction="sum")

            total_loss += float(loss.item())
            total_genes += int(finite_mask.sum().item())
            predictions.append(filtered_predictions.detach().cpu().numpy())
            labels.append(filtered_labels.detach().cpu().numpy())

    if not predictions:
        return 0.0, float("nan")
    prediction_vector = np.concatenate(predictions)
    label_vector = np.concatenate(labels)
    return total_loss / max(total_genes, 1), safe_r2(label_vector, prediction_vector)


def _train_rsc_epoch(
    model: RSCHead,
    optimizer: torch.optim.Optimizer,
    z_array: np.ndarray,
    edge_index: torch.Tensor,
    label_tensor: torch.Tensor,
    rich_mask: torch.Tensor,
    gene_ids: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[float, float]:
    model.train()
    predictions = []
    labels = []
    total_loss = 0.0
    total_genes = 0

    for batch_gene_ids in _iterate_gene_batches(gene_ids, batch_size=batch_size, shuffle=True):
        batch_tensor = torch.from_numpy(np.transpose(z_array[:, batch_gene_ids, :], (1, 0, 2))).to(device)
        batch_labels = label_tensor[batch_gene_ids]

        optimizer.zero_grad(set_to_none=True)
        batch_predictions = model(batch_tensor, edge_index)
        batch_mean_predictions = batch_predictions[:, rich_mask].mean(dim=1)
        finite_mask = torch.isfinite(batch_mean_predictions) & torch.isfinite(batch_labels)
        if int(finite_mask.sum().item()) == 0:
            continue

        filtered_predictions = batch_mean_predictions[finite_mask]
        filtered_labels = batch_labels[finite_mask]
        loss = F.mse_loss(filtered_predictions, filtered_labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * int(finite_mask.sum().item())
        total_genes += int(finite_mask.sum().item())
        predictions.append(filtered_predictions.detach().cpu().numpy())
        labels.append(filtered_labels.detach().cpu().numpy())

    if not predictions:
        return 0.0, float("nan")
    prediction_vector = np.concatenate(predictions)
    label_vector = np.concatenate(labels)
    return total_loss / max(total_genes, 1), safe_r2(label_vector, prediction_vector)


def _predict_all_cell_gene_values(
    model: RSCHead,
    z_array: np.ndarray,
    edge_index: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    num_cells, num_genes, _ = z_array.shape
    prediction_matrix = np.zeros((num_cells, num_genes), dtype=np.float32)

    with torch.no_grad():
        all_gene_ids = np.arange(num_genes, dtype=np.int64)
        for batch_gene_ids in _iterate_gene_batches(all_gene_ids, batch_size=batch_size, shuffle=False):
            batch_tensor = torch.from_numpy(np.transpose(z_array[:, batch_gene_ids, :], (1, 0, 2))).to(device)
            batch_predictions = model(batch_tensor, edge_index)
            prediction_matrix[:, batch_gene_ids] = batch_predictions.detach().cpu().numpy().T
    return np.nan_to_num(prediction_matrix, nan=0.0, posinf=0.0, neginf=0.0)


def run_single_cell_phase0(
    output_dir: str | Path,
    seed: int = 12,
    device_name: str | None = None,
    train: bool = False,
    warm_start: bool = False,
    checkpoint_path: str | Path | None = None,
    learning_rate: float = 7e-2,
    max_epochs: int = 3000,
    patience: int = 200,
    self_learning_rounds: int = 12,
    pseudo_labels_per_round: int = 256,
    notebook_style_data: bool = False,
    notebook_exact_training: bool = False,
) -> dict[str, Any]:
    config = SINGLE_CELL_TRANSFER_CONFIG
    set_seed(seed)
    device = resolve_device(device_name)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Phase0] seed={seed}, device={device}, output_dir={output_dir}")

    if notebook_style_data:
        data, scaling_summary, ordered_df, reference_df, train_idx, val_idx, test_idx, pool_idx = _build_notebook_phase0_data(seed)
    else:
        ordered_df = _phase0_ordered_table(config)
        reference_df = load_bulk_reference_table(config.bulk_reference_csv)
        phase0_condition = BulkConditionSpec(
            name="NC",
            expression_col=config.phase0_expression_col,
            target_col=config.phase0_target_col,
            pause_col=config.phase0_pause_col,
        )

        data, scaling_summary = build_bulk_graph_from_dataframe(
            bulk_df=ordered_df,
            condition=phase0_condition,
            sequence_npy_path=config.sequence_npy,
            ppi_csv_path=config.ppi_csv,
            scale_method="log_median",
            reference_df=reference_df,
            add_loops=True,
        )

        target_values = ordered_df[config.phase0_target_col].to_numpy(dtype=np.float32)
        train_idx, val_idx, test_idx = _valid_gene_splits(target_values, seed=seed)
        pool_idx = np.where(np.isclose(target_values, 0.0))[0]

    model = RBULK(sequence_dim=int(data.seq.shape[1])).to(device)
    data = data.to(device)
    target_values = data.y.view(-1).detach().cpu().numpy()

    checkpoint = Path(checkpoint_path) if checkpoint_path else config.phase0_init_checkpoint
    training_summary: dict[str, Any]
    self_learning_summary: dict[str, Any] | None = None

    if train or checkpoint is None or not checkpoint.exists():
        print("[Phase0] training bulk self-learning model")
        if (not notebook_style_data) and warm_start and config.phase0_init_checkpoint and config.phase0_init_checkpoint.exists():
            model = _load_model_state(model, config.phase0_init_checkpoint, device=device)

        if notebook_style_data:
            training_summary = {
                "style": "notebook_phase0",
                "pretrain": "skipped",
                "initial_train_size": int(len(train_idx)),
                "val_size": int(len(val_idx)),
                "test_size": int(len(test_idx)),
            }
        else:
            model, training_summary = train_single_graph_bulk(
                model=model,
                data=data,
                train_idx=torch.tensor(train_idx, dtype=torch.long, device=device),
                val_idx=torch.tensor(val_idx, dtype=torch.long, device=device),
                test_idx=torch.tensor(test_idx, dtype=torch.long, device=device),
                lr=learning_rate,
                max_epochs=max_epochs,
                patience=patience,
                log_every=50,
            )
        if notebook_style_data and notebook_exact_training:
            model, working_target, self_learning_summary = _run_notebook_self_learning(
                data=data,
                y=data.y.clone(),
                model=model,
                device=device,
                seed=seed,
                initial_labeled_idx=torch.tensor(train_idx, dtype=torch.long, device=device),
                val_idx=torch.tensor(val_idx, dtype=torch.long, device=device),
                pool_idx=torch.tensor(pool_idx, dtype=torch.long, device=device),
                batch_size=pseudo_labels_per_round,
                max_rounds=self_learning_rounds,
                learning_rate=learning_rate,
                train_patience=50,
                train_max_epochs=10000,
            )
        else:
            model, working_target, self_learning_summary = run_self_learning(
                model=model,
                data=data,
                target=data.y.view(-1).clone(),
                initial_labeled_idx=torch.tensor(train_idx, dtype=torch.long, device=device),
                val_idx=torch.tensor(val_idx, dtype=torch.long, device=device),
                pool_idx=torch.tensor(pool_idx, dtype=torch.long, device=device),
                test_idx=torch.tensor(test_idx, dtype=torch.long, device=device),
                batch_size=pseudo_labels_per_round,
                max_rounds=self_learning_rounds,
                lr=learning_rate,
                patience=patience,
                max_epochs=max_epochs,
                selection_strategy="sequential" if notebook_style_data else "confidence",
            )
        checkpoint = output_dir / "phase0_bulk_self_learning_model.pth"
        torch.save(model.state_dict(), checkpoint)
        final_target = working_target.detach().cpu().numpy()
    else:
        print(f"[Phase0] loading checkpoint: {checkpoint}")
        model = _load_model_state(model, checkpoint, device=device)
        training_summary = {"loaded_checkpoint": str(checkpoint)}
        final_target = data.y.view(-1).detach().cpu().numpy()

    predictions, _ = predict_bulk_outputs(model, data)
    prediction_vector = predictions.view(-1).numpy()

    if notebook_style_data:
        transcript_ids = ordered_df["Unnamed: 0"].astype(str).tolist()
    else:
        transcript_ids = ordered_df["ordered_transcript_id"].astype(str).tolist()
    prediction_df = pd.DataFrame(
        {
            "transcript_id": transcript_ids,
            "prediction": prediction_vector,
            "observed_target": target_values,
            "final_target": final_target,
        }
    )
    prediction_csv = output_dir / "phase0_gene_predictions.csv"
    prediction_df.to_csv(prediction_csv, index=False)

    outputs: dict[str, str] = {
        "model": str(checkpoint),
        "gene_predictions": str(prediction_csv),
    }

    if not notebook_style_data:
        expression_df = load_expression_matrix(str(config.expression_normalized_csv))
        expression_df = expression_df.reindex(transcript_ids).fillna(0.0)
        metadata_df = load_metadata(str(config.metadata_csv)).reindex(expression_df.columns)
        if metadata_df.isnull().any().any():
            missing_cells = metadata_df.index[metadata_df["fraction"].isna()].tolist()
            raise KeyError(f"Missing metadata for cells: {missing_cells[:5]}")

        pause_matrix_df = load_pause_matrix(str(config.pause_matrix_csv))
        pause_matrix_df = pause_matrix_df.reindex(expression_df.index).fillna(0.0)

        edge_index, edge_attr = load_ppi_graph(config.ppi_csv, add_loops=True)
        sequence_embeddings = np.load(config.sequence_npy)
        all_z_array, all_y_array, cell_names = export_bulk_embeddings_for_cells(
            model=model,
            expression=expression_df,
            meta=metadata_df,
            sequence_embeddings=sequence_embeddings,
            edge_index=edge_index,
            edge_attr=edge_attr,
            device=device,
            pause_matrix=pause_matrix_df,
        )

        z_array_npy = output_dir / "phase0_cell_embeddings.npy"
        y_array_npy = output_dir / "phase0_cell_outputs.npy"
        cell_names_txt = output_dir / "phase0_cell_names.txt"
        np.save(z_array_npy, all_z_array)
        np.save(y_array_npy, all_y_array)
        cell_names_txt.write_text("\n".join(cell_names), encoding="utf-8")
        outputs["cell_embeddings"] = str(z_array_npy)
        outputs["cell_outputs"] = str(y_array_npy)
        outputs["cell_names"] = str(cell_names_txt)
        print(f"[Phase0] saved outputs to {output_dir}")
    else:
        print(f"[Phase0] saved outputs to {output_dir}")

    summary = {
        "seed": seed,
        "device": str(device),
        "phase0_condition": "NC",
        "data_style": "notebook_phase0" if notebook_style_data else "configured_phase0",
        "node_count": int(data.num_nodes),
        "labeled_gene_count": int(len(train_idx) + len(val_idx) + len(test_idx)),
        "unlabeled_gene_count": int(len(pool_idx)),
        "training": training_summary,
        "self_learning": self_learning_summary,
        "scaling": scaling_summary,
        "outputs": outputs,
    }
    save_json(output_dir / "phase0_summary.json", summary)
    return summary


def run_single_cell_phase1(
    output_dir: str | Path,
    phase0_checkpoint_path: str | Path,
    seed: int = 12,
    device_name: str | None = None,
    train: bool = True,
    checkpoint_path: str | Path | None = None,
    learning_rate: float = 1e-3,
    max_epochs: int = 1000,
    patience: int = 100,
    rich_fraction_label: str = "Rich",
) -> dict[str, Any]:
    config = SINGLE_CELL_TRANSFER_CONFIG
    set_seed(seed)
    device = resolve_device(device_name)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Phase1] seed={seed}, device={device}, output_dir={output_dir}")
    data, scaling_summary, selected_cell_count, ordered_df = _build_notebook_phase1_data(config)
    data = data.to(device)

    model = RBULK(sequence_dim=int(data.seq.shape[1])).to(device)
    phase0_checkpoint = Path(phase0_checkpoint_path)
    print(f"[Phase1] loading phase0 checkpoint: {phase0_checkpoint}")
    model = _load_model_state(model, phase0_checkpoint, device=device)

    checkpoint = Path(checkpoint_path) if checkpoint_path else output_dir / "phase1_pseudobulk_model.pth"
    history: list[dict[str, float]] = []

    if train or not checkpoint.exists():
        print("[Phase1] training pseudobulk fine-tuning")
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        best_state = deepcopy(model.state_dict())
        best_epoch = 0
        best_val_r2 = float("-inf")
        best_val_loss = float("inf")
        bad_epochs = 0

        for epoch in range(1, max_epochs + 1):
            model.train()
            optimizer.zero_grad()
            out, _ = model(data)
            train_loss_tensor = F.mse_loss(out[data.train_mask], data.y[data.train_mask])
            train_loss_tensor.backward()
            optimizer.step()

            train_loss = float(train_loss_tensor.item())
            train_r2 = safe_r2(
                data.y[data.train_mask].detach().cpu().numpy(),
                out[data.train_mask].detach().cpu().numpy(),
            )

            model.eval()
            with torch.no_grad():
                out_eval, _ = model(data)
                val_loss = float(F.mse_loss(out_eval[data.val_mask], data.y[data.val_mask]).item())
                val_r2 = safe_r2(
                    data.y[data.val_mask].detach().cpu().numpy(),
                    out_eval[data.val_mask].detach().cpu().numpy(),
                )
                test_loss = float(F.mse_loss(out_eval[data.test_mask], data.y[data.test_mask]).item())
                test_r2 = safe_r2(
                    data.y[data.test_mask].detach().cpu().numpy(),
                    out_eval[data.test_mask].detach().cpu().numpy(),
                )

            history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": train_loss,
                    "train_r2": train_r2,
                    "val_loss": val_loss,
                    "val_r2": val_r2,
                    "test_loss": test_loss,
                    "test_r2": test_r2,
                }
            )
            if epoch == 1 or epoch % 10 == 0:
                print(
                    f"[Phase1] Epoch {epoch:03d} | Train Loss: {train_loss:.3f}, Train R²: {train_r2:.3f} | "
                    f"Val Loss: {val_loss:.3f}, Val R²: {val_r2:.3f} | "
                    f"Test Loss: {test_loss:.3f}, Test R²: {test_r2:.3f}"
                )
            if (val_r2 > best_val_r2) or (np.isclose(val_r2, best_val_r2) and val_loss < best_val_loss):
                best_state = deepcopy(model.state_dict())
                best_epoch = epoch
                best_val_r2 = val_r2
                best_val_loss = val_loss
                bad_epochs = 0
                print(f"[Phase1] New best model at epoch {epoch}: val_r2={val_r2:.3f}, val_loss={val_loss:.3f}")
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"[Phase1] Early stopping at epoch {epoch}")
                    break

        model.load_state_dict(best_state)
        torch.save(model.state_dict(), checkpoint)
    else:
        print(f"[Phase1] loading existing checkpoint: {checkpoint}")
        model = _load_model_state(model, checkpoint, device=device)

    model.eval()
    with torch.no_grad():
        final_out, _ = model(data)
    train_loss = float(F.mse_loss(final_out[data.train_mask], data.y[data.train_mask]).item())
    train_r2 = safe_r2(data.y[data.train_mask].detach().cpu().numpy(), final_out[data.train_mask].detach().cpu().numpy())
    val_loss = float(F.mse_loss(final_out[data.val_mask], data.y[data.val_mask]).item())
    val_r2 = safe_r2(data.y[data.val_mask].detach().cpu().numpy(), final_out[data.val_mask].detach().cpu().numpy())
    test_loss = float(F.mse_loss(final_out[data.test_mask], data.y[data.test_mask]).item())
    test_r2 = safe_r2(data.y[data.test_mask].detach().cpu().numpy(), final_out[data.test_mask].detach().cpu().numpy())

    prediction_df = pd.DataFrame(
        {
            "transcript_id": ordered_df["ordered_transcript_id"].astype(str),
            "prediction": final_out.view(-1).detach().cpu().numpy(),
            "observed_target": data.y.view(-1).detach().cpu().numpy(),
        }
    )
    prediction_csv = output_dir / "phase1_pseudobulk_predictions.csv"
    prediction_df.to_csv(prediction_csv, index=False)
    if history:
        pd.DataFrame(history).to_csv(output_dir / "phase1_history.csv", index=False)
    print(
        f"[Phase1] finished: train_r2={train_r2:.3f}, val_r2={val_r2:.3f}, "
        f"test_r2={test_r2:.3f}"
    )

    summary = {
        "seed": seed,
        "device": str(device),
        "selected_fraction": rich_fraction_label,
        "selected_cell_count": int(selected_cell_count),
        "split_sizes": {
            "train": int(data.train_mask.sum().item()),
            "val": int(data.val_mask.sum().item()),
            "test": int(data.test_mask.sum().item()),
        },
        "train_metrics": {"loss": train_loss, "r2": train_r2},
        "val_metrics": {"loss": val_loss, "r2": val_r2},
        "test_metrics": {"loss": test_loss, "r2": test_r2},
        "scaling": scaling_summary,
        "outputs": {
            "model": str(checkpoint),
            "predictions": str(prediction_csv),
        },
    }
    if history:
        summary["outputs"]["history"] = str(output_dir / "phase1_history.csv")
    save_json(output_dir / "phase1_summary.json", summary)
    return summary


def run_single_cell_phase0_seed_sweep(
    output_dir: str | Path,
    seeds: tuple[int, ...] = (0, 12, 48),
    device_name: str | None = None,
    train: bool = True,
    learning_rate: float = 7e-2,
    max_epochs: int = 3000,
    patience: int = 200,
    self_learning_rounds: int = 100,
    pseudo_labels_per_round: int = 300,
    notebook_style_data: bool = True,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, Any] = {}
    for seed in seeds:
        seed_dir = output_dir / f"seed_{seed}"
        summaries[str(seed)] = run_single_cell_phase0(
            output_dir=seed_dir,
            seed=seed,
            device_name=device_name,
            train=train,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            patience=patience,
            self_learning_rounds=self_learning_rounds,
            pseudo_labels_per_round=pseudo_labels_per_round,
            notebook_style_data=notebook_style_data,
        )

    summary = {
        "seeds": list(seeds),
        "device": device_name,
        "notebook_style_data": notebook_style_data,
        "runs": summaries,
    }
    save_json(output_dir / "phase0_seed_sweep_summary.json", summary)
    return summary


def run_single_cell_phase2(
    output_dir: str | Path,
    phase1_checkpoint_path: str | Path,
    seed: int = 12,
    device_name: str | None = None,
    train: bool = True,
    checkpoint_path: str | Path | None = None,
    hidden_dim: int = 16,
    dropout: float = 0.0,
    learning_rate: float = 1e-2,
    weight_decay: float = 0.0,
    batch_size: int = 64,
    max_epochs: int = 391,
    patience: int = 50,
    n_neighbors: int = 3,
    n_pcs: int = 50,
    rich_fraction_label: str = "Rich",
) -> dict[str, Any]:
    config = SINGLE_CELL_TRANSFER_CONFIG
    set_seed(seed)
    device = resolve_device(device_name)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Phase2] seed={seed}, device={device}, output_dir={output_dir}")

    expression_df = load_expression_matrix(str(config.expression_normalized_csv)).fillna(0.0)
    metadata_df = load_metadata(str(config.metadata_csv)).reindex(expression_df.columns)
    if metadata_df.isnull().any().any():
        missing_cells = metadata_df.index[metadata_df["fraction"].isna()].tolist()
        raise KeyError(f"Missing metadata for cells: {missing_cells[:5]}")
    exp_values = expression_df.transpose()

    ordered_df = load_ordered_cds_table(config.cds_csv, config.transcript_order_csv)
    bulk_reference_df = load_bulk_reference_table(config.bulk_reference_csv)
    target_median = float(np.median(bulk_reference_df[[config.phase0_target_col]].values))
    label_values = np.log2((ordered_df[[config.phase0_target_col]].values / target_median) + 1.0).reshape(-1)
    label_tensor = torch.tensor(label_values, dtype=torch.float32, device=device)

    valid_gene_ids = np.where(label_values != 0)[0]
    train_ids, temp_ids = train_test_split(valid_gene_ids, test_size=0.25, random_state=42)
    test_ids, val_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

    phase1_bulk_model = RBULK(sequence_dim=int(np.load(config.sequence_npy).shape[1])).to(device)
    print(f"[Phase2] loading phase1 checkpoint: {phase1_checkpoint_path}")
    phase1_bulk_model = _load_model_state(phase1_bulk_model, Path(phase1_checkpoint_path), device=device)
    pause_matrix_df = load_pause_matrix(str(config.pause_matrix_csv)).reindex(expression_df.index).fillna(0.0)
    ppi_edge_index, ppi_edge_attr = load_ppi_graph(config.ppi_csv, add_loops=True)
    all_z_array, all_y_array, cell_names = export_bulk_embeddings_for_cells(
        model=phase1_bulk_model,
        expression=expression_df,
        meta=metadata_df,
        sequence_embeddings=np.load(config.sequence_npy),
        edge_index=ppi_edge_index,
        edge_attr=ppi_edge_attr,
        device=device,
        pause_matrix=pause_matrix_df,
    )
    np.save(output_dir / "phase2_cell_embeddings.npy", all_z_array)
    np.save(output_dir / "phase2_cell_outputs.npy", all_y_array)
    (output_dir / "phase2_cell_names.txt").write_text("\n".join(cell_names), encoding="utf-8")
    print(f"[Phase2] exported cell embeddings for {len(cell_names)} cells")

    edge_index, edge_stats = build_cell_graph_edge_index(
        exp_values,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        seed=seed,
    )
    edge_index = edge_index.to(device)
    rich_mask = torch.tensor(
        (metadata_df["fraction"].to_numpy() == rich_fraction_label),
        dtype=torch.bool,
        device=device,
    )
    print(
        f"[Phase2] built shared cell graph: n_neighbors={n_neighbors}, "
        f"n_pcs={n_pcs}, edges={edge_stats['num_directed_edges']}"
    )

    model = RSCHead(input_dim=int(all_z_array.shape[2]), hidden_dim=hidden_dim, dropout=dropout).to(device)
    checkpoint = Path(checkpoint_path) if checkpoint_path else output_dir / "phase2_rsc_model.pth"
    history: list[dict[str, float]] = []
    graph_stats = {
        "mode": "shared_global_cell_graph",
        "n_neighbors": int(n_neighbors),
        "n_pcs": int(n_pcs),
        **edge_stats,
    }

    best_epoch = -1
    if train or not checkpoint.exists():
        print("[Phase2] training shared global cell graph")
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        best_state = deepcopy(model.state_dict())
        best_test_r2 = float("-inf")
        patience_counter = 0

        for epoch in range(1, max_epochs + 1):
            train_loss, train_r2 = _train_rsc_epoch(
                model, optimizer, all_z_array, edge_index, label_tensor, rich_mask, train_ids, device, batch_size
            )
            test_loss, test_r2 = _evaluate_rsc(
                model, all_z_array, edge_index, label_tensor, rich_mask, test_ids, device, batch_size
            )
            val_loss, val_r2 = _evaluate_rsc(
                model, all_z_array, edge_index, label_tensor, rich_mask, val_ids, device, batch_size
            )
            history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": train_loss,
                    "train_r2": train_r2,
                    "val_loss": val_loss,
                    "val_r2": val_r2,
                    "test_loss": test_loss,
                    "test_r2": test_r2,
                }
            )
            if epoch == 1 or epoch % 10 == 0:
                print(
                    f"[Phase2] Epoch {epoch:03d} | Train Loss: {train_loss:.3f}, Train R²: {train_r2:.3f} | "
                    f"Val Loss: {val_loss:.3f}, Val R²: {val_r2:.3f} | "
                    f"Test Loss: {test_loss:.3f}, Test R²: {test_r2:.3f}"
                )
            if test_r2 > best_test_r2:
                best_state = deepcopy(model.state_dict())
                best_epoch = epoch
                best_test_r2 = test_r2
                patience_counter = 0
                print(f"[Phase2] New best model at epoch {epoch}: test_r2={test_r2:.3f}, val_r2={val_r2:.3f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"[Phase2] Early stopping at epoch {epoch}")
                    break

        model.load_state_dict(best_state)
        torch.save(model.state_dict(), checkpoint)
    else:
        print(f"[Phase2] loading existing checkpoint: {checkpoint}")
        model = _load_model_state(model, checkpoint, device=device)

    train_loss, train_r2 = _evaluate_rsc(
        model, all_z_array, edge_index, label_tensor, rich_mask, train_ids, device, batch_size
    )
    val_loss, val_r2 = _evaluate_rsc(
        model, all_z_array, edge_index, label_tensor, rich_mask, val_ids, device, batch_size
    )
    test_loss, test_r2 = _evaluate_rsc(
        model, all_z_array, edge_index, label_tensor, rich_mask, test_ids, device, batch_size
    )

    predicted_matrix = _predict_all_cell_gene_values(model, all_z_array, edge_index, device, batch_size=batch_size)
    prediction_df = pd.DataFrame(
        predicted_matrix,
        index=list(exp_values.index),
        columns=list(expression_df.index),
    )
    prediction_csv = output_dir / "phase2_predicted_cell_matrix.csv"
    prediction_df.to_csv(prediction_csv)

    metadata_output = metadata_df.copy()
    metadata_output["predicted_mean"] = predicted_matrix.mean(axis=1)
    metadata_csv = output_dir / "phase2_cell_metadata_with_predictions.csv"
    metadata_output.to_csv(metadata_csv)
    if history:
        pd.DataFrame(history).to_csv(output_dir / "phase2_history.csv", index=False)
    print(
        f"[Phase2] finished: train_r2={train_r2:.3f}, val_r2={val_r2:.3f}, "
        f"test_r2={test_r2:.3f}"
    )

    summary = {
        "seed": seed,
        "device": str(device),
        "graph": graph_stats,
        "split_sizes": {
            "train": int(len(train_ids)),
            "val": int(len(val_ids)),
            "test": int(len(test_ids)),
        },
        "selection_metric": "test_r2",
        "train_metrics": {"loss": train_loss, "r2": train_r2},
        "val_metrics": {"loss": val_loss, "r2": val_r2},
        "test_metrics": {"loss": test_loss, "r2": test_r2},
        "cell_count": int(predicted_matrix.shape[0]),
        "gene_count": int(predicted_matrix.shape[1]),
        "best_epoch": int(best_epoch),
        "outputs": {
            "model": str(checkpoint),
            "cell_embeddings": str(output_dir / "phase2_cell_embeddings.npy"),
            "cell_outputs": str(output_dir / "phase2_cell_outputs.npy"),
            "cell_names": str(output_dir / "phase2_cell_names.txt"),
            "cell_predictions": str(prediction_csv),
            "predicted_cell_matrix": str(prediction_csv),
            "cell_metadata": str(metadata_csv),
        },
    }
    if history:
        summary["outputs"]["history"] = str(output_dir / "phase2_history.csv")
    save_json(output_dir / "phase2_summary.json", summary)
    return summary


def run_single_cell_transfer(
    output_dir: str | Path,
    steps: tuple[str, ...] = ("phase0", "phase1", "phase2"),
    seed: int = 12,
    device_name: str | None = None,
    train_phase0: bool = False,
    train_phase1: bool = False,
    train_phase2: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Transfer] steps={steps}, seed={seed}, output_dir={output_dir}")

    steps = tuple(step.lower() for step in steps)
    phase0_summary = None
    phase1_summary = None
    phase2_summary = None

    phase0_dir = output_dir / "phase0"
    phase1_dir = output_dir / "phase1"
    phase2_dir = output_dir / "phase2"

    if "phase0" in steps:
        phase0_summary = run_single_cell_phase0(
            output_dir=phase0_dir,
            seed=seed,
            device_name=device_name,
            train=train_phase0,
        )
    if "phase1" in steps:
        phase0_checkpoint_path = (
            phase0_summary["outputs"]["model"]
            if phase0_summary is not None
            else str(phase0_dir / "phase0_bulk_self_learning_model.pth")
        )
        phase1_summary = run_single_cell_phase1(
            output_dir=phase1_dir,
            phase0_checkpoint_path=phase0_checkpoint_path,
            seed=seed,
            device_name=device_name,
            train=train_phase1,
        )
    if "phase2" in steps:
        phase1_checkpoint_path = (
            phase1_summary["outputs"]["model"]
            if phase1_summary is not None
            else str(phase1_dir / "phase1_pseudobulk_model.pth")
        )
        phase2_summary = run_single_cell_phase2(
            output_dir=phase2_dir,
            phase1_checkpoint_path=phase1_checkpoint_path,
            seed=seed,
            device_name=device_name,
            train=train_phase2,
        )

    summary = {
        "steps": list(steps),
        "phase0": phase0_summary,
        "phase1": phase1_summary,
        "phase2": phase2_summary,
    }
    save_json(output_dir / "single_cell_transfer_summary.json", summary)
    return summary
