from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from torch_geometric.data import Data

from .bulk_data import load_bulk_reference_table, load_ppi_graph, merge_single_pause_file
from .utils import safe_r2


@dataclass(frozen=True)
class BulkConditionSpec:
    name: str
    expression_col: str
    target_col: str
    pause_col: str


def load_bulk_dataframe(
    reference_csv_path: str | Path,
    pause_csv_path: str | Path | None = None,
    pause_col_name: str | None = None,
    merge_on: str = "transcript_id",
) -> pd.DataFrame:
    bulk_df = load_bulk_reference_table(reference_csv_path)
    if pause_csv_path and pause_col_name:
        bulk_df = merge_single_pause_file(
            bulk_df,
            pause_csv_path,
            pause_col_name,
            merge_on=merge_on,
        )
    return bulk_df


def _scale_values(
    values: np.ndarray,
    reference_values: np.ndarray,
    method: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    values_2d = np.asarray(values, dtype=np.float32).reshape(-1, 1)
    reference_2d = np.asarray(reference_values, dtype=np.float32).reshape(-1, 1)

    if method == "log_median":
        denom = float(np.median(reference_2d))
        if np.isclose(denom, 0.0):
            denom = 1.0
        scaled = np.log2((values_2d / denom) + 1.0)
        return scaled.astype(np.float32), {"method": method, "reference_median": denom}

    if method == "standard":
        scaler = StandardScaler()
        scaler.fit(reference_2d)
        scaled = scaler.transform(values_2d)
        return scaled.astype(np.float32), {
            "method": method,
            "mean": float(scaler.mean_[0]),
            "scale": float(scaler.scale_[0]),
        }

    if method == "maxabs":
        scaler = MaxAbsScaler()
        scaler.fit(reference_2d)
        scaled = scaler.transform(values_2d)
        return scaled.astype(np.float32), {
            "method": method,
            "max_abs": float(scaler.max_abs_[0]),
        }

    if method == "none":
        return values_2d.astype(np.float32), {"method": method}

    raise ValueError(f"Unsupported scale method: {method}")


def build_bulk_graph_from_dataframe(
    bulk_df: pd.DataFrame,
    condition: BulkConditionSpec,
    sequence_npy_path: str | Path,
    ppi_csv_path: str | Path,
    scale_method: str = "log_median",
    reference_df: pd.DataFrame | None = None,
    add_loops: bool = True,
) -> tuple[Data, dict[str, Any]]:
    reference_df = bulk_df if reference_df is None else reference_df

    for required_col in (condition.expression_col, condition.target_col, condition.pause_col):
        if required_col not in bulk_df.columns:
            raise KeyError(f"Missing required column '{required_col}' in bulk dataframe.")

    x_values, x_scaling = _scale_values(
        bulk_df[[condition.expression_col]].values,
        reference_df[[condition.expression_col]].values,
        method=scale_method,
    )
    y_values, y_scaling = _scale_values(
        bulk_df[[condition.target_col]].values,
        reference_df[[condition.target_col]].values,
        method=scale_method,
    )

    sequence_embedding = np.load(sequence_npy_path)
    if sequence_embedding.shape[0] != len(bulk_df):
        raise ValueError(
            f"Sequence embedding rows ({sequence_embedding.shape[0]}) "
            f"do not match dataframe rows ({len(bulk_df)})."
        )

    edge_index, edge_weight = load_ppi_graph(ppi_csv_path, add_loops=add_loops)
    pause_values = bulk_df[condition.pause_col].to_numpy(dtype=np.float32).reshape(-1, 1)

    data = Data(
        x=torch.tensor(x_values, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_weight,
        y=torch.tensor(y_values, dtype=torch.float32),
    )
    data.seq = torch.tensor(sequence_embedding, dtype=torch.float32)
    data.pause = torch.tensor(pause_values, dtype=torch.float32)

    scaling_summary = {
        "x": x_scaling,
        "y": y_scaling,
        "expression_col": condition.expression_col,
        "target_col": condition.target_col,
        "pause_col": condition.pause_col,
    }
    return data, scaling_summary


def split_node_indices(
    node_count: int,
    seed: int,
    first_test_size: float = 0.25,
    second_test_size: float = 1.0 / 3.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    node_indices = np.arange(node_count)
    train_idx, temp_idx = train_test_split(node_indices, test_size=first_test_size, random_state=seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=second_test_size, random_state=seed)
    return (
        torch.tensor(train_idx, dtype=torch.long),
        torch.tensor(val_idx, dtype=torch.long),
        torch.tensor(test_idx, dtype=torch.long),
    )


def evaluate_graph_regression(
    model,
    data,
    eval_idx: torch.Tensor | None = None,
) -> dict[str, float]:
    criterion = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        out, _ = model(data)

    pred = out.view(-1)
    target = data.y.view(-1)
    if eval_idx is not None:
        pred = pred[eval_idx]
        target = target[eval_idx]

    loss = float(criterion(pred, target).item())
    r2 = safe_r2(target.detach().cpu().numpy(), pred.detach().cpu().numpy())
    return {"loss": loss, "r2": r2}


def predict_bulk_outputs(model, data) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        predictions, embeddings = model(data)
    return predictions.detach().cpu(), embeddings.detach().cpu()


def train_cross_condition_bulk(
    model,
    train_data,
    eval_data,
    lr: float = 7e-2,
    max_epochs: int = 20000,
    patience: int = 500,
    log_every: int = 50,
) -> tuple[Any, dict[str, Any]]:
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_state = deepcopy(model.state_dict())
    best_epoch = 0
    best_train_r2 = float("-inf")
    best_train_loss = float("inf")
    best_eval_loss = float("inf")
    best_eval_r2 = float("-inf")
    patience_counter = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        train_out, _ = model(train_data)
        train_loss = criterion(train_out.view(-1), train_data.y.view(-1))
        train_loss.backward()
        optimizer.step()

        train_loss_value = float(train_loss.item())
        train_r2 = safe_r2(
            train_data.y.detach().cpu().numpy(),
            train_out.detach().cpu().numpy(),
        )
        monitored_r2 = float("-inf") if np.isnan(train_r2) else float(train_r2)

        eval_metrics = evaluate_graph_regression(model, eval_data)
        history_entry = {
            "epoch": float(epoch),
            "train_loss": train_loss_value,
            "train_r2": float(train_r2),
            "eval_loss": eval_metrics["loss"],
            "eval_r2": eval_metrics["r2"],
        }
        history.append(history_entry)

        if monitored_r2 > best_train_r2:
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            best_train_r2 = monitored_r2
            best_train_loss = train_loss_value
            best_eval_loss = eval_metrics["loss"]
            best_eval_r2 = eval_metrics["r2"]
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

        if log_every > 0 and (epoch == 1 or epoch % log_every == 0):
            print(
                f"Epoch: {epoch:04d}, Train Loss: {train_loss_value:.3f}, Train R²: {train_r2:.3f}"
                f" | Eval Loss: {best_eval_loss:.3f}, Eval R²: {best_eval_r2:.3f}"
            )

    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "best_train_loss": best_train_loss,
        "best_train_r2": best_train_r2,
        "best_eval_loss": best_eval_loss,
        "best_eval_r2": best_eval_r2,
        "history": history,
    }


def train_single_graph_bulk(
    model,
    data,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
    lr: float = 7e-2,
    max_epochs: int = 3000,
    patience: int = 200,
    log_every: int = 50,
) -> tuple[Any, dict[str, Any]]:
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_state = deepcopy(model.state_dict())
    best_epoch = 0
    best_val_r2 = float("-inf")
    best_train_loss = float("inf")
    best_val_loss = float("inf")
    best_test_loss = float("inf")
    best_test_r2 = float("-inf")
    patience_counter = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out, _ = model(data)
        train_loss = criterion(out[train_idx], data.y[train_idx])
        train_loss.backward()
        optimizer.step()

        train_loss_value = float(train_loss.item())
        train_r2 = safe_r2(
            data.y[train_idx].detach().cpu().numpy(),
            out[train_idx].detach().cpu().numpy(),
        )
        val_metrics = evaluate_graph_regression(model, data, val_idx)
        test_metrics = evaluate_graph_regression(model, data, test_idx)

        history_entry = {
            "epoch": float(epoch),
            "train_loss": train_loss_value,
            "train_r2": float(train_r2),
            "val_loss": val_metrics["loss"],
            "val_r2": val_metrics["r2"],
            "test_loss": test_metrics["loss"],
            "test_r2": test_metrics["r2"],
        }
        history.append(history_entry)

        monitored_r2 = float("-inf") if np.isnan(val_metrics["r2"]) else float(val_metrics["r2"])
        if monitored_r2 > best_val_r2:
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            best_val_r2 = monitored_r2
            best_train_loss = train_loss_value
            best_val_loss = val_metrics["loss"]
            best_test_loss = test_metrics["loss"]
            best_test_r2 = test_metrics["r2"]
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

        if log_every > 0 and (epoch == 1 or epoch % log_every == 0):
            print(
                f"Epoch: {epoch:04d}, Train Loss: {train_loss_value:.3f}, Train R²: {train_r2:.3f}"
                f" | Val Loss: {best_val_loss:.3f}, Val R²: {best_val_r2:.3f}"
                f" | Test Loss: {best_test_loss:.3f}, Test R²: {best_test_r2:.3f}"
            )

    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "best_train_loss": best_train_loss,
        "best_val_loss": best_val_loss,
        "best_val_r2": best_val_r2,
        "best_test_loss": best_test_loss,
        "best_test_r2": best_test_r2,
        "history": history,
    }
