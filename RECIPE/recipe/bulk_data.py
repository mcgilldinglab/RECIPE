from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
from scipy import sparse as sp
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, from_scipy_sparse_matrix


def strip_version(values: pd.Series) -> pd.Series:
    return values.astype(str).str.split(".").str[0]


def load_order_ids(order_csv_path: Union[str, Path], id_col: Optional[str] = None) -> list[str]:
    order_df = pd.read_csv(order_csv_path)
    order_col = id_col or order_df.columns[0]
    return order_df[order_col].astype(str).tolist()


def align_to_order(df: pd.DataFrame, order_ids: list[str], id_col: str) -> pd.DataFrame:
    aligned = df.copy()
    aligned[id_col] = strip_version(aligned[id_col])
    aligned = aligned.drop_duplicates(subset=[id_col])
    aligned = aligned.set_index(id_col).reindex(pd.Index(order_ids, name=id_col)).reset_index()
    return aligned


def load_ordered_cds_table(cds_df_path: Union[str, Path], order_csv_path: Union[str, Path]) -> pd.DataFrame:
    cds_df = pd.read_csv(cds_df_path).iloc[:, 1:9].copy()
    cds_df["transcript_id"] = strip_version(cds_df["transcript_id_x"])
    order_ids = load_order_ids(order_csv_path)
    ordered = align_to_order(cds_df, order_ids, "transcript_id")
    ordered.insert(0, "ordered_transcript_id", order_ids)
    ordered.fillna(0, inplace=True)
    return ordered


def load_bulk_reference_table(reference_csv_path: Union[str, Path]) -> pd.DataFrame:
    reference_df = pd.read_csv(reference_csv_path)
    if "transcript_id" in reference_df.columns:
        reference_df["transcript_id"] = strip_version(reference_df["transcript_id"])
    return reference_df


def merge_single_pause_file(
    df: pd.DataFrame,
    pause_csv_path: Union[str, Path],
    pause_col_name: str,
    merge_on: str = "transcript_id",
) -> pd.DataFrame:
    pause_df = pd.read_csv(pause_csv_path)
    pause_key_col = pause_df.columns[2]
    pause_value_col = pause_df.columns[1]
    pause_df = pause_df.rename(
        columns={
            pause_key_col: merge_on,
            pause_value_col: pause_col_name,
        }
    )
    pause_df[merge_on] = strip_version(pause_df[merge_on])
    merged = df.merge(pause_df[[merge_on, pause_col_name]], on=merge_on, how="left")
    merged[pause_col_name] = merged[pause_col_name].fillna(0)
    return merged


def merge_multiple_pause_files(
    df: pd.DataFrame,
    pause_files: dict[str, Union[str, Path]],
    pause_columns: dict[str, str],
    merge_on: str = "transcript_id",
) -> pd.DataFrame:
    merged = df.copy()
    for condition, pause_path in pause_files.items():
        merged = merge_single_pause_file(
            merged,
            pause_path,
            pause_columns[condition],
            merge_on=merge_on,
        )
    return merged


def load_ppi_graph(ppi_csv_path: Union[str, Path], add_loops: bool = True):
    ppi_matrix = pd.read_csv(ppi_csv_path)
    edge_index, edge_weight = from_scipy_sparse_matrix(sp.coo_matrix(ppi_matrix).astype("float32"))
    if add_loops:
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight)
    return edge_index, edge_weight


def build_bulk_graph_data(
    node_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    sequence_npy_path: Union[str, Path],
    ppi_csv_path: Union[str, Path],
    expression_col: str,
    target_col: str,
    pause_col: str | None = None,
    add_loops: bool = True,
) -> Data:
    expr_median = float(np.median(reference_df[[expression_col]].values))
    target_median = float(np.median(reference_df[[target_col]].values))

    x_values = np.log2((node_df[[expression_col]].values / expr_median) + 1.0)
    y_values = np.log2((node_df[[target_col]].values / target_median) + 1.0)
    sequence_embedding = np.load(sequence_npy_path)

    if sequence_embedding.shape[0] != len(node_df):
        raise ValueError(
            f"Sequence embedding rows ({sequence_embedding.shape[0]}) "
            f"do not match node count ({len(node_df)})."
        )

    edge_index, edge_weight = load_ppi_graph(ppi_csv_path, add_loops=add_loops)

    data = Data(
        x=torch.tensor(x_values, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_weight,
        y=torch.tensor(y_values, dtype=torch.float32),
    )
    data.seq = torch.tensor(sequence_embedding, dtype=torch.float32)
    if pause_col is not None:
        pause_values = node_df[pause_col].to_numpy(dtype=np.float32).reshape(-1, 1)
        data.pause = torch.tensor(pause_values, dtype=torch.float32)
    return data


def find_labeled_indices(
    ordered_transcript_ids: Union[pd.Series, list[str]],
    labeled_df: pd.DataFrame,
    labeled_id_col: str = "transcript_id",
) -> torch.Tensor:
    labeled_ids = set(strip_version(labeled_df[labeled_id_col]).tolist())
    indices = [idx for idx, tid in enumerate(pd.Series(ordered_transcript_ids).astype(str)) if tid in labeled_ids]
    return torch.tensor(indices, dtype=torch.long)


def split_index_tensor(
    indices: torch.Tensor,
    seed: int,
    first_test_size: float = 0.25,
    second_test_size: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train_idx, temp_idx = train_test_split(indices.cpu().numpy(), test_size=first_test_size, random_state=seed)
    test_idx, val_idx = train_test_split(temp_idx, test_size=second_test_size, random_state=seed)
    return (
        torch.tensor(train_idx, dtype=torch.long),
        torch.tensor(val_idx, dtype=torch.long),
        torch.tensor(test_idx, dtype=torch.long),
    )


def build_masks_from_target_values(
    target_values: Union[np.ndarray, pd.Series],
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw_target = torch.tensor(np.asarray(target_values), dtype=torch.float32).view(-1)
    valid_mask = (~torch.isnan(raw_target)) & (raw_target != 0)
    valid_indices = valid_mask.nonzero(as_tuple=True)[0]
    train_idx, val_idx, test_idx = split_index_tensor(valid_indices, seed=seed)

    train_mask = torch.zeros(raw_target.numel(), dtype=torch.bool)
    val_mask = torch.zeros(raw_target.numel(), dtype=torch.bool)
    test_mask = torch.zeros(raw_target.numel(), dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


def load_fraction_average_expression(
    expression_csv_path: Union[str, Path],
    meta_csv_path: Union[str, Path],
    fraction: str = "Rich",
    gene_id_col: str = "Unnamed: 0",
    cell_id_col: str = "cell_names",
    fraction_col: str = "fraction",
    feature_name: str = "scribo",
) -> tuple[pd.DataFrame, list[str]]:
    expression_df = pd.read_csv(expression_csv_path)
    meta_df = pd.read_csv(meta_csv_path)
    selected_cells = meta_df.loc[meta_df[fraction_col] == fraction, cell_id_col].astype(str).tolist()
    present_cells = [cell for cell in selected_cells if cell in expression_df.columns]

    averaged = expression_df[[gene_id_col] + present_cells].copy()
    averaged[feature_name] = averaged.iloc[:, 1:].mean(axis=1)
    return averaged[[gene_id_col, feature_name]], present_cells
