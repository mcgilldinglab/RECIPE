from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from typing import Optional, Union

from .utils import safe_r2


def load_expression_matrix(expression_csv_path: str, gene_id_col: str = "Unnamed: 0") -> pd.DataFrame:
    expression_df = pd.read_csv(expression_csv_path)
    if gene_id_col not in expression_df.columns:
        gene_id_col = expression_df.columns[0]
    expression_df[gene_id_col] = expression_df[gene_id_col].astype(str).str.split(".").str[0]
    return expression_df.set_index(gene_id_col)


def load_metadata(meta_csv_path: str, cell_name_col: str = "cell_names") -> pd.DataFrame:
    meta_df = pd.read_csv(meta_csv_path)
    if cell_name_col not in meta_df.columns and meta_df.columns[0].startswith("Unnamed"):
        meta_df = meta_df.rename(columns={meta_df.columns[0]: cell_name_col})
    meta_df = meta_df.loc[:, ~meta_df.columns.duplicated()]
    meta_df[cell_name_col] = meta_df[cell_name_col].astype(str)
    meta_df = meta_df.drop_duplicates(subset=[cell_name_col]).set_index(cell_name_col)
    return meta_df


def load_pause_matrix(pause_csv_path: str, gene_id_col: str = "transcript_id") -> pd.DataFrame:
    pause_df = pd.read_csv(pause_csv_path)
    if gene_id_col not in pause_df.columns:
        gene_id_col = pause_df.columns[0]
    pause_df[gene_id_col] = pause_df[gene_id_col].astype(str).str.split(".").str[0]
    return pause_df.set_index(gene_id_col)


def create_knn_edge_index(values: np.ndarray, n_neighbors: int = 3) -> torch.Tensor:
    values = np.asarray(values, dtype=np.float32).reshape(-1, 1)
    effective_neighbors = min(len(values), n_neighbors + 1)
    knn = NearestNeighbors(n_neighbors=effective_neighbors)
    knn.fit(values)
    _, indices = knn.kneighbors(values)

    edges = []
    for node, neighbors in enumerate(indices):
        for neighbor in neighbors[1:]:
            edges.append([node, neighbor])

    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def build_pause_vector_for_cell(
    cell_name: str,
    meta: pd.DataFrame,
    pause_matrix: Optional[pd.DataFrame] = None,
    fraction_table: Optional[pd.DataFrame] = None,
    fraction_to_pause_col: Optional[dict[str, str]] = None,
    fraction_col: str = "fraction",
) -> torch.Tensor:
    if pause_matrix is not None:
        if cell_name not in pause_matrix.columns:
            raise KeyError(f"Cell {cell_name} is missing from the pause matrix.")
        pause_values = pause_matrix[cell_name].to_numpy(dtype=np.float32)
        return torch.tensor(pause_values, dtype=torch.float32).view(-1, 1)

    if fraction_table is None or fraction_to_pause_col is None:
        raise ValueError("fraction_table and fraction_to_pause_col are required when pause_matrix is not used.")

    fraction_name = meta.loc[cell_name, fraction_col]
    pause_col = fraction_to_pause_col[fraction_name]
    pause_values = fraction_table[pause_col].to_numpy(dtype=np.float32)
    return torch.tensor(pause_values, dtype=torch.float32).view(-1, 1)


def export_bulk_embeddings_for_cells(
    model,
    expression: pd.DataFrame,
    meta: pd.DataFrame,
    sequence_embeddings: np.ndarray,
    edge_index: torch.Tensor,
    edge_attr: Optional[torch.Tensor],
    device: torch.device,
    pause_matrix: Optional[pd.DataFrame] = None,
    fraction_table: Optional[pd.DataFrame] = None,
    fraction_to_pause_col: Optional[dict[str, str]] = None,
    fraction_col: str = "fraction",
):
    model.eval()
    sequence_tensor = torch.tensor(sequence_embeddings, dtype=torch.float32, device=device)
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device) if edge_attr is not None else None

    cell_names = list(expression.columns)
    all_z = []
    all_y = []

    for cell_name in cell_names:
        x_tensor = torch.tensor(
            expression[cell_name].to_numpy(dtype=np.float32).reshape(-1, 1),
            dtype=torch.float32,
            device=device,
        )
        pause_tensor = build_pause_vector_for_cell(
            cell_name,
            meta,
            pause_matrix=pause_matrix,
            fraction_table=fraction_table,
            fraction_to_pause_col=fraction_to_pause_col,
            fraction_col=fraction_col,
        ).to(device)

        data = Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_attr)
        data.seq = sequence_tensor
        data.pause = pause_tensor

        with torch.no_grad():
            y_pred, z = model(data)
        all_y.append(y_pred.detach().cpu().numpy())
        all_z.append(z.detach().cpu().numpy())

    return np.stack(all_z), np.stack(all_y), cell_names


def create_cell_graphs_subset_genes(
    exp_values: pd.DataFrame,
    z_array: np.ndarray,
    meta: pd.DataFrame,
    gene_indices: np.ndarray,
    n_neighbors: int = 3,
    fraction_col: str = "fraction",
) -> list[Data]:
    graphs: list[Data] = []
    gene_indices = np.asarray(gene_indices, dtype=np.int64)

    for cell_idx, cell_name in enumerate(exp_values.index):
        cell_exp = exp_values.iloc[cell_idx, gene_indices].to_numpy(dtype=np.float32)
        edge_index = create_knn_edge_index(cell_exp, n_neighbors=n_neighbors)
        node_features = torch.tensor(z_array[cell_idx, gene_indices, :], dtype=torch.float32)

        is_rich = bool(meta.loc[cell_name, fraction_col] == "Rich")
        rich_mask = torch.full((len(gene_indices),), is_rich, dtype=torch.bool)
        graph = Data(
            x=node_features,
            edge_index=edge_index,
            rich_mask=rich_mask,
            gene_ids=torch.tensor(gene_indices, dtype=torch.long),
        )
        graph.cell_name = cell_name
        graphs.append(graph)
    return graphs


def create_gene_graphs_subset_cells(
    exp_values: pd.DataFrame,
    z_array: np.ndarray,
    meta: pd.DataFrame,
    gene_indices: np.ndarray,
    n_neighbors: int = 3,
    fraction_col: str = "fraction",
) -> list[Data]:
    graphs: list[Data] = []
    gene_indices = np.asarray(gene_indices, dtype=np.int64)
    rich_mask = torch.tensor((meta[fraction_col].to_numpy() == "Rich"), dtype=torch.bool)
    num_cells = exp_values.shape[0]

    for gene_idx in gene_indices:
        gene_exp = exp_values.iloc[:, gene_idx].to_numpy(dtype=np.float32)
        edge_index = create_knn_edge_index(gene_exp, n_neighbors=n_neighbors)
        node_features = torch.tensor(z_array[:, gene_idx, :], dtype=torch.float32)
        graph = Data(
            x=node_features,
            edge_index=edge_index,
            rich_mask=rich_mask.clone(),
            gene_ids=torch.full((num_cells,), int(gene_idx), dtype=torch.long),
        )
        graph.gene_idx = int(gene_idx)
        graphs.append(graph)
    return graphs


def split_gene_ids(label_values: np.ndarray | pd.Series, seed: int = 42):
    label_array = np.asarray(label_values)
    valid_gene_ids = np.where(label_array != 0)[0]
    train_gene_ids, temp_gene_ids = train_test_split(valid_gene_ids, test_size=0.25, random_state=seed)
    test_gene_ids, val_gene_ids = train_test_split(temp_gene_ids, test_size=0.5, random_state=seed)
    return train_gene_ids, val_gene_ids, test_gene_ids


def compute_rich_loss_and_metrics(
    predictions: torch.Tensor,
    batch,
    label_tensor: torch.Tensor,
) -> tuple[Optional[torch.Tensor], float, float]:
    rich_mask = batch.rich_mask.view(-1).bool()
    gene_ids = batch.gene_ids.view(-1)

    if int(rich_mask.sum().item()) == 0:
        return None, 0.0, float("nan")

    rich_preds = predictions[rich_mask]
    rich_gene_ids = gene_ids[rich_mask]

    losses = []
    all_preds = []
    all_labels = []

    for gid in torch.unique(rich_gene_ids):
        gid_mask = rich_gene_ids == gid
        preds = rich_preds[gid_mask]
        if preds.numel() == 0:
            continue

        mean_pred = preds.mean()
        label = label_tensor[gid]
        losses.append(F.mse_loss(mean_pred, label))
        all_preds.append(float(mean_pred.detach().cpu().item()))
        all_labels.append(float(label.detach().cpu().item()))

    if not losses:
        return None, 0.0, float("nan")

    total_loss = torch.stack(losses).mean()
    return total_loss, float(total_loss.detach().cpu().item()), safe_r2(all_labels, all_preds)


def run_fsc_epoch(model, loader, label_tensor, device, optimizer=None) -> tuple[float, float]:
    batch = next(iter(loader)).to(device)
    if optimizer is None:
        model.eval()
        with torch.no_grad():
            predictions = model(batch.x, batch.edge_index)
            _, avg_loss, r2 = compute_rich_loss_and_metrics(predictions, batch, label_tensor)
        return avg_loss, r2

    model.train()
    optimizer.zero_grad()
    predictions = model(batch.x, batch.edge_index)
    total_loss, avg_loss, r2 = compute_rich_loss_and_metrics(predictions, batch, label_tensor)
    if total_loss is not None:
        total_loss.backward()
        optimizer.step()
    return avg_loss, r2


def predict_cell_gene_matrix(
    model,
    graphs: list[Data],
    cell_names: list[str],
    gene_labels: list[str],
    device: torch.device,
    batch_size: int = 32,
) -> pd.DataFrame:
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    predicted_matrix = np.zeros((len(graphs), len(gene_labels)), dtype=np.float32)

    model.eval()
    cell_offset = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            predictions = model(batch.x, batch.edge_index)
            num_graphs = int(batch.ptr.numel() - 1)

            for graph_idx in range(num_graphs):
                start = int(batch.ptr[graph_idx].item())
                end = int(batch.ptr[graph_idx + 1].item())
                gene_ids = batch.gene_ids[start:end].detach().cpu().numpy().astype(int)
                predicted_matrix[cell_offset + graph_idx, gene_ids] = (
                    predictions[start:end].detach().cpu().numpy().reshape(-1)
                )
            cell_offset += num_graphs

    return pd.DataFrame(predicted_matrix, index=cell_names, columns=gene_labels)


def predict_gene_cell_matrix(
    model,
    graphs: list[Data],
    cell_names: list[str],
    gene_labels: list[str],
    device: torch.device,
    batch_size: int = 32,
) -> pd.DataFrame:
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    predicted_matrix = np.zeros((len(cell_names), len(gene_labels)), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            predictions = model(batch.x, batch.edge_index)
            num_graphs = int(batch.ptr.numel() - 1)

            for graph_idx in range(num_graphs):
                start = int(batch.ptr[graph_idx].item())
                end = int(batch.ptr[graph_idx + 1].item())
                gene_idx = int(batch.gene_ids[start].detach().cpu().item())
                predicted_matrix[:, gene_idx] = predictions[start:end].detach().cpu().numpy().reshape(-1)

    return pd.DataFrame(predicted_matrix, index=cell_names, columns=gene_labels)
