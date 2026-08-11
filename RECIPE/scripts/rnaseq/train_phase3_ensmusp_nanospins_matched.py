#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv


BASE_DIR = Path(__file__).resolve().parent


def set_seed(seed: int = 0) -> None:
    print(f"seed = {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


class Phase3CellGraph(torch.nn.Module):
    def __init__(self, input_dim: int = 64, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim, aggr="mean")
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, aggr="mean")
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.head = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return self.head(x).view(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "ENSMUSP phase3 trainer using phase2 hidden z with matched nanoSPINS single-cell protein supervision. "
            "RNA barcodes are aligned to nanoSPINS protein sample IDs via d5lc01008j2.xlsx."
        )
    )
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--hidden-cache-root", type=Path, required=True)
    parser.add_argument("--truth-csv", type=Path, required=True)
    parser.add_argument("--mapping-xlsx", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--phase3-hidden-dim", type=int, default=64)
    parser.add_argument("--phase3-dropout", type=float, default=0.1)
    parser.add_argument("--phase3-batch-size", type=int, default=16)
    parser.add_argument("--phase3-epochs", type=int, default=1000)
    parser.add_argument("--phase3-patience", type=int, default=100)
    parser.add_argument("--phase3-lr", type=float, default=1e-3)
    parser.add_argument("--phase3-weight-decay", type=float, default=1e-4)
    parser.add_argument("--phase3-k-neighbors", type=int, default=7)
    parser.add_argument("--phase3-n-pcs", type=int, default=20)
    parser.add_argument("--sc-normalize-target-sum", type=float, default=1e4)
    parser.add_argument("--condition", choices=["both", "C10", "SVEC"], default="both")
    return parser.parse_args()


def read_ordered_frame(path: Path, order_ids: pd.Index) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["protein_id"] = df["protein_id"].astype(str)
    return df.drop_duplicates("protein_id").set_index("protein_id").reindex(order_ids).reset_index()


def normalize_total_rows(matrix: np.ndarray, target_sum: float) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float32)
    row_sums = arr.sum(axis=1, keepdims=True)
    scales = np.zeros_like(row_sums, dtype=np.float32)
    nz = row_sums > 0
    scales[nz] = float(target_sum) / row_sums[nz]
    return arr * scales


def build_pca_knn_edge_index(cell_by_gene: np.ndarray, n_neighbors: int, n_pcs: int, seed: int) -> torch.Tensor:
    if cell_by_gene.shape[0] < 2:
        raise ValueError("Need at least 2 cells to build KNN graph.")
    values = np.nan_to_num(np.asarray(cell_by_gene, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    values = np.log1p(normalize_total_rows(values, target_sum=1e4))
    n_pcs = max(1, min(int(n_pcs), values.shape[0], values.shape[1]))
    reduced = PCA(n_components=n_pcs, random_state=seed).fit_transform(values)
    effective_neighbors = min(int(n_neighbors) + 1, reduced.shape[0])
    knn = NearestNeighbors(n_neighbors=effective_neighbors, metric="euclidean")
    knn.fit(reduced)
    _, indices = knn.kneighbors(reduced)
    rows: list[int] = []
    cols: list[int] = []
    for src in range(indices.shape[0]):
        for dst in indices[src, 1:]:
            rows.append(src)
            cols.append(int(dst))
            rows.append(int(dst))
            cols.append(src)
    return torch.tensor(np.vstack([rows, cols]), dtype=torch.long)


def load_mapping_table(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="Supplmentary file 3", header=None)
    header = raw.iloc[1].fillna("").astype(str).tolist()
    df = raw.iloc[2:].copy()
    df.columns = header
    keep_cols = ["Cell Barcode", "Annotation", "Common with Proteomics samples", "Cell type", "Condition"]
    df = df[keep_cols].copy()
    for col in keep_cols:
        df[col] = df[col].astype(str)
    df = df[(df["Cell Barcode"] != "nan") & (df["Common with Proteomics samples"] != "nan")].copy()
    return df.drop_duplicates(subset=["Cell Barcode", "Common with Proteomics samples"])


def masked_mse(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.bool()
    if int(mask.sum().item()) == 0:
        return pred.new_tensor(0.0)
    return F.mse_loss(pred[mask], y[mask])


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return float("nan")
    if np.allclose(y_true, y_true[0]):
        return float("nan")
    return float(r2_score(y_true, y_pred))


def evaluate_split(model: Phase3CellGraph, graphs: list[Data], device: torch.device) -> tuple[dict[str, float], pd.DataFrame]:
    if not graphs:
        return {"loss": float("nan"), "r2": float("nan"), "count": 0}, pd.DataFrame()
    loader = DataLoader(graphs, batch_size=32, shuffle=False)
    losses: list[float] = []
    pred_all: list[np.ndarray] = []
    truth_all: list[np.ndarray] = []
    rows: list[dict] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index)
            loss = masked_mse(pred, batch.y.view(-1), batch.mask.view(-1))
            losses.append(float(loss.item()))
            pred_np = pred.detach().cpu().numpy().astype(np.float32, copy=False)
            y_np = batch.y.view(-1).detach().cpu().numpy().astype(np.float32, copy=False)
            mask_np = batch.mask.view(-1).detach().cpu().numpy().astype(bool, copy=False)
            pred_all.append(pred_np[mask_np])
            truth_all.append(y_np[mask_np])
            ptr = batch.ptr.detach().cpu().numpy().astype(int, copy=False)
            target_idx = batch.target_idx.detach().cpu().numpy().astype(int, copy=False)
            condition_idx = batch.condition_idx.detach().cpu().numpy().astype(int, copy=False)
            for i in range(len(ptr) - 1):
                start = ptr[i]
                end = ptr[i + 1]
                local_mask = mask_np[start:end]
                rows.append(
                    {
                        "target_idx": int(target_idx[i]),
                        "condition_idx": int(condition_idx[i]),
                        "labeled_nodes": int(local_mask.sum()),
                    }
                )
    truth_flat = np.concatenate(truth_all) if truth_all else np.asarray([], dtype=np.float32)
    pred_flat = np.concatenate(pred_all) if pred_all else np.asarray([], dtype=np.float32)
    metrics = {
        "loss": float(np.mean(losses)),
        "r2": safe_r2(truth_flat, pred_flat),
        "count": int(truth_flat.size),
    }
    return metrics, pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    for subdir in ("models", "tables", "reports"):
        (output_root / subdir).mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = resolve_device(args.device)

    hidden_npy = Path(args.hidden_cache_root) / "phase2_hidden_all.npy"
    cell_csv = Path(args.hidden_cache_root) / "phase2_hidden_cell_names.csv"
    gene_csv = Path(args.hidden_cache_root) / "phase2_hidden_gene_names.csv"
    if not (hidden_npy.exists() and cell_csv.exists() and gene_csv.exists()):
        raise FileNotFoundError("Missing phase2 hidden cache files.")

    hidden_all = np.load(hidden_npy, mmap_mode="r")
    cell_names = pd.read_csv(cell_csv)["cell_name"].astype(str).tolist()
    order_ids = pd.Index(pd.read_csv(gene_csv)["protein_id"].astype(str), name="protein_id")

    sc_rna_all = read_ordered_frame(Path(args.bundle_dir) / "scRNA_qc_cells_by_ENSMUSP_all.bulk_intersection.zero_filled.csv", order_ids)
    if [col for col in sc_rna_all.columns if col != "protein_id"] != cell_names:
        raise ValueError("Hidden cache cell names do not match scRNA matrix order.")
    expr_gene_by_cell = sc_rna_all.drop(columns=["protein_id"]).to_numpy(dtype=np.float32, copy=False)
    expr_cell_by_gene = expr_gene_by_cell.T.copy()

    truth_df = read_ordered_frame(Path(args.truth_csv), order_ids)
    truth_cols = [c for c in truth_df.columns if c != "protein_id"]

    mapping_df = load_mapping_table(Path(args.mapping_xlsx))
    mapping_df = mapping_df[
        mapping_df["Cell Barcode"].isin(cell_names) & mapping_df["Common with Proteomics samples"].isin(truth_cols)
    ].copy()
    mapping_df = mapping_df.drop_duplicates(subset=["Cell Barcode"])
    mapping_df = mapping_df.drop_duplicates(subset=["Common with Proteomics samples"])
    cell_to_idx = {name: idx for idx, name in enumerate(cell_names)}
    mapping_df["cell_idx"] = mapping_df["Cell Barcode"].map(cell_to_idx)
    mapping_df = mapping_df.sort_values("cell_idx").reset_index(drop=True)
    if mapping_df.empty:
        raise ValueError("No matched nanoSPINS cells found.")

    matched_truth_cols = mapping_df["Common with Proteomics samples"].astype(str).tolist()
    matched_cell_idx = mapping_df["cell_idx"].astype(int).to_numpy()
    matched_cell_type = mapping_df["Cell type"].astype(str).to_numpy()

    truth_matrix = truth_df.loc[:, matched_truth_cols].to_numpy(dtype=np.float32, copy=False)  # genes x matched_cells
    c10_pos = np.where(matched_cell_type == "C10")[0].astype(np.int64)
    svec_pos = np.where(matched_cell_type == "SVEC")[0].astype(np.int64)
    if args.condition == "both":
        if c10_pos.size < 2 or svec_pos.size < 2:
            raise ValueError("Need at least two matched cells in each condition.")
    elif args.condition == "C10":
        if c10_pos.size < 2:
            raise ValueError("Need at least two matched C10 cells.")
    else:
        if svec_pos.size < 2:
            raise ValueError("Need at least two matched SVEC cells.")

    c10_full_idx = matched_cell_idx[c10_pos]
    svec_full_idx = matched_cell_idx[svec_pos]
    c10_edge_index = None
    svec_edge_index = None
    if args.condition in {"both", "C10"}:
        c10_edge_index = build_pca_knn_edge_index(
            expr_cell_by_gene[c10_full_idx],
            n_neighbors=int(args.phase3_k_neighbors),
            n_pcs=int(args.phase3_n_pcs),
            seed=int(args.seed),
        )
    if args.condition in {"both", "SVEC"}:
        svec_edge_index = build_pca_knn_edge_index(
            expr_cell_by_gene[svec_full_idx],
            n_neighbors=int(args.phase3_k_neighbors),
            n_pcs=int(args.phase3_n_pcs),
            seed=int(args.seed),
        )

    finite_counts = np.isfinite(truth_matrix).sum(axis=1)
    eligible_idx = np.flatnonzero(finite_counts > 0).astype(np.int64)
    if eligible_idx.size < 20:
        raise ValueError(f"Too few eligible proteins for matched nanoSPINS phase3: {eligible_idx.size}")

    train_frac = float(args.train_frac)
    val_frac = float(args.val_frac)
    test_frac = 1.0 - train_frac - val_frac
    if test_frac <= 0:
        raise ValueError("train_frac + val_frac must be < 1.")
    train_idx, temp_idx = train_test_split(eligible_idx, test_size=(1.0 - train_frac), random_state=args.seed)
    val_relative = val_frac / (val_frac + test_frac)
    val_idx, test_idx = train_test_split(temp_idx, train_size=val_relative, random_state=args.seed)

    split_map = np.full((len(order_ids),), "unused", dtype=object)
    split_map[train_idx] = "train"
    split_map[val_idx] = "val"
    split_map[test_idx] = "test"

    def build_graphs(split_indices: np.ndarray) -> list[Data]:
        graphs: list[Data] = []
        for gene_idx in split_indices.tolist():
            if args.condition in {"both", "C10"}:
                y_c10 = truth_matrix[gene_idx, c10_pos].astype(np.float32, copy=False)
                mask_c10 = np.isfinite(y_c10)
                if int(mask_c10.sum()) > 0:
                    graph_c10 = Data(
                        x=torch.from_numpy(np.asarray(hidden_all[c10_full_idx, gene_idx, :], dtype=np.float32)),
                        edge_index=c10_edge_index.clone(),
                        y=torch.from_numpy(np.nan_to_num(y_c10, nan=0.0)).float(),
                        mask=torch.from_numpy(mask_c10.astype(np.float32)),
                    )
                    graph_c10.target_idx = torch.tensor([gene_idx], dtype=torch.long)
                    graph_c10.condition_idx = torch.tensor([0], dtype=torch.long)
                    graphs.append(graph_c10)

            if args.condition in {"both", "SVEC"}:
                y_svec = truth_matrix[gene_idx, svec_pos].astype(np.float32, copy=False)
                mask_svec = np.isfinite(y_svec)
                if int(mask_svec.sum()) > 0:
                    graph_svec = Data(
                        x=torch.from_numpy(np.asarray(hidden_all[svec_full_idx, gene_idx, :], dtype=np.float32)),
                        edge_index=svec_edge_index.clone(),
                        y=torch.from_numpy(np.nan_to_num(y_svec, nan=0.0)).float(),
                        mask=torch.from_numpy(mask_svec.astype(np.float32)),
                    )
                    graph_svec.target_idx = torch.tensor([gene_idx], dtype=torch.long)
                    graph_svec.condition_idx = torch.tensor([1], dtype=torch.long)
                    graphs.append(graph_svec)
        return graphs

    train_graphs = build_graphs(train_idx)
    val_graphs = build_graphs(val_idx)
    test_graphs = build_graphs(test_idx)
    if not train_graphs:
        raise ValueError("No training graphs were created.")

    train_loader = DataLoader(train_graphs, batch_size=int(args.phase3_batch_size), shuffle=True)
    model = Phase3CellGraph(input_dim=int(hidden_all.shape[2]), hidden_dim=int(args.phase3_hidden_dim), dropout=float(args.phase3_dropout)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.phase3_lr), weight_decay=float(args.phase3_weight_decay))

    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_val_r2 = float("-inf")
    patience_counter = 0
    history: list[dict] = []

    for epoch in range(1, int(args.phase3_epochs) + 1):
        model.train()
        train_loss_loader = 0.0
        train_graph_count = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch.x, batch.edge_index)
            loss = masked_mse(pred, batch.y.view(-1), batch.mask.view(-1))
            loss.backward()
            optimizer.step()
            train_loss_loader += float(loss.item()) * int(batch.num_graphs)
            train_graph_count += int(batch.num_graphs)

        train_metrics, _ = evaluate_split(model, train_graphs, device=device)
        val_metrics, _ = evaluate_split(model, val_graphs, device=device)
        test_metrics, _ = evaluate_split(model, test_graphs, device=device)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss_loader": train_loss_loader / float(max(train_graph_count, 1)),
                "train_loss_eval": float(train_metrics["loss"]),
                "train_r2": float(train_metrics["r2"]),
                "val_loss": float(val_metrics["loss"]),
                "val_r2": float(val_metrics["r2"]),
                "test_loss": float(test_metrics["loss"]),
                "test_r2": float(test_metrics["r2"]),
            }
        )
        print(
            f"Phase3Nano Epoch {epoch:03d} | "
            f"Train Loss {train_metrics['loss']:.4f} R2 {train_metrics['r2']:.4f} | "
            f"Val Loss {val_metrics['loss']:.4f} R2 {val_metrics['r2']:.4f} | "
            f"Test Loss {test_metrics['loss']:.4f} R2 {test_metrics['r2']:.4f}"
        )
        if val_metrics["r2"] > best_val_r2:
            best_val_r2 = float(val_metrics["r2"])
            best_epoch = int(epoch)
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= int(args.phase3_patience):
            print(f"Phase3Nano early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), output_root / "models" / "phase3_nanospins_best.pth")
    pd.DataFrame(history).to_csv(output_root / "tables" / "phase3_nanospins_history.csv", index=False)

    train_metrics, _ = evaluate_split(model, train_graphs, device=device)
    val_metrics, _ = evaluate_split(model, val_graphs, device=device)
    test_metrics, _ = evaluate_split(model, test_graphs, device=device)
    pd.DataFrame(
        {
            "protein_id": order_ids.astype(str),
            "split": split_map,
            "finite_label_count_matched_cells": finite_counts.astype(int),
        }
    ).to_csv(output_root / "tables" / "phase3_nanospins_targets.csv", index=False)
    mapping_df.to_csv(output_root / "tables" / "phase3_nanospins_matched_cells.csv", index=False)

    summary = {
        "device": str(device),
        "seed": int(args.seed),
        "condition": str(args.condition),
        "hidden_cache_root": str(args.hidden_cache_root),
        "truth_csv": str(args.truth_csv),
        "mapping_xlsx": str(args.mapping_xlsx),
        "matched_cell_count": int(len(matched_cell_idx)),
        "matched_c10_count": int(len(c10_full_idx)),
        "matched_svec_count": int(len(svec_full_idx)),
        "eligible_protein_count": int(len(eligible_idx)),
        "train_protein_count": int(len(train_idx)),
        "val_protein_count": int(len(val_idx)),
        "test_protein_count": int(len(test_idx)),
        "train_graph_count": int(len(train_graphs)),
        "val_graph_count": int(len(val_graphs)),
        "test_graph_count": int(len(test_graphs)),
        "phase3_k_neighbors": int(args.phase3_k_neighbors),
        "phase3_n_pcs": int(args.phase3_n_pcs),
        "phase3_cell_graph_source": "normalized_scRNA_log1p_PCA_KNN",
        "phase3_best_epoch": int(best_epoch),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "saved_model_path": str(output_root / "models" / "phase3_nanospins_best.pth"),
        "history_csv": str(output_root / "tables" / "phase3_nanospins_history.csv"),
        "targets_csv": str(output_root / "tables" / "phase3_nanospins_targets.csv"),
    }
    (output_root / "reports" / "phase3_nanospins_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("Phase3Nano training finished.")
    print(f"Phase3Nano summary: {output_root / 'reports' / 'phase3_nanospins_summary.json'}")


if __name__ == "__main__":
    main()
