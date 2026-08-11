#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import sparse as sp
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import add_self_loops, from_scipy_sparse_matrix


BASE_DIR = Path(__file__).resolve().parent
PHASE0_SCRIPT = BASE_DIR / "train_phase0_ensmusp_pseudobulk_raw_bulkprot.py"


def set_seed(seed: int = 0) -> None:
    print(f"seed = {seed}", flush=True)
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


class Phase2CellGraph(torch.nn.Module):
    def __init__(self, input_dim: int = 32, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim, aggr="mean")
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, aggr="mean")
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.head = torch.nn.Linear(hidden_dim, 1)

    def encode_hidden(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return x

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode_hidden(x, edge_index)).view(-1)


def load_phase0_module():
    spec = importlib.util.spec_from_file_location("phase0_ensmusp_module", PHASE0_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {PHASE0_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run ENSMUSP phase1 export with fixed phase0 bulk backbone, then phase2 cell-split "
            "bulk-to-cell duplicated supervision, and export split-safe phase2 hidden cache."
        )
    )
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--phase0-summary", type=Path, required=True)
    parser.add_argument("--phase0-model", type=Path, required=True)
    parser.add_argument("--ppi-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--condition", choices=["both", "C10", "SVEC"], default="both")
    parser.add_argument("--sc-normalize-target-sum", type=float, default=1e4)
    parser.add_argument("--phase1-progress-every", type=int, default=25)
    parser.add_argument("--phase2-k-neighbors", type=int, default=7)
    parser.add_argument("--phase2-n-pcs", type=int, default=20)
    parser.add_argument("--phase2-hidden-dim", type=int, default=64)
    parser.add_argument("--phase2-dropout", type=float, default=0.1)
    parser.add_argument("--phase2-batch-size", type=int, default=16)
    parser.add_argument("--phase2-epochs", type=int, default=1000)
    parser.add_argument("--phase2-patience", type=int, default=100)
    parser.add_argument("--phase2-lr", type=float, default=1e-3)
    parser.add_argument("--phase2-weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--bulk-target-transform", choices=["none", "log2_median_nonzero"], default="none")
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


def build_pca_knn_edge_index(
    cell_by_gene: np.ndarray,
    n_neighbors: int,
    n_pcs: int,
    seed: int,
    target_sum: float,
) -> torch.Tensor:
    if cell_by_gene.shape[0] < 2:
        raise ValueError("Need at least 2 cells to build KNN graph.")
    values = np.nan_to_num(np.asarray(cell_by_gene, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    values = np.log1p(normalize_total_rows(values, target_sum=target_sum))
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


def transform_bulk_targets(values: np.ndarray, mode: str) -> tuple[np.ndarray, float | None]:
    arr = np.asarray(values, dtype=np.float32)
    if mode == "none":
        return arr.copy(), None
    if mode != "log2_median_nonzero":
        raise ValueError(f"Unsupported bulk target transform: {mode}")
    positive = np.isfinite(arr) & (arr > 0)
    if not np.any(positive):
        return np.full_like(arr, np.nan, dtype=np.float32), None
    median = float(np.median(arr[positive]))
    out = np.full_like(arr, np.nan, dtype=np.float32)
    finite = np.isfinite(arr)
    out[finite] = np.log2((np.maximum(arr[finite], 0.0) / max(median, 1e-6)) + 1.0).astype(np.float32)
    return out, median


def split_cell_positions(
    cell_positions: np.ndarray,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cell_positions.size < 6:
        raise ValueError(f"Need at least 6 cells for cell split, got {cell_positions.size}.")
    test_frac = 1.0 - train_frac - val_frac
    if test_frac <= 0:
        raise ValueError("train_frac + val_frac must be < 1.")
    train_pos, temp_pos = train_test_split(
        cell_positions,
        test_size=(1.0 - train_frac),
        random_state=seed,
    )
    val_relative = val_frac / (val_frac + test_frac)
    val_pos, test_pos = train_test_split(
        temp_pos,
        train_size=val_relative,
        random_state=seed,
    )
    return (
        np.sort(np.asarray(train_pos, dtype=np.int64)),
        np.sort(np.asarray(val_pos, dtype=np.int64)),
        np.sort(np.asarray(test_pos, dtype=np.int64)),
    )


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


def evaluate_split(model: Phase2CellGraph, graphs: list[Data], device: torch.device) -> tuple[dict[str, float], pd.DataFrame]:
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
            split_idx = batch.cell_split_idx.detach().cpu().numpy().astype(int, copy=False)
            for i in range(len(ptr) - 1):
                start = ptr[i]
                end = ptr[i + 1]
                local_mask = mask_np[start:end]
                local_pred = pred_np[start:end][local_mask]
                local_truth = y_np[start:end][local_mask]
                rows.append(
                    {
                        "target_idx": int(target_idx[i]),
                        "condition_idx": int(condition_idx[i]),
                        "cell_split_idx": int(split_idx[i]),
                        "node_count": int(local_mask.sum()),
                        "pred_mean": float(np.mean(local_pred)) if local_pred.size else float("nan"),
                        "truth_mean": float(np.mean(local_truth)) if local_truth.size else float("nan"),
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


def export_phase1_all_gene_z(
    phase0_model,
    edge_index: torch.Tensor,
    seq_tensor: torch.Tensor,
    expr_log2_cells: np.ndarray,
    device: torch.device,
    progress_every: int,
) -> np.ndarray:
    z_all = np.zeros((expr_log2_cells.shape[0], expr_log2_cells.shape[1], 32), dtype=np.float32)
    phase0_model.eval()
    with torch.no_grad():
        for cell_idx in range(expr_log2_cells.shape[0]):
            data = Data(
                x=torch.from_numpy(expr_log2_cells[cell_idx]).float().view(-1, 1),
                edge_index=edge_index,
            )
            data.seq = seq_tensor
            data = data.to(device)
            _, z = phase0_model(data)
            z_all[cell_idx] = z.detach().cpu().numpy().astype(np.float32, copy=False)
            if (cell_idx + 1) % progress_every == 0 or (cell_idx + 1) == expr_log2_cells.shape[0]:
                print(f"Phase1 export z: {cell_idx + 1}/{expr_log2_cells.shape[0]} cells", flush=True)
    return z_all


def main() -> None:
    args = parse_args()
    phase12_root = Path(args.output_root)
    phase12_root.mkdir(parents=True, exist_ok=True)
    phase1_dir = phase12_root / "phase1"
    phase2_dir = phase12_root / "phase2"
    hidden_root = phase12_root.parent / "phase2_hidden_cache"
    for subdir in (phase1_dir, phase2_dir, hidden_root):
        subdir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = resolve_device(args.device)

    phase0_module = load_phase0_module()
    phase0_summary = json.loads(args.phase0_summary.read_text(encoding="utf-8"))
    c10_scale = float(phase0_summary["c10_scale_median"])
    svec_scale = float(phase0_summary["svec_scale_median"])

    bundle_dir = Path(args.bundle_dir)
    order_df = pd.read_csv(bundle_dir / "bulk_pseudobulk_intersection_order_by_ENSMUSP.csv")
    order_ids = pd.Index(order_df["protein_id"].astype(str), name="protein_id")
    seq = np.load(bundle_dir / "all_sequence_outputs_by_bulk_pseudobulk_intersection_order_ENSMUSP.npy").astype(np.float32)
    if seq.shape[0] != len(order_ids):
        raise ValueError("Sequence embedding rows do not match ordered protein rows.")

    sc_rna_all = read_ordered_frame(
        bundle_dir / "scRNA_qc_cells_by_ENSMUSP_all.bulk_intersection.zero_filled.csv",
        order_ids,
    )
    meta_all = pd.read_csv(bundle_dir / "scRNA_cell_metadata_all.for_training.csv")
    meta_all["cell_names"] = meta_all["cell_names"].astype(str)
    cell_names = [col for col in sc_rna_all.columns if col != "protein_id"]
    meta_all = meta_all.drop_duplicates("cell_names").set_index("cell_names").reindex(cell_names).reset_index()
    if meta_all["predicted_cell_type"].isna().any():
        raise ValueError("Some scRNA cells are missing metadata after alignment.")

    expr_gene_by_cell = sc_rna_all.drop(columns=["protein_id"]).to_numpy(dtype=np.float32)
    expr_cell_by_gene_raw = expr_gene_by_cell.T.copy()
    cell_type = meta_all["predicted_cell_type"].astype(str).to_numpy()
    cell_scale = np.where(cell_type == "SVEC", svec_scale, c10_scale).astype(np.float32)
    expr_cell_by_gene_phase1 = np.log2(
        (expr_cell_by_gene_raw / np.maximum(cell_scale[:, None], 1e-6)) + 1.0
    ).astype(np.float32)

    ppi_path = Path(args.ppi_path)
    if ppi_path.suffix.lower() == ".npz":
        ppi_matrix = sp.load_npz(ppi_path).tocoo().astype(np.float32)
    else:
        ppi_matrix = sp.coo_matrix(pd.read_csv(ppi_path)).astype(np.float32)
    if ppi_matrix.shape != (len(order_ids), len(order_ids)):
        raise ValueError(f"PPI shape {ppi_matrix.shape} does not match ordered protein rows {len(order_ids)}.")
    edge_index, edge_weight = from_scipy_sparse_matrix(ppi_matrix)
    edge_index, edge_weight = add_self_loops(edge_index, edge_weight)
    edge_index = edge_index.to(device)

    phase0_model = phase0_module.NeuralGraphLinear().to(device)
    phase0_model.load_state_dict(torch.load(args.phase0_model, map_location=device))
    seq_tensor = torch.from_numpy(seq).float().to(device)

    print(f"Selected phase0 model: {args.phase0_model}", flush=True)
    print(f"Cells: {len(cell_names)} | Genes: {len(order_ids)} | Device: {device}", flush=True)
    print(f"Phase1 scales: C10={c10_scale:.4f}, SVEC={svec_scale:.4f}", flush=True)

    z_all = export_phase1_all_gene_z(
        phase0_model=phase0_model,
        edge_index=edge_index,
        seq_tensor=seq_tensor,
        expr_log2_cells=expr_cell_by_gene_phase1,
        device=device,
        progress_every=args.phase1_progress_every,
    )
    np.save(phase1_dir / "arrays_phase1_all_gene_z.npy", z_all)
    pd.DataFrame({"cell_name": cell_names, "cell_type": cell_type}).to_csv(phase1_dir / "phase1_cell_names.csv", index=False)
    pd.DataFrame({"protein_id": order_ids.astype(str)}).to_csv(phase1_dir / "phase1_all_gene_names.csv", index=False)
    phase1_summary = {
        "selected_phase0_summary": str(args.phase0_summary),
        "selected_phase0_model": str(args.phase0_model),
        "cell_count": int(len(cell_names)),
        "gene_count": int(len(order_ids)),
        "z_shape": list(z_all.shape),
        "c10_scale": float(c10_scale),
        "svec_scale": float(svec_scale),
        "phase1_all_gene_z_npy": str(phase1_dir / "arrays_phase1_all_gene_z.npy"),
        "phase1_cell_names_csv": str(phase1_dir / "phase1_cell_names.csv"),
        "phase1_gene_names_csv": str(phase1_dir / "phase1_all_gene_names.csv"),
    }
    (phase1_dir / "phase1_export_summary.json").write_text(json.dumps(phase1_summary, indent=2), encoding="utf-8")

    c10_cell_idx = np.where(cell_type == "C10")[0].astype(np.int64)
    svec_cell_idx = np.where(cell_type == "SVEC")[0].astype(np.int64)

    split_positions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    if args.condition in {"both", "C10"}:
        split_positions["C10"] = split_cell_positions(c10_cell_idx, float(args.train_frac), float(args.val_frac), int(args.seed))
    if args.condition in {"both", "SVEC"}:
        split_positions["SVEC"] = split_cell_positions(svec_cell_idx, float(args.train_frac), float(args.val_frac), int(args.seed))

    edge_index_by_condition_split: dict[tuple[str, str], torch.Tensor] = {}
    full_idx_by_condition_split: dict[tuple[str, str], np.ndarray] = {}
    split_name_to_idx = {"train": 0, "val": 1, "test": 2}

    for condition_name, (train_pos, val_pos, test_pos) in split_positions.items():
        for split_name, pos in (("train", train_pos), ("val", val_pos), ("test", test_pos)):
            full_idx = np.asarray(pos, dtype=np.int64)
            full_idx_by_condition_split[(condition_name, split_name)] = full_idx
            edge_index_by_condition_split[(condition_name, split_name)] = build_pca_knn_edge_index(
                expr_cell_by_gene_raw[full_idx],
                n_neighbors=int(args.phase2_k_neighbors),
                n_pcs=int(args.phase2_n_pcs),
                seed=int(args.seed),
                target_sum=float(args.sc_normalize_target_sum),
            )

    bulk_ref = pd.read_csv(bundle_dir / "bulk_training_reference_by_ENSMUSP.csv")
    bulk_ref["protein_id"] = bulk_ref["protein_id"].astype(str)
    bulk_ref = bulk_ref.drop_duplicates("protein_id").set_index("protein_id").reindex(order_ids).reset_index()
    bulk_c10_raw = pd.to_numeric(bulk_ref["bulk_protein_mean_C10"], errors="coerce").to_numpy(dtype=np.float32)
    bulk_svec_raw = pd.to_numeric(bulk_ref["bulk_protein_mean_SVEC"], errors="coerce").to_numpy(dtype=np.float32)
    bulk_c10, bulk_c10_median = transform_bulk_targets(bulk_c10_raw, args.bulk_target_transform)
    bulk_svec, bulk_svec_median = transform_bulk_targets(bulk_svec_raw, args.bulk_target_transform)

    if args.condition == "C10":
        supervised_union = np.where(np.isfinite(bulk_c10))[0].astype(np.int64)
    elif args.condition == "SVEC":
        supervised_union = np.where(np.isfinite(bulk_svec))[0].astype(np.int64)
    else:
        supervised_union = np.where(np.isfinite(bulk_c10) | np.isfinite(bulk_svec))[0].astype(np.int64)
    if supervised_union.size < 20:
        raise ValueError(f"Too few supervised proteins: {supervised_union.size}")

    def build_graphs(split_name: str) -> list[Data]:
        graphs: list[Data] = []
        for gene_idx in supervised_union.tolist():
            if args.condition in {"both", "C10"} and np.isfinite(bulk_c10[gene_idx]):
                full_idx = full_idx_by_condition_split[("C10", split_name)]
                y = np.full(full_idx.shape[0], bulk_c10[gene_idx], dtype=np.float32)
                graph = Data(
                    x=torch.from_numpy(np.asarray(z_all[full_idx, gene_idx, :], dtype=np.float32)),
                    edge_index=edge_index_by_condition_split[("C10", split_name)].clone(),
                    y=torch.from_numpy(y).float(),
                    mask=torch.ones(full_idx.shape[0], dtype=torch.float32),
                )
                graph.target_idx = torch.tensor([gene_idx], dtype=torch.long)
                graph.condition_idx = torch.tensor([0], dtype=torch.long)
                graph.cell_split_idx = torch.tensor([split_name_to_idx[split_name]], dtype=torch.long)
                graphs.append(graph)
            if args.condition in {"both", "SVEC"} and np.isfinite(bulk_svec[gene_idx]):
                full_idx = full_idx_by_condition_split[("SVEC", split_name)]
                y = np.full(full_idx.shape[0], bulk_svec[gene_idx], dtype=np.float32)
                graph = Data(
                    x=torch.from_numpy(np.asarray(z_all[full_idx, gene_idx, :], dtype=np.float32)),
                    edge_index=edge_index_by_condition_split[("SVEC", split_name)].clone(),
                    y=torch.from_numpy(y).float(),
                    mask=torch.ones(full_idx.shape[0], dtype=torch.float32),
                )
                graph.target_idx = torch.tensor([gene_idx], dtype=torch.long)
                graph.condition_idx = torch.tensor([1], dtype=torch.long)
                graph.cell_split_idx = torch.tensor([split_name_to_idx[split_name]], dtype=torch.long)
                graphs.append(graph)
        return graphs

    train_graphs = build_graphs("train")
    val_graphs = build_graphs("val")
    test_graphs = build_graphs("test")
    if not train_graphs:
        raise ValueError("Phase2 requires at least one training graph.")

    train_loader = DataLoader(train_graphs, batch_size=int(args.phase2_batch_size), shuffle=True)
    model = Phase2CellGraph(
        input_dim=int(z_all.shape[2]),
        hidden_dim=int(args.phase2_hidden_dim),
        dropout=float(args.phase2_dropout),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.phase2_lr), weight_decay=float(args.phase2_weight_decay))

    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_val_r2 = float("-inf")
    patience_counter = 0
    history: list[dict] = []

    for epoch in range(1, int(args.phase2_epochs) + 1):
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
            f"Phase2CellSplit Epoch {epoch:03d} | "
            f"Train Loss {train_metrics['loss']:.4f} R2 {train_metrics['r2']:.4f} | "
            f"Val Loss {val_metrics['loss']:.4f} R2 {val_metrics['r2']:.4f} | "
            f"Test Loss {test_metrics['loss']:.4f} R2 {test_metrics['r2']:.4f}",
            flush=True,
        )
        if val_metrics["r2"] > best_val_r2:
            best_val_r2 = float(val_metrics["r2"])
            best_epoch = int(epoch)
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= int(args.phase2_patience):
            print(f"Phase2CellSplit early stopping at epoch {epoch}", flush=True)
            break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), phase2_dir / "phase2_cell_graph_best.pth")
    pd.DataFrame(history).to_csv(phase2_dir / "phase2_history.csv", index=False)

    train_metrics, train_pred_df = evaluate_split(model, train_graphs, device=device)
    val_metrics, val_pred_df = evaluate_split(model, val_graphs, device=device)
    test_metrics, test_pred_df = evaluate_split(model, test_graphs, device=device)
    pred_df = pd.concat(
        [
            train_pred_df.assign(split="train"),
            val_pred_df.assign(split="val"),
            test_pred_df.assign(split="test"),
        ],
        ignore_index=True,
    )
    pred_df["protein_id"] = pred_df["target_idx"].map(lambda idx: str(order_ids[int(idx)]))
    pred_df["condition"] = pred_df["condition_idx"].map({0: "C10", 1: "SVEC"})
    pred_df.to_csv(phase2_dir / "phase2_bulk_predictions_cellsplit.csv", index=False)

    target_df = pd.DataFrame(
        {
            "protein_id": order_ids.astype(str),
            "bulk_mean_C10_raw": bulk_c10_raw,
            "bulk_mean_SVEC_raw": bulk_svec_raw,
            "bulk_mean_C10": bulk_c10,
            "bulk_mean_SVEC": bulk_svec,
            "is_supervised": np.isin(np.arange(len(order_ids)), supervised_union).astype(np.int8),
        }
    )
    target_df.to_csv(phase2_dir / "phase2_supervision_targets.csv", index=False)

    split_cell_rows: list[dict[str, object]] = []
    for condition_name, (train_pos, val_pos, test_pos) in split_positions.items():
        for split_name, pos in (("train", train_pos), ("val", val_pos), ("test", test_pos)):
            for idx in pos.tolist():
                split_cell_rows.append(
                    {
                        "cell_name": str(cell_names[int(idx)]),
                        "cell_type": str(cell_type[int(idx)]),
                        "condition": condition_name,
                        "cell_split": split_name,
                        "cell_index": int(idx),
                    }
                )
    pd.DataFrame(split_cell_rows).to_csv(phase2_dir / "phase2_cell_split.csv", index=False)

    hidden_all = np.zeros((len(cell_names), len(order_ids), int(args.phase2_hidden_dim)), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for gene_idx in range(len(order_ids)):
            if args.condition in {"both", "C10"}:
                for split_name in ("train", "val", "test"):
                    full_idx = full_idx_by_condition_split[("C10", split_name)]
                    x = torch.from_numpy(np.asarray(z_all[full_idx, gene_idx, :], dtype=np.float32)).to(device)
                    h = model.encode_hidden(x, edge_index_by_condition_split[("C10", split_name)].to(device))
                    hidden_all[full_idx, gene_idx, :] = h.detach().cpu().numpy().astype(np.float32, copy=False)
            if args.condition in {"both", "SVEC"}:
                for split_name in ("train", "val", "test"):
                    full_idx = full_idx_by_condition_split[("SVEC", split_name)]
                    x = torch.from_numpy(np.asarray(z_all[full_idx, gene_idx, :], dtype=np.float32)).to(device)
                    h = model.encode_hidden(x, edge_index_by_condition_split[("SVEC", split_name)].to(device))
                    hidden_all[full_idx, gene_idx, :] = h.detach().cpu().numpy().astype(np.float32, copy=False)
            if (gene_idx + 1) % 500 == 0 or (gene_idx + 1) == len(order_ids):
                print(f"Export phase2 hidden z: {gene_idx + 1}/{len(order_ids)} genes", flush=True)

    np.save(hidden_root / "phase2_hidden_all.npy", hidden_all)
    pd.DataFrame({"cell_name": cell_names, "cell_type": cell_type}).to_csv(hidden_root / "phase2_hidden_cell_names.csv", index=False)
    pd.DataFrame({"protein_id": order_ids.astype(str)}).to_csv(hidden_root / "phase2_hidden_gene_names.csv", index=False)
    hidden_summary = {
        "phase2_model_path": str(phase2_dir / "phase2_cell_graph_best.pth"),
        "phase1_z_path": str(phase1_dir / "arrays_phase1_all_gene_z.npy"),
        "shape": [int(hidden_all.shape[0]), int(hidden_all.shape[1]), int(hidden_all.shape[2])],
        "split_safe_export": True,
        "split_axis": "cells",
        "hidden_npy": str(hidden_root / "phase2_hidden_all.npy"),
        "cell_csv": str(hidden_root / "phase2_hidden_cell_names.csv"),
        "gene_csv": str(hidden_root / "phase2_hidden_gene_names.csv"),
    }
    (hidden_root / "phase2_hidden_export_summary.json").write_text(
        json.dumps(hidden_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    phase2_summary = {
        "selected_phase0_model": str(args.phase0_model),
        "selected_phase0_summary": str(args.phase0_summary),
        "condition": str(args.condition),
        "device": str(device),
        "seed": int(args.seed),
        "split_axis": "cells",
        "supervision_source": "bulk protein duplicated to each cell in split graph",
        "cell_count_total": int(len(cell_names)),
        "cell_count_c10": int(len(c10_cell_idx)),
        "cell_count_svec": int(len(svec_cell_idx)),
        "gene_count_total": int(len(order_ids)),
        "supervised_gene_union_count": int(len(supervised_union)),
        "train_cell_count": int(sum(len(v[0]) for v in split_positions.values())),
        "val_cell_count": int(sum(len(v[1]) for v in split_positions.values())),
        "test_cell_count": int(sum(len(v[2]) for v in split_positions.values())),
        "train_graph_count": int(len(train_graphs)),
        "val_graph_count": int(len(val_graphs)),
        "test_graph_count": int(len(test_graphs)),
        "phase2_k_neighbors": int(args.phase2_k_neighbors),
        "phase2_n_pcs": int(args.phase2_n_pcs),
        "phase2_cell_graph_source": "normalized_scRNA_log1p_PCA_KNN",
        "bulk_target_transform": str(args.bulk_target_transform),
        "bulk_target_median_c10": None if bulk_c10_median is None else float(bulk_c10_median),
        "bulk_target_median_svec": None if bulk_svec_median is None else float(bulk_svec_median),
        "phase2_best_epoch": int(best_epoch),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "phase1_all_gene_z_npy": str(phase1_dir / "arrays_phase1_all_gene_z.npy"),
        "phase2_model_path": str(phase2_dir / "phase2_cell_graph_best.pth"),
        "phase2_history_csv": str(phase2_dir / "phase2_history.csv"),
        "phase2_predictions_csv": str(phase2_dir / "phase2_bulk_predictions_cellsplit.csv"),
        "phase2_supervision_targets_csv": str(phase2_dir / "phase2_supervision_targets.csv"),
        "phase2_cell_split_csv": str(phase2_dir / "phase2_cell_split.csv"),
        "phase2_hidden_cache_root": str(hidden_root),
    }
    (phase2_dir / "phase2_summary.json").write_text(json.dumps(phase2_summary, indent=2), encoding="utf-8")

    print(f"Phase2CellSplit summary: {phase2_dir / 'phase2_summary.json'}", flush=True)
    print(f"Phase2 hidden cache: {hidden_root / 'phase2_hidden_all.npy'}", flush=True)


if __name__ == "__main__":
    main()
