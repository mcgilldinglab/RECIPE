from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import sparse as sp
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import add_self_loops, from_scipy_sparse_matrix


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


class NeuralGraphLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc_x = nn.Sequential(
            nn.Linear(1, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.encoder = nn.Sequential(
            nn.Linear(9216, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.conv = SAGEConv(32, 32, aggr="sum")
        self.conv_activation = nn.Sequential(
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.regressor = nn.Linear(32, 1)

    def forward(self, data: Data):
        x = self.fc_x(data.x) + self.encoder(data.seq)
        x = self.fc(x)
        z = self.conv(x, data.edge_index)
        z = self.conv_activation(z)
        out = self.regressor(z)
        return out, z


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return float("nan")
    if np.allclose(y_true, y_true[0]):
        return float("nan")
    return float(r2_score(y_true, y_pred))


def choose_positive_safe_stat(values: np.ndarray, name: str) -> float:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError(f"No finite values found for {name}.")
    full_median = float(np.median(finite_values))
    if full_median > 0:
        return full_median
    positive_values = finite_values[finite_values > 0]
    if positive_values.size == 0:
        raise ValueError(f"{name} has no positive values, cannot apply log2(value / median + 1).")
    positive_median = float(np.median(positive_values))
    print(f"{name} full median <= 0; fallback to positive-only median {positive_median:.6f}")
    return positive_median


def align_to_order(df: pd.DataFrame, order_ids: pd.Index, label: str) -> pd.DataFrame:
    aligned = (
        df.drop_duplicates("protein_id")
        .assign(protein_id=df["protein_id"].astype(str))
        .set_index("protein_id")
        .reindex(order_ids)
        .reset_index()
    )
    if aligned["protein_id"].isna().any():
        raise ValueError(f"{label} failed to align to the requested order.")
    return aligned


def build_graph(x_values: np.ndarray, seq: np.ndarray, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> Data:
    data = Data(
        x=torch.tensor(x_values, dtype=torch.float32).reshape(-1, 1),
        edge_index=edge_index.clone(),
        edge_attr=edge_weight.clone(),
    )
    data.seq = torch.tensor(seq, dtype=torch.float32)
    data.edge_index, data.edge_attr = add_self_loops(data.edge_index, data.edge_attr)
    return data


def gather_split(pred: torch.Tensor, target: torch.Tensor, indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    if indices.size == 0:
        empty = torch.empty((0, 1), dtype=pred.dtype, device=pred.device)
        return empty, empty
    idx = torch.tensor(indices, dtype=torch.long, device=pred.device)
    return pred[idx], target[idx]


def concat_condition_tensors(items: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    preds = [p for p, y in items if p.numel() > 0 and y.numel() > 0]
    targets = [y for p, y in items if p.numel() > 0 and y.numel() > 0]
    if not preds:
        empty = torch.empty((0, 1), dtype=torch.float32)
        return empty, empty
    return torch.cat(preds, dim=0), torch.cat(targets, dim=0)


def make_eval_block(pred_np: np.ndarray, target_np: np.ndarray) -> dict:
    loss = float(np.mean((pred_np - target_np) ** 2)) if pred_np.size else float("nan")
    r2 = safe_r2(target_np.reshape(-1), pred_np.reshape(-1))
    return {"loss": loss, "r2": r2, "count": int(target_np.size)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--ppi-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--lr", type=float, default=7e-2)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--condition", choices=["both", "C10", "SVEC"], default="both")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    bundle_dir = Path(args.bundle_dir)
    pseudobulk_raw = pd.read_csv(bundle_dir / "pseudobulk_raw_counts_by_ENSMUSP.bulk_intersection.csv")
    bulk_protein = pd.read_csv(bundle_dir / "bulkProteomics_by_ENSMUSP.bulk_intersection.csv")
    order_df = pd.read_csv(bundle_dir / "bulk_pseudobulk_intersection_order_by_ENSMUSP.csv")
    seq = np.load(bundle_dir / "all_sequence_outputs_by_bulk_pseudobulk_intersection_order_ENSMUSP.npy")

    order_ids = pd.Index(order_df["protein_id"].astype(str), name="protein_id")
    pseudobulk_raw = align_to_order(pseudobulk_raw, order_ids, label="pseudobulk_raw")
    bulk_protein = align_to_order(bulk_protein, order_ids, label="bulk_protein")

    if seq.shape[0] != len(order_df):
        raise ValueError(f"Embedding rows {seq.shape[0]} do not match order rows {len(order_df)}.")

    c10_cols = [c for c in bulk_protein.columns if c.startswith("iBAQ.c10_")]
    svec_cols = [c for c in bulk_protein.columns if c.startswith("iBAQ.svec_")]
    if not c10_cols or not svec_cols:
        raise ValueError("Missing C10/SVEC bulk proteomics replicate columns.")

    bulk_c10_mean = bulk_protein[c10_cols].mean(axis=1, skipna=True).to_numpy(dtype=np.float32)
    bulk_svec_mean = bulk_protein[svec_cols].mean(axis=1, skipna=True).to_numpy(dtype=np.float32)
    rna_c10_raw = pseudobulk_raw["C10"].to_numpy(dtype=np.float32)
    rna_svec_raw = pseudobulk_raw["SVEC"].to_numpy(dtype=np.float32)

    c10_scale = choose_positive_safe_stat(rna_c10_raw, "pseudobulk_raw_C10")
    svec_scale = choose_positive_safe_stat(rna_svec_raw, "pseudobulk_raw_SVEC")
    rna_c10_norm = np.log2((np.nan_to_num(rna_c10_raw, nan=0.0) / c10_scale) + 1.0).astype(np.float32)
    rna_svec_norm = np.log2((np.nan_to_num(rna_svec_raw, nan=0.0) / svec_scale) + 1.0).astype(np.float32)

    ppi_path = Path(args.ppi_path)
    if ppi_path.suffix == ".npz":
        ppi_matrix = sp.load_npz(ppi_path).tocoo()
    else:
        ppi_matrix = sp.coo_matrix(pd.read_csv(ppi_path))
    if ppi_matrix.shape[0] != len(order_df):
        raise ValueError(f"PPI shape {ppi_matrix.shape} does not match order rows {len(order_df)}.")
    edge_index, edge_weight = from_scipy_sparse_matrix(ppi_matrix.astype("float32"))

    data_c10 = build_graph(rna_c10_norm, seq, edge_index, edge_weight)
    data_svec = build_graph(rna_svec_norm, seq, edge_index, edge_weight)
    data_c10.y = torch.tensor(np.nan_to_num(bulk_c10_mean, nan=0.0), dtype=torch.float32).view(-1, 1)
    data_svec.y = torch.tensor(np.nan_to_num(bulk_svec_mean, nan=0.0), dtype=torch.float32).view(-1, 1)

    mask_c10 = np.isfinite(bulk_c10_mean)
    mask_svec = np.isfinite(bulk_svec_mean)
    labeled_union = np.where(mask_c10 | mask_svec)[0].astype(np.int64)
    if args.condition == "C10":
        labeled_union = np.where(mask_c10)[0].astype(np.int64)
    elif args.condition == "SVEC":
        labeled_union = np.where(mask_svec)[0].astype(np.int64)
    train_idx, temp_idx = train_test_split(labeled_union, test_size=0.25, random_state=args.seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=1.0 / 3.0, random_state=args.seed)

    def subset(mask: np.ndarray, idx: np.ndarray) -> np.ndarray:
        return np.asarray([i for i in idx if mask[i]], dtype=np.int64)

    split_map = {
        "train_c10": subset(mask_c10, train_idx) if args.condition in {"both", "C10"} else np.asarray([], dtype=np.int64),
        "val_c10": subset(mask_c10, val_idx) if args.condition in {"both", "C10"} else np.asarray([], dtype=np.int64),
        "test_c10": subset(mask_c10, test_idx) if args.condition in {"both", "C10"} else np.asarray([], dtype=np.int64),
        "train_svec": subset(mask_svec, train_idx) if args.condition in {"both", "SVEC"} else np.asarray([], dtype=np.int64),
        "val_svec": subset(mask_svec, val_idx) if args.condition in {"both", "SVEC"} else np.asarray([], dtype=np.int64),
        "test_svec": subset(mask_svec, test_idx) if args.condition in {"both", "SVEC"} else np.asarray([], dtype=np.int64),
    }

    device = resolve_device(args.device)
    model = NeuralGraphLinear().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    data_c10 = data_c10.to(device)
    data_svec = data_svec.to(device)

    best_val_r2 = float("-inf")
    best_val_loss = float("inf")
    best_test_r2 = float("-inf")
    best_test_loss = float("inf")
    best_epoch = -1
    patience_counter = 0
    history: list[dict] = []
    best_predictions = {}
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred_c10, _ = model(data_c10)
        pred_svec, _ = model(data_svec)

        train_pred, train_true = concat_condition_tensors(
            [
                gather_split(pred_c10, data_c10.y, split_map["train_c10"]),
                gather_split(pred_svec, data_svec.y, split_map["train_svec"]),
            ]
        )
        train_loss = criterion(train_pred, train_true)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_c10, z_c10 = model(data_c10)
            pred_svec, z_svec = model(data_svec)

        pred_c10_np = pred_c10.detach().cpu().numpy().reshape(-1)
        pred_svec_np = pred_svec.detach().cpu().numpy().reshape(-1)
        true_c10_np = data_c10.y.detach().cpu().numpy().reshape(-1)
        true_svec_np = data_svec.y.detach().cpu().numpy().reshape(-1)

        train_block = make_eval_block(
            np.concatenate([pred_c10_np[split_map["train_c10"]], pred_svec_np[split_map["train_svec"]]]),
            np.concatenate([true_c10_np[split_map["train_c10"]], true_svec_np[split_map["train_svec"]]]),
        )
        val_block = make_eval_block(
            np.concatenate([pred_c10_np[split_map["val_c10"]], pred_svec_np[split_map["val_svec"]]]),
            np.concatenate([true_c10_np[split_map["val_c10"]], true_svec_np[split_map["val_svec"]]]),
        )
        test_block = make_eval_block(
            np.concatenate([pred_c10_np[split_map["test_c10"]], pred_svec_np[split_map["test_svec"]]]),
            np.concatenate([true_c10_np[split_map["test_c10"]], true_svec_np[split_map["test_svec"]]]),
        )
        val_c10_block = make_eval_block(pred_c10_np[split_map["val_c10"]], true_c10_np[split_map["val_c10"]])
        val_svec_block = make_eval_block(pred_svec_np[split_map["val_svec"]], true_svec_np[split_map["val_svec"]])
        test_c10_block = make_eval_block(pred_c10_np[split_map["test_c10"]], true_c10_np[split_map["test_c10"]])
        test_svec_block = make_eval_block(pred_svec_np[split_map["test_svec"]], true_svec_np[split_map["test_svec"]])

        improved = val_block["r2"] > best_val_r2
        if improved:
            best_val_r2 = float(val_block["r2"])
            best_val_loss = float(val_block["loss"])
            best_test_r2 = float(test_block["r2"])
            best_test_loss = float(test_block["loss"])
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(model.state_dict(), output_dir / "best_model.pth")
            best_predictions = {
                "pred_c10": pred_c10_np.astype(np.float32),
                "pred_svec": pred_svec_np.astype(np.float32),
                "z_c10": z_c10.detach().cpu().numpy().astype(np.float32),
                "z_svec": z_svec.detach().cpu().numpy().astype(np.float32),
            }
            patience_counter = 0
        else:
            patience_counter += 1

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_block["loss"],
                "train_r2": train_block["r2"],
                "val_loss": val_block["loss"],
                "val_r2": val_block["r2"],
                "test_loss": test_block["loss"],
                "test_r2": test_block["r2"],
                "val_c10_r2": val_c10_block["r2"],
                "val_svec_r2": val_svec_block["r2"],
                "test_c10_r2": test_c10_block["r2"],
                "test_svec_r2": test_svec_block["r2"],
                "best_val_r2": best_val_r2,
                "best_test_r2": best_test_r2,
            }
        )
        print(
            f"Epoch: {epoch:03d}, Train Loss: {train_block['loss']:.3f}, Train R²: {train_block['r2']:.3f}| "
            f"Val Loss: {val_block['loss']:.3f}, Val R²: {val_block['r2']:.3f}|"
            f"Test Loss: {test_block['loss']:.3f}, Test R²: {test_block['r2']:.3f}"
        )

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a best checkpoint.")

    split_df = order_df.copy()
    split_df["split"] = "unused"
    split_df.loc[train_idx, "split"] = "train"
    split_df.loc[val_idx, "split"] = "val"
    split_df.loc[test_idx, "split"] = "test"
    split_df["has_bulk_c10"] = mask_c10.astype(np.int8)
    split_df["has_bulk_svec"] = mask_svec.astype(np.int8)
    split_df.to_csv(output_dir / "protein_split.csv", index=False)

    pred_df = order_df.copy()
    pred_df["pseudobulk_raw_C10"] = rna_c10_raw.astype(np.float32)
    pred_df["pseudobulk_raw_SVEC"] = rna_svec_raw.astype(np.float32)
    pred_df["pseudobulk_log2median_C10"] = rna_c10_norm.astype(np.float32)
    pred_df["pseudobulk_log2median_SVEC"] = rna_svec_norm.astype(np.float32)
    pred_df["bulk_target_mean_C10"] = bulk_c10_mean.astype(np.float32)
    pred_df["bulk_target_mean_SVEC"] = bulk_svec_mean.astype(np.float32)
    pred_df["pred_C10"] = best_predictions["pred_c10"]
    pred_df["pred_SVEC"] = best_predictions["pred_svec"]
    pred_df.to_csv(output_dir / "phase0_all_predictions.csv", index=False)

    np.save(output_dir / "phase0_z_C10.npy", best_predictions["z_c10"])
    np.save(output_dir / "phase0_z_SVEC.npy", best_predictions["z_svec"])

    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "history.csv", index=False)

    summary = {
        "seed": args.seed,
        "lr": args.lr,
        "epochs": args.epochs,
        "patience": args.patience,
        "device": str(device),
        "bundle_dir": str(bundle_dir),
        "ppi_path": str(ppi_path),
        "condition": args.condition,
        "target_definition": {
            "C10": c10_cols,
            "SVEC": svec_cols,
            "aggregation": "rowwise mean across finite bulk replicates",
        },
        "input_definition": {
            "C10": "pseudobulk_raw_counts_by_ENSMUSP.bulk_intersection.csv::C10",
            "SVEC": "pseudobulk_raw_counts_by_ENSMUSP.bulk_intersection.csv::SVEC",
            "normalization": "log2((raw_count / median_ref) + 1)",
        },
        "best_epoch": int(best_epoch),
        "best_val_r2": float(best_val_r2),
        "best_val_loss": float(best_val_loss),
        "best_test_r2": float(best_test_r2),
        "best_test_loss": float(best_test_loss),
        "total_nodes": int(len(order_df)),
        "train_union_count": int(len(train_idx)),
        "val_union_count": int(len(val_idx)),
        "test_union_count": int(len(test_idx)),
        "labeled_union_count": int(len(labeled_union)),
        "c10_scale_median": float(c10_scale),
        "svec_scale_median": float(svec_scale),
        "c10_labeled_count": int(mask_c10.sum()),
        "svec_labeled_count": int(mask_svec.sum()),
        "train_c10_count": int(len(split_map["train_c10"])),
        "val_c10_count": int(len(split_map["val_c10"])),
        "test_c10_count": int(len(split_map["test_c10"])),
        "train_svec_count": int(len(split_map["train_svec"])),
        "val_svec_count": int(len(split_map["val_svec"])),
        "test_svec_count": int(len(split_map["test_svec"])),
        "ppi_shape": list(ppi_matrix.shape),
        "ppi_nnz": int(ppi_matrix.nnz),
        "sequence_shape": list(seq.shape),
        "outputs": {
            "model": str(output_dir / "best_model.pth"),
            "history": str(output_dir / "history.csv"),
            "predictions": str(output_dir / "phase0_all_predictions.csv"),
            "split": str(output_dir / "protein_split.csv"),
            "z_c10": str(output_dir / "phase0_z_C10.npy"),
            "z_svec": str(output_dir / "phase0_z_SVEC.npy"),
        },
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
