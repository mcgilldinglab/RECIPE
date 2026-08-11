#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from benchmark_utils import (  # noqa: E402
    build_aligned_targets,
    compute_metrics,
    read_expression_matrix,
    read_protein_map,
    split_cells_and_genes,
    write_json,
)


class VanillaNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the notebook-style single-cell VanillaNN benchmark.")
    parser.add_argument("--expression-csv", type=Path, required=True)
    parser.add_argument("--protein-map-csv", type=Path, required=True)
    parser.add_argument("--bulk-table-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--protein-column", type=str, default="NC3")
    parser.add_argument("--cell-test-size", type=float, default=0.1)
    parser.add_argument("--gene-temp-size", type=float, default=0.25)
    parser.add_argument("--gene-test-size-from-temp", type=float, default=0.5)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    gene_ids, _, matrix = read_expression_matrix(args.expression_csv)
    protein_map_df = read_protein_map(args.protein_map_csv)
    y, _ = build_aligned_targets(
        gene_ids=gene_ids,
        protein_map_df=protein_map_df,
        bulk_table_csv=args.bulk_table_csv,
        protein_column=args.protein_column,
    )
    split = split_cells_and_genes(
        matrix=matrix,
        y=y,
        seed=args.seed,
        cell_test_size=args.cell_test_size,
        gene_test_size=args.gene_temp_size,
        gene_val_size_from_temp=args.gene_test_size_from_temp,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = VanillaNN(input_dim=matrix.shape[1], hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    X_tensor = torch.tensor(matrix, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.float32, device=device)
    train_cells = torch.tensor(split["train_cells"], dtype=torch.long, device=device)
    test_cells = torch.tensor(split["test_cells"], dtype=torch.long, device=device)
    train_genes = torch.tensor(split["train_genes"], dtype=torch.long, device=device)
    val_genes = torch.tensor(split["val_genes"], dtype=torch.long, device=device)
    test_genes = torch.tensor(split["test_genes"], dtype=torch.long, device=device)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(args.epochs):
        model.train()
        train_pred = model(X_tensor[train_cells])
        train_mean = train_pred[:, train_genes].mean(dim=0)
        train_target = y_tensor[train_genes]
        train_loss = F.mse_loss(train_mean, train_target)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_mean = model(X_tensor[train_cells])[:, val_genes].mean(dim=0)
            val_target = y_tensor[val_genes]
            val_loss = F.mse_loss(val_mean, val_target)

            test_mean = model(X_tensor[test_cells])[:, test_genes].mean(dim=0)
            test_target = y_tensor[test_genes]
            test_loss = F.mse_loss(test_mean, test_target)

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss.item()),
                "val_loss": float(val_loss.item()),
                "test_loss": float(test_loss.item()),
                "train_r2": float(r2_score(train_target.detach().cpu().numpy(), train_mean.detach().cpu().numpy())),
                "val_r2": float(r2_score(val_target.detach().cpu().numpy(), val_mean.detach().cpu().numpy())),
                "test_r2": float(r2_score(test_target.detach().cpu().numpy(), test_mean.detach().cpu().numpy())),
            }
        )

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        y_hat_test = model(X_tensor[test_cells]).cpu().numpy()
    y_true_all = y_tensor.cpu().numpy()

    train_pred_genes = y_hat_test[:, split["train_genes"]].mean(axis=0)
    train_true_genes = y_true_all[split["train_genes"]]
    test_pred_genes = y_hat_test[:, split["test_genes"]].mean(axis=0)
    test_true_genes = y_true_all[split["test_genes"]]

    rows = []
    for split_name, y_true, y_pred in [
        ("train_genes_in_test_cells", train_true_genes, train_pred_genes),
        ("test_genes_in_test_cells", test_true_genes, test_pred_genes),
    ]:
        metrics = compute_metrics(y_true, y_pred)
        metrics["split_name"] = split_name
        rows.append(metrics)

    pd.DataFrame(history).to_csv(args.output_dir / "vanillann_history.csv", index=False)
    pd.DataFrame(rows).to_csv(args.output_dir / "vanillann_metrics.csv", index=False)
    write_json(args.output_dir / "vanillann_metrics.json", {"seed": args.seed, "rows": rows})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()

