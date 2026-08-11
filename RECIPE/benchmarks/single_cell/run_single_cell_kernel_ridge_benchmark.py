#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the notebook-style single-cell KRR-new benchmark.")
    parser.add_argument("--expression-csv", type=Path, required=True)
    parser.add_argument("--transcript-column", default=None)
    parser.add_argument("--protein-map-csv", type=Path, required=True)
    parser.add_argument("--bulk-table-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--protein-column", type=str, default="NC3")
    parser.add_argument("--cell-test-size", type=float, default=0.1)
    parser.add_argument("--gene-test-size", type=float, default=0.25)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.01)
    parser.add_argument("--max-train-genes", type=int, default=None)
    parser.add_argument("--max-test-genes", type=int, default=None)
    return parser.parse_args()


def evaluate_gene_set(matrix: np.ndarray, y: np.ndarray, train_cells, test_cells, gene_indices, alpha: float, gamma: float):
    X_train = matrix[train_cells]
    X_test = matrix[test_cells]

    preds = []
    trues = []
    for gene_idx in gene_indices:
        y_train_gene = X_train[:, gene_idx]
        model = KernelRidge(kernel="rbf", alpha=alpha, gamma=gamma)
        model.fit(X_train, y_train_gene)
        preds.append(float(model.predict(X_test).mean()))
        trues.append(float(y[gene_idx]))

    preds = np.asarray(preds, dtype=np.float64)
    trues = np.asarray(trues, dtype=np.float64)
    return compute_metrics(trues, preds), trues, preds


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gene_ids, _, matrix = read_expression_matrix(args.expression_csv, transcript_column=args.transcript_column)
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
        gene_test_size=args.gene_test_size,
    )
    if args.max_train_genes is not None:
        split["train_genes"] = split["train_genes"][: args.max_train_genes]
    if args.max_test_genes is not None:
        split["test_genes"] = split["test_genes"][: args.max_test_genes]

    rows = []
    for split_name in ("train_genes", "test_genes"):
        metrics, trues, preds = evaluate_gene_set(
            matrix=matrix,
            y=y,
            train_cells=split["train_cells"],
            test_cells=split["test_cells"],
            gene_indices=split[split_name],
            alpha=args.alpha,
            gamma=args.gamma,
        )
        metrics["split_name"] = split_name
        rows.append(metrics)
        pd.DataFrame(
            {
                "gene_index": split[split_name],
                "transcript_id": gene_ids[split[split_name]],
                "y_true": trues,
                "y_pred_mean_test_cells": preds,
            }
        ).to_csv(args.output_dir / f"krr_{split_name}_predictions.csv", index=False)

    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "krr_metrics.csv", index=False)
    write_json(args.output_dir / "krr_metrics.json", {"seed": args.seed, "rows": rows})
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
