#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from benchmark_utils import (  # noqa: E402
    build_aligned_targets,
    create_anndata,
    read_expression_matrix,
    read_protein_map,
    split_cells_and_genes,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare clean scTranslator benchmark inputs from the notebook-style single-cell benchmark workflow."
    )
    parser.add_argument("--expression-csv", type=Path, required=True)
    parser.add_argument("--transcript-column", default=None)
    parser.add_argument("--protein-map-csv", type=Path, required=True)
    parser.add_argument("--bulk-table-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--protein-column", type=str, default="NC3")
    parser.add_argument("--cell-test-size", type=float, default=0.1)
    parser.add_argument("--gene-test-size", type=float, default=0.25)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--export-test-train-gene-chunks", action="store_true")
    parser.add_argument("--export-test-test-gene-chunks", action="store_true")
    parser.add_argument("--train-rna-name", type=str, default="X_train_genes_adata.h5ad")
    parser.add_argument("--train-protein-name", type=str, default="Y_train_protein_adata.h5ad")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gene_ids, cell_names, matrix = read_expression_matrix(
        args.expression_csv,
        transcript_column=args.transcript_column,
    )
    protein_map_df = read_protein_map(args.protein_map_csv)
    y, myid_map = build_aligned_targets(
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

    train_cells = split["train_cells"]
    test_cells = split["test_cells"]
    train_genes = split["train_genes"]
    test_genes = split["test_genes"]

    train_rna = matrix[np.ix_(train_cells, train_genes)]
    train_targets = np.tile(y[train_genes], (len(train_cells), 1))

    train_obs = [str(cell_names[i]) for i in train_cells]
    train_tx = gene_ids[train_genes]

    train_rna_adata = create_anndata(train_rna, train_obs, train_tx, myid_map)
    train_protein_adata = create_anndata(train_targets, train_obs, train_tx, myid_map)
    train_rna_adata.write(args.output_dir / args.train_rna_name)
    train_protein_adata.write(args.output_dir / args.train_protein_name)

    if args.export_test_train_gene_chunks:
        test_train_rna = matrix[np.ix_(test_cells, train_genes)]
        test_train_pro = np.tile(y[train_genes], (len(test_cells), 1))
        test_obs = [str(cell_names[i]) for i in test_cells]
        test_train_tx = gene_ids[train_genes]
        test_train_rna_adata = create_anndata(test_train_rna, test_obs, test_train_tx, myid_map)
        test_train_pro_adata = create_anndata(test_train_pro, test_obs, test_train_tx, myid_map)
        for start in range(0, len(test_train_tx), args.chunk_size):
            end = min(start + args.chunk_size, len(test_train_tx))
            suffix = f"{(start // args.chunk_size) + 1:04d}"
            test_train_rna_adata[:, start:end].write(args.output_dir / f"X_testcell_traingene_adata{suffix}.h5ad")
            test_train_pro_adata[:, start:end].write(args.output_dir / f"y_testcell_traingene_adata{suffix}.h5ad")

    if args.export_test_test_gene_chunks:
        test_test_rna = matrix[np.ix_(test_cells, test_genes)]
        test_test_pro = np.tile(y[test_genes], (len(test_cells), 1))
        test_obs = [str(cell_names[i]) for i in test_cells]
        test_test_tx = gene_ids[test_genes]
        test_test_rna_adata = create_anndata(test_test_rna, test_obs, test_test_tx, myid_map)
        test_test_pro_adata = create_anndata(test_test_pro, test_obs, test_test_tx, myid_map)
        for start in range(0, len(test_test_tx), args.chunk_size):
            end = min(start + args.chunk_size, len(test_test_tx))
            suffix = f"{(start // args.chunk_size) + 1:02d}"
            test_test_rna_adata[:, start:end].write(args.output_dir / f"X_testcell_testgene_adata_{suffix}.h5ad")
            test_test_pro_adata[:, start:end].write(args.output_dir / f"y_testcell_testgene_adata_{suffix}.h5ad")

    summary = {
        "seed": args.seed,
        "cell_count": int(matrix.shape[0]),
        "gene_count": int(matrix.shape[1]),
        "train_cell_count": int(len(train_cells)),
        "test_cell_count": int(len(test_cells)),
        "train_gene_count": int(len(train_genes)),
        "test_gene_count": int(len(test_genes)),
        "output_dir": str(args.output_dir),
    }
    write_json(args.output_dir / "prepare_single_cell_sctranslator_data_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
