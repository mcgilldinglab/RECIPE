#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


def read_expression_matrix(
    expression_csv: Path,
    transcript_column: str = "transcript_id",
    drop_columns: Iterable[str] = ("Unnamed: 0", "scribo"),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(expression_csv).fillna(0)
    if transcript_column not in df.columns:
        raise ValueError(f"Missing transcript column '{transcript_column}' in {expression_csv}")

    gene_ids = df[transcript_column].astype(str).str.split(".").str[0].to_numpy()
    keep_cols = [c for c in df.columns if c not in set(drop_columns) | {transcript_column}]
    cell_names = np.asarray(keep_cols, dtype=object)
    matrix = df[keep_cols].to_numpy(dtype=np.float32).T
    return gene_ids, cell_names, matrix


def read_protein_map(protein_map_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(protein_map_csv)
    if "transcript_id" in df.columns:
        tx_col = "transcript_id"
    elif "transcript_id_x" in df.columns:
        tx_col = "transcript_id_x"
    else:
        raise ValueError(f"No transcript id column found in {protein_map_csv}")
    df = df.copy()
    df["transcript_id_clean"] = df[tx_col].astype(str).str.split(".").str[0]
    return df


def build_aligned_targets(
    gene_ids: np.ndarray,
    protein_map_df: pd.DataFrame,
    bulk_table_csv: Path,
    protein_column: str = "NC3",
    myid_column: str = "my_Id",
) -> tuple[np.ndarray, dict[str, str]]:
    bulk_df = pd.read_csv(bulk_table_csv)
    if protein_column not in bulk_df.columns:
        raise ValueError(f"Missing bulk protein column '{protein_column}' in {bulk_table_csv}")
    if protein_column not in protein_map_df.columns:
        raise ValueError(f"Missing protein column '{protein_column}' in protein map")

    denom = float(np.median(bulk_df[protein_column].to_numpy(dtype=np.float64)))
    if denom == 0:
        denom = 1e-6

    protein_map_df = protein_map_df.copy()
    protein_map_df[f"{protein_column}_log2cpm"] = np.log2(
        protein_map_df[protein_column].to_numpy(dtype=np.float64) / denom + 1.0
    )
    protein_map_df = protein_map_df.drop_duplicates("transcript_id_clean").set_index("transcript_id_clean")

    y = pd.Series(index=pd.Index(gene_ids, dtype=str), dtype=np.float32)
    common_gene_ids = np.intersect1d(gene_ids, protein_map_df.index.to_numpy(dtype=str))
    y.loc[common_gene_ids] = protein_map_df.loc[common_gene_ids, f"{protein_column}_log2cpm"].to_numpy(
        dtype=np.float32
    )
    y = y.fillna(0.0)

    myid_map: dict[str, str] = {}
    if myid_column in protein_map_df.columns:
        myid_map = {
            str(tx): str(protein_map_df.loc[tx, myid_column])
            for tx in protein_map_df.index.astype(str)
            if pd.notna(protein_map_df.loc[tx, myid_column])
        }
    return y.to_numpy(dtype=np.float32), myid_map


def split_cells_and_genes(
    matrix: np.ndarray,
    y: np.ndarray,
    seed: int,
    cell_test_size: float = 0.1,
    gene_test_size: float = 0.25,
    gene_val_size_from_temp: float | None = None,
) -> dict[str, np.ndarray]:
    n_cells = matrix.shape[0]
    all_cells = np.arange(n_cells)
    train_cells, test_cells = train_test_split(all_cells, test_size=cell_test_size, random_state=seed)

    valid_gene_indices = np.where(y != 0)[0]
    if gene_val_size_from_temp is None:
        train_genes, test_genes = train_test_split(
            valid_gene_indices, test_size=gene_test_size, random_state=seed
        )
        return {
            "train_cells": np.asarray(train_cells),
            "test_cells": np.asarray(test_cells),
            "train_genes": np.asarray(train_genes),
            "test_genes": np.asarray(test_genes),
        }

    train_genes, temp_genes = train_test_split(valid_gene_indices, test_size=gene_test_size, random_state=seed)
    val_genes, test_genes = train_test_split(temp_genes, test_size=gene_val_size_from_temp, random_state=seed)
    return {
        "train_cells": np.asarray(train_cells),
        "test_cells": np.asarray(test_cells),
        "train_genes": np.asarray(train_genes),
        "val_genes": np.asarray(val_genes),
        "test_genes": np.asarray(test_genes),
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    flat_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    flat_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    return {
        "count": int(flat_true.size),
        "r2": float(r2_score(flat_true, flat_pred)),
        "pearson_r": float(pearsonr(flat_true, flat_pred)[0]),
        "spearman_r": float(spearmanr(flat_true, flat_pred)[0]),
        "cosine_similarity": float(cosine_similarity(flat_pred.reshape(1, -1), flat_true.reshape(1, -1))[0, 0]),
    }


def create_anndata(
    matrix: np.ndarray,
    obs_names: Iterable[str],
    transcript_ids: Iterable[str],
    myid_map: dict[str, str] | None = None,
) -> ad.AnnData:
    transcript_ids = [str(x) for x in transcript_ids]
    myid_map = myid_map or {}
    my_ids = [myid_map.get(tx, tx) for tx in transcript_ids]

    adata = ad.AnnData(np.asarray(matrix, dtype=np.float32))
    adata.obs_names = np.asarray(list(obs_names), dtype=object)
    adata.var_names = np.asarray(my_ids, dtype=object)
    adata.var["my_Id"] = adata.var_names.astype(str)
    adata.var["transcript_id"] = np.asarray(transcript_ids, dtype=object)
    return adata


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
