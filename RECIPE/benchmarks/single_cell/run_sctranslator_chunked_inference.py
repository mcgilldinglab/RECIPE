#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import cosine_similarity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run chunked scTranslator inference and aggregate notebook-style metrics."
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--enc-max-seq-len", type=int, default=20000)
    parser.add_argument("--dec-max-seq-len", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--test-batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_dir = args.repo_root / "code" / "model"
    if str(model_dir) not in sys.path:
        sys.path.append(str(model_dir))
    from utils import SCDataset, fix_SCDataset, setup_seed, test  # type: ignore

    setup_seed(args.seed)
    setup_seed(args.seed + args.repeat)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = torch.load(args.checkpoint, map_location="cpu").to(device)

    all_truth = []
    all_pred = []
    data_pairs = []

    for rna_path in sorted(args.data_dir.glob("X_testcell_traingene_adata*.h5ad")):
        suffix = rna_path.stem.replace("X_testcell_traingene_adata", "")
        pro_path = args.data_dir / f"y_testcell_traingene_adata{suffix}.h5ad"
        if pro_path.exists():
            data_pairs.append((rna_path, pro_path))
    for rna_path in sorted(args.data_dir.glob("X_testcell_testgene_adata_*.h5ad")):
        suffix = rna_path.stem.replace("X_testcell_testgene_adata_", "")
        pro_path = args.data_dir / f"y_testcell_testgene_adata_{suffix}.h5ad"
        if pro_path.exists():
            data_pairs.append((rna_path, pro_path))

    if not data_pairs:
        raise ValueError(f"No matching chunk pairs found in {args.data_dir}")

    rows = []
    for rna_path, pro_path in data_pairs:
        rna_adata = sc.read_h5ad(rna_path)
        pro_adata = sc.read_h5ad(pro_path)
        dataset = fix_SCDataset(rna_adata, pro_adata, args.enc_max_seq_len, args.dec_max_seq_len)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
        test_loss, test_ccc, y_hat, y_true = test(model, device, loader)
        pred_df = pd.DataFrame(y_hat, columns=pro_adata.var.index.tolist())
        true_df = pd.DataFrame(y_true, columns=pro_adata.var.index.tolist())
        mean_pred = pred_df.mean(axis=0)
        mean_true = true_df.mean(axis=0)
        rows.append(
            {
                "chunk": rna_path.name,
                "count": int(len(mean_true)),
                "r2": float(r2_score(mean_true, mean_pred)),
                "pearson_r": float(pearsonr(mean_true, mean_pred)[0]),
                "spearman_r": float(spearmanr(mean_true, mean_pred)[0]),
                "cosine_similarity": float(
                    cosine_similarity(mean_pred.values.reshape(1, -1), mean_true.values.reshape(1, -1))[0, 0]
                ),
                "loss": float(test_loss),
                "ccc": float(test_ccc),
            }
        )
        all_truth.append(mean_true)
        all_pred.append(mean_pred)

    all_truth_series = pd.concat(all_truth, axis=0)
    all_pred_series = pd.concat(all_pred, axis=0)
    summary = {
        "count": int(len(all_truth_series)),
        "r2": float(r2_score(all_truth_series, all_pred_series)),
        "pearson_r": float(pearsonr(all_truth_series, all_pred_series)[0]),
        "spearman_r": float(spearmanr(all_truth_series, all_pred_series)[0]),
        "cosine_similarity": float(
            cosine_similarity(all_pred_series.values.reshape(1, -1), all_truth_series.values.reshape(1, -1))[0, 0]
        ),
        "seed": args.seed,
    }

    pd.DataFrame(rows).to_csv(args.output_dir / "chunk_metrics.csv", index=False)
    pd.DataFrame({"truth": all_truth_series, "pred": all_pred_series}).to_csv(
        args.output_dir / "combined_gene_means.csv", index=False
    )
    with open(args.output_dir / "summary_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

