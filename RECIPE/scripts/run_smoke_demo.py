from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from _bootstrap import add_src_to_path

PROJECT_ROOT = add_src_to_path()

from recipe.bulk_regression import (
    BulkConditionSpec,
    build_bulk_graph_from_dataframe,
    evaluate_graph_regression,
    train_single_graph_bulk,
)
from recipe.bulk_workflow import BulkSplitBundle, make_bulk_model, save_bulk_outputs
from recipe.utils import resolve_device, set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a CPU-friendly RECIPE smoke demo on tiny simulated data.")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "examples" / "smoke_data"))
    parser.add_argument("--output-dir", default="outputs/smoke_demo")
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=8)
    return parser


def split_demo_indices(node_count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if node_count < 6:
        raise ValueError("The smoke demo requires at least 6 nodes for train/val/test splits.")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(node_count)
    test_size = max(2, node_count // 4)
    val_size = max(2, node_count // 4)
    if test_size + val_size >= node_count:
        test_size = 2
        val_size = 2
    test_idx = indices[:test_size]
    val_idx = indices[test_size:test_size + val_size]
    train_idx = indices[test_size + val_size:]
    return (
        torch.tensor(train_idx, dtype=torch.long),
        torch.tensor(val_idx, dtype=torch.long),
        torch.tensor(test_idx, dtype=torch.long),
    )


def main() -> None:
    args = build_parser().parse_args()
    start_time = time.perf_counter()
    set_seed(args.seed)
    device = resolve_device(args.device)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bulk_df = pd.read_csv(data_dir / "bulk_reference.csv")
    sequence_embeddings = pd.read_csv(data_dir / "sequence_embeddings.csv").to_numpy(dtype=np.float32)
    sequence_npy = output_dir / "sequence_embeddings.npy"
    np.save(sequence_npy, sequence_embeddings)

    condition = BulkConditionSpec(
        name="SMOKE",
        expression_col="rNC2",
        target_col="NC3",
        pause_col="High_Pause_Countsc18nc",
    )
    data, scaling_summary = build_bulk_graph_from_dataframe(
        bulk_df=bulk_df,
        condition=condition,
        sequence_npy_path=sequence_npy,
        ppi_csv_path=data_dir / "ppi_matrix.csv",
        scale_method="log_median",
        reference_df=bulk_df,
    )
    train_idx, val_idx, test_idx = split_demo_indices(data.num_nodes, seed=args.seed)
    splits = BulkSplitBundle(
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        pool_idx=torch.empty(0, dtype=torch.long),
    )

    model = make_bulk_model(data, device=device)
    data = data.to(device)
    model, training_summary = train_single_graph_bulk(
        model=model,
        data=data,
        train_idx=train_idx.to(device),
        val_idx=val_idx.to(device),
        test_idx=test_idx.to(device),
        max_epochs=args.epochs,
        patience=args.patience,
        log_every=0,
    )

    predictions, embeddings = model(data)
    predictions = predictions.detach().cpu()
    embeddings = embeddings.detach().cpu()
    summary = {
        "demo": "smoke_bulk_graphsage",
        "seed": args.seed,
        "device": str(device),
        "node_count": int(data.num_nodes),
        "expression_col": condition.expression_col,
        "target_col": condition.target_col,
        "pause_col": condition.pause_col,
        "train_metrics": evaluate_graph_regression(model, data, train_idx.to(device)),
        "val_metrics": evaluate_graph_regression(model, data, val_idx.to(device)),
        "test_metrics": evaluate_graph_regression(model, data, test_idx.to(device)),
        "training": training_summary,
        "scaling": scaling_summary,
    }
    output_files = save_bulk_outputs(
        output_dir=output_dir,
        bulk_df=bulk_df,
        predictions=predictions,
        embeddings=embeddings,
        splits=splits,
        summary=summary,
        checkpoint_path=None,
    )
    summary["outputs"] = output_files
    summary["elapsed_seconds"] = round(time.perf_counter() - start_time, 3)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
