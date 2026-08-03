from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from _bootstrap import add_src_to_path

PROJECT_ROOT = add_src_to_path()

from recipe.bulk_data import load_ordered_cds_table, split_index_tensor, strip_version
from recipe.config import SINGLE_CELL_TRANSFER_CONFIG, get_bulk_task_config


def _first_present(columns: pd.Index, candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _split_labels(
    size: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    pool_idx: np.ndarray | None = None,
) -> np.ndarray:
    labels = np.full(size, "unlabeled", dtype=object)
    labels[np.asarray(train_idx, dtype=np.int64)] = "train"
    labels[np.asarray(val_idx, dtype=np.int64)] = "val"
    labels[np.asarray(test_idx, dtype=np.int64)] = "test"
    if pool_idx is not None:
        labels[np.asarray(pool_idx, dtype=np.int64)] = "unlabeled"
    return labels


def _write_split(path: Path, df: pd.DataFrame) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return {
        "file": str(path.relative_to(PROJECT_ROOT)),
        "rows": int(len(df)),
        "split_counts": df["split"].value_counts(dropna=False).to_dict(),
    }


def _bulk_split_dataframe(species: str, task: str, seed: int) -> pd.DataFrame:
    config = get_bulk_task_config(task=task, species=species)
    reference_df = pd.read_csv(config.reference_csv)
    condition = next(iter(config.conditions.values()))
    target = pd.to_numeric(reference_df[condition.target_col], errors="coerce").to_numpy(dtype=np.float32)
    labeled_idx = np.where(np.isfinite(target) & (~np.isclose(target, 0.0)))[0]
    pool_idx = np.where(~(np.isfinite(target) & (~np.isclose(target, 0.0))))[0]
    train_idx, val_idx, test_idx = split_index_tensor(torch.tensor(labeled_idx, dtype=torch.long), seed=seed)

    transcript_col = _first_present(reference_df.columns, ("transcript_id", "transcript_id_x", "ordered_transcript_id"))
    protein_col = _first_present(reference_df.columns, ("protein_id", "protein_id_x", "protein_x", "gene"))
    out = pd.DataFrame(
        {
            "node_index": np.arange(len(reference_df), dtype=np.int64),
            "split": _split_labels(
                len(reference_df),
                train_idx.numpy(),
                val_idx.numpy(),
                test_idx.numpy(),
                pool_idx=pool_idx,
            ),
            "is_labeled": np.isfinite(target) & (~np.isclose(target, 0.0)),
            "seed": seed,
            "species": species,
            "task": task,
            "target_column": condition.target_col,
        }
    )
    if transcript_col is not None:
        out.insert(1, "transcript_id", strip_version(reference_df[transcript_col]))
    if protein_col is not None:
        out.insert(2 if "transcript_id" in out.columns else 1, "protein_id", reference_df[protein_col].astype(str))
    return out


def _ordered_single_cell_table() -> pd.DataFrame:
    return load_ordered_cds_table(
        SINGLE_CELL_TRANSFER_CONFIG.cds_csv,
        SINGLE_CELL_TRANSFER_CONFIG.transcript_order_csv,
    )


def _single_cell_split_dataframe(seed: int, mode: str) -> pd.DataFrame:
    ordered_df = _ordered_single_cell_table()
    target = pd.to_numeric(ordered_df[SINGLE_CELL_TRANSFER_CONFIG.phase0_target_col], errors="coerce").to_numpy(dtype=np.float32)
    valid_idx = np.where(np.isfinite(target) & (~np.isclose(target, 0.0)))[0]
    pool_idx = np.where(~(np.isfinite(target) & (~np.isclose(target, 0.0))))[0]

    if mode == "self_learning":
        train_idx, val_idx, test_idx = split_index_tensor(torch.tensor(valid_idx, dtype=torch.long), seed=seed)
        train_idx = train_idx.numpy()
        val_idx = val_idx.numpy()
        test_idx = test_idx.numpy()
    elif mode == "module_a":
        train_val_idx, test_idx = train_test_split(valid_idx, test_size=0.2, random_state=seed)
        train_idx, val_idx = train_test_split(train_val_idx, test_size=0.25, random_state=seed)
    elif mode == "cell_graph":
        train_idx, temp_idx = train_test_split(valid_idx, test_size=0.25, random_state=seed)
        test_idx, val_idx = train_test_split(temp_idx, test_size=0.5, random_state=seed)
    else:
        raise ValueError(f"Unsupported single-cell split mode: {mode}")

    out = pd.DataFrame(
        {
            "node_index": np.arange(len(ordered_df), dtype=np.int64),
            "transcript_id": ordered_df["ordered_transcript_id"].astype(str),
            "protein_id": ordered_df["protein_id"].astype(str),
            "split": _split_labels(len(ordered_df), train_idx, val_idx, test_idx, pool_idx=pool_idx),
            "is_labeled": np.isfinite(target) & (~np.isclose(target, 0.0)),
            "seed": seed,
            "workflow": mode,
            "target_column": SINGLE_CELL_TRANSFER_CONFIG.phase0_target_col,
        }
    )
    return out


def _parse_seeds(raw: str) -> list[int]:
    return [int(seed.strip()) for seed in raw.split(",") if seed.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fixed train/val/test split CSV files used by training notebooks.")
    parser.add_argument("--output-dir", default="data/splits")
    parser.add_argument("--bulk-mouse-seeds", default="0,8,12,24")
    parser.add_argument("--bulk-human-seeds", default="0,5,8,12,42")
    parser.add_argument("--single-cell-self-learning-seeds", default="0,5,8,12,42")
    parser.add_argument("--single-cell-module-a-seed", type=int, default=42)
    parser.add_argument("--single-cell-graph-seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    summary: list[dict[str, object]] = []
    for seed in _parse_seeds(args.bulk_mouse_seeds):
        summary.append(
            _write_split(
                output_dir / f"bulk_mouse_unknown_seed{seed}.csv",
                _bulk_split_dataframe("mouse", "unknown", seed),
            )
        )
    for seed in _parse_seeds(args.bulk_human_seeds):
        summary.append(
            _write_split(
                output_dir / f"bulk_human_known_seed{seed}.csv",
                _bulk_split_dataframe("human", "known", seed),
            )
        )
    for seed in _parse_seeds(args.single_cell_self_learning_seeds):
        summary.append(
            _write_split(
                output_dir / f"single_cell_self_learning_seed{seed}.csv",
                _single_cell_split_dataframe(seed, mode="self_learning"),
            )
        )
    summary.append(
        _write_split(
            output_dir / f"single_cell_module_a_seed{args.single_cell_module_a_seed}.csv",
            _single_cell_split_dataframe(args.single_cell_module_a_seed, mode="module_a"),
        )
    )
    summary.append(
        _write_split(
            output_dir / f"single_cell_graph_seed{args.single_cell_graph_seed}.csv",
            _single_cell_split_dataframe(args.single_cell_graph_seed, mode="cell_graph"),
        )
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
