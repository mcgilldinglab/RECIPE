from __future__ import annotations

import argparse
import json

from _bootstrap import add_src_to_path

add_src_to_path()

from recipe.bulk_workflow import run_bulk_module
from recipe.utils import json_sanitize


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run RECIPE module B: known bulk protein prediction.")
    parser.add_argument("--species", choices=("human", "mouse"), default="mouse")
    parser.add_argument("--condition", default="KD", help="Condition name, for example NC or KD.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train", action="store_true", help="Force model training even if a checkpoint exists.")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--data-root", default=None, help="Directory containing RECIPE data subfolders.")
    parser.add_argument("--model-root", default=None, help="Directory containing RECIPE checkpoint subfolders.")
    parser.add_argument("--reference-csv", default=None, help="Bulk reference CSV.")
    parser.add_argument("--sequence-npy", default=None, help="Sequence embedding NPY.")
    parser.add_argument("--ppi-csv", default=None, help="PPI adjacency CSV.")
    parser.add_argument("--pause-csv", default=None, help="Optional pausing CSV.")
    parser.add_argument("--split-csv", default=None, help="Optional fixed train/val/test split CSV.")
    parser.add_argument(
        "--input-col",
        "--expression-col",
        dest="expression_col",
        default=None,
        help="Input signal column, for example RNA-seq, Ribo-seq, or another transcript-level feature.",
    )
    parser.add_argument("--target-col", default=None, help="Protein abundance target column.")
    parser.add_argument("--pause-col", default=None, help="Pausing-count column.")
    parser.add_argument("--no-pause", action="store_true", help="Use zero pausing features when no pausing column is available.")
    parser.add_argument("--max-epochs", type=int, default=3000)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=7e-2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_bulk_module(
        species=args.species,
        task="known",
        condition_name=args.condition,
        output_dir=args.output_dir,
        seed=args.seed,
        device_name=args.device,
        train=args.train,
        checkpoint_path=args.checkpoint_path,
        data_root=args.data_root,
        model_root=args.model_root,
        reference_csv=args.reference_csv,
        sequence_npy=args.sequence_npy,
        ppi_csv=args.ppi_csv,
        pause_csv=args.pause_csv,
        split_csv=args.split_csv,
        expression_col=args.expression_col,
        target_col=args.target_col,
        pause_col=args.pause_col,
        use_pause=not args.no_pause,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
    )
    print(json.dumps(json_sanitize(summary), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
