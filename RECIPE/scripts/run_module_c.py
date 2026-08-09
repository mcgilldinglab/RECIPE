from __future__ import annotations

import argparse
import json

from _bootstrap import add_src_to_path

add_src_to_path()

from recipe.ppi_workflow import run_ppi_refinement


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run RECIPE module C: PPI refinement.")
    parser.add_argument("--species", choices=("human", "mouse"), default="mouse")
    parser.add_argument("--condition", default="KD", help="Condition name, for example NC or KD.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bulk-checkpoint-path", default=None)
    parser.add_argument("--data-root", default=None, help="Directory containing RECIPE data subfolders.")
    parser.add_argument("--reference-csv", default=None, help="Bulk reference CSV used by the known-protein model.")
    parser.add_argument("--sequence-npy", default=None, help="Sequence embedding NPY used by the known-protein model.")
    parser.add_argument("--ppi-csv", default=None, help="Known PPI adjacency CSV.")
    parser.add_argument("--coexpression-csv", default=None, help="Optional coexpression CSV for edge-score summaries.")
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
    parser.add_argument("--edge-max-epochs", type=int, default=1000)
    parser.add_argument("--edge-patience", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--export-score-matrix", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_ppi_refinement(
        species=args.species,
        condition_name=args.condition,
        output_dir=args.output_dir,
        seed=args.seed,
        device_name=args.device,
        bulk_checkpoint_path=args.bulk_checkpoint_path,
        edge_max_epochs=args.edge_max_epochs,
        edge_patience=args.edge_patience,
        threshold=args.threshold,
        export_score_matrix=args.export_score_matrix,
        data_root=args.data_root,
        reference_csv=args.reference_csv,
        sequence_npy=args.sequence_npy,
        ppi_csv=args.ppi_csv,
        coexpression_csv=args.coexpression_csv,
        expression_col=args.expression_col,
        target_col=args.target_col,
        pause_col=args.pause_col,
        use_pause=not args.no_pause,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
