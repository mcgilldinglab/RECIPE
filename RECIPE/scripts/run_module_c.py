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
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
