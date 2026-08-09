from __future__ import annotations

import argparse
import json

from _bootstrap import add_src_to_path

add_src_to_path()

from recipe.pipeline import run_recipe_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run multiple RECIPE modules.")
    parser.add_argument("--modules", default="A,B,C,D")
    parser.add_argument("--species", choices=("human", "mouse"), default="mouse")
    parser.add_argument("--condition", default="KD", help="Condition name, for example NC or KD.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default=None, help="Directory containing RECIPE data subfolders.")
    parser.add_argument("--bulk-unknown-split-csv", default=None)
    parser.add_argument("--bulk-known-split-csv", default=None)
    parser.add_argument(
        "--bulk-input-col",
        "--bulk-expression-col",
        dest="bulk_expression_col",
        default=None,
        help="Bulk input signal column, for example RNA-seq, Ribo-seq, or another transcript-level feature.",
    )
    parser.add_argument("--bulk-target-col", default=None, help="Bulk protein abundance target column.")
    parser.add_argument("--bulk-pause-col", default=None, help="Bulk pausing-count column.")
    parser.add_argument("--no-bulk-pause", action="store_true", help="Use zero pausing features for bulk modules.")
    parser.add_argument("--phase0-split-csv", default=None)
    parser.add_argument("--phase1-split-csv", default=None)
    parser.add_argument("--phase2-split-csv", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    modules = tuple(module.strip() for module in args.modules.split(",") if module.strip())
    summary = run_recipe_pipeline(
        modules=modules,
        output_root=args.output_root,
        species=args.species,
        condition=args.condition,
        seed=args.seed,
        device_name=args.device,
        data_root=args.data_root,
        bulk_unknown_split_csv=args.bulk_unknown_split_csv,
        bulk_known_split_csv=args.bulk_known_split_csv,
        bulk_expression_col=args.bulk_expression_col,
        bulk_target_col=args.bulk_target_col,
        bulk_pause_col=args.bulk_pause_col,
        use_bulk_pause=not args.no_bulk_pause,
        phase0_split_csv=args.phase0_split_csv,
        phase1_split_csv=args.phase1_split_csv,
        phase2_split_csv=args.phase2_split_csv,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
