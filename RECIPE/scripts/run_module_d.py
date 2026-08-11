from __future__ import annotations

import argparse
import json

from _bootstrap import add_src_to_path

add_src_to_path()

from recipe.single_cell_riboseq_workflow import run_single_cell_transfer
from recipe.utils import json_sanitize


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run RECIPE module D: single-cell transfer.")
    parser.add_argument("--steps", default="phase0,phase1,phase2")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-phase0", action="store_true")
    parser.add_argument("--train-phase1", action="store_true")
    parser.add_argument("--train-phase2", action="store_true")
    parser.add_argument("--data-root", default=None, help="Directory containing RECIPE data subfolders.")
    parser.add_argument("--model-root", default=None, help="Directory containing RECIPE checkpoint subfolders.")
    parser.add_argument("--bulk-reference-csv", default=None)
    parser.add_argument("--transcript-order-csv", default=None)
    parser.add_argument("--sequence-npy", default=None)
    parser.add_argument("--ppi-csv", default=None)
    parser.add_argument("--cds-csv", default=None)
    parser.add_argument("--phase0-pause-csv", default=None)
    parser.add_argument("--phase1-pause-csv", default=None)
    parser.add_argument("--expression-csv", default=None)
    parser.add_argument("--expression-normalized-csv", default=None)
    parser.add_argument("--metadata-csv", default=None)
    parser.add_argument("--pause-matrix-csv", default=None)
    parser.add_argument("--phase0-init-checkpoint", default=None)
    parser.add_argument("--phase0-split-csv", default=None)
    parser.add_argument("--phase1-split-csv", default=None)
    parser.add_argument("--phase2-split-csv", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    steps = tuple(step.strip() for step in args.steps.split(",") if step.strip())
    summary = run_single_cell_transfer(
        output_dir=args.output_dir,
        steps=steps,
        seed=args.seed,
        device_name=args.device,
        train_phase0=args.train_phase0,
        train_phase1=args.train_phase1,
        train_phase2=args.train_phase2,
        data_root=args.data_root,
        model_root=args.model_root,
        bulk_reference_csv=args.bulk_reference_csv,
        transcript_order_csv=args.transcript_order_csv,
        sequence_npy=args.sequence_npy,
        ppi_csv=args.ppi_csv,
        cds_csv=args.cds_csv,
        phase0_pause_csv=args.phase0_pause_csv,
        phase1_pause_csv=args.phase1_pause_csv,
        expression_csv=args.expression_csv,
        expression_normalized_csv=args.expression_normalized_csv,
        metadata_csv=args.metadata_csv,
        pause_matrix_csv=args.pause_matrix_csv,
        phase0_init_checkpoint=args.phase0_init_checkpoint,
        phase0_split_csv=args.phase0_split_csv,
        phase1_split_csv=args.phase1_split_csv,
        phase2_split_csv=args.phase2_split_csv,
    )
    print(json.dumps(json_sanitize(summary), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
