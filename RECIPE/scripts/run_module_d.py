from __future__ import annotations

import argparse
import json
import shlex

from _bootstrap import add_src_to_path

add_src_to_path()

from recipe.single_cell_rnaseq_workflow import (
    run_phase0 as run_scrnaseq_phase0,
    run_phase12 as run_scrnaseq_phase12,
    run_phase3 as run_scrnaseq_phase3,
)
from recipe.single_cell_riboseq_workflow import run_single_cell_transfer
from recipe.utils import json_sanitize


def _split_forwarded_args(value: str) -> list[str]:
    return shlex.split(value) if value else []


def _run_scrnaseq_module_d(args: argparse.Namespace, steps: tuple[str, ...]) -> dict[str, object]:
    normalized_steps: list[str] = []
    for step in steps:
        key = step.lower()
        if key in {"all", "scrnaseq", "scrnaseq_workflow", "phase023"}:
            normalized_steps.extend(["phase0", "phase12", "phase3"])
        elif key == "phase1":
            normalized_steps.append("phase12")
        elif key == "phase2":
            normalized_steps.append("phase3")
        elif key in {"phase0", "phase12", "phase3"}:
            normalized_steps.append(key)
        else:
            raise ValueError(f"Unsupported scRNA-seq Module D step: {step}")

    forwarded = {
        "phase0": _split_forwarded_args(args.rnaseq_phase0_args or args.bulk_module_args),
        "phase12": _split_forwarded_args(
            args.rnaseq_phase12_args or args.phase1_rnaseq_pseudo_bulk_finetuning_args
        ),
        "phase3": _split_forwarded_args(
            args.rnaseq_phase3_args or args.phase2_single_cell_protein_finetuning_args
        ),
    }
    runners = {
        "phase0": run_scrnaseq_phase0,
        "phase12": run_scrnaseq_phase12,
        "phase3": run_scrnaseq_phase3,
    }
    for step in normalized_steps:
        runners[step](forwarded[step])

    return {
        "assay": "scrnaseq",
        "steps": normalized_steps,
        "forwarded_args": forwarded,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run RECIPE module D: single-cell transfer.")
    parser.add_argument(
        "--assay",
        choices=("scriboseq", "scrnaseq"),
        default="scriboseq",
        help="Module D input assay. scriboseq runs the original phase0/phase1/phase2 branch; scrnaseq runs the RNA-seq phase0/phase12/phase3 branch.",
    )
    parser.add_argument("--steps", default="phase0,phase1,phase2")
    parser.add_argument("--output-dir", default=None, help="Required for --assay scriboseq.")
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
    parser.add_argument("--phase1-checkpoint", default=None)
    parser.add_argument("--phase2-checkpoint", default=None)
    parser.add_argument("--phase0-split-csv", default=None)
    parser.add_argument("--phase1-split-csv", default=None)
    parser.add_argument("--phase2-split-csv", default=None)
    parser.add_argument(
        "--use-bundled-cell-embeddings",
        action="store_true",
        help="Use bundled phase2 cell embeddings instead of recomputing them from the phase1 bulk model.",
    )
    parser.add_argument(
        "--rnaseq-phase0-args",
        default="",
        help="Shell-style argument string forwarded to the scRNA-seq phase0 bulk-module script.",
    )
    parser.add_argument(
        "--rnaseq-phase12-args",
        default="",
        help="Shell-style argument string forwarded to the scRNA-seq phase12 RNA pseudo-bulk finetuning script.",
    )
    parser.add_argument(
        "--rnaseq-phase3-args",
        default="",
        help="Shell-style argument string forwarded to the scRNA-seq phase3 matched single-cell protein script.",
    )
    parser.add_argument("--bulk-module-args", default="", help="Deprecated alias for --rnaseq-phase0-args.")
    parser.add_argument(
        "--phase1-rnaseq-pseudo-bulk-finetuning-args",
        default="",
        help="Deprecated alias for --rnaseq-phase12-args.",
    )
    parser.add_argument(
        "--phase2-single-cell-protein-finetuning-args",
        default="",
        help="Deprecated alias for --rnaseq-phase3-args.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    steps = tuple(step.strip() for step in args.steps.split(",") if step.strip())
    if args.assay == "scrnaseq":
        summary = _run_scrnaseq_module_d(args, steps=steps)
        print(json.dumps(json_sanitize(summary), indent=2, ensure_ascii=False, allow_nan=False))
        return

    if args.output_dir is None:
        raise SystemExit("--output-dir is required for --assay scriboseq")
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
        phase1_checkpoint=args.phase1_checkpoint,
        phase2_checkpoint=args.phase2_checkpoint,
        phase0_split_csv=args.phase0_split_csv,
        phase1_split_csv=args.phase1_split_csv,
        phase2_split_csv=args.phase2_split_csv,
        use_bundled_cell_embeddings=args.use_bundled_cell_embeddings,
    )
    print(json.dumps(json_sanitize(summary), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
