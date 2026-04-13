from __future__ import annotations

import argparse
import importlib.util
import shlex
import sys
from pathlib import Path
from types import ModuleType
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PHASE0_SCRIPT = PROJECT_ROOT / "train_phase0_ensmusp_pseudobulk_raw_bulkprot.py"
PHASE12_SCRIPT = PROJECT_ROOT / "train_phase12_ensmusp_scRNA_bulkprot.py"
PHASE3_SCRIPT = PROJECT_ROOT / "train_phase3_ensmusp_nanospins_matched.py"

BULK_MODULE_STEP = "Bulk Module"
RNASEQ_PSEUDOBULK_FINETUNING_STEP = "Phase 1: RNA-seq Pseudo-Bulk Finetuning"
SINGLE_CELL_PROTEIN_FINETUNING_STEP = "Phase 2: Single-Cell Protein Finetuning"


def _load_script_module(script_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"recipe_wrapped_{script_path.stem}", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load workflow script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _normalize_script_args(script_args: Sequence[str] | None) -> list[str]:
    args = list(script_args or [])
    if args and args[0] == "--":
        return args[1:]
    return args


def _run_script_main(script_path: Path, script_args: Sequence[str] | None = None) -> None:
    args = _normalize_script_args(script_args)
    module = _load_script_module(script_path)
    if not hasattr(module, "main"):
        raise AttributeError(f"{script_path} does not define a main() entry point.")

    original_argv = sys.argv[:]
    sys.argv = [str(script_path), *args]
    try:
        module.main()
    finally:
        sys.argv = original_argv


def run_phase0(script_args: Sequence[str] | None = None) -> None:
    _run_script_main(PHASE0_SCRIPT, script_args)


def run_phase12(script_args: Sequence[str] | None = None) -> None:
    _run_script_main(PHASE12_SCRIPT, script_args)


def run_phase3(script_args: Sequence[str] | None = None) -> None:
    _run_script_main(PHASE3_SCRIPT, script_args)


def run_scrnaseq_workflow(
    bulk_module_args: Sequence[str] | None = None,
    phase1_rnaseq_pseudo_bulk_finetuning_args: Sequence[str] | None = None,
    phase2_single_cell_protein_finetuning_args: Sequence[str] | None = None,
    *,
    phase0_args: Sequence[str] | None = None,
    phase12_args: Sequence[str] | None = None,
    phase3_args: Sequence[str] | None = None,
) -> None:
    run_phase0(bulk_module_args if bulk_module_args is not None else phase0_args)
    run_phase12(
        phase1_rnaseq_pseudo_bulk_finetuning_args
        if phase1_rnaseq_pseudo_bulk_finetuning_args is not None
        else phase12_args
    )
    run_phase3(
        phase2_single_cell_protein_finetuning_args
        if phase2_single_cell_protein_finetuning_args is not None
        else phase3_args
    )


def run_phase023(
    bulk_module_args: Sequence[str] | None = None,
    phase1_rnaseq_pseudo_bulk_finetuning_args: Sequence[str] | None = None,
    phase2_single_cell_protein_finetuning_args: Sequence[str] | None = None,
    *,
    phase0_args: Sequence[str] | None = None,
    phase12_args: Sequence[str] | None = None,
    phase3_args: Sequence[str] | None = None,
) -> None:
    run_scrnaseq_workflow(
        bulk_module_args=bulk_module_args,
        phase1_rnaseq_pseudo_bulk_finetuning_args=phase1_rnaseq_pseudo_bulk_finetuning_args,
        phase2_single_cell_protein_finetuning_args=phase2_single_cell_protein_finetuning_args,
        phase0_args=phase0_args,
        phase12_args=phase12_args,
        phase3_args=phase3_args,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Unified RNA-seq single-cell workflow wrapper for the current ENSMUSP pipeline. "
            "This workflow keeps the training logic in the original bulk module / "
            "RNA-seq pseudo-bulk finetuning / single-cell protein finetuning "
            "scripts and centralizes entry points under RECIPE."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    phase0_parser = subparsers.add_parser("phase0", help=f"Run {BULK_MODULE_STEP}.")
    phase0_parser.add_argument("script_args", nargs=argparse.REMAINDER, help=f"Args forwarded to {BULK_MODULE_STEP}.")

    phase12_parser = subparsers.add_parser("phase12", help=f"Run {RNASEQ_PSEUDOBULK_FINETUNING_STEP}.")
    phase12_parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help=f"Args forwarded to {RNASEQ_PSEUDOBULK_FINETUNING_STEP}.",
    )

    phase3_parser = subparsers.add_parser("phase3", help=f"Run {SINGLE_CELL_PROTEIN_FINETUNING_STEP}.")
    phase3_parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help=f"Args forwarded to {SINGLE_CELL_PROTEIN_FINETUNING_STEP}.",
    )

    phase023_parser = subparsers.add_parser(
        "scrnaseq_workflow",
        aliases=["phase023"],
        help=(
            f"Run {BULK_MODULE_STEP}, {RNASEQ_PSEUDOBULK_FINETUNING_STEP}, "
            f"and {SINGLE_CELL_PROTEIN_FINETUNING_STEP} sequentially."
        ),
    )
    phase023_parser.add_argument(
        "--bulk-module-args",
        default="",
        help=f"Shell-style argument string forwarded to {BULK_MODULE_STEP}.",
    )
    phase023_parser.add_argument(
        "--phase1-rnaseq-pseudo-bulk-finetuning-args",
        default="",
        help=f"Shell-style argument string forwarded to {RNASEQ_PSEUDOBULK_FINETUNING_STEP}.",
    )
    phase023_parser.add_argument(
        "--phase2-single-cell-protein-finetuning-args",
        default="",
        help=f"Shell-style argument string forwarded to {SINGLE_CELL_PROTEIN_FINETUNING_STEP}.",
    )
    phase023_parser.add_argument(
        "--phase0-args",
        default="",
        help=f"Deprecated alias for --bulk-module-args.",
    )
    phase023_parser.add_argument(
        "--phase12-args",
        default="",
        help=f"Deprecated alias for --phase1-rnaseq-pseudo-bulk-finetuning-args.",
    )
    phase023_parser.add_argument(
        "--phase3-args",
        default="",
        help=f"Deprecated alias for --phase2-single-cell-protein-finetuning-args.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "phase0":
        run_phase0(args.script_args)
        return
    if args.command == "phase12":
        run_phase12(args.script_args)
        return
    if args.command == "phase3":
        run_phase3(args.script_args)
        return
    if args.command in {"scrnaseq_workflow", "phase023"}:
        bulk_module_args = (
            args.bulk_module_args if args.bulk_module_args else args.phase0_args
        )
        phase1_args = (
            args.phase1_rnaseq_pseudo_bulk_finetuning_args
            if args.phase1_rnaseq_pseudo_bulk_finetuning_args
            else args.phase12_args
        )
        phase2_args = (
            args.phase2_single_cell_protein_finetuning_args
            if args.phase2_single_cell_protein_finetuning_args
            else args.phase3_args
        )
        run_scrnaseq_workflow(
            bulk_module_args=shlex.split(bulk_module_args),
            phase1_rnaseq_pseudo_bulk_finetuning_args=shlex.split(phase1_args),
            phase2_single_cell_protein_finetuning_args=shlex.split(phase2_args),
        )
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
