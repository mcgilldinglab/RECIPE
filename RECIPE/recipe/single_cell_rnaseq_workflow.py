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


def run_phase023(
    phase0_args: Sequence[str] | None = None,
    phase12_args: Sequence[str] | None = None,
    phase3_args: Sequence[str] | None = None,
) -> None:
    run_phase0(phase0_args)
    run_phase12(phase12_args)
    run_phase3(phase3_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Unified RNA-seq single-cell workflow wrapper for the current ENSMUSP pipeline. "
            "This workflow keeps the training logic in the original phase0/phase12/phase3 "
            "scripts and centralizes entry points under RECIPE."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    phase0_parser = subparsers.add_parser("phase0", help="Run bulk/pseudobulk phase0 training.")
    phase0_parser.add_argument("script_args", nargs=argparse.REMAINDER, help="Args forwarded to phase0.")

    phase12_parser = subparsers.add_parser("phase12", help="Run phase1 export + phase2 training.")
    phase12_parser.add_argument("script_args", nargs=argparse.REMAINDER, help="Args forwarded to phase12.")

    phase3_parser = subparsers.add_parser("phase3", help="Run matched nanoSPINS phase3 training.")
    phase3_parser.add_argument("script_args", nargs=argparse.REMAINDER, help="Args forwarded to phase3.")

    phase023_parser = subparsers.add_parser(
        "phase023",
        help="Run phase0, phase12, and phase3 sequentially.",
    )
    phase023_parser.add_argument(
        "--phase0-args",
        default="",
        help="Shell-style argument string forwarded to phase0.",
    )
    phase023_parser.add_argument(
        "--phase12-args",
        default="",
        help="Shell-style argument string forwarded to phase12.",
    )
    phase023_parser.add_argument(
        "--phase3-args",
        default="",
        help="Shell-style argument string forwarded to phase3.",
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
    if args.command == "phase023":
        run_phase023(
            phase0_args=shlex.split(args.phase0_args),
            phase12_args=shlex.split(args.phase12_args),
            phase3_args=shlex.split(args.phase3_args),
        )
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
