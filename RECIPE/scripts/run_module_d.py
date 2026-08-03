from __future__ import annotations

import argparse
import json

from _bootstrap import add_src_to_path

add_src_to_path()

from recipe.single_cell_riboseq_workflow import run_single_cell_transfer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run RECIPE module D: single-cell transfer.")
    parser.add_argument("--steps", default="phase0,phase1,phase2")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-phase0", action="store_true")
    parser.add_argument("--train-phase1", action="store_true")
    parser.add_argument("--train-phase2", action="store_true")
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
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
