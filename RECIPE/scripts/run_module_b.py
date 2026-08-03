from __future__ import annotations

import argparse
import json

from _bootstrap import add_src_to_path

add_src_to_path()

from recipe.bulk_workflow import run_bulk_module


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run RECIPE module B: known bulk protein prediction.")
    parser.add_argument("--species", choices=("human", "mouse"), default="mouse")
    parser.add_argument("--condition", default="KD", help="Condition name, for example NC or KD.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train", action="store_true", help="Force model training even if a checkpoint exists.")
    parser.add_argument("--checkpoint-path", default=None)
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
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
