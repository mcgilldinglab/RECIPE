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
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
