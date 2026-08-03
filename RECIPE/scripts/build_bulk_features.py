from __future__ import annotations

import argparse
import json

from _bootstrap import add_src_to_path

add_src_to_path()

from recipe.data_construction import build_bulk_feature_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a packaged RECIPE bulk feature table.")
    parser.add_argument("--species", choices=("human", "mouse"), required=True)
    parser.add_argument("--task", choices=("known", "unknown"), required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()
    summary = build_bulk_feature_table(species=args.species, task=args.task, output_csv=args.output_csv)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
