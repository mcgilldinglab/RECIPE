from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from recipe.config import get_bulk_task_config
from recipe.data_construction import build_coexpression_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a coexpression matrix from the packaged reference table.")
    parser.add_argument("--species", choices=("human", "mouse"), required=True)
    parser.add_argument("--data-root", default=None, help="Directory containing RECIPE data subfolders.")
    parser.add_argument("--reference-csv", default=None, help="Reference CSV used to compute coexpression.")
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    if args.data_root is not None:
        data_root = Path(args.data_root).expanduser().resolve()
        reference_csv = data_root / "bulk" / f"{args.species}_reference.csv"
        output_csv = args.output_csv or data_root / "networks" / f"{args.species}_coexpression.csv"
    elif args.reference_csv is not None:
        reference_csv = Path(args.reference_csv).expanduser().resolve()
        if args.output_csv is None:
            parser.error("--output-csv is required when --reference-csv is used without --data-root.")
        output_csv = Path(args.output_csv).expanduser().resolve()
    else:
        config = get_bulk_task_config(task="known", species=args.species)
        reference_csv = config.reference_csv
        output_csv = args.output_csv or Path(config.ppi_csv).with_name(f"{args.species}_coexpression.csv")

    summary = build_coexpression_matrix(reference_csv=reference_csv, output_csv=output_csv)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
