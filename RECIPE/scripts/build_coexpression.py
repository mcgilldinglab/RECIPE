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
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()
    config = get_bulk_task_config(task="known", species=args.species)
    output_csv = args.output_csv or str(Path(config.ppi_csv).with_name(f"{args.species}_coexpression.csv"))
    summary = build_coexpression_matrix(reference_csv=config.reference_csv, output_csv=output_csv)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
