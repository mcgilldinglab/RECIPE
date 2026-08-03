from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from recipe.config import SINGLE_CELL_TRANSFER_CONFIG
from recipe.data_construction import normalize_gene_by_cell_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize the packaged single-cell expression matrix.")
    parser.add_argument("--expression-csv", default=str(SINGLE_CELL_TRANSFER_CONFIG.expression_csv))
    parser.add_argument("--output-csv", default=str(SINGLE_CELL_TRANSFER_CONFIG.expression_normalized_csv))
    parser.add_argument("--target-sum", type=float, default=1e6)
    parser.add_argument("--log1p", action="store_true")
    args = parser.parse_args()
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    summary = normalize_gene_by_cell_matrix(
        expression_csv=args.expression_csv,
        output_csv=args.output_csv,
        target_sum=args.target_sum,
        log1p=args.log1p,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
