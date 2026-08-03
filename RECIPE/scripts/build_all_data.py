from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from recipe.config import SINGLE_CELL_TRANSFER_CONFIG, get_bulk_task_config
from recipe.data_construction import (
    build_bulk_feature_table,
    build_coexpression_matrix,
    build_data_aliases,
    normalize_gene_by_cell_matrix,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the lightweight packaged data-build steps.")
    parser.add_argument("--rebuild-aliases", action="store_true")
    parser.add_argument("--manifest-json", default="data/alias_manifest.json")
    args = parser.parse_args()

    summary = {}
    if args.rebuild_aliases:
        summary["aliases"] = build_data_aliases(output_manifest_json=args.manifest_json)

    summary["bulk_features"] = {}
    for species in ("human", "mouse"):
        for task in ("known", "unknown"):
            output_csv = Path("data") / "bulk" / f"{species}_{task}_features.csv"
            summary["bulk_features"][f"{species}_{task}"] = build_bulk_feature_table(species, task, output_csv)
        config = get_bulk_task_config(task="known", species=species)
        summary[f"{species}_coexpression"] = build_coexpression_matrix(
            reference_csv=config.reference_csv,
            output_csv=Path(config.ppi_csv).with_name(f"{species}_coexpression.csv"),
        )

    summary["single_cell_normalized"] = normalize_gene_by_cell_matrix(
        expression_csv=SINGLE_CELL_TRANSFER_CONFIG.expression_csv,
        output_csv=SINGLE_CELL_TRANSFER_CONFIG.expression_normalized_csv,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
