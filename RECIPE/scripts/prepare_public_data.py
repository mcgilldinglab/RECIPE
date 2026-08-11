from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _bootstrap import add_src_to_path

add_src_to_path()

from recipe.data_construction import build_coexpression_matrix, normalize_gene_by_cell_matrix
from recipe.utils import json_sanitize


REQUIRED_PUBLIC_INPUTS = (
    "bulk/mouse_reference.csv",
    "bulk/mouse_sequence_unknown.npy",
    "bulk/mouse_sequence_known.npy",
    "bulk/human_reference.csv",
    "bulk/single_cell_transfer_sequence.npy",
    "networks/mouse_ppi_unknown.csv",
    "networks/mouse_ppi_known.csv",
    "networks/single_cell_transfer_ppi.csv",
    "pausing/cds_annotations.csv",
    "pausing/human_nc2_pause.csv",
    "pausing/fraction_rich_pause.csv",
    "pausing/pseudobulk_pause_matrix.csv",
    "single_cell/expression_raw.csv",
    "single_cell/expression_normalized.csv",
    "single_cell/metadata.csv",
    "splits/bulk_mouse_unknown_seed12.csv",
    "splits/bulk_mouse_known_seed12.csv",
    "splits/single_cell_self_learning_seed12.csv",
    "splits/single_cell_module_a_seed42.csv",
    "splits/single_cell_graph_seed42.csv",
)

REQUIRED_PUBLIC_MODELS = (
    "bulk/mouse_unknown_seed1.pth",
    "bulk/mouse_known_seed5.pth",
    "single_cell/bulk_self_learning.pth",
)


def _is_lfs_pointer(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    with path.open("rb") as handle:
        prefix = handle.read(128)
    return prefix.startswith(b"version https://git-lfs.github.com/spec/v1")


def _file_status(root: Path, relative_path: str) -> dict[str, Any]:
    path = root / relative_path
    is_pointer = _is_lfs_pointer(path)
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": int(path.stat().st_size) if path.exists() else 0,
        "is_lfs_pointer": is_pointer,
        "ready": path.exists() and path.stat().st_size > 0 and not is_pointer,
    }


def _prepare_normalized_expression(data_root: Path, force: bool) -> dict[str, Any]:
    expression_csv = data_root / "single_cell" / "expression_raw.csv"
    output_csv = data_root / "single_cell" / "expression_normalized.csv"

    if output_csv.exists() and not force:
        return {"status": "present", "output_csv": str(output_csv)}
    if not expression_csv.exists():
        return {"status": "missing_raw_expression", "expression_csv": str(expression_csv)}

    return {
        "status": "generated",
        **normalize_gene_by_cell_matrix(
            expression_csv=expression_csv,
            output_csv=output_csv,
            target_sum=1e6,
        ),
    }


def _prepare_mouse_coexpression(data_root: Path, force: bool) -> dict[str, Any]:
    reference_csv = data_root / "bulk" / "mouse_reference.csv"
    output_csv = data_root / "networks" / "mouse_coexpression.csv"

    if output_csv.exists() and not force:
        return {"status": "present", "output_csv": str(output_csv)}
    if not reference_csv.exists():
        return {"status": "missing_reference", "reference_csv": str(reference_csv)}

    return {
        "status": "generated",
        **build_coexpression_matrix(
            reference_csv=reference_csv,
            output_csv=output_csv,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and validate data files for RECIPE public reproduction tasks.")
    parser.add_argument("--data-root", default="data", help="Directory containing RECIPE data subfolders.")
    parser.add_argument("--model-root", default="models", help="Directory containing RECIPE checkpoint subfolders.")
    parser.add_argument(
        "--build-mouse-coexpression",
        action="store_true",
        help="Build data/networks/mouse_coexpression.csv for Module C coexpression summaries.",
    )
    parser.add_argument("--force-normalize", action="store_true", help="Rebuild single_cell/expression_normalized.csv.")
    parser.add_argument("--force-coexpression", action="store_true", help="Rebuild networks/mouse_coexpression.csv.")
    parser.add_argument("--manifest-json", default=None, help="Optional JSON path for the preparation summary.")
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    model_root = Path(args.model_root).expanduser().resolve()
    data_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "data_root": str(data_root),
        "model_root": str(model_root),
        "normalized_expression": _prepare_normalized_expression(data_root, force=args.force_normalize),
        "mouse_coexpression": {"status": "skipped"},
    }
    if args.build_mouse_coexpression or args.force_coexpression:
        summary["mouse_coexpression"] = _prepare_mouse_coexpression(data_root, force=args.force_coexpression)

    required_status = [_file_status(data_root, relative_path) for relative_path in REQUIRED_PUBLIC_INPUTS]
    model_status = [_file_status(model_root, relative_path) for relative_path in REQUIRED_PUBLIC_MODELS]
    missing = [item for item in required_status if not item["ready"]]
    missing_models = [item for item in model_status if not item["ready"]]
    summary["required_inputs"] = required_status
    summary["required_models"] = model_status
    summary["missing_required_inputs"] = missing
    summary["missing_required_models"] = missing_models
    summary["ready"] = not missing and not missing_models

    if args.manifest_json is not None:
        manifest_json = Path(args.manifest_json).expanduser()
        manifest_json.parent.mkdir(parents=True, exist_ok=True)
        manifest_json.write_text(
            json.dumps(json_sanitize(summary), indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )

    print(json.dumps(json_sanitize(summary), indent=2, ensure_ascii=False, allow_nan=False))
    if missing or missing_models:
        missing_paths = ", ".join(item["path"] for item in [*missing, *missing_models])
        raise SystemExit(f"Missing required public reproduction inputs: {missing_paths}")


if __name__ == "__main__":
    main()
