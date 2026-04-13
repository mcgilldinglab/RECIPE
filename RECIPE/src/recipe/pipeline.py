from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .bulk_workflow import run_bulk_module
from .ppi_workflow import run_ppi_refinement
from .single_cell_riboseq_workflow import run_single_cell_transfer


def run_recipe_pipeline(
    modules: Iterable[str],
    output_root: str | Path,
    species: str = "human",
    condition: str = "KD",
    seed: int = 12,
    device_name: str | None = None,
) -> dict[str, object]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    normalized_modules = [module.upper() for module in modules]
    summary: dict[str, object] = {}

    if "A" in normalized_modules:
        summary["A"] = run_bulk_module(
            species=species,
            task="unknown",
            condition_name=condition,
            output_dir=output_root / "module_a",
            seed=seed,
            device_name=device_name,
        )
    if "B" in normalized_modules:
        summary["B"] = run_bulk_module(
            species=species,
            task="known",
            condition_name=condition,
            output_dir=output_root / "module_b",
            seed=seed,
            device_name=device_name,
        )
    if "C" in normalized_modules:
        summary["C"] = run_ppi_refinement(
            species=species,
            condition_name=condition,
            output_dir=output_root / "module_c",
            seed=seed,
            device_name=device_name,
        )
    if "D" in normalized_modules:
        summary["D"] = run_single_cell_transfer(
            output_dir=output_root / "module_d",
            seed=seed,
            device_name=device_name,
        )

    return summary
