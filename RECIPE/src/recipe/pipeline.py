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
    data_root: str | Path | None = None,
    model_root: str | Path | None = None,
    bulk_unknown_split_csv: str | Path | None = None,
    bulk_known_split_csv: str | Path | None = None,
    bulk_expression_col: str | None = None,
    bulk_target_col: str | None = None,
    bulk_pause_col: str | None = None,
    use_bulk_pause: bool = True,
    bulk_train: bool = False,
    bulk_max_epochs: int = 3000,
    bulk_patience: int = 200,
    bulk_learning_rate: float = 7e-2,
    edge_max_epochs: int = 1000,
    edge_patience: int = 50,
    train_phase0: bool = False,
    train_phase1: bool = False,
    train_phase2: bool = False,
    phase0_split_csv: str | Path | None = None,
    phase1_split_csv: str | Path | None = None,
    phase2_split_csv: str | Path | None = None,
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
            data_root=data_root,
            model_root=model_root,
            split_csv=bulk_unknown_split_csv,
            expression_col=bulk_expression_col,
            target_col=bulk_target_col,
            pause_col=bulk_pause_col,
            use_pause=use_bulk_pause,
            train=bulk_train,
            max_epochs=bulk_max_epochs,
            patience=bulk_patience,
            learning_rate=bulk_learning_rate,
        )
    if "B" in normalized_modules:
        summary["B"] = run_bulk_module(
            species=species,
            task="known",
            condition_name=condition,
            output_dir=output_root / "module_b",
            seed=seed,
            device_name=device_name,
            data_root=data_root,
            model_root=model_root,
            split_csv=bulk_known_split_csv,
            expression_col=bulk_expression_col,
            target_col=bulk_target_col,
            pause_col=bulk_pause_col,
            use_pause=use_bulk_pause,
            train=bulk_train,
            max_epochs=bulk_max_epochs,
            patience=bulk_patience,
            learning_rate=bulk_learning_rate,
        )
    if "C" in normalized_modules:
        module_b_checkpoint = output_root / "module_b" / "model.pth"
        bulk_checkpoint_path = module_b_checkpoint if module_b_checkpoint.exists() else None
        summary["C"] = run_ppi_refinement(
            species=species,
            condition_name=condition,
            output_dir=output_root / "module_c",
            seed=seed,
            device_name=device_name,
            bulk_checkpoint_path=bulk_checkpoint_path,
            data_root=data_root,
            model_root=model_root,
            expression_col=bulk_expression_col,
            target_col=bulk_target_col,
            pause_col=bulk_pause_col,
            use_pause=use_bulk_pause,
            edge_max_epochs=edge_max_epochs,
            edge_patience=edge_patience,
        )
    if "D" in normalized_modules:
        summary["D"] = run_single_cell_transfer(
            output_dir=output_root / "module_d",
            seed=seed,
            device_name=device_name,
            data_root=data_root,
            model_root=model_root,
            train_phase0=train_phase0,
            train_phase1=train_phase1,
            train_phase2=train_phase2,
            phase0_split_csv=phase0_split_csv,
            phase1_split_csv=phase1_split_csv,
            phase2_split_csv=phase2_split_csv,
        )

    return summary
