from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

from .assets import (
    BULK_DATA_DIR,
    DATA_ROOT,
    BULK_MODEL_DIR,
    NETWORK_DATA_DIR,
    PAUSING_DATA_DIR,
    PPI_MODEL_DIR,
    SINGLE_CELL_DATA_DIR,
    SINGLE_CELL_MODEL_DIR,
)
from .bulk_regression import BulkConditionSpec


@dataclass(frozen=True)
class BulkTaskConfig:
    species: str
    task: str
    reference_csv: Path
    sequence_npy: Path
    ppi_csv: Path
    default_checkpoint: Path | None
    conditions: Mapping[str, BulkConditionSpec]
    pause_csv: Path | None = None


@dataclass(frozen=True)
class SingleCellTransferConfig:
    bulk_reference_csv: Path
    transcript_order_csv: Path
    sequence_npy: Path
    ppi_csv: Path
    cds_csv: Path
    phase0_pause_csv: Path
    phase1_pause_csv: Path
    phase0_expression_col: str
    phase0_target_col: str
    phase0_pause_col: str
    phase0_init_checkpoint: Path | None
    expression_csv: Path
    expression_normalized_csv: Path
    metadata_csv: Path
    scriboseq_metadata_csv: Path
    pause_matrix_csv: Path
    bundled_cell_embeddings_npy: Path
    bundled_cell_outputs_npy: Path
    bundled_prediction_csv: Path
    bundled_prediction_seed123_csv: Path
    bundled_phase2_checkpoint: Path | None


HUMAN_KNOWN_CONDITIONS = {
    "NC": BulkConditionSpec("NC", "rNC2", "NC3", "High_Pause_Countsnc"),
    "KD": BulkConditionSpec("KD", "rKD2", "KD3", "High_Pause_Countskd"),
}

HUMAN_UNKNOWN_CONDITIONS = {
    "NC": BulkConditionSpec("NC", "rNC2", "NC3", "High_Pause_Countssc"),
    "KD": BulkConditionSpec("KD", "rKD2", "KD3", "High_Pause_Countssc"),
}

MOUSE_KNOWN_CONDITIONS = {
    "NC": BulkConditionSpec("NC", "rNC2", "NC3", "High_Pause_Countsc18nc"),
    "KD": BulkConditionSpec("KD", "rKD2", "KD3", "High_Pause_Countsc18ko"),
}

MOUSE_UNKNOWN_CONDITIONS = {
    "NC": BulkConditionSpec("NC", "rNC2", "NC3", "High_Pause_Countsc18nc"),
    "KD": BulkConditionSpec("KD", "rKD2", "KD3", "High_Pause_Countsc18ko"),
}

BULK_KNOWN_CONFIGS: dict[str, BulkTaskConfig] = {
    "human": BulkTaskConfig(
        species="human",
        task="known",
        reference_csv=BULK_DATA_DIR / "human_reference.csv",
        sequence_npy=BULK_DATA_DIR / "human_sequence_known.npy",
        ppi_csv=NETWORK_DATA_DIR / "human_ppi_known.csv",
        default_checkpoint=BULK_MODEL_DIR / "human_known_seed12.pth",
        conditions=HUMAN_KNOWN_CONDITIONS,
    ),
    "mouse": BulkTaskConfig(
        species="mouse",
        task="known",
        reference_csv=BULK_DATA_DIR / "mouse_reference.csv",
        sequence_npy=BULK_DATA_DIR / "mouse_sequence_known.npy",
        ppi_csv=NETWORK_DATA_DIR / "mouse_ppi_known.csv",
        default_checkpoint=BULK_MODEL_DIR / "mouse_known_seed5.pth",
        conditions=MOUSE_KNOWN_CONDITIONS,
    ),
}

BULK_UNKNOWN_CONFIGS: dict[str, BulkTaskConfig] = {
    "human": BulkTaskConfig(
        species="human",
        task="unknown",
        reference_csv=BULK_DATA_DIR / "human_reference.csv",
        sequence_npy=BULK_DATA_DIR / "human_sequence_unknown.npy",
        ppi_csv=NETWORK_DATA_DIR / "human_ppi_unknown.csv",
        default_checkpoint=BULK_MODEL_DIR / "human_unknown_seed0.pth",
        conditions=HUMAN_UNKNOWN_CONDITIONS,
        pause_csv=PAUSING_DATA_DIR / "human_scribo_pause.csv",
    ),
    "mouse": BulkTaskConfig(
        species="mouse",
        task="unknown",
        reference_csv=BULK_DATA_DIR / "mouse_reference.csv",
        sequence_npy=BULK_DATA_DIR / "mouse_sequence_unknown.npy",
        ppi_csv=NETWORK_DATA_DIR / "mouse_ppi_unknown.csv",
        default_checkpoint=BULK_MODEL_DIR / "mouse_unknown_seed1.pth",
        conditions=MOUSE_UNKNOWN_CONDITIONS,
    ),
}

SINGLE_CELL_TRANSFER_CONFIG = SingleCellTransferConfig(
    bulk_reference_csv=BULK_DATA_DIR / "human_reference.csv",
    transcript_order_csv=SINGLE_CELL_DATA_DIR / "expression_normalized.csv",
    sequence_npy=BULK_DATA_DIR / "single_cell_transfer_sequence.npy",
    ppi_csv=NETWORK_DATA_DIR / "single_cell_transfer_ppi.csv",
    cds_csv=PAUSING_DATA_DIR / "cds_annotations.csv",
    phase0_pause_csv=PAUSING_DATA_DIR / "human_nc2_pause.csv",
    phase1_pause_csv=PAUSING_DATA_DIR / "fraction_rich_pause.csv",
    phase0_expression_col="rNC2",
    phase0_target_col="NC3",
    phase0_pause_col="phase0_pause",
    phase0_init_checkpoint=SINGLE_CELL_MODEL_DIR / "bulk_self_learning.pth",
    expression_csv=SINGLE_CELL_DATA_DIR / "expression_raw.csv",
    expression_normalized_csv=SINGLE_CELL_DATA_DIR / "expression_normalized.csv",
    metadata_csv=SINGLE_CELL_DATA_DIR / "metadata.csv",
    scriboseq_metadata_csv=SINGLE_CELL_DATA_DIR / "scriboseq_metadata.csv",
    pause_matrix_csv=PAUSING_DATA_DIR / "pseudobulk_pause_matrix.csv",
    bundled_cell_embeddings_npy=SINGLE_CELL_DATA_DIR / "cell_embeddings.npy",
    bundled_cell_outputs_npy=SINGLE_CELL_DATA_DIR / "cell_outputs.npy",
    bundled_prediction_csv=SINGLE_CELL_DATA_DIR / "predicted_cell_matrix.csv",
    bundled_prediction_seed123_csv=SINGLE_CELL_DATA_DIR / "predicted_cell_matrix_seed123.csv",
    bundled_phase2_checkpoint=None,
)


def get_bulk_task_config(task: str, species: str) -> BulkTaskConfig:
    normalized_task = task.lower()
    normalized_species = species.lower()

    if normalized_task == "known":
        return BULK_KNOWN_CONFIGS[normalized_species]
    if normalized_task == "unknown":
        return BULK_UNKNOWN_CONFIGS[normalized_species]
    raise KeyError(f"Unsupported bulk task: {task}")


def _resolve_optional_path(path: str | Path | None) -> Path | None:
    return Path(path).expanduser().resolve() if path is not None else None


def _replace_data_root(path: Path | None, data_root: str | Path | None) -> Path | None:
    if path is None or data_root is None:
        return path
    root = Path(data_root).expanduser().resolve()
    try:
        relative = path.resolve().relative_to(DATA_ROOT.resolve())
    except ValueError:
        relative = Path(path.name)
    return root / relative


def with_bulk_input_paths(
    config: BulkTaskConfig,
    data_root: str | Path | None = None,
    reference_csv: str | Path | None = None,
    sequence_npy: str | Path | None = None,
    ppi_csv: str | Path | None = None,
    pause_csv: str | Path | None = None,
) -> BulkTaskConfig:
    """Return a bulk config with user-supplied input paths overriding package defaults."""

    updated = replace(
        config,
        reference_csv=_replace_data_root(config.reference_csv, data_root),
        sequence_npy=_replace_data_root(config.sequence_npy, data_root),
        ppi_csv=_replace_data_root(config.ppi_csv, data_root),
        pause_csv=_replace_data_root(config.pause_csv, data_root),
    )
    overrides = {
        "reference_csv": _resolve_optional_path(reference_csv),
        "sequence_npy": _resolve_optional_path(sequence_npy),
        "ppi_csv": _resolve_optional_path(ppi_csv),
        "pause_csv": _resolve_optional_path(pause_csv),
    }
    overrides = {key: value for key, value in overrides.items() if value is not None}
    return replace(updated, **overrides)


def with_single_cell_input_paths(
    config: SingleCellTransferConfig,
    data_root: str | Path | None = None,
    bulk_reference_csv: str | Path | None = None,
    transcript_order_csv: str | Path | None = None,
    sequence_npy: str | Path | None = None,
    ppi_csv: str | Path | None = None,
    cds_csv: str | Path | None = None,
    phase0_pause_csv: str | Path | None = None,
    phase1_pause_csv: str | Path | None = None,
    expression_csv: str | Path | None = None,
    expression_normalized_csv: str | Path | None = None,
    metadata_csv: str | Path | None = None,
    pause_matrix_csv: str | Path | None = None,
    phase0_init_checkpoint: str | Path | None = None,
) -> SingleCellTransferConfig:
    """Return a single-cell config with user-supplied paths overriding package defaults."""

    updated = replace(
        config,
        bulk_reference_csv=_replace_data_root(config.bulk_reference_csv, data_root),
        transcript_order_csv=_replace_data_root(config.transcript_order_csv, data_root),
        sequence_npy=_replace_data_root(config.sequence_npy, data_root),
        ppi_csv=_replace_data_root(config.ppi_csv, data_root),
        cds_csv=_replace_data_root(config.cds_csv, data_root),
        phase0_pause_csv=_replace_data_root(config.phase0_pause_csv, data_root),
        phase1_pause_csv=_replace_data_root(config.phase1_pause_csv, data_root),
        expression_csv=_replace_data_root(config.expression_csv, data_root),
        expression_normalized_csv=_replace_data_root(config.expression_normalized_csv, data_root),
        metadata_csv=_replace_data_root(config.metadata_csv, data_root),
        pause_matrix_csv=_replace_data_root(config.pause_matrix_csv, data_root),
    )
    overrides = {
        "bulk_reference_csv": _resolve_optional_path(bulk_reference_csv),
        "transcript_order_csv": _resolve_optional_path(transcript_order_csv),
        "sequence_npy": _resolve_optional_path(sequence_npy),
        "ppi_csv": _resolve_optional_path(ppi_csv),
        "cds_csv": _resolve_optional_path(cds_csv),
        "phase0_pause_csv": _resolve_optional_path(phase0_pause_csv),
        "phase1_pause_csv": _resolve_optional_path(phase1_pause_csv),
        "expression_csv": _resolve_optional_path(expression_csv),
        "expression_normalized_csv": _resolve_optional_path(expression_normalized_csv),
        "metadata_csv": _resolve_optional_path(metadata_csv),
        "pause_matrix_csv": _resolve_optional_path(pause_matrix_csv),
        "phase0_init_checkpoint": _resolve_optional_path(phase0_init_checkpoint),
    }
    overrides = {key: value for key, value in overrides.items() if value is not None}
    return replace(updated, **overrides)
