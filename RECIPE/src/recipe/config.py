from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .assets import (
    BULK_DATA_DIR,
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
