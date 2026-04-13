from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .assets import BULK_DATA_DIR, DATA_ROOT, NETWORK_DATA_DIR, PAUSING_DATA_DIR, SINGLE_CELL_DATA_DIR
from .bulk_regression import load_bulk_dataframe
from .config import get_bulk_task_config
from .utils import ensure_parent_dir, save_json


@dataclass(frozen=True)
class AliasSpec:
    source: Path
    target: Path


def _safe_link_or_copy(source: Path, target: Path) -> str:
    ensure_parent_dir(target)
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def alias_specs() -> list[AliasSpec]:
    return [
        AliasSpec(DATA_ROOT / "project_data/24077132kdncmergedf.csv", BULK_DATA_DIR / "human_reference.csv"),
        AliasSpec(DATA_ROOT / "project_data/24msc18allmergedDF7594.csv", BULK_DATA_DIR / "mouse_reference.csv"),
        AliasSpec(DATA_ROOT / "project_data/all_sequence_outputsnew7132.npy", BULK_DATA_DIR / "human_sequence_known.npy"),
        AliasSpec(DATA_ROOT / "project_data/all_sequence_outputs7132.npy", BULK_DATA_DIR / "human_sequence_unknown.npy"),
        AliasSpec(DATA_ROOT / "project_data/all_sequence_outputsnewms7594.npy", BULK_DATA_DIR / "mouse_sequence_known.npy"),
        AliasSpec(DATA_ROOT / "project_data/all_sequence_outputsnewms7594.npy", BULK_DATA_DIR / "mouse_sequence_unknown.npy"),
        AliasSpec(DATA_ROOT / "project_data/all_sequence_outputsnewbulk11619.npy", BULK_DATA_DIR / "single_cell_transfer_sequence.npy"),
        AliasSpec(DATA_ROOT / "project_data/9606ppi_matrix.csv", NETWORK_DATA_DIR / "human_ppi_known.csv"),
        AliasSpec(DATA_ROOT / "project_data/ppi_ebi_string_ppi3ensp_lr_IntAct_corummatrix4p_p.csv", NETWORK_DATA_DIR / "human_ppi_unknown.csv"),
        AliasSpec(DATA_ROOT / "project_data/msppi_matrixppiebilr2BIOGRID_corum_uniport.csv", NETWORK_DATA_DIR / "mouse_ppi_known.csv"),
        AliasSpec(DATA_ROOT / "project_data/msppi_matrixppiebilr2BIOGRID_corum_uniport.csv", NETWORK_DATA_DIR / "mouse_ppi_unknown.csv"),
        AliasSpec(DATA_ROOT / "project_data/ppi_ebi_string_ppi3ensp_lr_IntAct_corummatrix4p_pbulk11619.csv", NETWORK_DATA_DIR / "single_cell_transfer_ppi.csv"),
        AliasSpec(DATA_ROOT / "project_data/240917co_expression_network_matrix.csv", NETWORK_DATA_DIR / "human_coexpression.csv"),
        AliasSpec(DATA_ROOT / "project_data/240917co_expression_network_matrixms.csv", NETWORK_DATA_DIR / "mouse_coexpression.csv"),
        AliasSpec(DATA_ROOT / "pausing/cds_df38510.csv", PAUSING_DATA_DIR / "cds_annotations.csv"),
        AliasSpec(DATA_ROOT / "pausing/pause_scorescdsallnewnohupNC1_38510FINAL.csv", PAUSING_DATA_DIR / "human_nc1_pause.csv"),
        AliasSpec(DATA_ROOT / "pausing/pause_scorescdsallnewnohupNC2_38510FINAL.csv", PAUSING_DATA_DIR / "human_nc2_pause.csv"),
        AliasSpec(DATA_ROOT / "pausing/pause_scorescdsallscribo293SRR13125084_withoutcontam_AlignedsortedByCoorddedup3sb.csv", PAUSING_DATA_DIR / "human_scribo_pause.csv"),
        AliasSpec(DATA_ROOT / "pausing/pause_scorescdsallscribo293Rich_dedup3sball.csv", PAUSING_DATA_DIR / "fraction_rich_pause.csv"),
        AliasSpec(DATA_ROOT / "pausing/pause_scorescdsallscribo293Leu6h_dedup3sball.csv", PAUSING_DATA_DIR / "fraction_leu6h_pause.csv"),
        AliasSpec(DATA_ROOT / "pausing/pause_scorescdsallscribo293Leu3h_dedup3sball.csv", PAUSING_DATA_DIR / "fraction_leu3h_pause.csv"),
        AliasSpec(DATA_ROOT / "pausing/pause_scorescdsallscribo293Arg3h_dedup3sball.csv", PAUSING_DATA_DIR / "fraction_arg3h_pause.csv"),
        AliasSpec(DATA_ROOT / "pausing/pause_scorescdsallscribo293Arg6h_dedup3sball.csv", PAUSING_DATA_DIR / "fraction_arg6h_pause.csv"),
        AliasSpec(DATA_ROOT / "pausing/data/250429scribonew11619_422.csv", PAUSING_DATA_DIR / "pseudobulk_pause_matrix.csv"),
        AliasSpec(DATA_ROOT / "project_data/sc11619genes422cell.csv", SINGLE_CELL_DATA_DIR / "expression_raw.csv"),
        AliasSpec(DATA_ROOT / "project_data/sc11619genes422cell_normalized.csv", SINGLE_CELL_DATA_DIR / "expression_normalized.csv"),
        AliasSpec(DATA_ROOT / "project_root/brforepridictmeta_dataall.csv", SINGLE_CELL_DATA_DIR / "metadata.csv"),
        AliasSpec(DATA_ROOT / "project_data/GSE162060_HEK293Tscriboseq_meta.csv", SINGLE_CELL_DATA_DIR / "scriboseq_metadata.csv"),
        AliasSpec(DATA_ROOT / "project_root/predicted_expression_421_11619_250504.csv", SINGLE_CELL_DATA_DIR / "predicted_cell_matrix.csv"),
        AliasSpec(DATA_ROOT / "project_root/predicted_expression_421_11619_250504s123.csv", SINGLE_CELL_DATA_DIR / "predicted_cell_matrix_seed123.csv"),
        AliasSpec(DATA_ROOT / "project_data/all_z_array_0503test.npy", SINGLE_CELL_DATA_DIR / "cell_embeddings.npy"),
        AliasSpec(DATA_ROOT / "project_data/all_y_array_0503test.npy", SINGLE_CELL_DATA_DIR / "cell_outputs.npy"),
    ]


def build_data_aliases(output_manifest_json: str | Path | None = None) -> dict[str, Any]:
    manifest: list[dict[str, str]] = []
    for spec in alias_specs():
        if not spec.source.exists():
            raise FileNotFoundError(f"Missing alias source: {spec.source}")
        mode = _safe_link_or_copy(spec.source, spec.target)
        manifest.append(
            {
                "source": str(spec.source),
                "target": str(spec.target),
                "mode": mode,
            }
        )

    summary = {"alias_count": len(manifest), "items": manifest}
    if output_manifest_json is not None:
        save_json(output_manifest_json, summary)
    return summary


def detect_expression_columns(reference_df: pd.DataFrame) -> list[str]:
    pattern = re.compile(r"^(?:r)?(?:NC|KD)\d+$")
    return [column for column in reference_df.columns if pattern.match(str(column))]


def build_coexpression_matrix(
    reference_csv: str | Path,
    output_csv: str | Path,
    expression_columns: Iterable[str] | None = None,
) -> dict[str, Any]:
    reference_df = pd.read_csv(reference_csv)
    selected_columns = list(expression_columns) if expression_columns is not None else detect_expression_columns(reference_df)
    if not selected_columns:
        raise ValueError("No expression columns were detected for coexpression construction.")

    value_matrix = reference_df[selected_columns].to_numpy(dtype=np.float32)
    coexpression = np.corrcoef(value_matrix)
    coexpression = np.nan_to_num(coexpression, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    output_csv = Path(output_csv)
    ensure_parent_dir(output_csv)
    pd.DataFrame(coexpression).to_csv(output_csv, index=False)
    return {
        "expression_columns": selected_columns,
        "node_count": int(coexpression.shape[0]),
        "output_csv": str(output_csv),
    }


def build_bulk_feature_table(
    species: str,
    task: str,
    output_csv: str | Path,
) -> dict[str, Any]:
    config = get_bulk_task_config(task=task, species=species)
    pause_col_name = None
    if config.pause_csv is not None:
        first_condition = next(iter(config.conditions.values()))
        pause_col_name = first_condition.pause_col

    bulk_df = load_bulk_dataframe(
        reference_csv_path=config.reference_csv,
        pause_csv_path=config.pause_csv,
        pause_col_name=pause_col_name,
    )
    output_csv = Path(output_csv)
    ensure_parent_dir(output_csv)
    bulk_df.to_csv(output_csv, index=False)
    return {
        "species": species,
        "task": task,
        "rows": int(len(bulk_df)),
        "columns": list(bulk_df.columns),
        "output_csv": str(output_csv),
    }


def normalize_gene_by_cell_matrix(
    expression_csv: str | Path,
    output_csv: str | Path,
    target_sum: float = 1e6,
    gene_id_col: str | None = None,
    log1p: bool = False,
) -> dict[str, Any]:
    expression_df = pd.read_csv(expression_csv)
    gene_column = gene_id_col or expression_df.columns[0]
    value_df = expression_df.drop(columns=[gene_column]).astype(np.float32)
    column_sums = value_df.sum(axis=0).replace(0, np.nan)
    normalized = value_df.divide(column_sums, axis=1) * float(target_sum)
    normalized = normalized.fillna(0.0)
    if log1p:
        normalized = np.log1p(normalized)

    normalized_df = pd.concat([expression_df[[gene_column]].copy(), normalized], axis=1)
    output_csv = Path(output_csv)
    ensure_parent_dir(output_csv)
    normalized_df.to_csv(output_csv, index=False)
    return {
        "target_sum": float(target_sum),
        "log1p": bool(log1p),
        "shape": [int(normalized_df.shape[0]), int(normalized_df.shape[1])],
        "output_csv": str(output_csv),
    }


def _load_fraction_pause_profile(pause_csv: str | Path) -> pd.DataFrame:
    pause_df = pd.read_csv(pause_csv)
    transcript_column = pause_df.columns[2]
    value_column = pause_df.columns[1]
    result = pause_df[[transcript_column, value_column]].copy()
    result.columns = ["transcript_id", "pause_value"]
    result["transcript_id"] = result["transcript_id"].astype(str).str.split(".").str[0]
    result = result.drop_duplicates(subset=["transcript_id"]).fillna(0.0)
    return result


def build_fraction_pause_table(output_csv: str | Path) -> dict[str, Any]:
    fraction_sources = {
        "Rich": PAUSING_DATA_DIR / "fraction_rich_pause.csv",
        "Leu6h": PAUSING_DATA_DIR / "fraction_leu6h_pause.csv",
        "Leu3h": PAUSING_DATA_DIR / "fraction_leu3h_pause.csv",
        "Arg3h": PAUSING_DATA_DIR / "fraction_arg3h_pause.csv",
        "Arg6h": PAUSING_DATA_DIR / "fraction_arg6h_pause.csv",
    }

    merged_df: pd.DataFrame | None = None
    for fraction_name, pause_csv in fraction_sources.items():
        pause_profile = _load_fraction_pause_profile(pause_csv).rename(columns={"pause_value": fraction_name})
        merged_df = pause_profile if merged_df is None else merged_df.merge(pause_profile, on="transcript_id", how="outer")

    assert merged_df is not None
    merged_df = merged_df.fillna(0.0)
    output_csv = Path(output_csv)
    ensure_parent_dir(output_csv)
    merged_df.to_csv(output_csv, index=False)
    return {
        "fractions": list(fraction_sources.keys()),
        "transcript_count": int(len(merged_df)),
        "output_csv": str(output_csv),
    }


def build_pseudobulk_pause_matrix(
    metadata_csv: str | Path,
    fraction_pause_csv: str | Path,
    output_csv: str | Path,
    transcript_order_csv: str | Path | None = None,
    cell_name_col: str = "cell_names",
    fraction_col: str = "fraction",
) -> dict[str, Any]:
    metadata_df = pd.read_csv(metadata_csv)
    fraction_df = pd.read_csv(fraction_pause_csv)

    if cell_name_col not in metadata_df.columns or fraction_col not in metadata_df.columns:
        raise KeyError(f"Metadata must contain '{cell_name_col}' and '{fraction_col}'.")

    fraction_df["transcript_id"] = fraction_df["transcript_id"].astype(str).str.split(".").str[0]
    if transcript_order_csv is not None:
        order_df = pd.read_csv(transcript_order_csv)
        order_col = order_df.columns[0]
        ordered_ids = order_df[order_col].astype(str).str.split(".").str[0]
        fraction_df = fraction_df.set_index("transcript_id").reindex(ordered_ids).fillna(0.0).reset_index()
        fraction_df.columns = ["transcript_id"] + list(fraction_df.columns[1:])

    pause_columns: dict[str, Any] = {"transcript_id": fraction_df["transcript_id"].astype(str).to_numpy()}
    fraction_lookup = fraction_df.set_index("transcript_id")

    for _, row in metadata_df[[cell_name_col, fraction_col]].drop_duplicates(subset=[cell_name_col]).iterrows():
        cell_name = str(row[cell_name_col])
        fraction_name = str(row[fraction_col])
        if fraction_name not in fraction_lookup.columns:
            raise KeyError(f"Unknown fraction '{fraction_name}' for cell '{cell_name}'.")
        pause_columns[cell_name] = fraction_lookup[fraction_name].to_numpy(dtype=np.float32)

    pause_matrix = pd.DataFrame(pause_columns)

    output_csv = Path(output_csv)
    ensure_parent_dir(output_csv)
    pause_matrix.to_csv(output_csv, index=False)
    return {
        "cell_count": int(metadata_df[cell_name_col].nunique()),
        "transcript_count": int(len(pause_matrix)),
        "output_csv": str(output_csv),
    }
