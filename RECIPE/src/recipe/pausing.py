from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .utils import ensure_parent_dir


PROTEIN_COLUMN_CANDIDATES = ("protein_id", "protein", "ENSP")
TRANSCRIPT_COLUMN_CANDIDATES = ("transcript_id", "transcript_id_x", "Transcript_ID")


def _import_pysam():
    try:
        import pysam
    except ImportError as exc:
        raise ImportError("Pausing calculation requires pysam. Install it with `python -m pip install pysam`.") from exc
    return pysam


def detect_column(df: pd.DataFrame, candidates: Iterable[str], requested: str | None = None) -> str:
    if requested is not None:
        if requested not in df.columns:
            raise KeyError(f"Column '{requested}' is missing.")
        return requested

    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise KeyError(f"None of the candidate columns are present: {', '.join(candidates)}")


def parse_interval_column(value: Any) -> list[int]:
    if pd.isna(value):
        return []
    return [int(float(item)) for item in str(value).split(";") if str(item).strip()]


def trimmed_cds_segments(
    starts: list[int],
    ends: list[int],
    trim_start_nt: int = 60,
    trim_end_nt: int = 60,
) -> list[tuple[int, int]]:
    if len(starts) != len(ends):
        raise ValueError(f"Start and End segment counts differ: {len(starts)} != {len(ends)}")
    if not starts:
        return []

    trimmed_starts = list(starts)
    trimmed_ends = list(ends)
    trimmed_starts[0] += int(trim_start_nt)
    trimmed_ends[-1] -= int(trim_end_nt)
    return [(start, end) for start, end in zip(trimmed_starts, trimmed_ends) if start < end]


def _denominator_codons(
    row: pd.Series,
    position_count: int,
    length_col: str | None,
    codon_step: int,
    average_denominator: str,
) -> float:
    if average_denominator == "length" and length_col and length_col in row.index:
        try:
            length_value = float(row[length_col])
        except (TypeError, ValueError):
            length_value = 0.0
        if np.isfinite(length_value) and length_value > 0:
            return length_value / float(codon_step)

    return float(position_count)


def _pause_records_for_row(
    row: pd.Series,
    bam_file: Any,
    protein_col: str,
    reference_col: str,
    start_col: str,
    end_col: str,
    length_col: str | None,
    trim_start_nt: int,
    trim_end_nt: int,
    codon_step: int,
    count_width: int,
    average_denominator: str,
) -> tuple[list[dict[str, Any]], str | None]:
    starts = parse_interval_column(row[start_col])
    ends = parse_interval_column(row[end_col])
    if not starts or not ends:
        return [], "empty_cds"

    try:
        segments = trimmed_cds_segments(starts, ends, trim_start_nt=trim_start_nt, trim_end_nt=trim_end_nt)
    except ValueError:
        return [], "invalid_segments"
    if not segments:
        return [], "empty_trimmed_cds"

    reference_name = str(row[reference_col])
    protein_id = row[protein_col]
    position_counts: list[tuple[int, int]] = []
    total_reads = 0

    for start, end in segments:
        for position in range(start, end, codon_step):
            reads_at_position = int(bam_file.count(reference=reference_name, start=position, end=position + count_width))
            position_counts.append((position, reads_at_position))
            total_reads += reads_at_position

    if not position_counts:
        return [], "empty_positions"
    if total_reads == 0:
        return [], "zero_reads"

    denominator = _denominator_codons(
        row=row,
        position_count=len(position_counts),
        length_col=length_col,
        codon_step=codon_step,
        average_denominator=average_denominator,
    )
    if denominator <= 0:
        return [], "zero_denominator"

    average_count = total_reads / denominator
    if average_count <= 0:
        return [], "zero_average"

    return [
        {
            "ENSP": protein_id,
            "Position": int(position),
            "Pause_Score": float(reads_at_position / average_count),
        }
        for position, reads_at_position in position_counts
    ], None


def iter_pause_score_records(
    cds_df: pd.DataFrame,
    bam_path: str | Path,
    protein_col: str | None = None,
    reference_col: str = "seqnames",
    start_col: str = "Start",
    end_col: str = "End",
    length_col: str | None = "Length",
    trim_start_nt: int = 60,
    trim_end_nt: int = 60,
    codon_step: int = 3,
    count_width: int = 3,
    average_denominator: str = "length",
    max_records: int | None = None,
):
    resolved_protein_col = detect_column(cds_df, PROTEIN_COLUMN_CANDIDATES, protein_col)
    for required_col in (reference_col, start_col, end_col):
        if required_col not in cds_df.columns:
            raise KeyError(f"Column '{required_col}' is missing from the CDS table.")

    pysam = _import_pysam()
    emitted = 0
    with pysam.AlignmentFile(str(bam_path), "rb") as bam_file:
        for _, row in cds_df.iterrows():
            records, _ = _pause_records_for_row(
                row=row,
                bam_file=bam_file,
                protein_col=resolved_protein_col,
                reference_col=reference_col,
                start_col=start_col,
                end_col=end_col,
                length_col=length_col,
                trim_start_nt=trim_start_nt,
                trim_end_nt=trim_end_nt,
                codon_step=codon_step,
                count_width=count_width,
                average_denominator=average_denominator,
            )
            for record in records:
                yield record
                emitted += 1
                if max_records is not None and emitted >= max_records:
                    return


def compute_pause_scores_from_bam(
    cds_df: pd.DataFrame,
    bam_path: str | Path,
    **kwargs,
) -> pd.DataFrame:
    return pd.DataFrame(iter_pause_score_records(cds_df=cds_df, bam_path=bam_path, **kwargs))


def write_pause_scores_from_bam(
    cds_csv: str | Path,
    bam_path: str | Path,
    output_csv: str | Path,
    protein_col: str | None = None,
    reference_col: str = "seqnames",
    start_col: str = "Start",
    end_col: str = "End",
    length_col: str | None = "Length",
    trim_start_nt: int = 60,
    trim_end_nt: int = 60,
    codon_step: int = 3,
    count_width: int = 3,
    average_denominator: str = "length",
    chunk_size: int = 100_000,
    max_records: int | None = None,
) -> dict[str, Any]:
    cds_df = pd.read_csv(cds_csv, dtype={reference_col: str}, low_memory=False)
    resolved_protein_col = detect_column(cds_df, PROTEIN_COLUMN_CANDIDATES, protein_col)

    output_csv = ensure_parent_dir(output_csv)
    rows: list[dict[str, Any]] = []
    written_rows = 0
    first_chunk = True
    for record in iter_pause_score_records(
        cds_df=cds_df,
        bam_path=bam_path,
        protein_col=resolved_protein_col,
        reference_col=reference_col,
        start_col=start_col,
        end_col=end_col,
        length_col=length_col,
        trim_start_nt=trim_start_nt,
        trim_end_nt=trim_end_nt,
        codon_step=codon_step,
        count_width=count_width,
        average_denominator=average_denominator,
        max_records=max_records,
    ):
        rows.append(record)
        if len(rows) >= chunk_size:
            pd.DataFrame(rows).to_csv(output_csv, index=False, mode="w" if first_chunk else "a", header=first_chunk)
            written_rows += len(rows)
            rows = []
            first_chunk = False

    if rows or first_chunk:
        pd.DataFrame(rows, columns=["ENSP", "Position", "Pause_Score"]).to_csv(
            output_csv,
            index=False,
            mode="w" if first_chunk else "a",
            header=first_chunk,
        )
        written_rows += len(rows)

    return {
        "cds_csv": str(cds_csv),
        "bam_path": str(bam_path),
        "output_csv": str(output_csv),
        "protein_col": resolved_protein_col,
        "reference_col": reference_col,
        "score_rows": int(written_rows),
        "trim_start_nt": int(trim_start_nt),
        "trim_end_nt": int(trim_end_nt),
        "average_denominator": average_denominator,
    }


def summarize_high_pause_counts(
    scores_df: pd.DataFrame,
    id_col: str = "ENSP",
    score_col: str = "Pause_Score",
    group_cols: list[str] | None = None,
    threshold: float = 3.3,
    threshold_mode: str = "absolute",
    output_id_col: str = "protein_id",
    count_col: str = "High_Pause_Counts",
) -> pd.DataFrame:
    group_cols = [id_col] if group_cols is None else list(group_cols)
    missing_cols = [column for column in group_cols + [score_col] if column not in scores_df.columns]
    if missing_cols:
        raise KeyError(f"Missing score columns: {', '.join(missing_cols)}")

    work_df = scores_df[group_cols + [score_col]].copy()
    work_df[score_col] = work_df[score_col].astype(float)
    if threshold_mode == "absolute":
        work_df["_is_high_pause"] = work_df[score_col] > float(threshold)
    elif threshold_mode == "relative_to_mean":
        means = work_df.groupby(group_cols, dropna=False)[score_col].transform("mean")
        work_df["_is_high_pause"] = work_df[score_col] > (float(threshold) * means)
    else:
        raise ValueError("threshold_mode must be 'absolute' or 'relative_to_mean'.")

    summary_df = work_df.groupby(group_cols, dropna=False)["_is_high_pause"].sum().reset_index()
    summary_df = summary_df.rename(columns={"_is_high_pause": count_col, id_col: output_id_col})
    summary_df[count_col] = summary_df[count_col].astype(int)
    return summary_df


def _combine_grouped_counts(existing: pd.Series | None, new_counts: pd.Series) -> pd.Series:
    if existing is None:
        return new_counts
    return existing.add(new_counts, fill_value=0)


def summarize_high_pause_counts_csv(
    scores_csv: str | Path,
    output_csv: str | Path,
    id_col: str = "ENSP",
    score_col: str = "Pause_Score",
    group_cols: list[str] | None = None,
    threshold: float = 3.3,
    threshold_mode: str = "absolute",
    output_id_col: str = "protein_id",
    count_col: str = "High_Pause_Counts",
    cds_csv: str | Path | None = None,
    protein_col: str | None = None,
    transcript_col: str | None = None,
    chunksize: int = 1_000_000,
) -> dict[str, Any]:
    group_cols = [id_col] if group_cols is None else list(group_cols)
    high_counts: pd.Series | None = None

    if threshold_mode == "absolute":
        for chunk_df in pd.read_csv(scores_csv, chunksize=chunksize):
            summary_df = summarize_high_pause_counts(
                chunk_df,
                id_col=id_col,
                score_col=score_col,
                group_cols=group_cols,
                threshold=threshold,
                threshold_mode=threshold_mode,
                output_id_col=id_col,
                count_col=count_col,
            )
            chunk_counts = summary_df.set_index(group_cols)[count_col]
            high_counts = _combine_grouped_counts(high_counts, chunk_counts)
    elif threshold_mode == "relative_to_mean":
        score_sums: pd.Series | None = None
        score_counts: pd.Series | None = None
        for chunk_df in pd.read_csv(scores_csv, chunksize=chunksize):
            chunk_df[score_col] = chunk_df[score_col].astype(float)
            grouped = chunk_df.groupby(group_cols, dropna=False)[score_col]
            score_sums = _combine_grouped_counts(score_sums, grouped.sum())
            score_counts = _combine_grouped_counts(score_counts, grouped.count())
        if score_sums is None or score_counts is None:
            means_df = pd.DataFrame(columns=group_cols + ["_mean"])
        else:
            means_df = (score_sums / score_counts).reset_index(name="_mean")

        for chunk_df in pd.read_csv(scores_csv, chunksize=chunksize):
            chunk_df[score_col] = chunk_df[score_col].astype(float)
            chunk_df = chunk_df.merge(means_df, on=group_cols, how="left")
            chunk_df["_is_high_pause"] = chunk_df[score_col] > (float(threshold) * chunk_df["_mean"])
            chunk_counts = chunk_df.groupby(group_cols, dropna=False)["_is_high_pause"].sum()
            high_counts = _combine_grouped_counts(high_counts, chunk_counts)
    else:
        raise ValueError("threshold_mode must be 'absolute' or 'relative_to_mean'.")

    if high_counts is None:
        summary_df = pd.DataFrame(columns=group_cols + [count_col])
    else:
        summary_df = high_counts.reset_index(name=count_col)
        summary_df[count_col] = summary_df[count_col].astype(int)
    summary_df = summary_df.rename(columns={id_col: output_id_col})

    if cds_csv is not None:
        cds_df = pd.read_csv(cds_csv, low_memory=False)
        summary_df = add_transcript_ids(
            summary_df,
            cds_df=cds_df,
            summary_protein_col=output_id_col,
            protein_col=protein_col,
            transcript_col=transcript_col,
        )

    output_csv = ensure_parent_dir(output_csv)
    summary_df.to_csv(output_csv, index=False)
    return {
        "scores_csv": str(scores_csv),
        "output_csv": str(output_csv),
        "rows": int(len(summary_df)),
        "group_cols": group_cols,
        "threshold": float(threshold),
        "threshold_mode": threshold_mode,
        "count_col": count_col,
        "cds_csv": str(cds_csv) if cds_csv is not None else None,
    }


def add_transcript_ids(
    summary_df: pd.DataFrame,
    cds_df: pd.DataFrame,
    summary_protein_col: str = "protein_id",
    protein_col: str | None = None,
    transcript_col: str | None = None,
) -> pd.DataFrame:
    resolved_protein_col = detect_column(cds_df, PROTEIN_COLUMN_CANDIDATES, protein_col)
    resolved_transcript_col = detect_column(cds_df, TRANSCRIPT_COLUMN_CANDIDATES, transcript_col)
    transcript_lookup = cds_df[[resolved_protein_col, resolved_transcript_col]].drop_duplicates(subset=[resolved_protein_col])
    transcript_lookup = transcript_lookup.rename(
        columns={
            resolved_protein_col: summary_protein_col,
            resolved_transcript_col: "transcript_id",
        }
    )
    return summary_df.merge(transcript_lookup, on=summary_protein_col, how="left")


def pivot_pause_counts(
    summary_df: pd.DataFrame,
    index_col: str = "protein_id",
    column_col: str = "CB",
    value_col: str = "High_Pause_Counts",
    fill_value: float = 0.0,
) -> pd.DataFrame:
    if column_col not in summary_df.columns:
        raise KeyError(f"Column '{column_col}' is missing and cannot be used for pivoting.")
    pivot_df = summary_df.pivot_table(index=index_col, columns=column_col, values=value_col, fill_value=fill_value)
    return pivot_df.reset_index()
