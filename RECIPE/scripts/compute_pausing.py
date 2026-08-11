from __future__ import annotations

import argparse
import json

import pandas as pd

from _bootstrap import add_src_to_path

add_src_to_path()

from recipe.pausing import (
    pivot_pause_counts,
    summarize_high_pause_counts_csv,
    write_pause_scores_from_bam,
)
from recipe.utils import json_sanitize


def _split_columns(value: str | None) -> list[str] | None:
    if value is None:
        return None
    columns = [item.strip() for item in value.split(",") if item.strip()]
    return columns or None


def _add_cds_score_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cds-csv", required=True, help="CDS annotation CSV with Start, End, Length, seqnames, and protein columns.")
    parser.add_argument("--bam", required=True, help="Coordinate-sorted and indexed BAM file.")
    parser.add_argument("--score-csv", required=True, help="Output CSV for per-position pause scores.")
    parser.add_argument("--protein-col", default=None, help="Protein identifier column. Auto-detects protein_id, protein, or ENSP.")
    parser.add_argument("--reference-col", default="seqnames", help="Reference/chromosome column in the CDS CSV.")
    parser.add_argument("--start-col", default="Start", help="Semicolon-separated CDS start column.")
    parser.add_argument("--end-col", default="End", help="Semicolon-separated CDS end column.")
    parser.add_argument("--length-col", default="Length", help="CDS length column used for average read-depth normalization.")
    parser.add_argument("--trim-start-nt", type=int, default=60, help="Nucleotides trimmed from the first CDS segment.")
    parser.add_argument("--trim-end-nt", type=int, default=60, help="Nucleotides trimmed from the last CDS segment.")
    parser.add_argument("--codon-step", type=int, default=3)
    parser.add_argument("--count-width", type=int, default=3)
    parser.add_argument(
        "--average-denominator",
        choices=("length", "positions"),
        default="length",
        help="Use Length/codon_step or the number of emitted CDS positions for average read-depth normalization.",
    )
    parser.add_argument("--chunk-size", type=int, default=100000)
    parser.add_argument("--max-records", type=int, default=None, help="Optional debug limit for emitted pause-score rows.")


def _add_summary_args(parser: argparse.ArgumentParser, require_scores: bool = True) -> None:
    parser.add_argument("--scores-csv", required=require_scores, help="Input per-position pause-score CSV.")
    parser.add_argument("--output-csv", required=True, help="Output high-pause count CSV.")
    parser.add_argument("--id-col", default="ENSP", help="Identifier column in the score CSV.")
    parser.add_argument("--score-col", default="Pause_Score")
    parser.add_argument("--group-cols", default=None, help="Comma-separated grouping columns. Defaults to the id column.")
    parser.add_argument("--threshold", type=float, default=3.3)
    parser.add_argument("--threshold-mode", choices=("absolute", "relative_to_mean"), default="absolute")
    parser.add_argument("--output-id-col", default="protein_id")
    parser.add_argument("--count-col", default="High_Pause_Counts")
    parser.add_argument("--summary-cds-csv", default=None, help="Optional CDS CSV used to add transcript_id to the output.")
    parser.add_argument("--summary-protein-col", default=None, help="Protein column in --summary-cds-csv.")
    parser.add_argument("--summary-transcript-col", default=None, help="Transcript column in --summary-cds-csv.")
    parser.add_argument("--summary-chunksize", type=int, default=1000000)
    parser.add_argument("--pivot-output-csv", default=None, help="Optional wide matrix output for grouped cell/barcode counts.")
    parser.add_argument("--pivot-col", default="CB")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute RECIPE pausing features from CDS annotations and BAM files.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    score_parser = subparsers.add_parser("score-bam", help="Compute per-position pause scores from a BAM file.")
    _add_cds_score_args(score_parser)

    summarize_parser = subparsers.add_parser("summarize", help="Summarize pause scores into high-pause counts.")
    _add_summary_args(summarize_parser)

    run_parser = subparsers.add_parser("run", help="Compute per-position scores and summarize them in one command.")
    _add_cds_score_args(run_parser)
    _add_summary_args(run_parser, require_scores=False)
    return parser


def _run_score(args: argparse.Namespace) -> dict[str, object]:
    return write_pause_scores_from_bam(
        cds_csv=args.cds_csv,
        bam_path=args.bam,
        output_csv=args.score_csv,
        protein_col=args.protein_col,
        reference_col=args.reference_col,
        start_col=args.start_col,
        end_col=args.end_col,
        length_col=args.length_col,
        trim_start_nt=args.trim_start_nt,
        trim_end_nt=args.trim_end_nt,
        codon_step=args.codon_step,
        count_width=args.count_width,
        average_denominator=args.average_denominator,
        chunk_size=args.chunk_size,
        max_records=args.max_records,
    )


def _run_summary(args: argparse.Namespace, scores_csv: str | None = None) -> dict[str, object]:
    score_path = scores_csv or args.scores_csv
    summary = summarize_high_pause_counts_csv(
        scores_csv=score_path,
        output_csv=args.output_csv,
        id_col=args.id_col,
        score_col=args.score_col,
        group_cols=_split_columns(args.group_cols),
        threshold=args.threshold,
        threshold_mode=args.threshold_mode,
        output_id_col=args.output_id_col,
        count_col=args.count_col,
        cds_csv=args.summary_cds_csv,
        protein_col=args.summary_protein_col,
        transcript_col=args.summary_transcript_col,
        chunksize=args.summary_chunksize,
    )
    if args.pivot_output_csv:
        summary_df = pd.read_csv(args.output_csv)
        pivot_df = pivot_pause_counts(
            summary_df,
            index_col=args.output_id_col,
            column_col=args.pivot_col,
            value_col=args.count_col,
        )
        pivot_df.to_csv(args.pivot_output_csv, index=False)
        summary["pivot_output_csv"] = args.pivot_output_csv
    return summary


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "score-bam":
        summary = {"score_bam": _run_score(args)}
    elif args.command == "summarize":
        summary = {"summarize": _run_summary(args)}
    elif args.command == "run":
        score_summary = _run_score(args)
        summary = {
            "score_bam": score_summary,
            "summarize": _run_summary(args, scores_csv=args.score_csv),
        }
    else:
        raise ValueError(f"Unsupported command: {args.command}")

    print(json.dumps(json_sanitize(summary), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
