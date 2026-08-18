from __future__ import annotations

import argparse
import json
import shlex

from _bootstrap import add_src_to_path

add_src_to_path()

from recipe.pipeline import run_recipe_pipeline
from recipe.utils import json_sanitize


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run multiple RECIPE modules.")
    parser.add_argument("--modules", default="A,B,C,D")
    parser.add_argument("--species", choices=("human", "mouse"), default="mouse")
    parser.add_argument("--condition", default="KD", help="Condition name, for example NC or KD.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default=None, help="Directory containing RECIPE data subfolders.")
    parser.add_argument("--model-root", default=None, help="Directory containing RECIPE checkpoint subfolders.")
    parser.add_argument("--bulk-unknown-split-csv", default=None)
    parser.add_argument("--bulk-known-split-csv", default=None)
    parser.add_argument(
        "--bulk-input-col",
        "--bulk-expression-col",
        dest="bulk_expression_col",
        default=None,
        help="Bulk input signal column, for example RNA-seq, Ribo-seq, or another transcript-level feature.",
    )
    parser.add_argument("--bulk-target-col", default=None, help="Bulk protein abundance target column.")
    parser.add_argument("--bulk-pause-col", default=None, help="Bulk pausing-count column.")
    parser.add_argument("--no-bulk-pause", action="store_true", help="Use zero pausing features for bulk modules.")
    parser.add_argument("--bulk-train", action="store_true", help="Force bulk model training for modules A/B.")
    parser.add_argument("--bulk-max-epochs", type=int, default=3000)
    parser.add_argument("--bulk-patience", type=int, default=200)
    parser.add_argument("--bulk-learning-rate", type=float, default=7e-2)
    parser.add_argument("--edge-max-epochs", type=int, default=1000)
    parser.add_argument("--edge-patience", type=int, default=50)
    parser.add_argument("--train-edge-classifier", action="store_true")
    parser.add_argument("--edge-checkpoint-path", default=None)
    parser.add_argument("--candidate-edge-csv", default=None)
    parser.add_argument("--skip-candidate-inference", action="store_true")
    parser.add_argument("--train-phase0", action="store_true")
    parser.add_argument("--train-phase1", action="store_true")
    parser.add_argument("--train-phase2", action="store_true")
    parser.add_argument(
        "--single-cell-assay",
        choices=("scriboseq", "scrnaseq"),
        default="scriboseq",
        help="Module D input assay. scriboseq uses phase0/phase1/phase2; scrnaseq uses phase0/phase12/phase3.",
    )
    parser.add_argument("--phase1-checkpoint", default=None)
    parser.add_argument("--phase2-checkpoint", default=None)
    parser.add_argument("--phase0-split-csv", default=None)
    parser.add_argument("--phase1-split-csv", default=None)
    parser.add_argument("--phase2-split-csv", default=None)
    parser.add_argument("--use-bundled-cell-embeddings", action="store_true")
    parser.add_argument(
        "--phase2-n-neighbors",
        "--phase2-k",
        dest="phase2_n_neighbors",
        type=int,
        default=3,
        help="KNN size for the scRibo-seq Module D phase2 shared cell graph.",
    )
    parser.add_argument("--phase2-n-pcs", type=int, default=50)
    parser.add_argument(
        "--phase2-selection-metric",
        choices=("train_loss", "train_r2", "val_loss", "val_r2", "test_loss", "test_r2"),
        default="val_r2",
    )
    parser.add_argument("--rnaseq-phase0-args", default="", help="Shell-style args forwarded to scRNA-seq phase0.")
    parser.add_argument("--rnaseq-phase12-args", default="", help="Shell-style args forwarded to scRNA-seq phase12.")
    parser.add_argument("--rnaseq-phase3-args", default="", help="Shell-style args forwarded to scRNA-seq phase3.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    modules = tuple(module.strip() for module in args.modules.split(",") if module.strip())
    summary = run_recipe_pipeline(
        modules=modules,
        output_root=args.output_root,
        species=args.species,
        condition=args.condition,
        seed=args.seed,
        device_name=args.device,
        data_root=args.data_root,
        model_root=args.model_root,
        bulk_unknown_split_csv=args.bulk_unknown_split_csv,
        bulk_known_split_csv=args.bulk_known_split_csv,
        bulk_expression_col=args.bulk_expression_col,
        bulk_target_col=args.bulk_target_col,
        bulk_pause_col=args.bulk_pause_col,
        use_bulk_pause=not args.no_bulk_pause,
        bulk_train=args.bulk_train,
        bulk_max_epochs=args.bulk_max_epochs,
        bulk_patience=args.bulk_patience,
        bulk_learning_rate=args.bulk_learning_rate,
        edge_max_epochs=args.edge_max_epochs,
        edge_patience=args.edge_patience,
        train_edge_classifier=args.train_edge_classifier,
        edge_checkpoint_path=args.edge_checkpoint_path,
        candidate_edge_csv=args.candidate_edge_csv,
        skip_candidate_inference=args.skip_candidate_inference,
        train_phase0=args.train_phase0,
        train_phase1=args.train_phase1,
        train_phase2=args.train_phase2,
        single_cell_assay=args.single_cell_assay,
        phase1_checkpoint=args.phase1_checkpoint,
        phase2_checkpoint=args.phase2_checkpoint,
        phase0_split_csv=args.phase0_split_csv,
        phase1_split_csv=args.phase1_split_csv,
        phase2_split_csv=args.phase2_split_csv,
        use_bundled_cell_embeddings=args.use_bundled_cell_embeddings,
        phase2_n_neighbors=args.phase2_n_neighbors,
        phase2_n_pcs=args.phase2_n_pcs,
        phase2_selection_metric=args.phase2_selection_metric,
        rnaseq_phase0_args=shlex.split(args.rnaseq_phase0_args),
        rnaseq_phase12_args=shlex.split(args.rnaseq_phase12_args),
        rnaseq_phase3_args=shlex.split(args.rnaseq_phase3_args),
    )
    print(json.dumps(json_sanitize(summary), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
