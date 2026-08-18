#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from train_phase3_ensmusp_nanospins_matched import (
    Phase3CellGraph,
    build_pca_knn_edge_index,
    load_mapping_table,
    normalize_total_rows,
    read_ordered_frame,
)


SCENARIO_CHOICES = (
    "c10_best_on_c10_test",
    "svec_best_on_svec_test",
    "c10_model_on_svec_test",
    "svec_model_on_c10_test",
)

SCATTER_COLOR = "#C7B0C4"
TOP_HIST_COLOR = "#C1CFBE"
SIDE_HIST_COLOR = "#BFD2DE"


@dataclass
class ModelBundle:
    condition: str
    seed: int
    summary_path: Path
    summary: dict
    hidden_all: np.ndarray
    cell_names: list[str]
    order_ids: pd.Index
    expr_cell_by_gene_knn: np.ndarray
    model: Phase3CellGraph
    device: torch.device


@dataclass(frozen=True)
class ScenarioSpec:
    source_condition: str
    target_condition: str
    x_label: str
    y_label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recreate Module D scRNA-seq phase3 C10/SVEC held-out and cross-condition "
            "scatter plots from archived nanoSPINS checkpoints, phase2 hidden caches, "
            "and phase3 summaries."
        )
    )
    parser.add_argument(
        "--phase23-root",
        type=Path,
        required=True,
        help="Root containing C10/seed*/phase3 and SVEC/seed*/phase3 outputs.",
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        required=True,
        help="Prepared ENSMUSP scRNA-seq bundle directory.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--c10-summary", type=Path, default=None)
    parser.add_argument("--svec-summary", type=Path, default=None)
    parser.add_argument(
        "--truth-csv",
        type=Path,
        default=None,
        help="Override nanoSPINS truth CSV stored in phase3 summaries.",
    )
    parser.add_argument(
        "--mapping-xlsx",
        type=Path,
        default=None,
        help="Override nanoSPINS cell-to-sample mapping workbook stored in phase3 summaries.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=SCENARIO_CHOICES,
        default=list(SCENARIO_CHOICES),
        help="Plots to generate.",
    )
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def select_best_phase3_summary(phase23_root: Path, condition: str) -> Path:
    candidates: list[tuple[float, int, Path]] = []
    for path in phase23_root.glob(f"{condition}/seed*/phase3/reports/phase3_nanospins_summary.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        candidates.append((float(data["test_metrics"]["r2"]), int(data["seed"]), path))
    if not candidates:
        raise FileNotFoundError(f"No phase3 summaries found for {condition} under {phase23_root}")
    candidates.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return candidates[0][2]


def _summary_seed_dir(summary_path: Path) -> Path:
    # .../<condition>/seed*/phase3/reports/phase3_nanospins_summary.json -> .../<condition>/seed*
    return summary_path.resolve().parents[2]


def resolve_hidden_cache_root(summary_path: Path, summary: dict, *, require_hidden_all: bool) -> Path:
    required_file = "phase2_hidden_all.npy" if require_hidden_all else "phase2_hidden_gene_names.csv"
    summary_root = Path(str(summary.get("hidden_cache_root", ""))).expanduser()
    if (summary_root / required_file).exists():
        return summary_root
    relative_root = _summary_seed_dir(summary_path) / "phase2_hidden_cache"
    if (relative_root / required_file).exists():
        return relative_root
    raise FileNotFoundError(
        "Missing phase2 hidden cache. Tried "
        f"{summary_root} and {relative_root}."
    )


def resolve_model_path(summary_path: Path, summary: dict) -> Path:
    summary_model = Path(str(summary.get("saved_model_path", ""))).expanduser()
    if summary_model.exists():
        return summary_model
    relative_model = summary_path.resolve().parents[1] / "models" / "phase3_nanospins_best.pth"
    if relative_model.exists():
        return relative_model
    raise FileNotFoundError(
        "Missing phase3 nanoSPINS checkpoint. Tried "
        f"{summary_model} and {relative_model}."
    )


def resolve_targets_csv(summary_path: Path, summary: dict) -> Path:
    summary_targets = Path(str(summary.get("targets_csv", ""))).expanduser()
    if summary_targets.exists():
        return summary_targets
    relative_targets = summary_path.resolve().parents[1] / "tables" / "phase3_nanospins_targets.csv"
    if relative_targets.exists():
        return relative_targets
    raise FileNotFoundError(
        "Missing phase3 nanoSPINS targets CSV. Tried "
        f"{summary_targets} and {relative_targets}."
    )


def load_model_bundle(
    summary_path: Path,
    *,
    bundle_dir: Path,
    device: torch.device,
) -> ModelBundle:
    summary_path = summary_path.resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    hidden_cache_root = resolve_hidden_cache_root(summary_path, summary, require_hidden_all=True)
    hidden_all = np.load(hidden_cache_root / "phase2_hidden_all.npy", mmap_mode="r")
    cell_names = pd.read_csv(hidden_cache_root / "phase2_hidden_cell_names.csv")["cell_name"].astype(str).tolist()
    order_ids = pd.Index(
        pd.read_csv(hidden_cache_root / "phase2_hidden_gene_names.csv")["protein_id"].astype(str).tolist(),
        name="protein_id",
    )
    sc_rna_all = read_ordered_frame(
        bundle_dir / "scRNA_qc_cells_by_ENSMUSP_all.bulk_intersection.zero_filled.csv",
        order_ids,
    )
    expr_gene_by_cell = sc_rna_all.drop(columns=["protein_id"]).to_numpy(dtype=np.float32, copy=False)
    expr_cell_by_gene_knn = normalize_total_rows(expr_gene_by_cell.T.copy(), target_sum=1e4)

    model = Phase3CellGraph(input_dim=int(hidden_all.shape[2]), hidden_dim=64, dropout=0.1).to(device)
    model.load_state_dict(torch.load(resolve_model_path(summary_path, summary), map_location=device))
    model.eval()
    return ModelBundle(
        condition=str(summary["condition"]),
        seed=int(summary["seed"]),
        summary_path=summary_path,
        summary=summary,
        hidden_all=hidden_all,
        cell_names=cell_names,
        order_ids=order_ids,
        expr_cell_by_gene_knn=expr_cell_by_gene_knn,
        model=model,
        device=device,
    )


def safe_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    pearson_r, pearson_p = pearsonr(y_true, y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearmanr(y_true, y_pred)[0]),
        "cosine_similarity": float(
            cosine_similarity(y_pred.reshape(1, -1), y_true.reshape(1, -1))[0, 0]
        ),
    }


def format_p_value(p_value: float) -> str:
    if p_value < 1e-300:
        return r"$P < 1 \times 10^{-300}$"
    base, exp = f"{p_value:.2e}".split("e")
    return rf"$P = {float(base):.2f} \times 10^{{{int(exp)}}}$"


def _summary_path_value(summary: dict, key: str, override: Path | None) -> Path:
    if override is not None:
        return override
    value = summary.get(key)
    if value is None:
        raise KeyError(f"Missing {key!r} in phase3 summary.")
    return Path(str(value)).expanduser()


def evaluate_on_target(
    source_bundle: ModelBundle,
    target_summary_path: Path,
    *,
    target_condition: str,
    truth_csv: Path | None,
    mapping_xlsx: Path | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    target_summary_path = target_summary_path.resolve()
    target_summary = json.loads(target_summary_path.read_text(encoding="utf-8"))
    target_hidden_root = resolve_hidden_cache_root(target_summary_path, target_summary, require_hidden_all=False)
    target_hidden_gene_names = pd.read_csv(target_hidden_root / "phase2_hidden_gene_names.csv")[
        "protein_id"
    ].astype(str).tolist()
    if target_hidden_gene_names != source_bundle.order_ids.astype(str).tolist():
        raise ValueError("Source and target gene order do not match.")

    truth_df = read_ordered_frame(_summary_path_value(target_summary, "truth_csv", truth_csv), source_bundle.order_ids)
    truth_cols = [col for col in truth_df.columns if col != "protein_id"]

    mapping_df = load_mapping_table(_summary_path_value(target_summary, "mapping_xlsx", mapping_xlsx))
    mapping_df = mapping_df[
        mapping_df["Cell Barcode"].isin(source_bundle.cell_names)
        & mapping_df["Common with Proteomics samples"].isin(truth_cols)
    ].copy()
    mapping_df = mapping_df.drop_duplicates(subset=["Cell Barcode"])
    mapping_df = mapping_df.drop_duplicates(subset=["Common with Proteomics samples"])
    cell_to_idx = {name: idx for idx, name in enumerate(source_bundle.cell_names)}
    mapping_df["cell_idx"] = mapping_df["Cell Barcode"].map(cell_to_idx)
    mapping_df = mapping_df.sort_values("cell_idx").reset_index(drop=True)
    if mapping_df.empty:
        raise ValueError("No matched nanoSPINS cells found.")

    matched_truth_cols = mapping_df["Common with Proteomics samples"].astype(str).tolist()
    matched_cell_idx = mapping_df["cell_idx"].astype(int).to_numpy()
    matched_cell_type = mapping_df["Cell type"].astype(str).to_numpy()
    truth_matrix = truth_df.loc[:, matched_truth_cols].to_numpy(dtype=np.float32, copy=False)

    target_pos = np.where(matched_cell_type == target_condition)[0].astype(np.int64)
    target_full_idx = matched_cell_idx[target_pos]
    edge_index = build_pca_knn_edge_index(
        source_bundle.expr_cell_by_gene_knn[target_full_idx],
        n_neighbors=int(source_bundle.summary["phase3_k_neighbors"]),
        n_pcs=int(source_bundle.summary["phase3_n_pcs"]),
        seed=int(source_bundle.summary["seed"]),
    ).to(source_bundle.device)

    targets_df = pd.read_csv(resolve_targets_csv(target_summary_path, target_summary))
    test_gene_idx = np.where(targets_df["split"].astype(str).to_numpy() == "test")[0].astype(np.int64)

    pred_all: list[np.ndarray] = []
    truth_all: list[np.ndarray] = []
    with torch.no_grad():
        for gene_idx in test_gene_idx.tolist():
            x = torch.from_numpy(
                np.asarray(source_bundle.hidden_all[target_full_idx, gene_idx, :], dtype=np.float32)
            ).to(source_bundle.device)
            pred = source_bundle.model(x, edge_index).detach().cpu().numpy().astype(np.float32, copy=False)
            truth = truth_matrix[gene_idx, target_pos].astype(np.float32, copy=False)
            mask = np.isfinite(truth)
            pred_all.append(pred[mask])
            truth_all.append(truth[mask])

    y_pred = np.concatenate(pred_all).astype(np.float64, copy=False)
    y_true = np.concatenate(truth_all).astype(np.float64, copy=False)
    return y_true, y_pred, safe_metrics(y_true, y_pred)


def style_axes(ax) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
    ax.tick_params(axis="both", labelsize=12, width=2.0, length=4, pad=1)


def plot_scatter_with_marginals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: dict[str, float],
    x_label: str,
    y_label: str,
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(6, 6))
    gs = GridSpec(4, 4)
    ax_scatter = fig.add_subplot(gs[1:4, 0:3])
    ax_histx = fig.add_subplot(gs[0, 0:3], sharex=ax_scatter)
    ax_histy = fig.add_subplot(gs[1:4, 3], sharey=ax_scatter)

    ax_scatter.scatter(y_true, y_pred, alpha=0.6, color=SCATTER_COLOR, s=55, edgecolors="none")
    min_val = float(min(y_true.min(), y_pred.min()))
    max_val = float(max(y_true.max(), y_pred.max()))
    ax_scatter.plot([min_val, max_val], [min_val, max_val], ls="--", color="black", linewidth=2.6)
    sns.regplot(
        x=y_true,
        y=y_pred,
        scatter=False,
        color="black",
        line_kws={"linewidth": 2.6},
        ax=ax_scatter,
    )

    ax_scatter.set_xlabel(x_label, fontsize=16)
    ax_scatter.set_ylabel(y_label, fontsize=16)
    ax_scatter.text(
        0.05,
        0.91,
        f"Correlation:  {metrics['pearson_r']:.4f}",
        transform=ax_scatter.transAxes,
        fontsize=18,
    )
    ax_scatter.text(
        0.05,
        0.83,
        format_p_value(metrics["pearson_p"]),
        transform=ax_scatter.transAxes,
        fontsize=18,
        style="italic",
    )

    ax_histx.hist(y_true, bins=40, color=TOP_HIST_COLOR, alpha=0.8, edgecolor="#B7C7B4", linewidth=0.4)
    ax_histy.hist(
        y_pred,
        bins=40,
        orientation="horizontal",
        color=SIDE_HIST_COLOR,
        alpha=0.8,
        edgecolor="#AFC3CF",
        linewidth=0.4,
    )
    ax_histy.yaxis.set_visible(False)
    ax_histx.set_ylabel("Count", fontsize=12)
    ax_histx.xaxis.set_visible(False)

    style_axes(ax_scatter)
    style_axes(ax_histx)
    style_axes(ax_histy)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format=out_path.suffix.lstrip(".") or "pdf", bbox_inches="tight")
    plt.close(fig)


def scenario_specs() -> dict[str, ScenarioSpec]:
    return {
        "c10_best_on_c10_test": ScenarioSpec(
            source_condition="C10",
            target_condition="C10",
            x_label="Real protein expression of C10",
            y_label="Predicted protein expression\nof C10 best model",
        ),
        "svec_best_on_svec_test": ScenarioSpec(
            source_condition="SVEC",
            target_condition="SVEC",
            x_label="Real protein expression of SVEC",
            y_label="Predicted protein expression\nof SVEC best model",
        ),
        "c10_model_on_svec_test": ScenarioSpec(
            source_condition="C10",
            target_condition="SVEC",
            x_label="Real protein expression of SVEC",
            y_label="Predicted protein expression\nby C10 model",
        ),
        "svec_model_on_c10_test": ScenarioSpec(
            source_condition="SVEC",
            target_condition="C10",
            x_label="Real protein expression of C10",
            y_label="Predicted protein expression\nby SVEC model",
        ),
    }


def run(args: argparse.Namespace) -> list[dict[str, object]]:
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.family"] = "Arial"

    phase23_root = args.phase23_root.expanduser().resolve()
    bundle_dir = args.bundle_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    c10_summary = (
        args.c10_summary.expanduser().resolve()
        if args.c10_summary is not None
        else select_best_phase3_summary(phase23_root, "C10")
    )
    svec_summary = (
        args.svec_summary.expanduser().resolve()
        if args.svec_summary is not None
        else select_best_phase3_summary(phase23_root, "SVEC")
    )
    device = resolve_device(str(args.device))

    selected_scenarios = _deduplicate(args.scenarios)
    summaries = {"C10": c10_summary, "SVEC": svec_summary}
    specs = scenario_specs()
    source_conditions = _deduplicate(specs[slug].source_condition for slug in selected_scenarios)
    bundles = {
        condition: load_model_bundle(summaries[condition], bundle_dir=bundle_dir, device=device)
        for condition in source_conditions
    }

    rows: list[dict[str, object]] = []
    for slug in selected_scenarios:
        spec = specs[slug]
        source_bundle = bundles[spec.source_condition]
        target_summary_path = summaries[spec.target_condition]
        y_true, y_pred, metrics = evaluate_on_target(
            source_bundle,
            target_summary_path,
            target_condition=spec.target_condition,
            truth_csv=args.truth_csv,
            mapping_xlsx=args.mapping_xlsx,
        )
        out_pdf = output_dir / f"{slug}.pdf"
        out_csv = output_dir / f"{slug}.csv"
        plot_scatter_with_marginals(y_true, y_pred, metrics, spec.x_label, spec.y_label, out_pdf)
        pd.DataFrame({"real": y_true, "pred": y_pred}).to_csv(out_csv, index=False)
        rows.append(
            {
                "scenario": slug,
                "x_label": spec.x_label,
                "y_label": spec.y_label,
                "source_condition": source_bundle.condition,
                "source_seed": source_bundle.seed,
                "source_summary_path": str(source_bundle.summary_path),
                "target_condition": spec.target_condition,
                "target_summary_path": str(target_summary_path),
                "pair_count": int(y_true.size),
                **metrics,
                "pdf": str(out_pdf),
                "csv": str(out_csv),
            }
        )

    pd.DataFrame(rows).to_csv(output_dir / "phase3_cross_condition_scatter_metrics.csv", index=False)
    (output_dir / "phase3_cross_condition_scatter_metrics.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return rows


def _deduplicate(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def main() -> None:
    rows = run(parse_args())
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
