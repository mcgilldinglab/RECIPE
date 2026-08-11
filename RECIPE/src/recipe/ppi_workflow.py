from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .assets import PPI_MODEL_DIR
from .bulk_regression import predict_bulk_outputs
from .bulk_workflow import build_bulk_graph_for_task, load_model_state, make_bulk_model
from .models import RBULK
from .ppi_inference import (
    infer_candidate_edges,
    load_edge_classifier_checkpoint,
    load_positive_ppi_edges,
    save_new_edges_csv,
    save_score_matrix,
    score_edge_index,
    train_edge_classifier as train_edge_classifier_model,
)
from .utils import resolve_device, save_json, set_seed


def _filter_new_edges(candidate_edges: torch.Tensor, candidate_scores: torch.Tensor, known_edges: torch.Tensor):
    known_pairs = {tuple(pair) for pair in known_edges.t().cpu().numpy().tolist()}
    keep_indices = []
    for idx, pair in enumerate(candidate_edges.t().cpu().numpy().tolist()):
        if tuple(pair) not in known_pairs:
            keep_indices.append(idx)

    if not keep_indices:
        return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.float32)
    keep_tensor = torch.tensor(keep_indices, dtype=torch.long)
    return candidate_edges[:, keep_tensor], candidate_scores[keep_tensor]


def _coexpression_summary(coexpression_csv: Path, edge_index: torch.Tensor) -> dict[str, float]:
    if not coexpression_csv.exists() or edge_index.numel() == 0:
        return {"mean": float("nan"), "median": float("nan")}

    matrix = pd.read_csv(coexpression_csv).to_numpy(dtype=np.float32)
    src = edge_index[0].cpu().numpy().astype(int)
    dst = edge_index[1].cpu().numpy().astype(int)
    values = matrix[src, dst]
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
    }


def _default_edge_checkpoint(species: str, model_root: str | Path | None = None) -> Path:
    checkpoint = PPI_MODEL_DIR / f"{species.lower()}_edge_classifier.pth"
    if model_root is None:
        return checkpoint
    return Path(model_root).expanduser().resolve() / "ppi" / checkpoint.name


def _read_candidate_edges_csv(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    df = pd.read_csv(path)
    if {"node1", "node2"}.issubset(df.columns):
        src_col, dst_col = "node1", "node2"
    elif {"source", "target"}.issubset(df.columns):
        src_col, dst_col = "source", "target"
    else:
        raise ValueError("Candidate edge CSV must contain node1/node2 or source/target columns.")

    edge_index = torch.tensor(df[[src_col, dst_col]].to_numpy(dtype=np.int64).T, dtype=torch.long)
    score_col = "probability" if "probability" in df.columns else "score" if "score" in df.columns else None
    if score_col is None:
        scores = torch.full((edge_index.size(1),), float("nan"), dtype=torch.float32)
    else:
        scores = torch.tensor(pd.to_numeric(df[score_col], errors="coerce").to_numpy(dtype=np.float32))
    return edge_index, scores


def run_ppi_refinement(
    species: str,
    condition_name: str,
    output_dir: str | Path,
    seed: int = 12,
    device_name: str | None = None,
    bulk_checkpoint_path: str | Path | None = None,
    edge_learning_rate: float = 1e-3,
    edge_batch_size: int = 512,
    edge_max_epochs: int = 1000,
    edge_patience: int = 50,
    negative_ratio: float = 1.0,
    threshold: float = 0.8,
    export_score_matrix: bool = False,
    train_edge_classifier: bool = False,
    edge_checkpoint_path: str | Path | None = None,
    candidate_edge_csv: str | Path | None = None,
    skip_candidate_inference: bool = False,
    log_every: int = 10,
    data_root: str | Path | None = None,
    model_root: str | Path | None = None,
    reference_csv: str | Path | None = None,
    sequence_npy: str | Path | None = None,
    ppi_csv: str | Path | None = None,
    coexpression_csv: str | Path | None = None,
    expression_col: str | None = None,
    target_col: str | None = None,
    pause_col: str | None = None,
    use_pause: bool = True,
) -> dict[str, Any]:
    set_seed(seed)
    device = resolve_device(device_name)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config, _, data, scaling_summary = build_bulk_graph_for_task(
        species=species,
        task="known",
        condition_name=condition_name,
        scale_method="log_median",
        data_root=data_root,
        model_root=model_root,
        reference_csv=reference_csv,
        sequence_npy=sequence_npy,
        ppi_csv=ppi_csv,
        expression_col=expression_col,
        target_col=target_col,
        pause_col=pause_col,
        use_pause=use_pause,
    )
    model: RBULK = make_bulk_model(data, device=device)
    checkpoint = Path(bulk_checkpoint_path) if bulk_checkpoint_path else config.default_checkpoint
    if checkpoint is None or not checkpoint.exists():
        raise FileNotFoundError("A trained bulk checkpoint is required before running module C.")

    model = load_model_state(model, checkpoint, device=device)
    bulk_checkpoint_load = getattr(model, "_recipe_load_report", None)
    data = data.to(device)
    _, node_embeddings = predict_bulk_outputs(model, data)
    positive_edges = load_positive_ppi_edges(config.ppi_csv)

    edge_checkpoint = Path(edge_checkpoint_path).expanduser().resolve() if edge_checkpoint_path else _default_edge_checkpoint(
        species=species,
        model_root=model_root,
    )
    if (not train_edge_classifier) and edge_checkpoint.exists():
        edge_model, edge_summary = load_edge_classifier_checkpoint(
            checkpoint_path=edge_checkpoint,
            embedding_dim=int(node_embeddings.size(1)),
            device=device,
        )
        edge_summary["source"] = "checkpoint"
    else:
        if (not train_edge_classifier) and not edge_checkpoint.exists():
            print(f"[Module C] Edge checkpoint not found, training a new classifier: {edge_checkpoint}")
        edge_model, edge_summary = train_edge_classifier_model(
            node_embeddings=node_embeddings,
            positive_edge_index=positive_edges,
            device=device,
            lr=edge_learning_rate,
            batch_size=edge_batch_size,
            max_epochs=edge_max_epochs,
            patience=edge_patience,
            negative_ratio=negative_ratio,
            log_every=log_every,
        )
        edge_summary["source"] = "trained"
        edge_summary["requested_checkpoint"] = str(edge_checkpoint)

    positive_scores = score_edge_index(
        model=edge_model,
        node_embeddings=node_embeddings,
        edge_index=positive_edges,
        device=device,
    )
    score_matrix = None
    candidate_source = "inferred"
    if candidate_edge_csv:
        candidate_edges, candidate_scores = _read_candidate_edges_csv(candidate_edge_csv)
        candidate_source = "csv"
    elif skip_candidate_inference:
        candidate_edges = torch.empty((2, 0), dtype=torch.long)
        candidate_scores = torch.empty(0, dtype=torch.float32)
        candidate_source = "skipped"
    else:
        candidate_edges, candidate_scores, score_matrix = infer_candidate_edges(
            model=edge_model,
            node_embeddings=node_embeddings,
            device=device,
            threshold=threshold,
            export_score_matrix=export_score_matrix,
        )
    new_edges, new_scores = _filter_new_edges(candidate_edges, candidate_scores, positive_edges)

    edge_model_path = output_dir / "edge_classifier.pth"
    known_edge_score_csv = output_dir / "known_edge_scores.csv"
    new_edge_csv = output_dir / "candidate_edges.csv"
    summary_json = output_dir / "summary.json"
    embedding_npy = output_dir / "bulk_node_embeddings.npy"

    torch.save(edge_model.state_dict(), edge_model_path)
    pd.DataFrame(
        {
            "source": positive_edges[0].cpu().numpy(),
            "target": positive_edges[1].cpu().numpy(),
            "score": positive_scores.cpu().numpy(),
        }
    ).to_csv(known_edge_score_csv, index=False)
    save_new_edges_csv(new_edge_csv, new_edges, new_scores)
    np.save(embedding_npy, node_embeddings.numpy())

    if score_matrix is not None:
        save_score_matrix(output_dir / "edge_score_matrix.npy", score_matrix)

    coexpression_path = (
        Path(coexpression_csv).expanduser().resolve()
        if coexpression_csv
        else config.ppi_csv.with_name(f"{species.lower()}_coexpression.csv")
    )
    summary = {
        "species": species,
        "condition": condition_name.upper(),
        "device": str(device),
        "bulk_checkpoint": str(checkpoint),
        "bulk_checkpoint_load": bulk_checkpoint_load,
        "edge_classifier": edge_summary,
        "candidate_source": candidate_source,
        "node_count": int(data.num_nodes),
        "known_edge_count": int(positive_edges.size(1)),
        "candidate_edge_count": int(new_edges.size(1)),
        "mean_positive_edge_score": float(positive_scores.mean().item()) if positive_scores.numel() else float("nan"),
        "mean_candidate_edge_score": float(new_scores.mean().item()) if new_scores.numel() else float("nan"),
        "coexpression_known": _coexpression_summary(coexpression_path, positive_edges),
        "coexpression_candidates": _coexpression_summary(coexpression_path, new_edges),
        "scaling": scaling_summary,
        "inputs": {
            "reference_csv": str(config.reference_csv),
            "sequence_npy": str(config.sequence_npy),
            "ppi_csv": str(config.ppi_csv),
            "bulk_checkpoint": str(checkpoint),
            "edge_checkpoint": str(edge_checkpoint),
            "candidate_edge_csv": str(Path(candidate_edge_csv).expanduser().resolve()) if candidate_edge_csv else None,
            "coexpression_csv": str(coexpression_path),
        },
        "outputs": {
            "edge_model": str(edge_model_path),
            "known_edge_scores": str(known_edge_score_csv),
            "candidate_edges": str(new_edge_csv),
            "bulk_node_embeddings": str(embedding_npy),
        },
    }
    if score_matrix is not None:
        summary["outputs"]["edge_score_matrix"] = str(output_dir / "edge_score_matrix.npy")
    save_json(summary_json, summary)
    return summary
