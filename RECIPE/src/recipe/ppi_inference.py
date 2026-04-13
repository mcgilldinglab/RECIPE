from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch_geometric.utils import negative_sampling

from .bulk_data import load_ppi_graph


class EdgePairDataset(Dataset):
    def __init__(self, node_embeddings: torch.Tensor, edge_index: torch.Tensor, labels: torch.Tensor) -> None:
        self.node_embeddings = node_embeddings.cpu().float()
        self.edge_pairs = edge_index.t().contiguous().cpu().long()
        self.labels = labels.view(-1).cpu().float()

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, idx: int):
        src, dst = self.edge_pairs[idx]
        return self.node_embeddings[src], self.node_embeddings[dst], self.labels[idx]


class EdgeMLP(nn.Module):
    def __init__(self, embedding_dim: int = 32, hidden_dim: int = 32) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x = torch.cat((x1, x2), dim=-1)
        return self.mlp(x).view(-1)


def unique_undirected_edge_index(
    edge_index: torch.Tensor,
    remove_self_loops: bool = True,
) -> torch.Tensor:
    if edge_index.numel() == 0:
        return edge_index

    src = edge_index[0].cpu().long()
    dst = edge_index[1].cpu().long()
    if remove_self_loops:
        keep_mask = src != dst
        src = src[keep_mask]
        dst = dst[keep_mask]

    undirected_pairs = torch.stack((torch.minimum(src, dst), torch.maximum(src, dst)), dim=1)
    if undirected_pairs.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)

    unique_pairs = torch.unique(undirected_pairs, dim=0)
    return unique_pairs.t().contiguous()


def sample_negative_edges(
    positive_edge_index: torch.Tensor,
    num_nodes: int,
    num_neg_samples: int,
) -> torch.Tensor:
    sampled = negative_sampling(
        edge_index=positive_edge_index,
        num_nodes=num_nodes,
        num_neg_samples=max(num_neg_samples * 2, num_neg_samples),
    )
    unique_negative = unique_undirected_edge_index(sampled, remove_self_loops=True)
    if unique_negative.size(1) >= num_neg_samples:
        return unique_negative[:, :num_neg_samples]

    extra_needed = num_neg_samples - unique_negative.size(1)
    extra_sampled = negative_sampling(
        edge_index=positive_edge_index,
        num_nodes=num_nodes,
        num_neg_samples=max(extra_needed * 4, extra_needed),
    )
    combined = torch.cat((unique_negative, unique_undirected_edge_index(extra_sampled, remove_self_loops=True)), dim=1)
    combined = unique_undirected_edge_index(combined, remove_self_loops=True)
    return combined[:, :num_neg_samples]


def build_edge_dataset(
    node_embeddings: torch.Tensor,
    positive_edge_index: torch.Tensor,
    negative_ratio: float = 1.0,
) -> tuple[EdgePairDataset, dict[str, int]]:
    positive_edge_index = unique_undirected_edge_index(positive_edge_index, remove_self_loops=True)
    num_pos = int(positive_edge_index.size(1))
    num_neg = max(1, int(round(num_pos * negative_ratio)))
    negative_edge_index = sample_negative_edges(positive_edge_index, node_embeddings.size(0), num_neg)

    edge_index = torch.cat((positive_edge_index, negative_edge_index), dim=1)
    labels = torch.cat(
        (
            torch.ones(positive_edge_index.size(1), dtype=torch.float32),
            torch.zeros(negative_edge_index.size(1), dtype=torch.float32),
        ),
        dim=0,
    )
    dataset = EdgePairDataset(node_embeddings=node_embeddings, edge_index=edge_index, labels=labels)
    return dataset, {
        "positive_edges": positive_edge_index.size(1),
        "negative_edges": negative_edge_index.size(1),
    }


def train_edge_classifier(
    node_embeddings: torch.Tensor,
    positive_edge_index: torch.Tensor,
    device: torch.device,
    lr: float = 1e-3,
    batch_size: int = 512,
    max_epochs: int = 1000,
    patience: int = 50,
    negative_ratio: float = 1.0,
    log_every: int = 10,
) -> tuple[EdgeMLP, dict[str, Any]]:
    dataset, edge_counts = build_edge_dataset(
        node_embeddings=node_embeddings,
        positive_edge_index=positive_edge_index,
        negative_ratio=negative_ratio,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = EdgeMLP(embedding_dim=node_embeddings.size(1)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_state = deepcopy(model.state_dict())
    best_epoch = 0
    best_loss = float("inf")
    patience_counter = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        sample_count = 0

        for x1_batch, x2_batch, label_batch in dataloader:
            x1_batch = x1_batch.to(device)
            x2_batch = x2_batch.to(device)
            label_batch = label_batch.to(device)

            optimizer.zero_grad()
            logits = model(x1_batch, x2_batch)
            loss = criterion(logits, label_batch)
            loss.backward()
            optimizer.step()

            batch_size_now = int(label_batch.numel())
            epoch_loss += float(loss.item()) * batch_size_now
            sample_count += batch_size_now

        mean_loss = epoch_loss / max(sample_count, 1)
        history.append({"epoch": float(epoch), "loss": mean_loss})

        if mean_loss < best_loss:
            best_loss = mean_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

        if log_every > 0 and (epoch == 1 or epoch % log_every == 0):
            print(f"Epoch {epoch:04d}, Edge Loss: {mean_loss:.4f}")

    model.load_state_dict(best_state)
    summary = {
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "history": history,
        **edge_counts,
    }
    return model, summary


def score_edge_index(
    model: EdgeMLP,
    node_embeddings: torch.Tensor,
    edge_index: torch.Tensor,
    device: torch.device,
    batch_size: int = 16384,
) -> torch.Tensor:
    model.eval()
    scores: list[torch.Tensor] = []
    edge_pairs = edge_index.t().contiguous().cpu()

    with torch.no_grad():
        for start in range(0, edge_pairs.size(0), batch_size):
            batch_pairs = edge_pairs[start : start + batch_size]
            x1 = node_embeddings[batch_pairs[:, 0]].to(device)
            x2 = node_embeddings[batch_pairs[:, 1]].to(device)
            probs = torch.sigmoid(model(x1, x2)).cpu()
            scores.append(probs)

    if not scores:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(scores, dim=0)


def infer_candidate_edges(
    model: EdgeMLP,
    node_embeddings: torch.Tensor,
    device: torch.device,
    threshold: float = 0.8,
    batch_size: int = 16384,
    export_score_matrix: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray | None]:
    num_nodes = int(node_embeddings.size(0))
    selected_src: list[torch.Tensor] = []
    selected_dst: list[torch.Tensor] = []
    selected_prob: list[torch.Tensor] = []
    score_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32) if export_score_matrix else None

    model.eval()
    with torch.no_grad():
        for src in range(num_nodes - 1):
            all_dst = torch.arange(src + 1, num_nodes, dtype=torch.long)
            if all_dst.numel() == 0:
                continue

            for start in range(0, all_dst.numel(), batch_size):
                batch_dst = all_dst[start : start + batch_size]
                batch_src = torch.full_like(batch_dst, fill_value=src)
                x1 = node_embeddings[batch_src].to(device)
                x2 = node_embeddings[batch_dst].to(device)
                probs = torch.sigmoid(model(x1, x2)).cpu()

                if score_matrix is not None:
                    src_idx = int(src)
                    dst_idx = batch_dst.numpy()
                    prob_np = probs.numpy()
                    score_matrix[src_idx, dst_idx] = prob_np
                    score_matrix[dst_idx, src_idx] = prob_np

                keep_mask = probs >= threshold
                if keep_mask.any():
                    selected_src.append(batch_src[keep_mask].cpu())
                    selected_dst.append(batch_dst[keep_mask].cpu())
                    selected_prob.append(probs[keep_mask])

    if not selected_src:
        return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.float32), score_matrix

    new_edge_index = torch.stack((torch.cat(selected_src), torch.cat(selected_dst)), dim=0)
    new_edge_probs = torch.cat(selected_prob, dim=0)
    return new_edge_index, new_edge_probs, score_matrix


def save_new_edges_csv(
    path: str | Path,
    edge_index: torch.Tensor,
    edge_probabilities: torch.Tensor | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "node1": edge_index[0].cpu().numpy() if edge_index.numel() else np.array([], dtype=np.int64),
        "node2": edge_index[1].cpu().numpy() if edge_index.numel() else np.array([], dtype=np.int64),
    }
    if edge_probabilities is not None:
        payload["probability"] = edge_probabilities.cpu().numpy()

    pd.DataFrame(payload).to_csv(path, index=False)


def save_score_matrix(path: str | Path, score_matrix: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        np.savetxt(path, score_matrix, delimiter=",", fmt="%.6f")
        return
    np.save(path, score_matrix)


def load_positive_ppi_edges(ppi_csv_path: str | Path) -> torch.Tensor:
    edge_index, _ = load_ppi_graph(ppi_csv_path, add_loops=False)
    return unique_undirected_edge_index(edge_index, remove_self_loops=True)
