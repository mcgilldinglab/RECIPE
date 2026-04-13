from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class RBULK(nn.Module):
    """GraphSAGE regressor used for bulk protein abundance inference."""

    def __init__(
        self,
        sequence_dim: int = 9216,
        hidden_dim: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.fc_x = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fc_pause = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.sequence_encoder = nn.Sequential(
            nn.Linear(sequence_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.conv = SAGEConv(hidden_dim, hidden_dim, aggr="sum")
        self.conv_activation = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.ReLU(),
        )

    def forward(self, data):
        x = data.x
        sequence_embedding = data.seq
        pause = data.pause

        if pause.dim() == 1:
            pause = pause.view(-1, 1)

        x = self.fc_x(x) + self.sequence_encoder(sequence_embedding)
        x = torch.cat((x, self.fc_pause(pause)), dim=1)
        x = self.fusion(x)

        z = self.conv(x, data.edge_index)
        z = self.conv_activation(z)
        out = self.regressor(z)
        return out, z


class CPPI(nn.Module):
    """Edge scoring network for self-supervised PPI refinement."""

    def __init__(self, embedding_dim: int = 32, hidden_dim: int = 32) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat((x1, x2), dim=-1)).view(-1)


class RSCHead(nn.Module):
    """Cell-level graph learner used for single-cell transfer."""

    def __init__(self, input_dim: int = 32, hidden_dim: int = 16, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim, aggr="mean")
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        batch_size, num_cells, feature_dim = x.shape
        flat_x = x.reshape(batch_size * num_cells, feature_dim)

        if batch_size == 1:
            batch_edge_index = edge_index
        else:
            repeated_edges = []
            for batch_idx in range(batch_size):
                offset = batch_idx * num_cells
                repeated_edges.append(edge_index + offset)
            batch_edge_index = torch.cat(repeated_edges, dim=1)

        hidden = self.conv1(flat_x, batch_edge_index)
        hidden = F.relu(hidden)
        hidden = self.dropout(hidden)
        output = self.mlp(hidden)
        output = F.relu(output)
        return output.view(batch_size, num_cells)


NeuralGraph = RBULK
