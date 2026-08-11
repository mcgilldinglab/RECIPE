from __future__ import annotations

import pandas as pd
import pytest
import torch

from recipe.ppi_workflow import _filter_new_edges, _read_candidate_edges_csv, _validate_candidate_edges


def test_candidate_edges_without_scores_are_marked_for_model_scoring(tmp_path):
    path = tmp_path / "candidates.csv"
    pd.DataFrame({"source": [0, 1], "target": [2, 3]}).to_csv(path, index=False)

    edge_index, scores = _read_candidate_edges_csv(path)

    assert edge_index.tolist() == [[0, 1], [2, 3]]
    assert scores is None
    _validate_candidate_edges(edge_index, node_count=4)


def test_candidate_edge_validation_and_undirected_filtering():
    candidates = torch.tensor([[1, 0, 2], [0, 2, 3]], dtype=torch.long)
    scores = torch.tensor([0.9, 0.8, 0.7])
    known = torch.tensor([[0], [1]], dtype=torch.long)

    new_edges, new_scores = _filter_new_edges(candidates, scores, known)

    assert new_edges.tolist() == [[0, 2], [2, 3]]
    assert torch.equal(new_scores, torch.tensor([0.8, 0.7]))
    with pytest.raises(ValueError, match="valid node range"):
        _validate_candidate_edges(torch.tensor([[0], [4]]), node_count=4)
