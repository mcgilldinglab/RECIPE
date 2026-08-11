from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse as sp

from recipe.bulk_data import load_ppi_graph, load_ppi_matrix


def test_load_ppi_matrix_supports_csv_and_npz(tmp_path):
    values = np.asarray([[0.0, 0.5], [0.5, 0.0]], dtype=np.float32)
    csv_path = tmp_path / "ppi.csv"
    npz_path = tmp_path / "ppi.npz"
    pd.DataFrame(values).to_csv(csv_path, index=False)
    sp.save_npz(npz_path, sp.csr_matrix(values))

    np.testing.assert_allclose(load_ppi_matrix(csv_path).toarray(), values)
    np.testing.assert_allclose(load_ppi_matrix(npz_path).toarray(), values)
    edge_index, edge_weight = load_ppi_graph(npz_path, add_loops=False)
    assert edge_index.shape == (2, 2)
    np.testing.assert_allclose(edge_weight.numpy(), [0.5, 0.5])


def test_load_ppi_matrix_rejects_invalid_csv(tmp_path):
    non_numeric = tmp_path / "non_numeric.csv"
    non_square = tmp_path / "non_square.csv"
    pd.DataFrame({"source": ["A"], "target": ["B"]}).to_csv(non_numeric, index=False)
    pd.DataFrame([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]).to_csv(non_square, index=False)

    with pytest.raises(ValueError, match="numeric"):
        load_ppi_matrix(non_numeric)
    with pytest.raises(ValueError, match="square"):
        load_ppi_matrix(non_square)
