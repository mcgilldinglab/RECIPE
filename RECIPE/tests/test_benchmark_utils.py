from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


BENCHMARK_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "single_cell"
sys.path.insert(0, str(BENCHMARK_DIR))

from benchmark_utils import read_expression_matrix  # noqa: E402


def test_read_expression_matrix_detects_bundled_identifier_column(tmp_path):
    path = tmp_path / "expression.csv"
    pd.DataFrame(
        {
            "Unnamed: 0": ["ENST1.1", "ENST2.2"],
            "cell_a": [1.0, 2.0],
            "cell_b": [3.0, 4.0],
        }
    ).to_csv(path, index=False)

    gene_ids, cell_names, matrix = read_expression_matrix(path)

    assert gene_ids.tolist() == ["ENST1", "ENST2"]
    assert cell_names.tolist() == ["cell_a", "cell_b"]
    assert matrix.shape == (2, 2)


def test_read_expression_matrix_accepts_custom_identifier_column(tmp_path):
    path = tmp_path / "expression.csv"
    pd.DataFrame({"gene": ["A", "B"], "cell": [1.0, 2.0]}).to_csv(path, index=False)

    gene_ids, _, _ = read_expression_matrix(path, transcript_column="gene")

    assert gene_ids.tolist() == ["A", "B"]
