# RECIPE Smoke Demo Data

This is a tiny simulated bulk/PPI dataset for checking that RECIPE, PyTorch, and PyTorch Geometric run correctly.

- `bulk_reference.csv`: 8 transcript/protein rows with RNA, protein, and pause-count columns.
- `sequence_embeddings.csv`: 8 rows of 4-dimensional simulated sequence embeddings.
- `ppi_matrix.csv`: 8 by 8 simulated PPI adjacency matrix.

Run from `RECIPE/`:

```bash
python scripts/run_smoke_demo.py --device cpu --output-dir outputs/smoke_demo
```
