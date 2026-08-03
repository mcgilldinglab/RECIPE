# Data

RECIPE uses two kinds of data:

- A tiny simulated demo dataset in `examples/smoke_data/`.
- Runtime data aliases under `data/` for the packaged workflows.

## Smoke Demo Dataset

The smoke demo dataset contains:

- `examples/smoke_data/bulk_reference.csv`
- `examples/smoke_data/sequence_embeddings.csv`
- `examples/smoke_data/ppi_matrix.csv`

Run it with:

```bash
python scripts/run_smoke_demo.py --device cpu --output-dir outputs/smoke_demo
```

Expected run time: under 1 minute. The tested `pyg` CPU run completed in about 4.7 seconds wall time.

## GitHub / Git LFS Strategy

Commit small CSV and metadata files directly. Track large runtime assets with Git LFS when they are suitable for GitHub:

- `data/bulk/human_sequence_known.npy`
- `data/bulk/human_sequence_unknown.npy`
- `data/bulk/mouse_sequence_known.npy`
- `data/bulk/mouse_sequence_unknown.npy`
- `data/bulk/single_cell_transfer_sequence.npy`
- `data/networks/human_ppi_known.csv`
- `data/networks/mouse_ppi_known.csv`
- `data/networks/mouse_ppi_unknown.csv`
- `data/networks/single_cell_transfer_ppi.csv`
- `data/single_cell/cell_embeddings.npy`

Do not upload this file to GitHub:

- `data/networks/human_ppi_unknown.csv`

That graph is about 51-54 GB locally and should be distributed through external storage.

## External Data Roots

For an installed package with data outside site-packages:

```bash
export RECIPE_DATA_ROOT=/path/to/RECIPE/RECIPE/data
export RECIPE_MODEL_ROOT=/path/to/RECIPE/RECIPE/models
```

To rebuild aliases from a private source data tree, arrange that tree with the same relative layout as `data/` and then run:

```bash
export RECIPE_SOURCE_DATA_ROOT=/path/to/source/project
python scripts/build_data_aliases.py --manifest-json data/alias_manifest.json
```
