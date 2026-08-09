# Data

RECIPE uses two kinds of data:

- A tiny simulated demo dataset in `examples/smoke_data/`.
- Runtime data under `data/` for the command-line workflows.

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

## Data Preparation

After cloning, download Git LFS files and check the reproduction inputs:

```bash
git lfs pull
python scripts/prepare_public_data.py --data-root data --manifest-json outputs/reproduce/data_preparation.json
```

To build the mouse coexpression matrix used in Module C summaries:

```bash
python scripts/prepare_public_data.py --data-root data --build-mouse-coexpression
```

## Explicit Reproduction Inputs

The public reproduction commands pass these input paths explicitly under `DATA_ROOT`:

- Module A: `bulk/mouse_reference.csv`, `bulk/mouse_sequence_unknown.npy`, `networks/mouse_ppi_unknown.csv`.
- Module B: `bulk/mouse_reference.csv`, `bulk/mouse_sequence_known.npy`, `networks/mouse_ppi_known.csv`.
- Module C: `bulk/mouse_reference.csv`, `bulk/mouse_sequence_known.npy`, `networks/mouse_ppi_known.csv`, optional generated `networks/mouse_coexpression.csv`, plus a Module B checkpoint such as `outputs/reproduce/module_b_mouse_known/model.pth`.
- Module D: `bulk/human_reference.csv`, `bulk/single_cell_transfer_sequence.npy`, `networks/single_cell_transfer_ppi.csv`, `pausing/cds_annotations.csv`, `pausing/human_nc2_pause.csv`, `pausing/fraction_rich_pause.csv`, `pausing/pseudobulk_pause_matrix.csv`, `single_cell/expression_raw.csv`, `single_cell/expression_normalized.csv`, and `single_cell/metadata.csv`.

Fixed split reference files are under `data/splits/`.

To rebuild aliases from a private source data tree, arrange that tree with the same relative layout as `data/` and then run:

```bash
export RECIPE_SOURCE_DATA_ROOT=/path/to/source/project
python scripts/build_data_aliases.py --manifest-json data/alias_manifest.json
```
