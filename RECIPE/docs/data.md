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

## Large Files

These larger runtime assets are tracked with Git LFS in the repository:

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

## External Human PPI Graph

This file is required only for the human unknown-protein workflow and is not stored in GitHub:

- `data/networks/human_ppi_unknown.csv`

Download the compressed graph from [Google Drive](https://drive.google.com/file/d/1UIefENLMUvWTJ9K8jxVmuGdPFD4Vtnrc/view?usp=sharing). The compressed file is 3,306,294,128 bytes (3.08 GiB) and expands to about 54 GB. Its SHA-256 checksum is:

```text
2c7b7cd3e3ca7de35354aa81a6caf34c37da8d4e20406329b49bde09dd48704e
```

From the package directory (`RECIPE/RECIPE`), download, verify, and extract it with:

```bash
mkdir -p data/networks
curl --location --fail --retry 3 --continue-at - \
  'https://drive.usercontent.google.com/download?id=1UIefENLMUvWTJ9K8jxVmuGdPFD4Vtnrc&export=download&confirm=t' \
  --output data/networks/human_ppi_unknown.csv.gz
printf '%s  %s\n' \
  '2c7b7cd3e3ca7de35354aa81a6caf34c37da8d4e20406329b49bde09dd48704e' \
  'data/networks/human_ppi_unknown.csv.gz' | sha256sum --check -
gzip --decompress --keep data/networks/human_ppi_unknown.csv.gz
```

Keep at least 60 GB of free disk space for the compressed and extracted copies. For publication, a stable archive such as Zenodo, Figshare, OSF, or an institutional repository is preferable to Google Drive.

## Data Preparation

After cloning, download Git LFS files and check the reproduction inputs:

```bash
git lfs pull
python scripts/prepare_public_data.py --data-root data --model-root models --manifest-json outputs/reproduce/data_preparation.json
```

To build the mouse coexpression matrix used in Module C summaries:

```bash
python scripts/prepare_public_data.py --data-root data --model-root models --build-mouse-coexpression
```

## Explicit Reproduction Inputs

The public reproduction commands pass these input paths explicitly under `DATA_ROOT`:

- Module A: known-protein bulk prediction with `bulk/{human,mouse}_reference.csv`, `bulk/{human,mouse}_sequence_known.npy`, and `networks/{human,mouse}_ppi_known.csv`.
- Module B: unknown-protein bulk inference with `bulk/{human,mouse}_reference.csv`, `bulk/{human,mouse}_sequence_unknown.npy`, and `networks/{human,mouse}_ppi_unknown.csv`. The human unknown PPI graph is external and not distributed through GitHub.
- Module C: PPI refinement with known-protein bulk inputs, optional generated `networks/{human,mouse}_coexpression.csv`, plus a Module A known-protein checkpoint such as `models/bulk/mouse_known_seed5.pth` or `outputs/reproduce/module_a_mouse_known/model.pth`.
- Module D scRibo-seq branch: `bulk/human_reference.csv`, `bulk/single_cell_transfer_sequence.npy`, `networks/single_cell_transfer_ppi.csv`, `pausing/cds_annotations.csv`, `pausing/human_nc2_pause.csv`, `pausing/fraction_rich_pause.csv`, `pausing/pseudobulk_pause_matrix.csv`, `single_cell/expression_raw.csv`, `single_cell/expression_normalized.csv`, and `single_cell/metadata.csv`.
- Module D scRNA-seq branch: an external ENSMUSP scRNA/bulk-protein bundle, the matching PPI CSV, a phase2 hidden-cache directory, and nanoSPINS truth/mapping files passed explicitly to `scripts/run_module_d.py --assay scrnaseq`.

Fixed split reference files are under `data/splits/`.
