# Data

RECIPE uses two kinds of data:

- A tiny simulated demo dataset in `examples/smoke_data/`.
- Runtime data under `data/` for the command-line workflows.

## Smoke Demo Dataset

The smoke demo dataset contains:

- `examples/smoke_data/bulk_reference.csv`
- `examples/smoke_data/sequence_embeddings.csv`
- `examples/smoke_data/ppi_matrix.csv`

The quick-check command and expected outputs are documented in [`reproduction.md`](reproduction.md#quick-check).

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

Run the preparation, validation, and split-generation commands in [`reproduction.md`](reproduction.md#data-preparation). That page also lists the explicit inputs for each module.
