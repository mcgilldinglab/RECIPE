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

Download the compressed graph from [Google Drive](https://drive.google.com/file/d/1UIefENLMUvWTJ9K8jxVmuGdPFD4Vtnrc/view?usp=sharing). The compressed file is 3,957,792 bytes (3.77 MiB) and expands to 101,765,398 bytes (97.05 MiB). Its SHA-256 checksum is:

```text
56d278c0244f1288989d9a0be929e8cd96deab0d4e0fdd1ecd4ff8596d73e95b
```

From the package directory (`RECIPE/RECIPE`), download, verify, and extract it with:

```bash
mkdir -p data/networks
curl --location --fail --retry 3 --continue-at - \
  'https://drive.usercontent.google.com/download?id=1UIefENLMUvWTJ9K8jxVmuGdPFD4Vtnrc&export=download&confirm=t' \
  --output data/networks/human_ppi_unknown.csv.gz
printf '%s  %s\n' \
  '56d278c0244f1288989d9a0be929e8cd96deab0d4e0fdd1ecd4ff8596d73e95b' \
  'data/networks/human_ppi_unknown.csv.gz' | sha256sum --check -
gzip --decompress --keep data/networks/human_ppi_unknown.csv.gz
```

Keep at least 200 MiB of free disk space for the compressed and extracted copies. For publication, a stable archive such as Zenodo, Figshare, OSF, or an institutional repository is preferable to Google Drive.

## Data Preparation

Run the preparation, validation, and split-generation commands in [`reproduction.md`](reproduction.md#data-preparation). That page also lists the explicit inputs for each module.
