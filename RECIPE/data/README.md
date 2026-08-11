# RECIPE Data Layout

This directory contains runtime data used by `recipe.config`.

## Smoke Demo

The small simulated demo dataset is stored outside this runtime directory:

- `../examples/smoke_data/bulk_reference.csv`
- `../examples/smoke_data/sequence_embeddings.csv`
- `../examples/smoke_data/ppi_matrix.csv`

Run it from the package root:

```bash
python scripts/run_smoke_demo.py --device cpu --output-dir outputs/smoke_demo
```

## Large Files

Large runtime assets in this directory are tracked with Git LFS when they are distributed through the repository.

## Training Splits

Fixed train/validation/test split CSV files are stored in `splits/`. They are small enough to commit directly and are used by the command-line runners and optional training references.

Regenerate them with:

```bash
python scripts/build_training_splits.py
```

## Public Data Preparation

From the package root, check the reproduction inputs with:

```bash
python scripts/prepare_public_data.py --data-root data --model-root models --manifest-json outputs/reproduce/data_preparation.json
```

Build the Module C mouse coexpression matrix with:

```bash
python scripts/prepare_public_data.py --data-root data --model-root models --build-mouse-coexpression
```

## Pausing Data

The pausing CSV files can be regenerated from CDS annotations and coordinate-sorted, indexed Ribo-seq BAM files with:

```bash
python scripts/compute_pausing.py score-bam \
  --cds-csv data/pausing/cds_annotations.csv \
  --bam /path/to/riboseq.sorted.bam \
  --score-csv outputs/pausing/pause_scores.csv

python scripts/compute_pausing.py summarize \
  --scores-csv outputs/pausing/pause_scores.csv \
  --summary-cds-csv data/pausing/cds_annotations.csv \
  --output-csv outputs/pausing/high_pause_counts.csv
```

The default `--average-denominator length` mode follows the original scripts and uses the `Length` column for average read-depth normalization. Use `--average-denominator positions` when `Length` is not the analyzed CDS length after trimming.

These files are tracked with Git LFS:

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

This file is external and is not uploaded to GitHub:

- `data/networks/human_ppi_unknown.csv`

That file is about 51-54 GB locally. It can be shared through Google Drive for review and should be deposited in a stable archive for publication when possible.

For a package installed outside the repository, pass `--data-root /path/to/RECIPE/RECIPE/data` or the file-level input arguments listed in `docs/reproduction.md`.
