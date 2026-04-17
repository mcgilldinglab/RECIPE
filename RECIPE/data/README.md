# RECIPE Data Layout

This directory contains the packaged runtime assets used by `recipe.config`.

## Recommended GitHub Strategy

- Commit small CSV and metadata files directly to Git.
- Track large runtime assets with Git LFS.
- Keep oversized assets outside GitHub and document how to fetch them.

## PPI Files To Upload

For GitHub, only the smaller PPI graphs needed for packaged training and inference should be uploaded.

Track these with Git LFS:

- `data/networks/human_ppi_known.csv`
- `data/networks/mouse_ppi_known.csv`
- `data/networks/mouse_ppi_unknown.csv`
- `data/networks/single_cell_transfer_ppi.csv`
- `data/networks/human_coexpression.csv`
- `data/networks/mouse_coexpression.csv`

Do not upload this file to GitHub:

- `data/networks/human_ppi_unknown.csv`

That file is about `54 GB` locally and should stay in external storage, with a documented download step.

## Other Runtime Assets

These packaged assets are also referenced by the code and should stay alongside the repository if you want the packaged workflows to run without rebuilding data:

- `data/bulk/human_reference.csv`
- `data/bulk/mouse_reference.csv`
- `data/bulk/human_sequence_known.npy`
- `data/bulk/human_sequence_unknown.npy`
- `data/bulk/mouse_sequence_known.npy`
- `data/bulk/mouse_sequence_unknown.npy`
- `data/bulk/single_cell_transfer_sequence.npy`
- `data/pausing/cds_annotations.csv`
- `data/pausing/human_nc2_pause.csv`
- `data/pausing/human_scribo_pause.csv`
- `data/pausing/pseudobulk_pause_matrix.csv`
- `data/single_cell/expression_raw.csv`
- `data/single_cell/expression_normalized.csv`
- `data/single_cell/metadata.csv`
- `data/single_cell/scriboseq_metadata.csv`
- `data/single_cell/cell_embeddings.npy`
- `data/single_cell/cell_outputs.npy`
- `data/single_cell/predicted_cell_matrix.csv`
- `data/single_cell/predicted_cell_matrix_seed123.csv`

## Git LFS Setup

From the repository root:

```bash
git lfs install
git add RECIPE/.gitattributes
git add RECIPE/data
```

If you later move the full-size `human_ppi_unknown.csv` to external storage, keep the packaged path documented and provide either:

- a download script, or
- a release asset / object-storage URL, or
- a separate institutional data location.
