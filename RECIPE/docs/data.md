# Data

RECIPE uses packaged data under `RECIPE/data/`.

## What Should Go To GitHub

Commit small text assets directly, and track large runtime assets with Git LFS.

For PPI uploads, keep only the smaller training and inference graphs in GitHub:

- `data/networks/human_ppi_known.csv`
- `data/networks/mouse_ppi_known.csv`
- `data/networks/mouse_ppi_unknown.csv`
- `data/networks/single_cell_transfer_ppi.csv`

These are the PPI files intended for the public repository. The very large human unknown graph is excluded:

- `data/networks/human_ppi_unknown.csv`

## Why `human_ppi_unknown.csv` Stays External

The packaged `human_ppi_unknown.csv` is much larger than the rest of the runtime assets and should be distributed outside the repository, with a documented download step.

## Large Files

The repository also contains large sequence embeddings, coexpression matrices, and single-cell embeddings. These should be tracked with Git LFS using `RECIPE/.gitattributes`.

See also:

- `RECIPE/data/README.md`
- `RECIPE/docs/script_index.md`
