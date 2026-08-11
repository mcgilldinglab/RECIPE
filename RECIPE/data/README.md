# RECIPE Data Layout

This directory contains runtime inputs used by the RECIPE command-line workflows:

- `bulk/`: reference tables and sequence embeddings for bulk prediction.
- `networks/`: known and unknown protein-interaction graphs.
- `pausing/`: CDS annotations and pausing features.
- `single_cell/`: expression matrices, metadata, embeddings, and targets.
- `splits/`: fixed train/validation/test assignments.

See [`docs/data.md`](../docs/data.md) for the complete data inventory, Git LFS information, and external data downloads. See [`docs/reproduction.md`](../docs/reproduction.md) for data preparation and workflow-specific commands.
