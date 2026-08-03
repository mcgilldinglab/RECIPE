# Training Notebooks

This directory contains public, sanitized training notebooks derived from the manuscript workflow notebooks. Execution outputs, execution counts, and local absolute paths were removed before adding them to the repository.

Before rerunning the notebooks, replace these placeholders with local paths:

- `<RECIPE_PROJECT_ROOT>`: local project workspace containing runtime data and outputs.
- `<PAUSING_SOURCE_ROOT>`: local source directory for pausing-score intermediate files.
- `<DNABERT_ROOT>`: local DNABERT checkout, if sequence-embedding utilities are reused.
- `<LOCAL_DATA_ROOT>`: local root for external datasets not distributed through GitHub.

The packaged command-line entry points in `../scripts/` are recommended for reproducible runs. Fixed train/validation/test CSV files are provided in `../data/splits/`. These notebooks are provided as training references and tutorials.

## Included Notebooks

- `training/bulk_mouse_unknown_training.ipynb`: bulk mouse unknown protein prediction with early stopping.
- `training/ppi_refinement_training.ipynb`: self-supervised PPI edge-refinement model training.
- `training/bulk_self_learning_training.ipynb`: bulk self-learning training for known and unknown targets.
- `training/single_cell_module_a_finetuning.ipynb`: single-cell Module A fine-tuning workflow.
- `training/single_cell_graph_training.ipynb`: single-cell graph model training workflow.
