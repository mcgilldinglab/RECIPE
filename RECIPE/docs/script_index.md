# Script Index

Demo:

- `scripts/run_smoke_demo.py`: CPU-friendly smoke test using `examples/smoke_data/`.

Packaged module entry points:

- `scripts/run_module_a.py`: module A, known bulk protein prediction.
- `scripts/run_module_b.py`: module B, bulk unknown protein inference.
- `scripts/run_module_c.py`: module C, self-supervised PPI refinement.
- `scripts/run_module_d.py`: module D, single-cell transfer.
- `scripts/run_recipe.py`: combined multi-module runner.

Data construction entry points:

- `scripts/prepare_public_data.py`: check reproduction inputs and build derived files used by the reproduction commands.
- `scripts/build_data_aliases.py`: rebuild data aliases from `RECIPE_SOURCE_DATA_ROOT`.
- `scripts/compute_pausing.py`: compute per-position pausing scores from CDS annotations and BAM files, then summarize high-pause counts.
- `scripts/build_bulk_features.py`: export bulk feature tables.
- `scripts/build_coexpression.py`: rebuild a coexpression matrix.
- `scripts/build_single_cell_inputs.py`: normalize the single-cell expression matrix.
- `scripts/build_training_splits.py`: export fixed train/validation/test CSV files into `data/splits/`.
- `scripts/build_all_data.py`: run lightweight data-build steps; pass `--rebuild-aliases` only when source data should be relinked.

Training notebooks:

- `notebooks/training/bulk_mouse_unknown_training.ipynb`: bulk mouse unknown protein prediction with early stopping.
- `notebooks/training/ppi_refinement_training.ipynb`: self-supervised PPI edge-refinement model training.
- `notebooks/training/bulk_self_learning_training.ipynb`: bulk self-learning training for known and unknown targets.
- `notebooks/training/single_cell_module_a_finetuning.ipynb`: single-cell Module A fine-tuning workflow.
- `notebooks/training/single_cell_graph_training.ipynb`: single-cell graph model training workflow.

Core package modules:

- `src/recipe/data_construction.py`: data aliasing and construction helpers.
- `src/recipe/assets.py`: data, model, and output path resolution.
- `src/recipe/config.py`: default task-level dataset and checkpoint configuration.
- `src/recipe/bulk_workflow.py`: bulk runner used by modules A and B.
- `src/recipe/ppi_workflow.py`: PPI refinement runner used by module C.
- `src/recipe/single_cell_riboseq_workflow.py`: three-stage single-cell workflow used by module D.
- `src/recipe/pipeline.py`: combined pipeline entry point.
