# Script Index

Packaged entry points:

- `scripts/build_data_aliases.py`: rebuild the packaged English alias tree under `data/`
- `scripts/build_bulk_features.py`: export packaged bulk feature tables for human or mouse, known or unknown
- `scripts/build_coexpression.py`: rebuild the packaged coexpression matrices
- `scripts/build_single_cell_inputs.py`: rebuild normalized single-cell expression and the pseudobulk pause matrix
- `scripts/build_all_data.py`: run the packaged data build end to end
- `scripts/run_module_a.py`: module A, proteomics-undetected bulk inference
- `scripts/run_module_b.py`: module B, known bulk protein prediction
- `scripts/run_module_c.py`: module C, self-supervised PPI refinement
- `scripts/run_module_d.py`: module D, single-cell transfer
- `scripts/run_recipe.py`: combined multi-module runner

Core package modules:

- `src/recipe/data_construction.py`: packaged data builders for aliases, bulk features, coexpression, and single-cell inputs
- `src/recipe/assets.py`: English path aliases for curated data and checkpoints
- `src/recipe/config.py`: module-level dataset and checkpoint configuration
- `src/recipe/bulk_workflow.py`: packaged bulk runner used by modules A and B
- `src/recipe/ppi_workflow.py`: packaged PPI refinement runner used by module C
- `src/recipe/single_cell_riboseq_workflow.py`: packaged three-stage single-cell workflow used by module D
- `src/recipe/pipeline.py`: combined pipeline entry for scripted orchestration
