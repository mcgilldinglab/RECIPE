# Script Index

Demo:

- `scripts/run_smoke_demo.py`: CPU-friendly smoke test using `examples/smoke_data/`.

Packaged module entry points:

- `scripts/run_module_a.py`: module A, known bulk protein prediction.
- `scripts/run_module_b.py`: module B, bulk unknown protein inference.
- `scripts/run_module_c.py`: module C, self-supervised PPI refinement.
- `scripts/run_module_d.py`: module D; choose `--assay scriboseq` for scRibo-seq input or `--assay scrnaseq` for scRNA-seq input. Use `--scriboseq-reproduction-preset seed7_npcs20_k7_all_labeled` for the archived scRibo-seq seed7/k7/npcs20 run.
- `scripts/run_recipe.py`: combined multi-module runner.

Data construction entry points:

- `scripts/prepare_public_data.py`: check reproduction inputs and build derived files used by the reproduction commands.
- `scripts/build_data_aliases.py`: internal helper for rebuilding data aliases from a private source tree.
- `scripts/compute_pausing.py`: compute per-position pausing scores from CDS annotations and BAM files, then summarize high-pause counts.
- `scripts/build_bulk_features.py`: export bulk feature tables.
- `scripts/build_coexpression.py`: rebuild a coexpression matrix.
- `scripts/build_single_cell_inputs.py`: normalize the single-cell expression matrix.
- `scripts/build_training_splits.py`: export fixed train/validation/test CSV files into `data/splits/`.
- `scripts/build_all_data.py`: run lightweight data-build steps; pass `--rebuild-aliases` only when source data should be relinked.

Benchmark wrappers:

- `benchmarks/single_cell/prepare_single_cell_sctranslator_data.py`: export h5ad files for scTranslator single-cell benchmarks.
- `benchmarks/single_cell/run_sctranslator_chunked_inference.py`: evaluate a fine-tuned scTranslator checkpoint on chunked test files.
- `benchmarks/single_cell/run_single_cell_kernel_ridge_benchmark.py`: run the single-cell KRR benchmark.
- `benchmarks/single_cell/run_single_cell_vanillann_benchmark.py`: run the single-cell VanillaNN benchmark.

Core package modules:

- `src/recipe/data_construction.py`: data aliasing and construction helpers.
- `src/recipe/assets.py`: data, model, and output path resolution.
- `src/recipe/config.py`: default task-level dataset and checkpoint configuration.
- `src/recipe/bulk_workflow.py`: bulk runner used by modules A and B.
- `src/recipe/ppi_workflow.py`: PPI refinement runner used by module C.
- `src/recipe/single_cell_riboseq_workflow.py`: scRibo-seq branch used by Module D.
- `src/recipe/single_cell_rnaseq_workflow.py`: scRNA-seq branch wrapper used by Module D.
- `src/recipe/pipeline.py`: combined pipeline entry point.
