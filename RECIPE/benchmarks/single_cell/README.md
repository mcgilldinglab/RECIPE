# Single-Cell Benchmark Wrappers

This directory contains lightweight wrappers used for the single-cell RNA-seq benchmark comparisons.
They do not include raw data, trained scTranslator checkpoints, or local result folders.

## Inputs

Prepare these files before running the wrappers:

- A gene-by-cell expression CSV with a `transcript_id` column.
- A protein mapping CSV with transcript IDs and protein target values.
- A bulk reference CSV containing the protein target column used for alignment. The examples below use `NC3`; replace it with the target column in your table.
- For scTranslator only: a local checkout of the upstream scTranslator repository and its pretrained checkpoint.

## Export scTranslator Inputs

```bash
python prepare_single_cell_sctranslator_data.py \
  --expression-csv /path/to/expression_normalized.csv \
  --protein-map-csv /path/to/protein_map.csv \
  --bulk-table-csv /path/to/human_reference.csv \
  --protein-column NC3 \
  --seed 12 \
  --export-test-train-gene-chunks \
  --export-test-test-gene-chunks \
  --output-dir outputs/benchmarks/sctranslator_inputs
```

## KRR

```bash
python run_single_cell_kernel_ridge_benchmark.py \
  --expression-csv /path/to/expression_normalized.csv \
  --protein-map-csv /path/to/protein_map.csv \
  --bulk-table-csv /path/to/human_reference.csv \
  --protein-column NC3 \
  --seed 12 \
  --output-dir outputs/benchmarks/krr
```

## VanillaNN

```bash
python run_single_cell_vanillann_benchmark.py \
  --expression-csv /path/to/expression_normalized.csv \
  --protein-map-csv /path/to/protein_map.csv \
  --bulk-table-csv /path/to/human_reference.csv \
  --protein-column NC3 \
  --seed 12 \
  --device cuda:0 \
  --output-dir outputs/benchmarks/vanillann
```

## scTranslator

Use the upstream scTranslator fine-tuning command from its own repository, then evaluate chunked test files with:

```bash
python run_sctranslator_chunked_inference.py \
  --repo-root /path/to/scTranslator \
  --checkpoint /path/to/fine_tuned_sctranslator.pt \
  --data-dir outputs/benchmarks/sctranslator_inputs \
  --seed 12 \
  --device cuda:0 \
  --output-dir outputs/benchmarks/sctranslator
```

The upstream fine-tuning command used in the local benchmark was:

```bash
CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 \
  --master_addr=127.0.0.1 --master_port=23333 \
  code/stage3_fine-tune.py \
  --epoch=100 \
  --frac_finetune_test=0.25 \
  --fix_set \
  --pretrain_checkpoint=/path/to/scTranslator/checkpoint/stage2_single-cell_scTranslator.pt \
  --RNA_path=/path/to/X_train_genes_adatas12.h5ad \
  --Pro_path=/path/to/Y_train_protein_adatas12.h5ad
```

## Figure Note

`c10_best_on_c10_test.pdf` was generated from the Module D scRNA-seq branch, not from scTranslator or the scRibo-seq branch. The figure evaluates the nanoSPINS cell-graph checkpoint `phase3_nanospins_best.pth` for C10/SVEC testing.
