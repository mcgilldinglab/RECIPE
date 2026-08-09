# Reproduction Commands

This page lists the commands a new user can run after cloning the repository.

## Clone And Install

```bash
git lfs install
git clone https://github.com/mcgilldinglab/RECIPE.git
cd RECIPE/RECIPE
git lfs pull

conda activate recipe
python -m pip install -e . --no-deps
```

Use the fresh-environment instructions in `installation.md` if PyTorch and PyTorch Geometric are not already installed.

Set these shell variables to the locations on your machine. The example below uses the packaged repository data, but the same commands work if `DATA_ROOT` points to another directory with the same files.

```bash
DATA_ROOT="${PWD}/data"
MODEL_ROOT="${PWD}/models"
OUTPUT_ROOT="${PWD}/outputs/reproduce"
```

## Quick Check

```bash
python scripts/run_smoke_demo.py --device cpu --output-dir outputs/smoke_demo
```

Expected run time: under 1 minute. Expected files:

- `outputs/smoke_demo/predictions.csv`
- `outputs/smoke_demo/embeddings.npy`
- `outputs/smoke_demo/metrics.json`

## Public Task Reproduction

Run commands from the `RECIPE/` package directory.

### Module A: Bulk Unknown Protein Prediction

Inputs passed explicitly:

- `${DATA_ROOT}/bulk/mouse_reference.csv`
- `${DATA_ROOT}/bulk/mouse_sequence_unknown.npy`
- `${DATA_ROOT}/networks/mouse_ppi_unknown.csv`
- `${DATA_ROOT}/splits/bulk_mouse_unknown_seed12.csv`

```bash
python scripts/run_module_a.py \
  --species mouse \
  --condition KD \
  --seed 12 \
  --device auto \
  --reference-csv "${DATA_ROOT}/bulk/mouse_reference.csv" \
  --sequence-npy "${DATA_ROOT}/bulk/mouse_sequence_unknown.npy" \
  --ppi-csv "${DATA_ROOT}/networks/mouse_ppi_unknown.csv" \
  --split-csv "${DATA_ROOT}/splits/bulk_mouse_unknown_seed12.csv" \
  --output-dir "${OUTPUT_ROOT}/module_a_mouse_unknown"
```

Expected files:

- `${OUTPUT_ROOT}/module_a_mouse_unknown/predictions.csv`
- `${OUTPUT_ROOT}/module_a_mouse_unknown/embeddings.npy`
- `${OUTPUT_ROOT}/module_a_mouse_unknown/metrics.json`
- `${OUTPUT_ROOT}/module_a_mouse_unknown/model.pth`

### Module B: Bulk Known Protein Prediction

Inputs passed explicitly:

- `${DATA_ROOT}/bulk/mouse_reference.csv`
- `${DATA_ROOT}/bulk/mouse_sequence_known.npy`
- `${DATA_ROOT}/networks/mouse_ppi_known.csv`
- `${DATA_ROOT}/splits/bulk_mouse_known_seed12.csv`

```bash
python scripts/run_module_b.py \
  --species mouse \
  --condition KD \
  --seed 12 \
  --device auto \
  --reference-csv "${DATA_ROOT}/bulk/mouse_reference.csv" \
  --sequence-npy "${DATA_ROOT}/bulk/mouse_sequence_known.npy" \
  --ppi-csv "${DATA_ROOT}/networks/mouse_ppi_known.csv" \
  --split-csv "${DATA_ROOT}/splits/bulk_mouse_known_seed12.csv" \
  --output-dir "${OUTPUT_ROOT}/module_b_mouse_known"
```

Expected files:

- `${OUTPUT_ROOT}/module_b_mouse_known/predictions.csv`
- `${OUTPUT_ROOT}/module_b_mouse_known/embeddings.npy`
- `${OUTPUT_ROOT}/module_b_mouse_known/metrics.json`
- `${OUTPUT_ROOT}/module_b_mouse_known/model.pth`

### Module C: PPI Refinement

Module C requires a trained bulk checkpoint. Run Module B first, then pass its checkpoint:

Inputs passed explicitly:

- `${OUTPUT_ROOT}/module_b_mouse_known/model.pth`
- `${DATA_ROOT}/bulk/mouse_reference.csv`
- `${DATA_ROOT}/bulk/mouse_sequence_known.npy`
- `${DATA_ROOT}/networks/mouse_ppi_known.csv`
- `${DATA_ROOT}/networks/mouse_coexpression.csv`

```bash
python scripts/run_module_c.py \
  --species mouse \
  --condition KD \
  --seed 12 \
  --device auto \
  --reference-csv "${DATA_ROOT}/bulk/mouse_reference.csv" \
  --sequence-npy "${DATA_ROOT}/bulk/mouse_sequence_known.npy" \
  --ppi-csv "${DATA_ROOT}/networks/mouse_ppi_known.csv" \
  --coexpression-csv "${DATA_ROOT}/networks/mouse_coexpression.csv" \
  --bulk-checkpoint-path "${OUTPUT_ROOT}/module_b_mouse_known/model.pth" \
  --output-dir "${OUTPUT_ROOT}/module_c_mouse_ppi"
```

Expected files:

- `${OUTPUT_ROOT}/module_c_mouse_ppi/candidate_edges.csv`
- `${OUTPUT_ROOT}/module_c_mouse_ppi/known_edge_scores.csv`
- `${OUTPUT_ROOT}/module_c_mouse_ppi/edge_classifier.pth`
- `${OUTPUT_ROOT}/module_c_mouse_ppi/bulk_node_embeddings.npy`
- `${OUTPUT_ROOT}/module_c_mouse_ppi/summary.json`

### Module D: Single-Cell Transfer

Inputs passed explicitly:

- `${DATA_ROOT}/bulk/human_reference.csv`
- `${DATA_ROOT}/bulk/single_cell_transfer_sequence.npy`
- `${DATA_ROOT}/networks/single_cell_transfer_ppi.csv`
- `${DATA_ROOT}/pausing/cds_annotations.csv`
- `${DATA_ROOT}/pausing/human_nc2_pause.csv`
- `${DATA_ROOT}/pausing/fraction_rich_pause.csv`
- `${DATA_ROOT}/pausing/pseudobulk_pause_matrix.csv`
- `${DATA_ROOT}/single_cell/expression_raw.csv`
- `${DATA_ROOT}/single_cell/expression_normalized.csv`
- `${DATA_ROOT}/single_cell/metadata.csv`
- `${DATA_ROOT}/splits/single_cell_self_learning_seed12.csv`
- `${DATA_ROOT}/splits/single_cell_module_a_seed42.csv`
- `${DATA_ROOT}/splits/single_cell_graph_seed42.csv`

```bash
python scripts/run_module_d.py \
  --steps phase0,phase1,phase2 \
  --seed 12 \
  --device auto \
  --bulk-reference-csv "${DATA_ROOT}/bulk/human_reference.csv" \
  --transcript-order-csv "${DATA_ROOT}/single_cell/expression_normalized.csv" \
  --sequence-npy "${DATA_ROOT}/bulk/single_cell_transfer_sequence.npy" \
  --ppi-csv "${DATA_ROOT}/networks/single_cell_transfer_ppi.csv" \
  --cds-csv "${DATA_ROOT}/pausing/cds_annotations.csv" \
  --phase0-pause-csv "${DATA_ROOT}/pausing/human_nc2_pause.csv" \
  --phase1-pause-csv "${DATA_ROOT}/pausing/fraction_rich_pause.csv" \
  --expression-csv "${DATA_ROOT}/single_cell/expression_raw.csv" \
  --expression-normalized-csv "${DATA_ROOT}/single_cell/expression_normalized.csv" \
  --metadata-csv "${DATA_ROOT}/single_cell/metadata.csv" \
  --pause-matrix-csv "${DATA_ROOT}/pausing/pseudobulk_pause_matrix.csv" \
  --phase0-init-checkpoint "${MODEL_ROOT}/single_cell/bulk_self_learning.pth" \
  --phase0-split-csv "${DATA_ROOT}/splits/single_cell_self_learning_seed12.csv" \
  --phase1-split-csv "${DATA_ROOT}/splits/single_cell_module_a_seed42.csv" \
  --phase2-split-csv "${DATA_ROOT}/splits/single_cell_graph_seed42.csv" \
  --output-dir "${OUTPUT_ROOT}/module_d_single_cell"
```

Expected files include:

- `${OUTPUT_ROOT}/module_d_single_cell/single_cell_transfer_summary.json`
- `${OUTPUT_ROOT}/module_d_single_cell/phase0/phase0_summary.json`
- `${OUTPUT_ROOT}/module_d_single_cell/phase1/phase1_summary.json`
- `${OUTPUT_ROOT}/module_d_single_cell/phase2/phase2_summary.json`

## One-Command Pipeline

To run modules A-D in order:

```bash
python scripts/run_recipe.py \
  --modules A,B,C,D \
  --species mouse \
  --condition KD \
  --seed 12 \
  --device auto \
  --data-root "${DATA_ROOT}" \
  --bulk-unknown-split-csv "${DATA_ROOT}/splits/bulk_mouse_unknown_seed12.csv" \
  --bulk-known-split-csv "${DATA_ROOT}/splits/bulk_mouse_known_seed12.csv" \
  --phase0-split-csv "${DATA_ROOT}/splits/single_cell_self_learning_seed12.csv" \
  --phase1-split-csv "${DATA_ROOT}/splits/single_cell_module_a_seed42.csv" \
  --phase2-split-csv "${DATA_ROOT}/splits/single_cell_graph_seed42.csv" \
  --output-root "${OUTPUT_ROOT}/all_modules"
```

When Module B and Module C are run together, Module C uses `${OUTPUT_ROOT}/all_modules/module_b/model.pth`.

## Fixed Split Files

Train/validation/test split files are included under `data/splits/`. Regenerate them with:

```bash
python scripts/build_training_splits.py
```

Pass these files with `--split-csv` for bulk modules, `--phase0-split-csv`, `--phase1-split-csv`, and `--phase2-split-csv` for Module D, or the matching split arguments in `run_recipe.py`. If a split file is not provided, the runners fall back to generating the train/validation/test partitions from the seed value.

## Notes

- The mouse workflows can be reproduced from files included in the repository.
- The human unknown workflow requires `data/networks/human_ppi_unknown.csv`, which is about 51-54 GB and is distributed outside GitHub.
- Full-size model training is intended for a CUDA-capable GPU. The tested full workflows were run on one NVIDIA RTX 4090 GPU.
