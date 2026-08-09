# Reproduction Commands

This page lists the commands a new user can run after cloning the repository.

## Clone And Install

```bash
git lfs install
git clone https://github.com/mcgilldinglab/RECIPE.git
cd RECIPE/RECIPE
git lfs pull

conda activate recipe
export RECIPE_DATA_ROOT="${PWD}/data"
export RECIPE_MODEL_ROOT="${PWD}/models"
export RECIPE_OUTPUT_ROOT="${PWD}/outputs"
python -m pip install -e . --no-deps
```

Use the fresh-environment instructions in `installation.md` if PyTorch and PyTorch Geometric are not already installed.

The command-line runners read inputs from `${RECIPE_DATA_ROOT}` and write outputs under `${RECIPE_OUTPUT_ROOT}` unless an explicit `--output-dir` or `--output-root` is passed.

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

Inputs:

- `${RECIPE_DATA_ROOT}/bulk/mouse_reference.csv`
- `${RECIPE_DATA_ROOT}/bulk/mouse_sequence_unknown.npy`
- `${RECIPE_DATA_ROOT}/networks/mouse_ppi_unknown.csv`
- `${RECIPE_DATA_ROOT}/splits/bulk_mouse_unknown_seed12.csv` as the fixed split reference.

```bash
python scripts/run_module_a.py \
  --species mouse \
  --condition KD \
  --seed 12 \
  --device auto \
  --output-dir outputs/reproduce/module_a_mouse_unknown
```

Expected files:

- `outputs/reproduce/module_a_mouse_unknown/predictions.csv`
- `outputs/reproduce/module_a_mouse_unknown/embeddings.npy`
- `outputs/reproduce/module_a_mouse_unknown/metrics.json`
- `outputs/reproduce/module_a_mouse_unknown/model.pth`

### Module B: Bulk Known Protein Prediction

Inputs:

- `${RECIPE_DATA_ROOT}/bulk/mouse_reference.csv`
- `${RECIPE_DATA_ROOT}/bulk/mouse_sequence_known.npy`
- `${RECIPE_DATA_ROOT}/networks/mouse_ppi_known.csv`
- `${RECIPE_DATA_ROOT}/splits/bulk_mouse_known_seed12.csv` as the fixed split reference.

```bash
python scripts/run_module_b.py \
  --species mouse \
  --condition KD \
  --seed 12 \
  --device auto \
  --output-dir outputs/reproduce/module_b_mouse_known
```

Expected files:

- `outputs/reproduce/module_b_mouse_known/predictions.csv`
- `outputs/reproduce/module_b_mouse_known/embeddings.npy`
- `outputs/reproduce/module_b_mouse_known/metrics.json`
- `outputs/reproduce/module_b_mouse_known/model.pth`

### Module C: PPI Refinement

Module C requires a trained bulk checkpoint. Run Module B first, then pass its checkpoint:

Inputs:

- `outputs/reproduce/module_b_mouse_known/model.pth`
- `${RECIPE_DATA_ROOT}/bulk/mouse_reference.csv`
- `${RECIPE_DATA_ROOT}/bulk/mouse_sequence_known.npy`
- `${RECIPE_DATA_ROOT}/networks/mouse_ppi_known.csv`

```bash
python scripts/run_module_c.py \
  --species mouse \
  --condition KD \
  --seed 12 \
  --device auto \
  --bulk-checkpoint-path outputs/reproduce/module_b_mouse_known/model.pth \
  --output-dir outputs/reproduce/module_c_mouse_ppi
```

Expected files:

- `outputs/reproduce/module_c_mouse_ppi/candidate_edges.csv`
- `outputs/reproduce/module_c_mouse_ppi/known_edge_scores.csv`
- `outputs/reproduce/module_c_mouse_ppi/edge_classifier.pth`
- `outputs/reproduce/module_c_mouse_ppi/bulk_node_embeddings.npy`
- `outputs/reproduce/module_c_mouse_ppi/summary.json`

### Module D: Single-Cell Transfer

Inputs:

- `${RECIPE_DATA_ROOT}/bulk/human_reference.csv`
- `${RECIPE_DATA_ROOT}/bulk/single_cell_transfer_sequence.npy`
- `${RECIPE_DATA_ROOT}/networks/single_cell_transfer_ppi.csv`
- `${RECIPE_DATA_ROOT}/pausing/cds_annotations.csv`
- `${RECIPE_DATA_ROOT}/pausing/human_nc2_pause.csv`
- `${RECIPE_DATA_ROOT}/pausing/pseudobulk_pause_matrix.csv`
- `${RECIPE_DATA_ROOT}/single_cell/expression_raw.csv`
- `${RECIPE_DATA_ROOT}/single_cell/expression_normalized.csv`
- `${RECIPE_DATA_ROOT}/single_cell/metadata.csv`
- `${RECIPE_DATA_ROOT}/splits/single_cell_self_learning_seed12.csv`, `${RECIPE_DATA_ROOT}/splits/single_cell_module_a_seed42.csv`, and `${RECIPE_DATA_ROOT}/splits/single_cell_graph_seed42.csv` as fixed split references.

```bash
python scripts/run_module_d.py \
  --steps phase0,phase1,phase2 \
  --seed 12 \
  --device auto \
  --output-dir outputs/reproduce/module_d_single_cell
```

Expected files include:

- `outputs/reproduce/module_d_single_cell/single_cell_transfer_summary.json`
- `outputs/reproduce/module_d_single_cell/phase0/phase0_summary.json`
- `outputs/reproduce/module_d_single_cell/phase1/phase1_summary.json`
- `outputs/reproduce/module_d_single_cell/phase2/phase2_summary.json`

## One-Command Pipeline

To run modules A-D in order:

```bash
python scripts/run_recipe.py \
  --modules A,B,C,D \
  --species mouse \
  --condition KD \
  --seed 12 \
  --device auto \
  --output-root outputs/reproduce/all_modules
```

When Module B and Module C are run together, Module C uses `outputs/reproduce/all_modules/module_b/model.pth`.

## Fixed Split Files

Train/validation/test split files are included under `data/splits/`. Regenerate them with:

```bash
python scripts/build_training_splits.py
```

The packaged runners generate the same train/validation/test partitions internally from the seed values. The CSV files are included so users can inspect and cite the exact row membership.

## Notes

- The mouse workflows can be reproduced from files included in the repository.
- The human unknown workflow requires `data/networks/human_ppi_unknown.csv`, which is about 51-54 GB and is distributed outside GitHub.
- Full-size model training is intended for a CUDA-capable GPU. The tested full workflows were run on one NVIDIA RTX 4090 GPU.
