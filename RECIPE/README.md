# RECIPE

RECIPE packages the workflow behind the manuscript into a Python project with reusable code, command-line entry points, packaged runtime data aliases, and a small smoke-test dataset.

The pipeline has four modules:

- Module A: bulk inference for proteomics-undetected or unknown proteins.
- Module B: bulk protein abundance prediction for proteins with measured labels.
- Module C: self-supervised PPI refinement.
- Module D: single-cell transfer with pseudo-bulk alignment and a cell-graph head.

## Repository Layout

- `src/recipe/`: reusable package code.
- `scripts/`: command-line entry points for modules A-D, data builders, and the smoke demo.
- `notebooks/`: sanitized training notebooks with outputs and local absolute paths removed.
- `examples/smoke_data/`: tiny simulated data for a CPU-friendly demo.
- `data/`: packaged runtime data aliases. Large arrays and graphs are tracked with Git LFS.
- `data/splits/`: fixed train/validation/test CSV files used by the training notebooks.
- `models/`: optional checkpoints. See `models/README.md`.
- `docs/`: Sphinx documentation source.

## System Requirements

Tested environment on the local `pyg` conda environment:

- Operating system: Linux `6.8.0-124-generic`, x86_64.
- Python: `3.8.0`.
- PyTorch: `2.1.1`.
- CUDA runtime reported by PyTorch: `12.1`.
- PyTorch Geometric: `2.5.3`.
- CUDA availability during test: `True`.
- Other tested Python packages: `numpy 1.24.3`, `pandas 2.0.3`, `scipy 1.10.1`, `scikit-learn 1.3.2`, `matplotlib 3.6.3`, `networkx 3.1`, `seaborn 0.13.2`, `openpyxl 3.1.5`, `typing_extensions 4.9.0`.

Package metadata supports Python `>=3.8`. The smoke demo runs on CPU. Full-size training and inference are much faster with an NVIDIA GPU; `--device auto` uses `cuda:0` when available and falls back to CPU. The full human unknown PPI graph is about 51-54 GB and is intentionally external, so that workflow needs substantially more disk and memory than the smoke demo or mouse packaged examples.

No non-standard hardware is required for installation or the smoke demo. A CUDA-capable GPU is recommended for full model training on the large graph assets.

## Installation

### Fresh Environment

Install PyTorch and PyTorch Geometric with wheels matching your CUDA or CPU setup first, then install RECIPE:

```bash
conda create -n recipe python=3.8 -y
conda activate recipe

# Example only; choose the PyTorch/PyG commands matching your CUDA or CPU setup.
python -m pip install "torch>=2.1" "torch-geometric>=2.5"

git clone https://github.com/mcgilldinglab/RECIPE.git
cd RECIPE/RECIPE
python -m pip install -r requirements.txt
python -m pip install -e .
```

Typical fresh environment setup time on a normal desktop or workstation is about 10-30 minutes, mostly depending on PyTorch/PyG wheel downloads.

### Install From GitHub

The Python package lives in the repository subdirectory `RECIPE/`:

```bash
python -m pip install "git+https://github.com/mcgilldinglab/RECIPE.git@main#subdirectory=RECIPE"
```

If you install this way and keep data outside site-packages, pass the data location when running a workflow, for example with `--data-root /path/to/RECIPE/RECIPE/data` or with the file-level arguments shown below.

## Data

The repository includes a small simulated demo dataset:

- `examples/smoke_data/bulk_reference.csv`
- `examples/smoke_data/sequence_embeddings.csv`
- `examples/smoke_data/ppi_matrix.csv`

The packaged runtime data are under `data/`. Large files are tracked with Git LFS where they are suitable for GitHub. The full `data/networks/human_ppi_unknown.csv` is not committed because it is about 51-54 GB; distribute it separately and place it at that path if you need the human unknown workflow.

To rebuild aliases from a private source data tree, arrange that tree with the same relative layout as `data/` and then run:

```bash
export RECIPE_SOURCE_DATA_ROOT=/path/to/source/project
python scripts/build_data_aliases.py --manifest-json data/alias_manifest.json
```

## Smoke Demo

Run the CPU-friendly demo:

```bash
cd /path/to/RECIPE/RECIPE
python scripts/run_smoke_demo.py --device cpu --output-dir outputs/smoke_demo
```

Expected run time on a normal desktop/workstation: under 1 minute. The tested `pyg` CPU run completed in about 4.7 seconds wall time.

Expected output:

- Console JSON containing `"demo": "smoke_bulk_graphsage"` and `"node_count": 8`.
- `outputs/smoke_demo/predictions.csv`
- `outputs/smoke_demo/embeddings.npy`
- `outputs/smoke_demo/metrics.json`
- `outputs/smoke_demo/sequence_embeddings.npy`

Example JSON fields:

```json
{
  "demo": "smoke_bulk_graphsage",
  "node_count": 8,
  "outputs": {
    "prediction_csv": "outputs/smoke_demo/predictions.csv",
    "embedding_npy": "outputs/smoke_demo/embeddings.npy",
    "metrics_json": "outputs/smoke_demo/metrics.json"
  }
}
```

## Reproduce Public Tasks

After cloning the repository, run commands from the package directory:

```bash
git lfs install
git lfs pull
python -m pip install -e . --no-deps
```

Set these paths to the data, model, and output locations on your machine:

```bash
DATA_ROOT="${PWD}/data"
MODEL_ROOT="${PWD}/models"
OUTPUT_ROOT="${PWD}/outputs/reproduce"
```

Prepare and validate public-task inputs:

```bash
python scripts/prepare_public_data.py \
  --data-root "${DATA_ROOT}" \
  --manifest-json "${OUTPUT_ROOT}/data_preparation.json"
```

For the full Module C coexpression summary, also generate the mouse coexpression matrix:

```bash
python scripts/prepare_public_data.py \
  --data-root "${DATA_ROOT}" \
  --build-mouse-coexpression \
  --manifest-json "${OUTPUT_ROOT}/data_preparation_with_coexpression.json"
```

For a minimal check:

```bash
python scripts/run_smoke_demo.py --device cpu --output-dir outputs/smoke_demo
```

For the packaged mouse and single-cell tasks:

```bash
# Module A: bulk mouse unknown protein prediction
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

# Module B: bulk mouse known protein prediction
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

# Module C: PPI refinement; run Module B first
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

# Module D: single-cell transfer
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

To run all modules in order:

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

When Module B and Module C are run together through `run_recipe.py`, Module C uses the Module B checkpoint at `${OUTPUT_ROOT}/all_modules/module_b/model.pth`. Detailed commands and expected output files are also listed in `docs/reproduction.md`.

Input data used by the commands above:

- Module A: `bulk/mouse_reference.csv`, `bulk/mouse_sequence_unknown.npy`, `networks/mouse_ppi_unknown.csv`, with split reference `splits/bulk_mouse_unknown_seed12.csv`.
- Module B: `bulk/mouse_reference.csv`, `bulk/mouse_sequence_known.npy`, `networks/mouse_ppi_known.csv`, with split reference `splits/bulk_mouse_known_seed12.csv`.
- Module C: Module B checkpoint `${OUTPUT_ROOT}/module_b_mouse_known/model.pth`, plus `bulk/mouse_reference.csv`, `bulk/mouse_sequence_known.npy`, `networks/mouse_ppi_known.csv`, and optional generated `networks/mouse_coexpression.csv`.
- Module D: `bulk/human_reference.csv`, `bulk/single_cell_transfer_sequence.npy`, `networks/single_cell_transfer_ppi.csv`, `pausing/cds_annotations.csv`, `pausing/human_nc2_pause.csv`, `pausing/fraction_rich_pause.csv`, `pausing/pseudobulk_pause_matrix.csv`, `single_cell/expression_raw.csv`, `single_cell/expression_normalized.csv`, `single_cell/metadata.csv`, and the three `splits/single_cell_*.csv` files.

## Outputs

Module A and B write:

- `predictions.csv`: transcript IDs, predictions, observed targets, split labels.
- `embeddings.npy`: learned node embeddings.
- `metrics.json`: train/validation/test metrics, scaling metadata, and checkpoint path.
- `model.pth`: created when training is run or no default checkpoint exists.

Module C writes:

- `candidate_edges.csv`
- `known_edge_scores.csv`
- `edge_classifier.pth`
- `bulk_node_embeddings.npy`
- `summary.json`

Module D writes phase-specific summaries and predictions under `phase0/`, `phase1/`, and `phase2/`.

## Running On Your Own Data

For bulk workflows, prepare:

- A reference CSV with `transcript_id`, RNA expression columns such as `rNC2` or `rKD2`, protein target columns such as `NC3` or `KD3`, and a pause-count column.
- A sequence embedding `.npy` file whose row count and order match the reference CSV.
- A square PPI adjacency CSV whose dimensions match the number of reference rows.

Use the existing workflow code directly if your column names match one of the packaged configs, or build a custom `BulkConditionSpec` and call `build_bulk_graph_from_dataframe`.

For command-line use, pass these files directly with `--reference-csv`, `--sequence-npy`, `--ppi-csv`, and optionally `--split-csv`. For a data directory that mirrors the packaged `data/` layout, pass `--data-root /path/to/data`.

## Reproduction Notes

For a minimal reproducibility check, run the smoke demo and confirm that the three output files above are produced. For manuscript-scale reproduction, run the module commands with the packaged or externally restored runtime data and record the generated `metrics.json` / `summary.json` files. Use a fixed `--seed` value, and record the exact PyTorch/PyG/CUDA versions from the tested environment section.
