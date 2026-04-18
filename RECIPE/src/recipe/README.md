# RECIPE `src/recipe` Code Guide

This directory contains the core Python package for RECIPE. It organizes the code used for bulk modeling, PPI refinement, and single-cell transfer workflows.

## High-Level Structure

The files in this directory can be grouped into five categories:

1. Entrypoints and workflows
2. Configuration and paths
3. Data processing
4. Models and training
5. PPI inference

---

## Installation

Install the packaged module before importing `recipe`:

```bash
python -m pip install "git+https://github.com/mcgilldinglab/RECIPE.git@main#subdirectory=RECIPE"
```

For local development:

```bash
cd /path/to/RECIPE
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Usage

Module A, human unknown:
```bash
python3 scripts/run_module_a.py \
  --species human \
  --condition KD \
  --output-dir outputs/module_a
```

Module A, mouse unknown:
```bash
python3 scripts/run_module_a.py \
  --species mouse \
  --condition KD \
  --output-dir outputs/module_a
```

Module B, human known:
```bash
python3 scripts/run_module_b.py \
  --species human \
  --condition KD \
  --output-dir outputs/module_b
```

Module B, mouse known:
```bash
python3 scripts/run_module_b.py \
  --species mouse \
  --condition KD \
  --output-dir outputs/module_b
```

Module C, human PPI refinement:
```bash
python3 scripts/run_module_c.py \
  --species human \
  --condition KD \
  --output-dir outputs/module_c
```

Module C, mouse PPI refinement:
```bash
python3 scripts/run_module_c.py \
  --species mouse \
  --condition KD \
  --output-dir outputs/module_c
```

Module D, packaged single-cell transfer:
```bash
python3 scripts/run_module_d.py \
  --steps phase0,phase1,phase2 \
  --output-dir outputs/module_d
```

Run the complete packaged workflow:
```bash
python3 scripts/run_recipe.py \
  --modules A,B,C,D \
  --species human \
  --condition KD \
  --output-root outputs/pipeline
```

---

## Entrypoints and Workflows

### `__init__.py`

Package entrypoint.

Responsibilities:
- Exports default configurations
- Exports main model classes
- Exports the unified RNA-seq workflow entrypoints

Currently exported RNA-seq workflow functions:
- `run_rnaseq_phase0`
- `run_rnaseq_phase12`
- `run_rnaseq_phase3`
- `run_rnaseq_phase023`

### `pipeline.py`

Top-level RECIPE pipeline orchestrator.

Responsibilities:
- Runs the pipeline by module name (`A / B / C / D`)
- Connects bulk, PPI, and single-cell workflows

Use this when:
- You want a single high-level entrypoint for the overall RECIPE pipeline

### `single_cell_riboseq_workflow.py`

Legacy single-cell transfer workflow for the Ribo-seq / pausing-based route.

Responsibilities:
- Manages the older phase0 / phase1 / phase2 logic
- Uses pausing, fraction, and Ribo-seq-related inputs
- Preserves earlier notebook-style transfer experiments

Use this when:
- You want to reproduce the older single-cell Ribo-seq transfer workflow

### `single_cell_rnaseq_workflow.py`

Unified workflow wrapper for the current RNA-seq-based ENSMUSP pipeline.

Responsibilities:
- Wraps the current RNA-seq training scripts under the RECIPE package
- Provides unified entrypoints for `phase0`, `phase12`, `phase3`, and `phase023`
- Keeps the training logic in the original scripts and only centralizes invocation

Currently wrapped scripts:
- `train_phase0_ensmusp_pseudobulk_raw_bulkprot.py`
- `train_phase12_ensmusp_scRNA_bulkprot.py`
- `train_phase3_ensmusp_nanospins_matched.py`

Use this when:
- You want to run the current mouse / nanoSPINS RNA-seq workflow through the RECIPE package

---

## Configuration and Paths

### `assets.py`

Defines directory constants.

Responsibilities:
- Defines repository-relative data, model, and output roots
- Exposes standard locations such as `data/`, `models/`, and `outputs/`
- Provides `ensure_output_dir()` for creating output folders

Use this when:
- You want a central place for filesystem layout instead of hardcoding paths

### `config.py`

Defines task-level default configurations.

Responsibilities:
- Defines `BulkTaskConfig`
- Defines `SingleCellTransferConfig`
- Stores default paths for human/mouse, known/unknown, and single-cell transfer tasks

Use this when:
- You want task-specific configuration objects instead of passing many paths manually

### `defaults.py`

Backward-compatible export layer for default paths and configs.

Responsibilities:
- Re-exports common objects from `assets.py` and `config.py`
- Provides a simpler import surface for external callers

Use this when:
- You want convenient access to default paths and configs

---

## Data Processing

### `bulk_data.py`

Bulk-side data preparation utilities.

Responsibilities:
- Removes version suffixes from transcript/protein IDs
- Loads ordering tables and realigns dataframes to a fixed order
- Loads bulk reference tables
- Merges one or more pausing files
- Loads PPI graphs
- Builds masks and train/val/test splits

Use this when:
- You are preparing aligned bulk graph inputs

### `single_cell.py`

Single-cell utility module.

Responsibilities:
- Loads single-cell expression matrices
- Loads metadata tables
- Loads pause matrices
- Builds KNN graphs
- Exports single-cell embeddings and prediction matrices

Use this when:
- You need shared utilities for single-cell data loading or cell graph construction

### `data_construction.py`

Dataset aliasing and organization utilities.

Responsibilities:
- Reorganizes scattered source files into the RECIPE directory layout
- Creates hardlinks or copies into standardized locations
- Writes dataset manifests

Use this when:
- You want to convert older project data into the standardized RECIPE layout

---

## Models and Training

### `models.py`

Model definitions.

Main classes:
- `RBULK`: bulk graph regression model
- `CPPI`: PPI edge scoring model
- `RSCHead`: single-cell graph head

Use this when:
- You need shared model architectures across workflows

### `bulk_regression.py`

Core training code for the bulk module.

Responsibilities:
- Defines `BulkConditionSpec`
- Builds bulk graph inputs
- Scales expression and target values
- Trains bulk graph regression models
- Predicts bulk outputs
- Evaluates train / val / test performance

Use this when:
- You are implementing or reusing phase0-style bulk training

### `bulk_workflow.py`

High-level workflow wrapper around `bulk_regression.py`.

Responsibilities:
- Builds the graph
- Splits labeled nodes
- Trains or loads a bulk model
- Saves checkpoints, predictions, embeddings, and metrics

Use this when:
- You want a workflow-level wrapper for bulk tasks

### `self_learning.py`

Semi-supervised / self-learning utilities.

Responsibilities:
- Early stopping
- Training on selected indices
- Evaluation on selected indices
- Pseudo-label selection
- Iterative self-learning support

Use this when:
- You want pseudo-label-based training over unlabeled nodes

### `utils.py`

General helper functions.

Responsibilities:
- Random seed setup
- Device resolution
- Parent directory creation
- Safe `R²` computation
- JSON saving

Use this when:
- You need shared low-level utilities used throughout the package

---

## PPI Inference

### `ppi_inference.py`

Low-level implementation for PPI edge inference.

Responsibilities:
- Defines edge datasets
- Defines the edge classifier MLP
- Samples negative edges
- Trains the edge classifier
- Infers candidate new edges
- Exports score matrices

Use this when:
- You want to infer new candidate PPI edges from node embeddings

### `ppi_workflow.py`

High-level workflow for PPI refinement.

Responsibilities:
- Loads a trained bulk checkpoint
- Extracts bulk node embeddings
- Trains an edge classifier
- Exports known-edge scores and candidate new edges

Use this when:
- You want to run RECIPE module C

---

## Current RNA-seq Workflow Mapping

For the current mouse / nanoSPINS RNA-seq route, the most relevant files in this package are:

- `single_cell_rnaseq_workflow.py`
- `models.py`
- `utils.py`

However, the actual training logic still lives in the project-root scripts:

- `train_phase0_ensmusp_pseudobulk_raw_bulkprot.py`
- `train_phase12_ensmusp_scRNA_bulkprot.py`
- `train_phase3_ensmusp_nanospins_matched.py`

In other words:
- `RECIPE/src/recipe/single_cell_rnaseq_workflow.py` provides the unified package-level entrypoints
- The actual training behavior is still implemented in the three project-root scripts

---

## Suggested Reading Order

If you want to understand this package quickly, a practical reading order is:

1. `README.md`
2. `assets.py`
3. `config.py`
4. `models.py`
5. `bulk_regression.py`
6. `bulk_workflow.py`
7. `single_cell.py`
8. `single_cell_riboseq_workflow.py`
9. `single_cell_rnaseq_workflow.py`
10. `pipeline.py`

---

## Notes

- `__pycache__/` is Python-generated cache, not handwritten logic.
- `single_cell_riboseq_workflow.py` and `single_cell_rnaseq_workflow.py` correspond to two different single-cell routes.
- The latest addition for your current project is the RNA-seq workflow wrapper; it does not replace the older riboseq workflow.
