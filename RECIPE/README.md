# RECIPE

RECIPE packages the workflow behind the project into a GitHub-style Python repository with English-only code and runnable entry scripts.

The pipeline integrates four modules:

- Module A: feature construction and extrapolation to proteomics-undetected proteins.
- Module B: bulk protein abundance prediction with the `RBULK` GraphSAGE regressor.
- Module C: self-supervised PPI refinement with the `CPPI` edge scorer.
- Module D: single-cell transfer with pseudo-bulk alignment and the `RSC` cell-graph head.

## Repository Layout

- `src/recipe/`: reusable package code.
- `scripts/`: direct entry points for modules `A`, `B`, `C`, `D`, and the combined pipeline.
- `data/`: English aliases for the curated datasets used by the packaged pipeline.
- `models/`: English aliases for bundled checkpoints.
- `figures/`: original exported figures preserved as project assets.
- `docs/`: copied manifests from the earlier notebook-to-repo curation pass.

## Installation

RECIPE is packaged as the `recipe` Python module and lives under the `RECIPE/` subdirectory of the GitHub repository.

Recommended runtime:

- Python `>=3.10`
- PyTorch with a matching `torch-geometric` install

Install directly from GitHub:

```bash
python -m pip install -U pip
python -m pip install "git+https://github.com/mcgilldinglab/RECIPE.git@main#subdirectory=RECIPE"
```

Install from a local clone in editable mode:

```bash
git clone https://github.com/mcgilldinglab/RECIPE.git
cd RECIPE/RECIPE
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Quick import check:

```bash
python -c "import recipe; print(recipe.__file__)"
```

## Documentation

Read the Docs should be configured from the repository root with `.readthedocs.yaml`, which points at `RECIPE/docs/conf.py`.

Build the docs locally:

```bash
cd RECIPE/RECIPE
python -m pip install -r docs/requirements.txt
python -m sphinx -b html docs docs/_build/html
```

## Data Availability

The packaged workflow uses both small text tables and large binary assets under `data/`.

- Small and medium files that are below GitHub's normal limit can be committed directly.
- Large runtime assets should be tracked with Git LFS via [`RECIPE/.gitattributes`](./.gitattributes).
- `data/networks/human_ppi_unknown.csv` is about `54 GB` locally and should stay external. It is too large for normal GitHub storage and too large for Git LFS per-file limits.

The current upload strategy is documented in [`data/README.md`](./data/README.md). For PPI files, only the smaller training-time graphs are intended for GitHub:

- `data/networks/human_ppi_known.csv`
- `data/networks/mouse_ppi_known.csv`
- `data/networks/mouse_ppi_unknown.csv`
- `data/networks/single_cell_transfer_ppi.csv`

## Data Construction

Rebuild the packaged English aliases under `data/`:

```bash
python3 scripts/build_data_aliases.py
```

Build packaged bulk feature tables:

```bash
python3 scripts/build_bulk_features.py --species human --task unknown --output-csv data/bulk/human_unknown_features.csv
python3 scripts/build_bulk_features.py --species mouse --task known --output-csv data/bulk/mouse_known_features.csv
```

Rebuild bulk coexpression matrices:

```bash
python3 scripts/build_coexpression.py --species human
python3 scripts/build_coexpression.py --species mouse
```

Rebuild packaged single-cell inputs:

```bash
python3 scripts/build_single_cell_inputs.py
```

Run the full packaged data build:

```bash
python3 scripts/build_all_data.py
```

## Direct Commands

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

## Retraining

Modules A and B reuse bundled `RBULK` checkpoints by default. Module D reuses the bundled phase 0 bulk checkpoint, then fits a fresh packaged `RSCHead` in phase 1. To retrain more aggressively, add the relevant training flag.

Examples:

```bash
python3 scripts/run_module_b.py --species human --condition KD --train
python3 scripts/run_module_d.py --steps phase0,phase1,phase2 --train-phase0 --train-phase1
```

## Input Mapping

The packaged scripts point to English aliases under `data/` and `models/`. Those aliases map to the original project assets used in the notebooks you listed.

Notebook lineage captured by the packaged modules:

- Module A, human unknown: `2411...unknown...newseed.ipynb`
- Module A, mouse unknown: `2411...unknown...ms copy.ipynb`
- Module B, human known: `2411...known...copy 3newseeds.ipynb`
- Module B, mouse known: `2411...known...ms ...seed.ipynb`
- Module C, human and mouse PPI refinement: `2411...ppi...ipynb`
- Module D, single-cell transfer: the three `2503/2504` self-learning and transfer notebooks

## Resource Notes

- `data/networks/human_ppi_unknown.csv` is about `51 GB`. Module A on the human unknown setting is the heaviest configuration.
- `data/networks/single_cell_transfer_ppi.csv` is about `258 MB`, so the packaged single-cell path is much lighter than the human unknown bulk graph.
- The scripts default to `--device auto`, which uses `cuda:0` when available and falls back to CPU otherwise.
- If the selected GPU runs out of memory, rerun the same command with `--device cpu`.

## Outputs

Each module writes structured outputs under `outputs/`:

- Module A and B: `predictions.csv`, `embeddings.npy`, `metrics.json`
- Module C: `candidate_edges.csv`, `known_edge_scores.csv`, `edge_classifier.pth`, `summary.json`
- Module D:
  - `phase0/`: bulk self-learning checkpoint, transcript predictions, cell embeddings
  - `phase1/`: `RSC` checkpoint, adjacency matrix, training history
  - `phase2/`: predicted cell-by-gene matrix and metadata table
