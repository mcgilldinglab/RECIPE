<h1 align="center">RECIPE bridges transcriptomics and proteomics with deep graph learning on Ribo-seq data</h1>

![logo](RECIPE_logo.png)

## Overview

RECIPE is a deep graph learning framework for estimating protein abundance from RNA/Ribo-seq-related signals, transcript sequence embeddings, and protein-protein interaction topology. The packaged Python project lives in the repository subdirectory [`RECIPE/`](./RECIPE).

![Workflow](riboseq_WORKFLOW.png)

## Modules

- Module A: bulk inference for proteomics-undetected or unknown proteins.
- Module B: bulk protein abundance prediction for proteins with measured labels.
- Module C: self-supervised PPI refinement.
- Module D: single-cell transfer with pseudo-bulk alignment and a cell-graph head.

## Tested Environment

- Linux `6.8.0-124-generic`, x86_64
- Python `3.8.0`
- PyTorch `2.1.1`
- CUDA runtime reported by PyTorch `12.1`
- PyTorch Geometric `2.5.3`
- `numpy 1.24.3`, `pandas 2.0.3`, `scipy 1.10.1`, `scikit-learn 1.3.2`

The smoke demo runs on CPU. Full-size training is faster with an NVIDIA GPU. The human unknown PPI graph is about 51-54 GB and must be distributed outside GitHub.

## Install

From an existing PyTorch/PyG environment:

```bash
git clone https://github.com/mcgilldinglab/RECIPE.git
cd RECIPE/RECIPE
python -m pip install -e . --no-deps
python -c "import recipe; print(recipe.__file__)"
```

From GitHub:

```bash
python -m pip install "git+https://github.com/mcgilldinglab/RECIPE.git@main#subdirectory=RECIPE"
```

If data are kept outside the installed package, set:

```bash
export RECIPE_DATA_ROOT=/path/to/RECIPE/RECIPE/data
export RECIPE_MODEL_ROOT=/path/to/RECIPE/RECIPE/models
```

## Smoke Demo

The repository includes a tiny simulated demo dataset in [`RECIPE/examples/smoke_data`](./RECIPE/examples/smoke_data). Run:

```bash
cd RECIPE/RECIPE
python scripts/run_smoke_demo.py --device cpu --output-dir outputs/smoke_demo
```

Expected run time: under 1 minute. Expected outputs:

- `outputs/smoke_demo/predictions.csv`
- `outputs/smoke_demo/embeddings.npy`
- `outputs/smoke_demo/metrics.json`

## Reproduce Public Tasks

Run commands from the package directory:

```bash
cd RECIPE/RECIPE
git lfs pull
export RECIPE_DATA_ROOT="${PWD}/data"
export RECIPE_MODEL_ROOT="${PWD}/models"
export RECIPE_OUTPUT_ROOT="${PWD}/outputs"
python -m pip install -e . --no-deps
```

The input data used by these commands are under `${RECIPE_DATA_ROOT}`. For example, Module A reads `data/bulk/mouse_reference.csv`, `data/bulk/mouse_sequence_unknown.npy`, and `data/networks/mouse_ppi_unknown.csv`. The full per-module input list is in [`RECIPE/docs/reproduction.md`](./RECIPE/docs/reproduction.md).

Then run the task entry points:

```bash
# Module A: bulk mouse unknown prediction
python scripts/run_module_a.py --species mouse --condition KD --seed 12 --device auto --output-dir outputs/reproduce/module_a_mouse_unknown

# Module B: bulk mouse known prediction
python scripts/run_module_b.py --species mouse --condition KD --seed 12 --device auto --output-dir outputs/reproduce/module_b_mouse_known

# Module C: PPI refinement; run Module B first
python scripts/run_module_c.py --species mouse --condition KD --seed 12 --device auto --bulk-checkpoint-path outputs/reproduce/module_b_mouse_known/model.pth --output-dir outputs/reproduce/module_c_mouse_ppi

# Module D: single-cell transfer
python scripts/run_module_d.py --steps phase0,phase1,phase2 --seed 12 --device auto --output-dir outputs/reproduce/module_d_single_cell
```

One-command pipeline:

```bash
python scripts/run_recipe.py --modules A,B,C,D --species mouse --condition KD --seed 12 --device auto --output-root outputs/reproduce/all_modules
```

Full command details and expected output files are in [`RECIPE/docs/reproduction.md`](./RECIPE/docs/reproduction.md).

## Documentation

Full installation, data, demo, expected output, runtime, and reproduction notes are in [`RECIPE/README.md`](./RECIPE/README.md).

Sanitized training notebooks are available in [`RECIPE/notebooks/training`](./RECIPE/notebooks/training). Outputs and local absolute paths were removed before adding them to the repository.

## Contact

Luying Su (luying.su@mail.mcgill.ca), Bowen Zhao (bowen.zhao@mail.mcgill.ca), Wei Song (songwei@ibms.pumc.edu.cn), Jun Ding (jun.ding@mcgill.ca)

Affiliations: Meakins-Christie Laboratories, RI-MUHC, McGill University
