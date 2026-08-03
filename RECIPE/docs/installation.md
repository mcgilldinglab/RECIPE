# Installation

## Tested Environment

The local tested environment is the `pyg` conda environment:

- Linux `6.8.0-124-generic`, x86_64
- Python `3.8.0`
- PyTorch `2.1.1`
- CUDA runtime reported by PyTorch `12.1`
- PyTorch Geometric `2.5.3`
- `numpy 1.24.3`, `pandas 2.0.3`, `scipy 1.10.1`, `scikit-learn 1.3.2`

## Requirements

- Python `>=3.8`
- PyTorch `>=2.1`
- `torch-geometric >=2.5` compatible with your PyTorch/CUDA build
- `numpy >=1.24`
- `pandas >=2.0`
- `scipy >=1.10`
- `scikit-learn >=1.3`
- `matplotlib >=3.6`
- `networkx >=3.1`
- `seaborn >=0.13`
- `openpyxl >=3.1`

No non-standard hardware is required for installation or the smoke demo. A CUDA-capable GPU is recommended for full-size training. The full human unknown PPI graph is about 51-54 GB and is not distributed in GitHub.

## Existing PyG Environment

```bash
conda activate pyg
cd /path/to/RECIPE/RECIPE
python -m pip install -e . --no-deps
python -c "import recipe; print(recipe.__file__)"
```

Typical install time in an environment that already has PyTorch and PyG: under 1 minute. Editable installation in the tested `pyg` environment completed successfully in about 11 seconds.

## Fresh Environment

Install PyTorch and PyTorch Geometric first with wheels matching your CUDA or CPU setup, then install RECIPE:

```bash
conda create -n recipe python=3.8 -y
conda activate recipe
python -m pip install "torch>=2.1" "torch-geometric>=2.5"
git clone https://github.com/mcgilldinglab/RECIPE.git
cd RECIPE/RECIPE
python -m pip install -r requirements.txt
python -m pip install -e .
```

Typical fresh setup time: about 10-30 minutes, depending mostly on PyTorch/PyG downloads.

## Install From GitHub

```bash
python -m pip install "git+https://github.com/mcgilldinglab/RECIPE.git@main#subdirectory=RECIPE"
```

If runtime data are outside the installed package:

```bash
export RECIPE_DATA_ROOT=/path/to/RECIPE/RECIPE/data
export RECIPE_MODEL_ROOT=/path/to/RECIPE/RECIPE/models
```

## Verify

```bash
python -c "import recipe; print(recipe.__file__)"
python scripts/run_smoke_demo.py --device cpu --output-dir outputs/smoke_demo
```

The smoke demo should finish in under 1 minute and create `predictions.csv`, `embeddings.npy`, and `metrics.json`. The tested `pyg` CPU run completed in about 4.7 seconds wall time.

## Build Documentation Locally

```bash
cd /path/to/RECIPE/RECIPE
python -m pip install -r docs/requirements.txt
python -m sphinx -b html docs docs/_build/html
```
