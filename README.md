<h1 align="center">RECIPE</h1>

<p align="center"><strong>Deep graph learning for linking Ribo-seq/RNA-seq signals with protein abundance</strong></p>

<p align="center">
  <img src="RECIPE_logo.png" alt="RECIPE logo" width="260">
</p>

## Overview

RECIPE is a Python package for estimating protein abundance from transcript-level signals, transcript sequence embeddings, and protein-protein interaction topology. The input signal can be RNA-seq, Ribo-seq, or another transcript-level measurement selected by the user.

The package provides four workflows:

- Module A: bulk inference for proteomics-undetected or unknown proteins.
- Module B: bulk protein abundance prediction for proteins with measured labels.
- Module C: self-supervised PPI refinement.
- Module D: single-cell transfer with pseudo-bulk alignment and a cell-graph head.

The package source, command-line runners, data layout, notebooks, and detailed documentation are in [`RECIPE/`](./RECIPE).

## Install

Clone the repository and install the Python package from the `RECIPE/` subdirectory:

```bash
git lfs install
git clone https://github.com/mcgilldinglab/RECIPE.git
cd RECIPE/RECIPE
git lfs pull

conda create -n recipe python=3.8 -y
conda activate recipe

# Install PyTorch and PyTorch Geometric for your CUDA or CPU setup first.
python -m pip install "torch>=2.1" "torch-geometric>=2.5"
python -m pip install -r requirements.txt
python -m pip install -e .
```

If PyTorch and PyTorch Geometric are already installed, run:

```bash
cd RECIPE/RECIPE
python -m pip install -e . --no-deps
```

Detailed environment notes are provided in [`RECIPE/README.md`](./RECIPE/README.md).

## Tutorials

Start with the smoke demo:

```bash
cd RECIPE/RECIPE
python scripts/run_smoke_demo.py --device cpu --output-dir outputs/smoke_demo
```

Additional tutorials and reproduction instructions:

- Full package guide: [`RECIPE/README.md`](./RECIPE/README.md)
- Public task reproduction commands with explicit input paths: [`RECIPE/docs/reproduction.md`](./RECIPE/docs/reproduction.md)
- Script index: [`RECIPE/docs/script_index.md`](./RECIPE/docs/script_index.md)
- Data layout: [`RECIPE/data/README.md`](./RECIPE/data/README.md)
- Training notebooks: [`RECIPE/notebooks/training`](./RECIPE/notebooks/training)

## Contact

- Luying Su: luying.su@mail.mcgill.ca
- Bowen Zhao: bowen.zhao@mail.mcgill.ca
- Wei Song: songwei@ibms.pumc.edu.cn
- Jun Ding: jun.ding@mcgill.ca

Meakins-Christie Laboratories, Research Institute of the McGill University Health Centre, McGill University.
