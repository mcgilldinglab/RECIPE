<h1 align="center">RECIPE bridges transcriptomics and proteomics with deep graph learning on Ribo-seq data</h1>

![logo](RECIPE_logo.png)

## Overview

RECIPE is a deep graph learning framework for estimating protein abundance from RNA/Ribo-seq-related signals, transcript sequence embeddings, and protein-protein interaction topology. The packaged Python project lives in the repository subdirectory [`RECIPE/`](./RECIPE).

![Workflow](riboseq_WORKFLOW.png)

## Package Directory

The installable package, command-line runners, packaged data layout, training notebooks, and detailed documentation are under [`RECIPE/`](./RECIPE).

## Modules

- Module A: bulk inference for proteomics-undetected or unknown proteins.
- Module B: bulk protein abundance prediction for proteins with measured labels.
- Module C: self-supervised PPI refinement.
- Module D: single-cell transfer with pseudo-bulk alignment and a cell-graph head.

## Quick Start

```bash
git lfs install
git clone https://github.com/mcgilldinglab/RECIPE.git
cd RECIPE/RECIPE
git lfs pull
export RECIPE_DATA_ROOT="${PWD}/data"
export RECIPE_MODEL_ROOT="${PWD}/models"
export RECIPE_OUTPUT_ROOT="${PWD}/outputs"
python -m pip install -e . --no-deps
python scripts/run_smoke_demo.py --device cpu --output-dir outputs/smoke_demo
```

The smoke demo should finish in under 1 minute on a normal desktop or workstation.

## Documentation

- Full package documentation: [`RECIPE/README.md`](./RECIPE/README.md)
- Reproduction commands and explicit input data paths: [`RECIPE/docs/reproduction.md`](./RECIPE/docs/reproduction.md)
- Data layout: [`RECIPE/docs/data.md`](./RECIPE/docs/data.md)
- Sanitized training notebooks: [`RECIPE/notebooks/training`](./RECIPE/notebooks/training)

## Contact

Luying Su (luying.su@mail.mcgill.ca), Bowen Zhao (bowen.zhao@mail.mcgill.ca), Wei Song (songwei@ibms.pumc.edu.cn), Jun Ding (jun.ding@mcgill.ca)

Affiliations: Meakins-Christie Laboratories, RI-MUHC, McGill University
