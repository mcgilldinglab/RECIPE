
<h1 align="center">RECIPE bridges transcriptomics and proteomics with deep graph learning on Ribo-seq data</h1>

![logo](RECIPE_logo.png)

## Overview

Direct protein quantification is limited by cost and coverage of mass spectrometry and antibody-based assays. These challenges are especially severe at the single-cell level, where proteomics remains extremely restricted. Computational inference from RNA-seq is attractive but hindered by the mismatch between transcript and protein abundance. Ribosome profiling (Ribo-seq) captures ribosome occupancy and translational dynamics, but not absolute protein abundance. Here we introduce RECIPE, a deep graph learning framework that integrates Ribo-seq signals, transcript features, and protein–protein interactions to estimate protein abundance. Built on a GraphSAGE architecture, RECIPE leverages network connectivity to generalize beyond training-observed proteins, enabling accurate prediction of low-abundance and undetected proteins. RECIPE also refines protein–protein interaction networks by identifying missing and spurious edges, extending utility. Benchmarking across human and mouse datasets shows that RECIPE outperforms state-of-the-art approaches. RECIPE enables generalizable protein estimation at bulk and single-cell resolution and advances translatomics by linking translational dynamics with interactome refinement.

![Workflow](riboseq_WORKFLOW.png)


### Principal Contributions

1. **Translation-aware graph learning.**  
   A unified GraphSAGE framework links transcriptomics, translationomics (Ribo-seq occupancy & pausing), and proteomics using DNABERT sequence embeddings and PPI topology, explicitly modeling information flow from mRNA translation to protein abundance.

2. **Generalizes beyond detected proteins.**  
   Graph message passing over PPI propagates translation/expression signals to neighbors, enabling accurate prediction of low-abundance or undetected proteins (e.g., Pearson R ≈ 0.96) and improving proteome coverage for downstream analyses.

3. **Bulk ↔ single-cell applicability.**  
   Two-phase strategy—pseudo-bulk adaptation then cell–cell GNN—yields translation-informed single-cell protein estimation, providing a scalable alternative where single-cell proteomics is sparse.

4. **Interactome refinement.**  
   A PPI inference module removes spurious edges and adds missing ones, producing dataset-specific networks with stronger co-expression and pathway coherence (GO/KEGG/Reactome).

5. **Mechanistic insight into translation–protein decoupling.**  
   Systematic comparison of predicted vs. measured proteins reveals modules (e.g., cytoskeletal, ribosomal, nucleoproteins) where stability/stoichiometry/chromatin binding decouple abundance from ribosome occupancy.

## Installation

The pip-installable RECIPE package lives in the repository subdirectory `RECIPE/`.

### Install from GitHub

```bash
python -m pip install -U pip
python -m pip install "git+https://github.com/mcgilldinglab/RECIPE.git@main#subdirectory=RECIPE"
```

### Editable install from a local clone

```bash
git clone https://github.com/mcgilldinglab/RECIPE.git
cd RECIPE
python -m pip install -U pip
python -m pip install -r RECIPE/requirements.txt
python -m pip install -e RECIPE
```

For package-specific details, see **[RECIPE/README.md](./RECIPE/README.md)**.

Note: Ensure that PyTorch Geometric and its dependencies are installed with versions compatible with your PyTorch and CUDA setup. For the correct wheel files, please consult the official PyTorch Geometric documentation.

## Documentation

Read the Docs is configured from the repository root via **[.readthedocs.yaml](./.readthedocs.yaml)** and builds the docs stored in `RECIPE/docs/`.


## Prerequisites

Python ≥ 3.10

PyTorch ≥ 2.0

PyTorch Geometric (GraphSAGE)

Transformers (for DNABERT/sequence embeddings)

NumPy, Pandas, SciPy, scikit-learn

matplotlib (figures)

networkx (PPI IO/ops)

(Optional) scanpy/anndata for single-cell workflows


## Usage

Install the package first, then import `recipe` directly.

```python
import recipe
```

## Module A: bulk known
```python
from recipe.bulk_workflow import run_bulk_module

summary = run_bulk_module(
    species="human",
    task="known",
    condition_name="KD",
    output_dir="/tmp/recipe_module_b",
    seed=12,
    device_name="cuda:0",
    train=True,
)
print(summary)
```

## Module B: bulk unknown
```python
from recipe.bulk_workflow import run_bulk_module

summary = run_bulk_module(
    species="human",
    task="unknown",
    condition_name="KD",
    output_dir="/tmp/recipe_module_a",
    seed=12,
    device_name="cuda:0",
    train=True,
)
print(summary)
```

## Module C: PPI refinement
```python
from recipe.ppi_workflow import run_ppi_refinement

summary = run_ppi_refinement(
    species="human",
    condition_name="KD",
    output_dir="/tmp/recipe_module_c",
    seed=12,
    device_name="cuda:0",
    bulk_checkpoint_path="/path/to/bulk_checkpoint.pth",
)
print(summary)
```

## Module D -1: single-cell riboseq workflow
```python
from recipe.single_cell_riboseq_workflow import run_single_cell_transfer

summary = run_single_cell_transfer(
    output_dir="/tmp/recipe_module_d",
    steps=(
        "Bulk Module",
        "Phase 1: Pseudo-Bulk Module Finetuning",
        "Phase 2: Single-Cell Finetuning",
    ),
    seed=12,
    device_name="cuda:0",
    train_phase0=True,
    train_phase1=True,
    train_phase2=True,
)
print(summary)
```

## Module D -2: RNA-seq workflow:
```python
from recipe.single_cell_rnaseq_workflow import run_scrnaseq_workflow

run_scrnaseq_workflow(
    bulk_module_args=[
        "--bundle-dir", "/path/to/bundle",
        "--output-dir", "/tmp/rnaseq_phase0",
        "--seed", "8",
        "--device", "cuda:0",
        "--condition", "C10",
    ],
    phase1_rnaseq_pseudo_bulk_finetuning_args=[
        "--bundle-dir", "/path/to/bundle",
        "--phase0-summary", "/tmp/rnaseq_phase0/summary.json",
        "--phase0-model", "/tmp/rnaseq_phase0/best_model.pth",
        "--output-root", "/tmp/rnaseq_phase12",
        "--seed", "8",
        "--device", "cuda:0",
        "--condition", "C10",
    ],
    phase2_single_cell_protein_finetuning_args=[
        "--bundle-dir", "/path/to/bundle",
        "--hidden-cache-root", "/tmp/rnaseq_phase12/phase2_hidden_cache",
        "--truth-csv", "/path/to/protein_truth.csv",
        "--mapping-xlsx", "/path/to/d5lc01008j2.xlsx",
        "--output-root", "/tmp/rnaseq_phase3",
        "--seed", "8",
        "--device", "cuda:0",
        "--condition", "C10",
    ],
)

```

## Pipeline for RECIPE modules
```python
from recipe.pipeline import run_recipe_pipeline

summary = run_recipe_pipeline(
    modules=("A", "B", "C", "D"),
    output_root="/tmp/recipe_pipeline",
    species="human",
    condition="KD",
    seed=12,
    device_name="cuda:0",
)
print(summary)
```

## Data

The repository documents a split between lightweight packaged assets committed directly and large runtime assets tracked with Git LFS or kept external.

The public repository now includes the smaller packaged PPI graphs needed for model training and inference:

- `RECIPE/data/networks/human_ppi_known.csv`
- `RECIPE/data/networks/mouse_ppi_known.csv`
- `RECIPE/data/networks/mouse_ppi_unknown.csv`
- `RECIPE/data/networks/single_cell_transfer_ppi.csv`

The very large `human_ppi_unknown.csv` is intentionally excluded from GitHub and should be distributed separately.
## Contact

Luying Su (luying.su@mail.mcgill.ca), Bowen Zhao (bowen.zhao@mail.mcgill.ca), Wei Song (songwei@ibms.pumc.edu.cn), Jun Ding (jun.ding@mcgill.ca) 
Affiliations: Meakins-Christie Laboratories, RI-MUHC, McGill University
