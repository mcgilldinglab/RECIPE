
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
### Conda (recommended for PyTorch/CUDA)
```bash
conda create -n recipe python=3.9 -y
conda activate recipe
# Install PyTorch/CUDA (match your GPU/driver)
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
# Then install the rest via pip
pip install -r requirements.txt
# Dev install
pip install -e .
```

See full list in **[requirements.txt](./requirements.txt)**.

Note: Ensure that PyTorch Geometric and its dependencies are installed with versions compatible with your PyTorch and CUDA setup. For the correct wheel files, please consult the official PyTorch Geometric documentation.


## Prerequisites

Python ≥ 3.9

PyTorch ≥ 2.0

PyTorch Geometric (GraphSAGE)

Transformers (for DNABERT/sequence embeddings)

NumPy, Pandas, SciPy, scikit-learn

matplotlib (figures)

networkx (PPI IO/ops)

(Optional) scanpy/anndata for single-cell workflows


## Usage

# RECIPE Usage

## Run RECIPE
```python
import sys
sys.path.append("/path/to/RECIPE/src")
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
## Contact

Luying Su (luying.su@mail.mcgill.ca), Bowen Zhao (bowen.zhao@mail.mcgill.ca), Wei Song (songwei@ibms.pumc.edu.cn), Jun Ding (jun.ding@mcgill.ca) 
Affiliations: Meakins-Christie Laboratories, RI-MUHC, McGill University
