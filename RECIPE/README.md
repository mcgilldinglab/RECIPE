# RECIPE

RECIPE provides the manuscript workflow as a Python project with reusable code, command-line entry points, runtime data, and a small smoke-test dataset.

The pipeline has four modules:

- Module A: bulk protein abundance prediction for proteins with measured labels.
- Module B: bulk inference for proteomics-undetected or unknown proteins.
- Module C: self-supervised PPI refinement.
- Module D: single-cell protein prediction from either scRibo-seq or scRNA-seq input.

## Repository Layout

- `src/recipe/`: reusable package code.
- `scripts/`: command-line entry points for modules A-D, scRNA-seq phase scripts, data builders, and the smoke demo.
- `examples/smoke_data/`: tiny simulated data for a CPU-friendly demo.
- `data/`: runtime data for the command-line workflows. Large arrays and graphs are tracked with Git LFS.
- `data/splits/`: fixed train/validation/test CSV files used by the command-line runners.
- `models/`: pretrained checkpoints tracked with Git LFS. See `models/README.md`.
- `benchmarks/`: lightweight benchmark wrappers for external baseline comparisons.
- `docs/`: Sphinx documentation source.

## System Requirements

Tested environment on the local `pyg` conda environment:

- Operating system: Linux `6.8.0-124-generic`, x86_64.
- Python: `3.8.0`.
- PyTorch: `2.1.1`.
- CUDA runtime reported by PyTorch: `12.1`.
- PyTorch Geometric: `2.5.3`.
- CUDA availability during test: `True`.
- Other tested Python packages: `numpy 1.24.3`, `pandas 2.0.3`, `pysam 0.22.1`, `scipy 1.10.1`, `scikit-learn 1.3.2`, `matplotlib 3.6.3`, `networkx 3.1`, `seaborn 0.13.2`, `openpyxl 3.1.5`, `typing_extensions 4.9.0`.

Package metadata supports Python `>=3.8`. The smoke demo runs on CPU. Full-size training and inference are much faster with an NVIDIA GPU; `--device auto` uses `cuda:0` when available and falls back to CPU. The external human unknown PPI graph is about 97 MiB after extraction.

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

Typical fresh environment setup time on a workstation is about 10-30 minutes, mostly depending on PyTorch/PyG wheel downloads.

### Install From GitHub

The Python package lives in the repository subdirectory `RECIPE/`:

```bash
python -m pip install "git+https://github.com/mcgilldinglab/RECIPE.git@main#subdirectory=RECIPE"
```

If you install this way and keep data or checkpoints outside site-packages, pass their locations when running a workflow, for example with `--data-root /path/to/RECIPE/RECIPE/data` and `--model-root /path/to/RECIPE/RECIPE/models`, or with the file-level arguments shown below. For full manuscript-scale reproduction, especially the Module D scRNA-seq phase scripts under `scripts/rnaseq/`, clone the repository and run from the checkout.

## Data

The repository includes a small simulated demo dataset:

- `examples/smoke_data/bulk_reference.csv`
- `examples/smoke_data/sequence_embeddings.csv`
- `examples/smoke_data/ppi_matrix.csv`

Runtime data are under `data/`. Large files and pretrained checkpoints are tracked with Git LFS. The full `data/networks/human_ppi_unknown.csv` is distributed separately. Download its compressed copy from [Google Drive](https://drive.google.com/file/d/1UIefENLMUvWTJ9K8jxVmuGdPFD4Vtnrc/view?usp=sharing) and follow the checksum and extraction instructions in [`docs/data.md`](docs/data.md#external-human-ppi-graph).

## Pausing Feature Calculation

The pausing feature can be rebuilt from a CDS annotation CSV and a coordinate-sorted, indexed BAM file. The CDS table should contain semicolon-separated `Start` and `End` columns, a reference column such as `seqnames`, a `Length` column, and a protein identifier column such as `protein_id` or `protein`.

Compute per-position pause scores:

```bash
python scripts/compute_pausing.py score-bam \
  --cds-csv data/pausing/cds_annotations.csv \
  --bam /path/to/riboseq.sorted.bam \
  --score-csv outputs/pausing/pause_scores.csv
```

Summarize position scores into the `High_Pause_Counts` table used by RECIPE:

```bash
python scripts/compute_pausing.py summarize \
  --scores-csv outputs/pausing/pause_scores.csv \
  --summary-cds-csv data/pausing/cds_annotations.csv \
  --output-csv outputs/pausing/high_pause_counts.csv \
  --threshold 3.3 \
  --threshold-mode absolute
```

The score calculation trims the first and last 60 nt of the CDS by default, matching the manuscript preprocessing scripts. For single-cell or barcode-level score files, pass grouped columns such as `--group-cols CB,ENSP` and use `--threshold-mode relative_to_mean` if the high-pause definition should be relative to each group mean.
By default `--average-denominator length` assumes the `Length` column is the CDS length used by the original preprocessing scripts. If your `Length` column is not the analyzed trimmed region length, use `--average-denominator positions` to normalize by the emitted CDS positions after trimming.

## Smoke Demo

Run the CPU-friendly demo:

```bash
cd /path/to/RECIPE/RECIPE
python scripts/run_smoke_demo.py --device cpu --output-dir outputs/smoke_demo
```

Expected run time on a workstation: under 1 minute. The tested `pyg` CPU run completed in about 4.7 seconds wall time.

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

Run commands from the package directory:

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

Prepare the reproduction inputs:

```bash
python scripts/prepare_public_data.py \
  --data-root "${DATA_ROOT}" \
  --model-root "${MODEL_ROOT}" \
  --manifest-json "${OUTPUT_ROOT}/data_preparation.json"
```

To include the Module C coexpression summary, generate the mouse coexpression matrix:

```bash
python scripts/prepare_public_data.py \
  --data-root "${DATA_ROOT}" \
  --model-root "${MODEL_ROOT}" \
  --build-mouse-coexpression \
  --manifest-json "${OUTPUT_ROOT}/data_preparation_with_coexpression.json"
```

Run a minimal check:

```bash
python scripts/run_smoke_demo.py --device cpu --output-dir outputs/smoke_demo
```

Run the mouse bulk/PPI tasks together with the human single-cell transfer task:

```bash
python scripts/run_recipe.py \
  --modules A,B,C,D \
  --species mouse \
  --condition KD \
  --seed 12 \
  --device auto \
  --data-root "${DATA_ROOT}" \
  --model-root "${MODEL_ROOT}" \
  --single-cell-assay scriboseq \
  --bulk-unknown-split-csv "${DATA_ROOT}/splits/bulk_mouse_unknown_seed12.csv" \
  --bulk-known-split-csv "${DATA_ROOT}/splits/bulk_mouse_known_seed12.csv" \
  --phase0-split-csv "${DATA_ROOT}/splits/single_cell_self_learning_seed12.csv" \
  --phase1-split-csv "${DATA_ROOT}/splits/single_cell_module_a_seed42.csv" \
  --phase2-split-csv "${DATA_ROOT}/splits/single_cell_graph_seed42.csv" \
  --skip-candidate-inference \
  --use-bundled-cell-embeddings \
  --output-root "${OUTPUT_ROOT}/all_modules"
```

In this combined command, `--species mouse` applies to Modules A-C. `--single-cell-assay scriboseq` makes Module D use the human scRibo-seq transfer inputs. The command uses bundled checkpoints unless the matching training flags are passed. Add `--bulk-train`, `--train-edge-classifier`, `--train-phase0`, `--train-phase1`, or `--train-phase2` to retrain those parts. When Module A is retrained before Module C, Module C uses `${OUTPUT_ROOT}/all_modules/module_a/model.pth`; otherwise it falls back to the bundled known-protein bulk checkpoint. `--skip-candidate-inference` avoids the all-pairs PPI candidate scan for quick review runs; remove it to generate the full `candidate_edges.csv`, or pass a precomputed file with `--candidate-edge-csv`. Per-module commands, explicit input paths, and expected output files are in `docs/reproduction.md`.

## Outputs

Module A and B write:

- `predictions.csv`: transcript IDs, predictions, observed targets, split labels.
- `embeddings.npy`: learned node embeddings.
- `metrics.json`: train/validation/test metrics, scaling metadata, and checkpoint path.
- `model.pth`: created when training is run or no default checkpoint exists; otherwise the bundled checkpoint path is recorded in `metrics.json`.

Module C writes:

- `candidate_edges.csv`
- `known_edge_scores.csv`
- `edge_classifier.pth`
- `bulk_node_embeddings.npy`
- `summary.json`

When Module C is run with `--skip-candidate-inference`, `candidate_edges.csv` is an empty placeholder and `known_edge_scores.csv` plus `summary.json` are the quick reproducibility outputs.

Module D has two assay branches. With `--assay scriboseq`, it uses scRibo-seq input to predict single-cell protein abundance; `--use-bundled-cell-embeddings` makes the bundled cell embeddings and outputs available without recomputing them. With `--assay scrnaseq`, it uses scRNA-seq input to predict single-cell protein abundance and writes the nanoSPINS model as `phase3/models/phase3_nanospins_best.pth`.

## Running On Your Own Data

For bulk workflows, prepare:

- A reference CSV with one row per transcript or protein. It must include a transcript identifier column and the feature columns used as model inputs, for example RNA-seq, Ribo-seq, or other translation-related measurements. If supervised training is needed, include a protein abundance target column. A pausing-count column can also be used when available.
- A sequence embedding `.npy` file whose row count and order match the reference CSV.
- A square numeric PPI adjacency matrix in CSV or SciPy sparse NPZ format whose dimensions match the number of reference rows.

Column names are not fixed by the package. The bundled configs use names such as `rNC2`, `rKD2`, `NC3`, and `KD3` only because those are the column names in the example datasets. For new data, pass your own input, target, and optional pausing columns on the command line, or define a `BulkConditionSpec` and call `build_bulk_graph_from_dataframe` from Python.

For command-line use, pass these files directly with `--reference-csv`, `--sequence-npy`, `--ppi-csv`, and optionally `--split-csv`. Use `--input-col` for the RNA-seq, Ribo-seq, or other transcript-level input column, `--target-col` for the protein abundance column, and `--pause-col` for a pausing-count column when available. If no pausing feature is available, pass `--no-pause`. For a data directory that mirrors the repository `data/` layout, pass `--data-root /path/to/data`. Use `run_module_a.py` for known-protein prediction and `run_module_b.py` for unknown-protein inference.

```bash
python scripts/run_module_a.py \
  --reference-csv /path/to/reference.csv \
  --sequence-npy /path/to/sequence.npy \
  --ppi-csv /path/to/ppi.csv \
  --input-col your_ribo_or_rna_column \
  --target-col your_protein_column \
  --no-pause \
  --output-dir outputs/custom_bulk
```

## Reproduction Notes

For a minimal reproducibility check, run the smoke demo and confirm that the three output files above are produced. For manuscript-scale reproduction, run the module commands with the repository data or externally restored data and record the generated `metrics.json` / `summary.json` files. Use a fixed `--seed` value, and record the exact PyTorch/PyG/CUDA versions from the tested environment section.
