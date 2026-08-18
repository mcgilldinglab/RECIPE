# Reproduction Commands

Use this page as the canonical command list for reproducing the public RECIPE tasks.

## Clone And Install

```bash
git lfs install
git clone https://github.com/mcgilldinglab/RECIPE.git
cd RECIPE/RECIPE
git lfs pull

conda activate recipe
python -m pip install -e . --no-deps
```

Use the fresh-environment instructions in `installation.md` if PyTorch and PyTorch Geometric are not already installed.

Set these paths for your local checkout. `DATA_ROOT` may point to the repository data directory or to another directory with the same file layout.

```bash
DATA_ROOT="${PWD}/data"
MODEL_ROOT="${PWD}/models"
OUTPUT_ROOT="${PWD}/outputs/reproduce"
```

## Data Preparation

Create derived inputs when needed and check that the reproduction files are present:

```bash
python scripts/prepare_public_data.py \
  --data-root "${DATA_ROOT}" \
  --model-root "${MODEL_ROOT}" \
  --manifest-json "${OUTPUT_ROOT}/data_preparation.json"
```

For the Module C coexpression summary, build the mouse coexpression matrix. This creates a large CSV file under `${DATA_ROOT}/networks/`.

```bash
python scripts/prepare_public_data.py \
  --data-root "${DATA_ROOT}" \
  --model-root "${MODEL_ROOT}" \
  --build-mouse-coexpression \
  --manifest-json "${OUTPUT_ROOT}/data_preparation_with_coexpression.json"
```

If the fixed split files need to be regenerated, run:

```bash
python scripts/build_training_splits.py --output-dir "${DATA_ROOT}/splits"
```

The complete data inventory and external download instructions are in [`data.md`](data.md).

## Quick Check

```bash
python scripts/run_smoke_demo.py --device cpu --output-dir outputs/smoke_demo
```

Expected run time: under 1 minute. Expected files:

- `outputs/smoke_demo/predictions.csv`
- `outputs/smoke_demo/embeddings.npy`
- `outputs/smoke_demo/metrics.json`

## Public Task Reproduction

Run commands from the `RECIPE/` package directory.

The bulk and PPI tasks are separated by species:

| Module | Task | Human status | Mouse status |
| --- | --- | --- | --- |
| A | Known bulk protein prediction | Reproducible from GitHub LFS files | Reproducible from GitHub LFS files |
| B | Unknown bulk protein inference | Requires external `data/networks/human_ppi_unknown.csv` | Reproducible from GitHub LFS files |
| C | PPI refinement from known bulk embeddings | Reproducible from GitHub LFS files | Reproducible from GitHub LFS files |
| D | Single-cell transfer | Human HEK293T single-cell workflow | Not used |

### Module A: Bulk Known Protein Prediction

Inputs passed explicitly:

- `${DATA_ROOT}/bulk/mouse_reference.csv`
- `${DATA_ROOT}/bulk/mouse_sequence_known.npy`
- `${DATA_ROOT}/networks/mouse_ppi_known.csv`
- `${DATA_ROOT}/splits/bulk_mouse_known_seed12.csv`
- `${MODEL_ROOT}/bulk/mouse_known_seed5.pth` unless `--train` is used.

```bash
python scripts/run_module_a.py \
  --species mouse \
  --condition KD \
  --seed 12 \
  --device auto \
  --model-root "${MODEL_ROOT}" \
  --reference-csv "${DATA_ROOT}/bulk/mouse_reference.csv" \
  --sequence-npy "${DATA_ROOT}/bulk/mouse_sequence_known.npy" \
  --ppi-csv "${DATA_ROOT}/networks/mouse_ppi_known.csv" \
  --split-csv "${DATA_ROOT}/splits/bulk_mouse_known_seed12.csv" \
  --output-dir "${OUTPUT_ROOT}/module_a_mouse_known"
```

Expected files:

- `${OUTPUT_ROOT}/module_a_mouse_known/predictions.csv`
- `${OUTPUT_ROOT}/module_a_mouse_known/embeddings.npy`
- `${OUTPUT_ROOT}/module_a_mouse_known/metrics.json`
- `${OUTPUT_ROOT}/module_a_mouse_known/model.pth` when `--train` is used or no bundled checkpoint is available.

For the human known-protein run, use the same command with `--species human`, `human_reference.csv`, `human_sequence_known.npy`, `human_ppi_known.csv`, `bulk_human_known_seed12.csv`, and output directory `${OUTPUT_ROOT}/module_a_human_known`.

### Module B: Bulk Unknown Protein Inference

Inputs passed explicitly:

- `${DATA_ROOT}/bulk/mouse_reference.csv`
- `${DATA_ROOT}/bulk/mouse_sequence_unknown.npy`
- `${DATA_ROOT}/networks/mouse_ppi_unknown.csv`
- `${DATA_ROOT}/splits/bulk_mouse_unknown_seed12.csv`
- `${MODEL_ROOT}/bulk/mouse_unknown_seed1.pth` unless `--train` is used.

```bash
python scripts/run_module_b.py \
  --species mouse \
  --condition KD \
  --seed 12 \
  --device auto \
  --model-root "${MODEL_ROOT}" \
  --reference-csv "${DATA_ROOT}/bulk/mouse_reference.csv" \
  --sequence-npy "${DATA_ROOT}/bulk/mouse_sequence_unknown.npy" \
  --ppi-csv "${DATA_ROOT}/networks/mouse_ppi_unknown.csv" \
  --split-csv "${DATA_ROOT}/splits/bulk_mouse_unknown_seed12.csv" \
  --output-dir "${OUTPUT_ROOT}/module_b_mouse_unknown"
```

Expected files:

- `${OUTPUT_ROOT}/module_b_mouse_unknown/predictions.csv`
- `${OUTPUT_ROOT}/module_b_mouse_unknown/embeddings.npy`
- `${OUTPUT_ROOT}/module_b_mouse_unknown/metrics.json`
- `${OUTPUT_ROOT}/module_b_mouse_unknown/model.pth` when `--train` is used or no bundled checkpoint is available.

The human unknown-protein run uses `human_reference.csv`, `human_sequence_unknown.npy`, `bulk/human_unknown_seed0.pth`, and the externally downloaded `data/networks/human_ppi_unknown.csv` file described in [`data.md`](data.md#external-human-ppi-graph).

### Module C: PPI Refinement

Module C requires a trained known-protein bulk checkpoint. Either use the bundled known-protein checkpoint or run Module A with `--train` first and pass its checkpoint:

Inputs passed explicitly:

- `${MODEL_ROOT}/bulk/mouse_known_seed5.pth` or `${OUTPUT_ROOT}/module_a_mouse_known/model.pth`
- `${MODEL_ROOT}/ppi/mouse_edge_classifier.pth` unless `--train-edge-classifier` is used.
- `${DATA_ROOT}/bulk/mouse_reference.csv`
- `${DATA_ROOT}/bulk/mouse_sequence_known.npy`
- `${DATA_ROOT}/networks/mouse_ppi_known.csv`
- `${DATA_ROOT}/networks/mouse_coexpression.csv` if generated during data preparation.

```bash
python scripts/run_module_c.py \
  --species mouse \
  --condition KD \
  --seed 12 \
  --device auto \
  --model-root "${MODEL_ROOT}" \
  --reference-csv "${DATA_ROOT}/bulk/mouse_reference.csv" \
  --sequence-npy "${DATA_ROOT}/bulk/mouse_sequence_known.npy" \
  --ppi-csv "${DATA_ROOT}/networks/mouse_ppi_known.csv" \
  --coexpression-csv "${DATA_ROOT}/networks/mouse_coexpression.csv" \
  --bulk-checkpoint-path "${MODEL_ROOT}/bulk/mouse_known_seed5.pth" \
  --edge-checkpoint-path "${MODEL_ROOT}/ppi/mouse_edge_classifier.pth" \
  --skip-candidate-inference \
  --output-dir "${OUTPUT_ROOT}/module_c_mouse_ppi"
```

For the human PPI run, use `--species human`, `human_reference.csv`, `human_sequence_known.npy`, `human_ppi_known.csv`, `${MODEL_ROOT}/bulk/human_known_seed12.pth`, and `${MODEL_ROOT}/ppi/human_edge_classifier.pth`.

The command above is the quick checkpoint-based review run. It scores known PPI edges and writes the node embeddings, but it does not scan all possible gene pairs. To generate the full candidate-edge output, remove `--skip-candidate-inference`. To reproduce from a precomputed candidate-edge table without rescanning all pairs, pass `--candidate-edge-csv /path/to/candidate_edges.csv`.

Expected files:

- `${OUTPUT_ROOT}/module_c_mouse_ppi/candidate_edges.csv`
- `${OUTPUT_ROOT}/module_c_mouse_ppi/known_edge_scores.csv`
- `${OUTPUT_ROOT}/module_c_mouse_ppi/edge_classifier.pth`
- `${OUTPUT_ROOT}/module_c_mouse_ppi/bulk_node_embeddings.npy`
- `${OUTPUT_ROOT}/module_c_mouse_ppi/summary.json`

### Module D: Single-Cell Transfer

Module D uses a single entry script with two assay options:

- `--assay scriboseq`: use scRibo-seq input to predict single-cell protein abundance.
- `--assay scrnaseq`: use scRNA-seq input to predict single-cell protein abundance.

```bash
python scripts/run_module_d.py \
  --assay scriboseq \
  --data-root "${DATA_ROOT}" \
  --model-root "${MODEL_ROOT}" \
  --output-dir "${OUTPUT_ROOT}/module_d_single_cell"
```

To reproduce the archived scRibo-seq phase2 result named `seed7_npcs20_k7_all_labeled`,
use the packaged preset. This uses the same settings as the original
`codex_runs/shared_global_phase2_runner.py` run: seed 7, 20 expression PCs, KNN k=7,
and best-checkpoint selection by `test_r2`.

```bash
python scripts/run_module_d.py \
  --assay scriboseq \
  --scriboseq-reproduction-preset seed7_npcs20_k7_all_labeled \
  --data-root "${DATA_ROOT}" \
  --model-root "${MODEL_ROOT}" \
  --output-dir "${OUTPUT_ROOT}/module_d_seed7_npcs20_k7"
```

Expected key outputs:

- `${OUTPUT_ROOT}/module_d_seed7_npcs20_k7/phase2/phase2_summary.json`
- `${OUTPUT_ROOT}/module_d_seed7_npcs20_k7/phase2/phase2_predicted_cell_matrix.csv`
- `${OUTPUT_ROOT}/module_d_seed7_npcs20_k7/phase2/seed7_npcs20_k7_all_labeled_predictions_vs_real.csv`
- `${OUTPUT_ROOT}/module_d_seed7_npcs20_k7/phase2/seed7_npcs20_k7_all_labeled_performance.pdf`

To retrain the phase2 checkpoint instead of loading the archived one, add
`--train-phase2`. Exact retraining also requires the archived
`${MODEL_ROOT}/single_cell/seed7_phase1_pseudobulk_model.pth` checkpoint because
phase2 starts from phase1-derived cell embeddings.

```bash
python scripts/run_module_d.py \
  --assay scrnaseq \
  --scrnaseq-bundle-dir "${SCRNASEQ_BUNDLE_DIR}" \
  --scrnaseq-ppi-path "${SCRNASEQ_PPI_PATH}" \
  --nanospins-truth-csv "${NANOSPINS_TRUTH_CSV}" \
  --nanospins-mapping-xlsx "${NANOSPINS_MAPPING_XLSX}" \
  --seed 0 \
  --device auto \
  --output-dir "${OUTPUT_ROOT}/module_d_scrnaseq"
```

For the scRNA-seq option, set `SCRNASEQ_BUNDLE_DIR`, `SCRNASEQ_PPI_PATH`, `NANOSPINS_TRUTH_CSV`, and `NANOSPINS_MAPPING_XLSX` to the corresponding local inputs before running. The PPI path may be a numeric CSV or SciPy sparse NPZ matrix.

## One-Command Pipeline

To run the mouse bulk/PPI tasks together with the human single-cell transfer task:

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

In this combined command, `--species mouse` applies to Modules A-C. `--single-cell-assay scriboseq` makes Module D use the human scRibo-seq transfer inputs listed above. Use the dedicated `run_module_d.py --assay scrnaseq` command above for the scRNA-seq workflow. When Module A and Module C are run together with `--bulk-train`, Module C uses `${OUTPUT_ROOT}/all_modules/module_a/model.pth`. Without `--bulk-train`, Module C falls back to the bundled known-protein bulk checkpoint.

Add `--bulk-train` to retrain Modules A and B, `--bulk-max-epochs`, `--bulk-patience`, and `--bulk-learning-rate` to change bulk training, and `--train-edge-classifier`, `--edge-max-epochs`, and `--edge-patience` to change Module C training. Module D branch-specific options are controlled through `run_module_d.py`.

## Fixed Split Files

Train/validation/test split files are included under `data/splits/`. The data preparation section above shows the regeneration command. Pass these files with `--split-csv` for bulk modules, `--phase0-split-csv`, `--phase1-split-csv`, and `--phase2-split-csv` for Module D, or the matching split arguments in `run_recipe.py`. If a split file is not provided, the runners fall back to generating the train/validation/test partitions from the seed value.

## Custom Bulk Columns

The example bulk data use column names such as `rNC2`, `rKD2`, `NC3`, and `KD3`, but these names are not required. For Modules A-C, pass `--input-col`, `--target-col`, and optionally `--pause-col` to select columns in your own reference CSV. The input column can contain RNA-seq, Ribo-seq, or another transcript-level feature. Use `--no-pause` when no pausing-count column is available.

For the combined runner, use the matching bulk arguments: `--bulk-input-col`, `--bulk-target-col`, `--bulk-pause-col`, and `--no-bulk-pause`.
