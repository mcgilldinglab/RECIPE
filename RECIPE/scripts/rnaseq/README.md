# Module D scRNA-seq Phase Scripts

These scripts implement the Module D scRNA-seq branch:

- `train_phase0_ensmusp_pseudobulk_raw_bulkprot.py`: phase0 bulk module using pseudo-bulk RNA and bulk protein.
- `train_phase12_ensmusp_scRNA_bulkprot.py`: phase12 cell-split RNA pseudo-bulk/cell-graph training and phase2 hidden-cache export.
- `train_phase3_ensmusp_nanospins_matched.py`: phase3 nanoSPINS matched single-cell protein training.

Run them through `scripts/run_module_d.py --assay scrnaseq` so the public entry point is consistent with the rest of RECIPE. All data locations must be passed explicitly; these scripts do not contain local absolute-path defaults.
