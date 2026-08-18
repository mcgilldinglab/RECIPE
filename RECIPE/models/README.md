# Model Checkpoints

Pretrained checkpoints used by the default workflows are included in this directory and tracked with Git LFS. After cloning, run:

```bash
git lfs pull
```

Bundled checkpoints:

- `bulk/human_known_seed12.pth`: human known-protein bulk model.
- `bulk/human_unknown_seed0.pth`: human unknown-protein bulk model.
- `bulk/mouse_known_seed5.pth`: mouse known-protein bulk model.
- `bulk/mouse_unknown_seed1.pth`: mouse unknown-protein bulk model.
- `ppi/human_edge_classifier.pth`: archived human PPI edge-classifier output.
- `ppi/mouse_edge_classifier.pth`: archived mouse PPI edge-classifier output.
- `single_cell/bulk_self_learning.pth`: Module D phase0 initialization checkpoint.
- `single_cell/bulk_self_learning_full.pth`: full phase0 self-learning checkpoint.
- `single_cell/pseudobulk_finetuned.pth`: phase1 pseudobulk fine-tuned checkpoint.
- `single_cell/cell_graph_seed12.pth`: phase2 cell-graph checkpoint.
- `single_cell/seed7_phase1_pseudobulk_model.pth`: archived phase1 checkpoint used to reproduce the scRibo-seq Module D `seed7_npcs20_k7_all_labeled` run.
- `single_cell/seed7_npcs20_k7_phase2_rsc_model.pth`: archived phase2 `RSCHead` checkpoint from the scRibo-seq Module D `seed7_npcs20_k7_all_labeled` run.

Modules A and B use the bulk checkpoints unless `--train` is passed or a different `--checkpoint-path` is provided. Module C uses the bundled PPI edge-classifier checkpoint unless `--train-edge-classifier` is passed; use `--edge-checkpoint-path` to override it. Module D `--assay scriboseq` uses the phase0, phase1, and phase2 single-cell checkpoints by default unless the matching `--train-phase*` flag is passed. Module D `--assay scrnaseq` writes its phase3 checkpoint as `phase3/models/phase3_nanospins_best.pth` under the requested output root.
