# Model Checkpoints

This directory is reserved for optional trained checkpoints used by the workflow defaults.

Expected checkpoint names:

- `bulk/human_known_seed12.pth`
- `bulk/human_unknown_seed0.pth`
- `bulk/mouse_known_seed5.pth`
- `bulk/mouse_unknown_seed1.pth`
- `single_cell/bulk_self_learning.pth`

The repository can run modules A and B without bundled checkpoints by training a new model into the selected output directory. Module C needs a trained bulk checkpoint; pass it with `--bulk-checkpoint-path` or generate it with module B first.
