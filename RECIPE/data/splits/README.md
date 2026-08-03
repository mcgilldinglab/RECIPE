# Training Split CSV Files

This directory contains fixed train/validation/test split tables used by the public training notebooks and packaged runners.

Each CSV contains:

- `node_index`: row index in the corresponding packaged feature table.
- `transcript_id`: transcript identifier, when available.
- `protein_id`: protein identifier, when available.
- `split`: `train`, `val`, `test`, or `unlabeled`.
- `is_labeled`: whether the node has a non-zero training target.
- `seed`: random seed used for the split.
- `target_column`: target column used to define labeled nodes.

Use the `split` column to select the rows for each training stage. These files can be regenerated with:

```bash
python scripts/build_training_splits.py
```

## Included Splits

- `bulk_mouse_unknown_seed*.csv`: bulk mouse unknown-protein training splits used by the mouse benchmark notebook.
- `bulk_human_known_seed*.csv`: bulk human known-protein splits used by the bulk self-learning notebook.
- `single_cell_self_learning_seed*.csv`: 11,619-node single-cell transfer/self-learning splits.
- `single_cell_module_a_seed42.csv`: single-cell Module A fine-tuning split with 60% train, 20% validation, and 20% test among labeled nodes.
- `single_cell_graph_seed42.csv`: single-cell graph training split with 75% train, 12.5% validation, and 12.5% test among labeled nodes.
