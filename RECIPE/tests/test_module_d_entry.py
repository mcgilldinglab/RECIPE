from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_module_d.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("run_module_d", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_scrnaseq_entry_connects_all_internal_stages(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(MODULE, "run_scrnaseq_phase0", lambda args: calls.append(("phase0", args)))
    monkeypatch.setattr(MODULE, "run_scrnaseq_phase12", lambda args: calls.append(("phase12", args)))
    monkeypatch.setattr(MODULE, "run_scrnaseq_phase3", lambda args: calls.append(("phase3", args)))
    args = MODULE.build_parser().parse_args(
        [
            "--assay", "scrnaseq",
            "--output-dir", str(tmp_path / "output"),
            "--scrnaseq-bundle-dir", str(tmp_path / "bundle"),
            "--scrnaseq-ppi-path", str(tmp_path / "ppi.npz"),
            "--nanospins-truth-csv", str(tmp_path / "truth.csv"),
            "--nanospins-mapping-xlsx", str(tmp_path / "mapping.xlsx"),
        ]
    )

    summary = MODULE._run_scrnaseq_module_d(args, ("all",))

    assert [name for name, _ in calls] == ["phase0", "phase12", "phase3"]
    assert summary["status"] == "completed"
    assert str(tmp_path / "output" / "phase0" / "best_model.pth") in calls[1][1]
    assert str(tmp_path / "output" / "phase2_hidden_cache") in calls[2][1]


def test_scriboseq_seed7_preset_sets_phase2_reproduction_args(tmp_path):
    model_root = tmp_path / "models"
    single_cell_model_dir = model_root / "single_cell"
    single_cell_model_dir.mkdir(parents=True)
    phase1_checkpoint = single_cell_model_dir / "seed7_phase1_pseudobulk_model.pth"
    phase2_checkpoint = single_cell_model_dir / "seed7_npcs20_k7_phase2_rsc_model.pth"
    phase1_checkpoint.write_bytes(b"phase1")
    phase2_checkpoint.write_bytes(b"phase2")

    args = MODULE.build_parser().parse_args(
        [
            "--assay", "scriboseq",
            "--scriboseq-reproduction-preset", "seed7_npcs20_k7_all_labeled",
            "--model-root", str(model_root),
            "--output-dir", str(tmp_path / "output"),
        ]
    )

    MODULE._apply_scriboseq_reproduction_preset(args)

    assert args.steps == "phase2"
    assert args.seed == 7
    assert args.phase2_n_neighbors == 7
    assert args.phase2_n_pcs == 20
    assert args.phase2_selection_metric == "test_r2"
    assert args.phase1_checkpoint == str(phase1_checkpoint)
    assert args.phase2_checkpoint == str(phase2_checkpoint)


def test_scrnaseq_reproduction_preset_routes_phase3_scatter_args(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        MODULE,
        "run_scrnaseq_phase3_c10_svec_scatter",
        lambda args: calls.append(args),
    )
    args = MODULE.build_parser().parse_args(
        [
            "--assay", "scrnaseq",
            "--scrnaseq-reproduction-preset", "phase3_c10_svec_test_scatter",
            "--scrnaseq-phase23-root", str(tmp_path / "phase23"),
            "--scrnaseq-bundle-dir", str(tmp_path / "bundle"),
            "--nanospins-truth-csv", str(tmp_path / "truth.csv"),
            "--nanospins-mapping-xlsx", str(tmp_path / "mapping.xlsx"),
            "--scrnaseq-reproduction-scenarios", "svec_best_on_svec_test,svec_model_on_c10_test",
            "--output-dir", str(tmp_path / "output"),
            "--device", "cpu",
        ]
    )

    summary = MODULE._run_scrnaseq_module_d(args, ("all",))

    assert summary["preset"] == "phase3_c10_svec_test_scatter"
    assert len(calls) == 1
    forwarded = calls[0]
    assert forwarded[forwarded.index("--phase23-root") + 1] == str((tmp_path / "phase23").resolve())
    assert forwarded[forwarded.index("--bundle-dir") + 1] == str((tmp_path / "bundle").resolve())
    assert forwarded[forwarded.index("--truth-csv") + 1] == str((tmp_path / "truth.csv").resolve())
    assert forwarded[forwarded.index("--mapping-xlsx") + 1] == str((tmp_path / "mapping.xlsx").resolve())
    assert forwarded[forwarded.index("--scenarios") + 1 :] == [
        "svec_best_on_svec_test",
        "svec_model_on_c10_test",
    ]
