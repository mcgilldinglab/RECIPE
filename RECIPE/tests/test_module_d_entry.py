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
