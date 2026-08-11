from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "single_cell" / "run_sctranslator_chunked_inference.py"
SPEC = importlib.util.spec_from_file_location("sctranslator_inference", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_load_sctranslator_model_from_full_checkpoint(tmp_path):
    checkpoint = tmp_path / "full.pt"
    torch.save(torch.nn.Linear(2, 1), checkpoint)

    model = MODULE.load_sctranslator_model(checkpoint, None, torch.device("cpu"))

    assert isinstance(model, torch.nn.Linear)


def test_load_sctranslator_model_from_state_dict(tmp_path):
    base_checkpoint = tmp_path / "base.pt"
    weights_checkpoint = tmp_path / "weights.pt"
    base = torch.nn.Linear(2, 1)
    expected = torch.nn.Linear(2, 1)
    torch.save(base, base_checkpoint)
    torch.save(expected.state_dict(), weights_checkpoint)

    model = MODULE.load_sctranslator_model(weights_checkpoint, base_checkpoint, torch.device("cpu"))

    for actual, target in zip(model.parameters(), expected.parameters()):
        assert torch.equal(actual, target)

    with pytest.raises(ValueError, match="base-checkpoint"):
        MODULE.load_sctranslator_model(weights_checkpoint, None, torch.device("cpu"))
