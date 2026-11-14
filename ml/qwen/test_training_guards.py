"""Run-state regression tests only; no Torch import or simulated quality scores."""
import json
from ml.qwen.train import record_failure, diagnostic_subset
import pytest


def test_paired_diagnostic_subset_uses_stable_prefix_without_label_selection():
    calls = [{"call_id": str(i), "facts": ["different labels"] * i} for i in range(6)]
    assert diagnostic_subset(calls, 2) == calls[:2]
    assert diagnostic_subset(calls, 10) == calls
    assert len(calls) == 6
    with pytest.raises(ValueError):
        diagnostic_subset(calls, 0)


def test_existing_successful_run_is_immutable_when_new_attempt_refused(tmp_path):
    path = tmp_path / "pilot.json"
    original = b'{"status":"success","real_measurement":"preserve exactly"}\n'
    path.write_bytes(original)
    assert record_failure(tmp_path, "pilot", "RuntimeError", False) is False
    assert path.read_bytes() == original


def test_fresh_failed_run_does_not_claim_training(tmp_path):
    out = tmp_path / "fresh"
    assert record_failure(out, "pilot", "RuntimeError", False)
    report = json.loads((out / "pilot.json").read_text())
    assert report["status"] == "failed" and not report["main_sft_ready"]
    assert report["successful_training_claimed"] is False


def test_current_run_failure_preserves_measured_fields(tmp_path):
    (tmp_path / "pilot.json").write_text('{"timing_steps":50,"main_sft_ready":true}')
    assert record_failure(tmp_path, "pilot", "OutOfMemoryError", True)
    report = json.loads((tmp_path / "pilot.json").read_text())
    assert report["timing_steps"] == 50 and report["main_sft_ready"] is False
