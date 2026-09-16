from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import runpy

import pytest
import torch

from music_critic.experiments.analysisgnn.corrected_model import CorrectedAnalysisGNNModel
from music_critic.experiments.context_diagnostics import fingerprint
from music_critic.experiments.context_diagnostics_run import (
    SuiteProgress, _predict_accounted, checkpoint_suite_protocol, main, run_checkpoint_suite, suite_manifest,
)
from music_critic.experiments.pop909_cl_frozen.checkpoint import create_model_only_payload
from music_critic.experiments.registry import canonical_json_bytes, seal_record


def test_suite_deterministic_and_review_limits(tmp_path):
    first = suite_manifest()
    assert first == suite_manifest()
    assert len(first["pairs"]) == 5
    assert first["musician_review"] == "not_reviewed"
    assert first["preferred_side"] is None
    output = tmp_path / "suite.json"
    assert main(["--suite", "--output", str(output)]) == 0
    assert json.loads(output.read_text()) == first
    with pytest.raises(FileExistsError):
        main(["--suite", "--output", str(output)])


@pytest.mark.parametrize("split", ["test", "TEST", "train", "validation"])
def test_cli_partition_guard_before_output_or_payload_read(tmp_path, split):
    output = tmp_path / "out.json"
    with pytest.raises(ValueError, match="TEST_closed"):
        main(["--checkpoint-suite", "--split", split, "--output", str(output)])
    assert not output.exists()


def test_preflight_binds_source_classes_and_device_without_model_reads(tmp_path):
    output = tmp_path / "preflight.json"
    assert main(["--preflight", "--checkpoint-sha256", "0"*64, "--checkpoint-size-bytes", "123",
                 "--device", "cuda:0", "--output", str(output)]) == 0
    protocol = json.loads(output.read_text())
    assert protocol["forward_budget"] == 20
    assert len(protocol["ordered_classes"]["quality"]) == 17
    assert protocol["training_updates"] == 0
    assert "graph/feature_registry.py" in protocol["source_sha256"]
    assert "models/encoder.py" in protocol["source_sha256"]
    assert "models/heads.py" in protocol["source_sha256"]
    assert "experiments/pop909_cl_frozen/checkpoint.py" in protocol["source_sha256"]
    wrapper = Path(__file__).parents[1] / protocol["external_resource_wrapper"]["path"]
    assert protocol["external_resource_wrapper"]["sha256"] == sha256(wrapper.read_bytes()).hexdigest()
    assert fingerprint(protocol) != fingerprint(checkpoint_suite_protocol("0"*64, 123, "cpu"))


def test_preflight_preserves_caller_cpu_rng():
    before = torch.random.get_rng_state().clone()
    checkpoint_suite_protocol("0"*64, 123, "cpu")
    assert torch.equal(before, torch.random.get_rng_state())


@pytest.mark.parametrize("error,status", [(ValueError("fixture failure"), "failed"), (KeyboardInterrupt(), "aborted")])
def test_cli_failure_and_abort_receipts_are_preserved(tmp_path, monkeypatch, error, status):
    from music_critic.experiments import context_diagnostics_run as module
    def fail(**kwargs):
        raise error
    monkeypatch.setattr(module, "run_checkpoint_suite", fail)
    output = tmp_path/"failed.json"
    args = ["--checkpoint-suite", "--checkpoint", "not-read.pt", "--experiment-record", "not-read.json",
            "--output", str(output)]
    with pytest.raises(type(error)):
        module.main(args)
    receipt = json.loads(output.read_text())
    assert receipt["status"] == status
    with pytest.raises(FileExistsError):
        module.main(args)


def _planned(tmp_path, protocol, *, status="planned"):
    template = Path(__file__).parents[1]/"docs/experiments/records/EXP-CTX-C2-ARTIFACT-002.json"
    record = json.loads(template.read_text())
    record.update(experiment_id="EXP-CTX-C2-SYNTHETIC-UNIT", status=status,
                  protocol={"context_binding_fingerprint": fingerprint(protocol)})
    if status == "completed":
        record.update(results={"smoke": True}, validity={"valid": True, "reason": None})
    path = tmp_path/"EXP-CTX-C2-SYNTHETIC-UNIT.json"
    path.write_bytes(canonical_json_bytes(seal_record(record))+b"\n")
    return path


def test_record_gate_before_checkpoint_read(tmp_path):
    protocol = checkpoint_suite_protocol("0"*64, 123, "cpu")
    protocol["checkpoint"]["sha256"] = "1"*64
    record = _planned(tmp_path, protocol)
    with pytest.raises(ValueError, match="matching_planned"):
        run_checkpoint_suite(checkpoint=tmp_path/"not-read.pt", checkpoint_sha256="0"*64,
                             size_bytes=123, device="cpu", experiment_record=record)


def test_tiny_random_checkpoint_suite_mechanics_only(tmp_path):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        model = CorrectedAnalysisGNNModel().eval()
    checkpoint = tmp_path/"fixture.pt"
    torch.save(create_model_only_payload(model), checkpoint)
    raw = checkpoint.read_bytes()
    digest = sha256(raw).hexdigest()
    record = _planned(tmp_path, checkpoint_suite_protocol(digest, len(raw), "cpu"))
    output = tmp_path / "result.json"
    journal = tmp_path / "progress.jsonl"
    assert main(["--checkpoint-suite", "--checkpoint", str(checkpoint), "--checkpoint-sha256", digest,
                 "--checkpoint-size-bytes", str(len(raw)), "--device", "cpu",
                 "--experiment-record", str(record), "--output", str(output),
                 "--progress", str(journal)]) == 0
    result = json.loads(output.read_text())
    wrapper = runpy.run_path(str(Path(__file__).parents[1] / "scripts/context_diagnostics_resource_wrapper.py"))
    recovered = wrapper["recover_progress"](journal)
    assert recovered["journal_partial"] is False
    assert recovered["last_valid_record"]["event"] == "result_written"
    assert recovered["accounting"]["forwards_started"] == recovered["accounting"]["forwards_completed"] == 20
    assert recovered["accounting"]["forwards_completion_unknown"] == 0
    assert recovered["accounting"]["views_recorded"] == 20
    assert result["model_forwards"] == 20
    assert result["progress"]["forwards_started"] == result["progress"]["forwards_completed"] == 20
    assert result["progress"]["forwards_completion_unknown"] == 0
    assert result["progress"]["views_recorded"] == 20
    assert result["correctness_claim"] is result["quality_preference_claim"] is None
    assert result["TEST"] == {"inputs": False, "targets": False, "metrics": False}
    for pair in result["observations"]:
        for side in pair["sides"].values():
            assert side["full"]["local_features"] == side["one_qn_note_evidence_same_clock_grid"]["local_features"]
            for view in side.values():
                for task in view["tasks"].values():
                    assert sum(task["probabilities"]) == pytest.approx(1., abs=1e-6)
        assert all(v["correctness"] is None for view in pair["sensitivity"].values() for v in view.values())


@pytest.mark.parametrize("failure_stage", ["predict", "postprocess", "interrupt"])
def test_failure_after_three_completed_forwards_has_durable_accounting(tmp_path, monkeypatch, failure_stage):
    from music_critic.experiments import context_diagnostics_run as module
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        model = CorrectedAnalysisGNNModel().eval()
    checkpoint = tmp_path/"fixture.pt"
    torch.save(create_model_only_payload(model), checkpoint)
    raw = checkpoint.read_bytes()
    digest = sha256(raw).hexdigest()
    record = _planned(tmp_path, checkpoint_suite_protocol(digest, len(raw), "cpu"))
    predict = CorrectedAnalysisGNNModel.predict
    count = 0
    journal = tmp_path/"progress.jsonl"

    def failing_predict(self, batch, **kwargs):
        nonlocal count
        count += 1
        if count == 4 and failure_stage != "postprocess":
            # The attempt must already be persisted before predict is entered.
            last = json.loads(journal.read_text().splitlines()[-1])
            assert (last["forwards_started"], last["forwards_completed"]) == (4, 3)
            if failure_stage == "interrupt":
                raise KeyboardInterrupt()
            raise RuntimeError("injected fourth predict failure")
        return predict(self, batch, **kwargs)

    original_graph_fingerprint = module.graph_fingerprint
    def failing_postprocess(graph):
        if count == 4 and failure_stage == "postprocess":
            raise RuntimeError("injected postprocessing failure")
        return original_graph_fingerprint(graph)
    monkeypatch.setattr(CorrectedAnalysisGNNModel, "predict", failing_predict)
    monkeypatch.setattr(module, "graph_fingerprint", failing_postprocess)
    # Source code hashing is unchanged by these injected in-memory failures.
    output = tmp_path/"failed.json"
    args = ["--checkpoint-suite", "--checkpoint", str(checkpoint), "--checkpoint-sha256", digest,
            "--checkpoint-size-bytes", str(len(raw)), "--experiment-record", str(record),
            "--progress", str(journal), "--output", str(output)]
    with pytest.raises(KeyboardInterrupt if failure_stage == "interrupt" else RuntimeError):
        module.main(args)
    result = json.loads(output.read_text())
    progress = result["progress"]
    assert progress["pair_id"] == suite_manifest()["pairs"][0]["pair_id"]
    assert progress["side"] == "right"
    assert progress["view"] == "one_qn_note_evidence_same_clock_grid"
    assert progress["forwards_started"] == 4
    assert progress["forwards_completed"] == (4 if failure_stage == "postprocess" else 3)
    assert progress["forwards_completion_unknown"] == (0 if failure_stage == "postprocess" else 1)
    assert progress["views_recorded"] == 3
    assert progress["applied_updates"] == 0
    assert progress["failure_stage"] == ("observation_validation" if failure_stage == "postprocess" else "model_predict")
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    assert rows[-1]["forwards_completed"] == progress["forwards_completed"]
    assert rows[-1]["failure_stage"] == progress["failure_stage"]
    assert [row["sequence"] for row in rows] == list(range(1, len(rows)+1))
    assert len(rows) <= 200 and all(len(line.encode()) <= 4096 for line in journal.read_text().splitlines())
    prior = journal.read_bytes()
    with pytest.raises(FileExistsError):
        module.main(args)
    assert journal.read_bytes() == prior


def test_cuda_completion_stays_unknown_until_successful_sync(monkeypatch, tmp_path):
    # No real CUDA calls or GPU tensors: isolate the synchronization boundary.
    from music_critic.experiments import context_diagnostics_run as module
    class FakeModel:
        def predict(self, batch):
            return "returned"
    def failed_sync(device):
        assert device == "cuda:0"
        raise KeyboardInterrupt()
    with (tmp_path/"journal.jsonl").open("x") as stream:
        progress = SuiteProgress(stream=stream, forwards_started=3, forwards_completed=3, views_recorded=3)
        monkeypatch.setattr(module.torch.cuda, "synchronize", failed_sync)
        with pytest.raises(KeyboardInterrupt):
            _predict_accounted(FakeModel(), None, "cuda:0", progress)
        snapshot = progress.snapshot()
        assert snapshot["prediction_returned"] is True
        assert snapshot["stage"] == "cuda_forward_synchronization"
        assert snapshot["forwards_started"] == 4 and snapshot["forwards_completed"] == 3
        assert snapshot["forwards_completion_unknown"] == 1
    persisted = json.loads((tmp_path/"journal.jsonl").read_text().splitlines()[-1])
    assert persisted["forwards_completion_unknown"] == 1
    fresh = SuiteProgress()
    devices = []
    monkeypatch.setattr(module.torch.cuda, "synchronize", lambda device: devices.append(device))
    assert _predict_accounted(FakeModel(), None, "cuda:0", fresh) == "returned"
    assert fresh.snapshot()["forwards_completed"] == 1
    assert fresh.snapshot()["forwards_completion_unknown"] == 0
    assert devices == ["cuda:0"]


def test_partial_result_write_gets_parseable_failure_receipt(tmp_path, monkeypatch):
    from music_critic.experiments import context_diagnostics_run as module
    output = tmp_path/"result.json"
    original_open = Path.open
    class BreakOnce:
        def __init__(self, stream):
            self.stream, self.broken = stream, False
        def __getattr__(self, key):
            return getattr(self.stream, key)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return self.stream.__exit__(*args)
        def write(self, value):
            if not self.broken:
                self.broken = True
                self.stream.write(value[:5])
                raise KeyboardInterrupt()
            return self.stream.write(value)
    def guarded_open(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        return BreakOnce(stream) if path == output and args and args[0] == "x" else stream
    def completed_fixture(**kwargs):
        progress = kwargs["progress"]
        progress.forwards_started = progress.forwards_completed = progress.views_recorded = 20
        progress.emit("suite_completed", "suite_completed")
        return {"synthetic_fixture": True}
    monkeypatch.setattr(module, "run_checkpoint_suite", completed_fixture)
    monkeypatch.setattr(Path, "open", guarded_open)
    with pytest.raises(KeyboardInterrupt):
        module.main(["--checkpoint-suite", "--checkpoint", "fixture-not-read.pt", "--experiment-record",
                     "fixture-not-read.json", "--output", str(output)])
    receipt = json.loads(output.read_text())
    assert receipt["status"] == "aborted"
    assert receipt["progress"]["forwards_completed"] == 20
    assert receipt["progress"]["forwards_completion_unknown"] == 0
    assert receipt["progress"]["failure_stage"] == "output_write"
    last = json.loads(output.with_name(output.name+".progress.jsonl").read_text().splitlines()[-1])
    assert last["failure_stage"] == "output_write"


def test_existing_progress_rejected_before_checkpoint_or_output(tmp_path):
    journal = tmp_path/"result.json.progress.jsonl"
    journal.write_text("retained evidence\n")
    output = tmp_path/"result.json"
    with pytest.raises(FileExistsError):
        main(["--checkpoint-suite", "--output", str(output)])
    assert not output.exists()
    assert journal.read_text() == "retained evidence\n"
