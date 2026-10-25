"""Complete synthetic C4 preparation/training/resume; no corpus or CUDA use."""
from pathlib import Path
import json

import torch

from music_critic.experiments.c4_pop909 import training
from music_critic.experiments.analysisgnn.performance_replay import recursive_state_fingerprint
from test_c4_pop909_data import synthetic_population  # Shared generated MIDI fixture.


def test_prepare_train_and_resume_keep_test_closed_and_exact_state(
        synthetic_population, tmp_path, monkeypatch):
    population, source_root = synthetic_population
    test_path = (source_root / "003.mid").resolve()
    original_reader = Path.read_bytes
    def deny_test(path):
        assert path.resolve() != test_path, "synthetic TEST source payload was accessed"
        return original_reader(path)
    monkeypatch.setattr(Path, "read_bytes", deny_test)
    # These controls stand in only for experiment registration and source-tree
    # snapshotting. All data, graph, model, optimizer and checkpoint paths are real.
    monkeypatch.setattr(training, "registered_record",
        lambda *args, **kwargs: {"experiment_id": "EXP-SYNTHETIC-C4-INTEGRATION"})
    monkeypatch.setattr(training, "implementation_binding",
        lambda: {"files": {}, "fingerprint": "0" * 64, "c3_source_id": "synthetic-snapshot"})
    previous_threads = torch.get_num_threads()
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    try:
        torch.set_num_threads(1)
        prepared = tmp_path / "prepared"
        report = training.prepare(index=source_root.parent / "pop.index.json",
            split=source_root.parent / "split.json", companion_indices=(),
            source_root=source_root, output=prepared, experiment_record=tmp_path / "synthetic-record.json")
        assert report["status"] == "completed"
        assert report["access"]["validation_accessed"] is False
        assert report["access"]["test_used_for_selection"] is False
        manifest = training.load_manifest(prepared / "manifest.json")
        assert len(manifest["orbit"]) == 12
        assert {row["record_id"] for row in manifest["orbit"]} == {"piece:pop909-cl-001"}
        assert manifest["validation_ids"] == ["piece:pop909-cl-002"]
        assert manifest["class_counts"]["root"] == [1] * 12
        assert manifest["class_counts"]["bass"] == [1] * 12
        config = training.TrainingConfig(total_updates=2, batch_size=1, workers=0,
            max_pending=2, cache_bytes=10_000_000, validation_interval=1,
            checkpoint_interval=1, progress_interval=1, max_wall_seconds=120.)
        first = training.train(manifest_path=prepared / "manifest.json", source_root=source_root,
            output=tmp_path / "first", experiment_record=tmp_path / "synthetic-record.json", config=config)
        assert first["status"] == "completed", first["error"]
        assert first["applied_updates"] == 2 and first["draws"] == 2
        assert first["validation_history"][-1]["update"] == 2
        resume_path = tmp_path / "first" / "checkpoint-000001.pt"
        resumed = training.train(manifest_path=prepared / "manifest.json", source_root=source_root,
            output=tmp_path / "resumed", experiment_record=tmp_path / "synthetic-record.json", config=config,
            resume=resume_path, resume_sha256=training.file_identity(resume_path)["sha256"])
        assert resumed["status"] == "completed", resumed["error"]
        assert resumed["applied_updates"] == 2 and resumed["this_attempt_updates"] == 1
        assert resumed["best_checkpoint"] is not None
        assert resumed["best_checkpoint"]["update"] == resumed["best_update"]
        assert resumed["access"]["validation_accessed"] is True
        assert resumed["access"]["test_used_for_selection"] is False
        full = torch.load(first["final_checkpoint"]["path"], weights_only=True)
        continued = torch.load(resumed["final_checkpoint"]["path"], weights_only=True)
        fields = ("model_state", "optimizer_state", "scheduler_state", "sampler_state", "rng")
        assert recursive_state_fingerprint(tuple(full[key] for key in fields)) == \
               recursive_state_fingerprint(tuple(continued[key] for key in fields))
        assert first["final_model_fingerprint"] == resumed["final_model_fingerprint"]
        assert first["validation_history"] == resumed["validation_history"]
    finally:
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_determinism)


def test_failed_partial_update_keeps_previous_safe_checkpoint(
        synthetic_population, tmp_path, monkeypatch):
    _, source_root = synthetic_population
    monkeypatch.setattr(training, "registered_record",
        lambda *args, **kwargs: {"experiment_id": "EXP-SYNTHETIC-C4-FAILURE"})
    monkeypatch.setattr(training, "implementation_binding",
        lambda: {"files": {}, "fingerprint": "0" * 64, "c3_source_id": "synthetic-snapshot"})
    previous_threads = torch.get_num_threads()
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    try:
        torch.set_num_threads(1)
        prepared = tmp_path / "prepared"
        training.prepare(index=source_root.parent / "pop.index.json",
            split=source_root.parent / "split.json", companion_indices=(),
            source_root=source_root, output=prepared, experiment_record=tmp_path / "synthetic-record.json")
        original_update = training.attempt_update
        calls = []
        def failing_update(model, *args, **kwargs):
            calls.append(True)
            if len(calls) == 1:
                return original_update(model, *args, **kwargs)
            torch.rand(7)  # A failed forward may already have consumed dropout RNG.
            with torch.no_grad():
                next(model.parameters()).add_(123.)
            raise RuntimeError("synthetic failure during update")
        monkeypatch.setattr(training, "attempt_update", failing_update)
        config = training.TrainingConfig(total_updates=3, batch_size=1, workers=0,
            max_pending=2, cache_bytes=10_000_000, validation_interval=3,
            checkpoint_interval=3, progress_interval=1, max_wall_seconds=120.)
        report = training.train(manifest_path=prepared / "manifest.json", source_root=source_root,
            output=tmp_path / "failed", experiment_record=tmp_path / "synthetic-record.json", config=config)
        assert report["status"] == "failed"
        assert report["applied_updates"] == 1 and report["attempted_updates"] == 2
        assert report["final_checkpoint"]["update"] == 0
        assert not (tmp_path / "failed" / "checkpoint-000001.pt").exists()
        safe = torch.load(report["final_checkpoint"]["path"], weights_only=True)
        assert safe["progress"]["applied_updates"] == 0
        assert safe["model_state_fingerprint"] == report["initial_model_fingerprint"]
    finally:
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_determinism)


def test_cli_synthetic_gate_covers_all_shifts_and_exact_resume(
        synthetic_population, tmp_path, monkeypatch):
    _, source_root = synthetic_population
    identity = {"experiment_id": "EXP-SYNTHETIC-C4-CLI"}
    planned_path = tmp_path / "synthetic-record.json"
    planned_path.write_text(json.dumps(identity))
    monkeypatch.setattr(training, "registered_record", lambda *args, **kwargs: identity)
    monkeypatch.setattr(training, "implementation_binding",
        lambda: {"files": {}, "fingerprint": "0" * 64, "c3_source_id": "synthetic-snapshot"})
    common = ["--source-root", str(source_root), "--experiment-record", str(planned_path)]
    previous_threads = torch.get_num_threads()
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    try:
        torch.set_num_threads(1)
        prepared, gated = tmp_path / "prepared", tmp_path / "gate"
        assert training.main(["prepare", "--index", str(source_root.parent / "pop.index.json"),
            "--split", str(source_root.parent / "split.json"), "--output", str(prepared), *common]) == 0
        result = training.main(["gate", "--manifest", str(prepared / "manifest.json"),
            "--output", str(gated), "--updates", "4", "--batch-size", "1", "--workers", "0",
            "--cache-bytes", "10000000", "--max-pending", "2", "--device", "cpu",
            "--max-wall-seconds", "120", *common])
        report = json.loads((gated / "report.json").read_text())
        assert result == 0, report.get("error")
        assert report["status"] == "completed"
        assert report["applied_updates"] == 32
        assert report["shift_coverage"] == list(range(12))
        assert report["exact_reference_pipeline_parity"]
        assert report["exact_full_resume_parity"]
    finally:
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_determinism)
