from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import probe_temporal_gpu_memory as probe


def mock_gpu_listing(monkeypatch, listing, executables):
    def run(command, **kwargs):
        assert command == ["nvidia-smi", "--id=0", "--query-compute-apps=pid,process_name,used_gpu_memory",
                           "--format=csv,noheader,nounits"]
        assert kwargs == dict(check=True, capture_output=True, text=True, timeout=10)
        return SimpleNamespace(stdout=listing)
    monkeypatch.setattr(probe.subprocess, "run", run)
    monkeypatch.setattr(probe.os, "getpid", lambda: 900000)
    monkeypatch.setattr(probe.os, "readlink", lambda path: executables[int(path.split("/")[2])])


def test_idle_guard_accepts_only_verified_observed_desktop_apps(monkeypatch):
    apps = {7482: "/usr/libexec/gnome-remote-desktop-daemon", 299037: "/usr/bin/nautilus",
            9065: "/snap/snapd-desktop-integration/391/usr/bin/snapd-desktop-integration"}
    memory = {7482: 369, 299037: 47, 9065: 13}
    mock_gpu_listing(monkeypatch, "\n".join(f"{pid}, {name}, {memory[pid]}" for pid, name in apps.items()), apps)
    rows = probe.assert_probe_gpu_idle()
    assert len(rows) == 3 and sum(r["used_memory_mib"] for r in rows) == 429
    assert all(r["proc_exe"] == apps[r["pid"]] and r["allowed_as"] == "desktop" for r in rows)


@pytest.mark.parametrize("listing,exe,reason", [
    ("123, /usr/bin/python3, 100", "/usr/bin/python3", "other_gpu_compute"),
    ("123, /usr/bin/nautilus, 601", "/usr/bin/nautilus", "desktop_gpu_memory"),
    ("123, /usr/bin/nautilus, 47", "/tmp/renamed-trainer", "executable_mismatch"),
    ("123, /usr/bin/nautilus, N/A", "/usr/bin/nautilus", "listing_invalid"),
    ("123, /usr/bin/nautilus, 1\n123, /usr/bin/nautilus, 1", "/usr/bin/nautilus", "pid_invalid"),
    ("123, /snap/snapd-desktop-integration/392/usr/bin/snapd-desktop-integration, 1", "other", "other_gpu_compute"),
])
def test_idle_guard_rejects_unknown_or_unverified_process(monkeypatch, listing, exe, reason):
    mock_gpu_listing(monkeypatch, listing, {123: exe})
    with pytest.raises(ValueError, match=reason):
        probe.assert_probe_gpu_idle()


def test_idle_guard_self_exemption_and_unreadable_proc(monkeypatch):
    mock_gpu_listing(monkeypatch, "900000, /usr/bin/python3, 2000", {900000: "/usr/bin/python3"})
    assert probe.assert_probe_gpu_idle()[0]["allowed_as"] == "self"
    def missing(path):
        raise FileNotFoundError(path)
    monkeypatch.setattr(probe.os, "readlink", missing)
    with pytest.raises(ValueError, match="identity_unverified"):
        probe.assert_probe_gpu_idle()


def inventory(distinct=True):
    rows = [dict(record_id=n, component_id="c-"+n, piece_sha256=n*64,
                 total_tensor_bytes=100-i, slice_nodes=20-i, base_nodes={"note": 5},
                 base_edges=10, extension_edges=20) for i, n in enumerate("abcd")]
    by_slices = list(reversed(rows)) if distinct else rows
    if distinct:
        for i, row in enumerate(rows):
            row["slice_nodes"] = 20+i
    report = dict(schema="TemporalFullTrainSizing@1.0.0", completed=True, records=1295,
                  validation_payload_accessed=False, test_payload_accessed=False,
                  top_by_tensor_bytes=rows, top_by_slice_nodes=by_slices)
    manifest = {"records": [dict(record_id=r["record_id"], source_component_id=r["component_id"],
                                 split="train", piece={"sha256": r["piece_sha256"]}) for r in rows]}
    return report, manifest


def test_selects_union_duplicate_batches_and_largest_pair():
    report, manifest = inventory()
    selected, jobs = probe.select_probe(report, manifest)
    assert set(selected) == set("abcd") and len(jobs) == 6
    assert [r["record_ids"] for r in jobs[:4]] == [[n, n] for n in "abcd"]
    assert jobs[-2]["record_ids"] == ["a", "b"] and jobs[-2]["shifts"] == [0, 0]
    assert jobs[-1]["shifts"] == [6, 6]
    report, manifest = inventory(False)
    selected, jobs = probe.select_probe(report, manifest)
    assert set(selected) == set("ab") and len(jobs) == 4


@pytest.mark.parametrize("change", ["test", "validation", "wrong_piece", "wrong_component", "duplicate", "rank", "incomplete"])
def test_selection_fails_closed_before_payload_access(change):
    report, manifest = inventory()
    if change == "test":
        manifest["records"][-1]["split"] = "test"
    elif change == "validation":
        manifest["records"][0]["split"] = "validation"
    elif change == "wrong_piece":
        manifest["records"][0]["piece"]["sha256"] = "0"*64
    elif change == "wrong_component":
        manifest["records"][0]["source_component_id"] = "different"
    elif change == "duplicate":
        report["top_by_tensor_bytes"][1] = deepcopy(report["top_by_tensor_bytes"][0])
    elif change == "rank":
        report["top_by_tensor_bytes"] = list(reversed(report["top_by_tensor_bytes"]))
    else:
        report["completed"] = False
    with pytest.raises(ValueError):
        probe.select_probe(report, manifest)


@pytest.mark.parametrize("values", [(17*1024**3, 17*1024**3), (1, 17*1024**3), (5, 4), (-1, 1), (True, 1)])
def test_memory_ceiling_checks_reserved_and_allocated(values):
    with pytest.raises(ValueError, match="ceiling"):
        probe.validate_ceiling(*values)


def test_valid_memory_and_exact_rebuild_dimensions():
    probe.validate_ceiling(100, probe.GPU_LIMIT_BYTES)
    expected = inventory()[0]["top_by_tensor_bytes"][0]
    probe.check_dimensions(expected, expected)
    for field in ("total_tensor_bytes", "slice_nodes", "base_edges", "extension_edges"):
        with pytest.raises(ValueError, match="rebuilt_sizing"):
            probe.check_dimensions({**expected, field: expected[field]+1}, expected)


def test_source_hash_path_and_probe_binding(tmp_path, monkeypatch):
    path = tmp_path / "probe.py"
    path.write_text("# frozen fixture\n")
    monkeypatch.setattr(probe, "__file__", str(path))
    source = tmp_path / "sources.json"
    source.write_text(json.dumps({"files": [{"path": "probe.py", "sha256": sha256(path.read_bytes()).hexdigest()}]}))
    config = dict(source_manifest=str(source), source_manifest_sha256=sha256(source.read_bytes()).hexdigest(), source_root=str(tmp_path))
    assert probe.verify_sources(config) == 1
    path.write_text("changed")
    with pytest.raises(ValueError, match="source_changed"):
        probe.verify_sources(config)
    source.write_text(json.dumps({"files": [{"path": "../outside.py", "sha256": "0"*64}]}))
    config["source_manifest_sha256"] = sha256(source.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="source_path"):
        probe.verify_sources(config)


def test_configuration_requires_planned_six_forwards_and_completed_parent(tmp_path, monkeypatch):
    sizing, manifest = inventory()
    def write(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return str(path), sha256(path.read_bytes()).hexdigest()
    mpath, msha = write("data.json", manifest)
    spath, ssha = write("sizing.json", sizing)
    monkeypatch.setattr(probe, "verify_sources", lambda config: 1)
    current = {"status": "planned", "experiment_id": "EXP-FIXTURE", "budget": {"planned": 6}}
    parent = {"status": "completed", "artifacts": [{"sha256": ssha, "size_bytes": Path(spath).stat().st_size}]}
    monkeypatch.setattr(probe, "load_record", lambda path: current if str(path) == "current" else parent)
    config = dict(record="current", sizing_record="parent", experiment_id="EXP-FIXTURE",
                  max_forwards=6, gpu_limit_bytes=probe.GPU_LIMIT_BYTES,
                  data_manifest=mpath, data_manifest_sha256=msha, sizing_report=spath, sizing_report_sha256=ssha)
    assert len(probe.check_configuration(config)[2]) == 6
    parent["status"] = "running"
    with pytest.raises(ValueError, match="completed_sizing_record"):
        probe.check_configuration(config)
    parent["status"] = "completed"
    parent["artifacts"][0]["sha256"] = "0"*64
    with pytest.raises(ValueError, match="sizing_record_artifact_binding"):
        probe.check_configuration(config)
    current["budget"]["planned"] = 7
    with pytest.raises(ValueError, match="planned_forward_budget"):
        probe.check_configuration(config)


def test_source_free_forward_backward_changes_no_model_weights(monkeypatch):
    import music_critic.tasks as tasks
    from music_critic.experiments.analysisgnn import corrected_training as old
    from music_critic.experiments.analysisgnn.corrected_model import model_state_fingerprint
    from music_critic.experiments.analysisgnn.temporal_slices import build_temporal_batch, TemporalSliceModel
    captured = []
    prepare = tasks.prepare_multisource_sample
    def capture(piece, *args, **kwargs):
        captured.append(piece)
        return prepare(piece, *args, **kwargs)
    monkeypatch.setattr(tasks, "prepare_multisource_sample", capture)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        _, sidecar = old.build_source_free_fixture()
        graph = build_temporal_batch([captured[0], captured[0]])
        model = TemporalSliceModel().train()
        before = model_state_fingerprint(model)
        result = probe.execute_probe(model, graph, [sidecar, sidecar], [0, 0], old.load_frozen_class_weights())
        assert result["finite_logits"] and result["finite_gradients"] and result["gradient_norm"] > 0
        assert len(result["valid_rows"]) == 18
        assert before == model_state_fingerprint(model)
        assert model.slice_projection.weight.grad.abs().sum() > 0
    finally:
        torch.set_num_threads(threads)
