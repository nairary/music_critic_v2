"""Synthetic-only tests of registered transfer and its metadata-first boundary."""

from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from music_critic.adapters.pop909_cl import Pop909ClConversionError, Pop909ClCorpusRecord, convert_pop909_cl_file
from music_critic.adapters.pop909_cl_synthetic import build_pop909_cl_smoke_midi
from music_critic.experiments.analysisgnn.corrected_model import CorrectedAnalysisGNNModel
from music_critic.experiments.pop909_cl_frozen import evaluation, transfer, transfer_run
from music_critic.experiments.pop909_cl_frozen.checkpoint import create_model_only_payload
from music_critic.experiments.pop909_cl_frozen.contracts import canonical_json, fingerprint
from music_critic.experiments.registry import load_record, seal_record, write_record_once
from music_critic.tasks import (
    CanonicalCorpusInput, CorpusCacheConfig, cache_canonical_corpus, create_split_manifest,
    dump_corpus_index, dump_split_manifest,
)


def bound(path):
    content = path.read_bytes()
    return transfer.BoundFile(path.resolve(), sha256(content).hexdigest(), len(content))


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    with torch.random.fork_rng(devices=[]):
        model = CorrectedAnalysisGNNModel().eval().requires_grad_(False)
        with torch.no_grad():
            for tensor in model.state_dict().values():
                tensor.zero_()
    path = tmp_path_factory.mktemp("transfer-checkpoint") / "synthetic.pt"
    torch.save(create_model_only_payload(model), path)
    return bound(path)


@pytest.fixture
def corpus(tmp_path, checkpoint):
    source = tmp_path / "source"
    directory = source / "POP909_processed"
    directory.mkdir(parents=True)
    accepted = []
    for song_id in ("001", "002", "003"):
        midi = build_pop909_cl_smoke_midi()
        if song_id == "003":
            for track in midi.tracks:
                for message in track:
                    if message.type in ("note_on", "note_off") and message.channel == 0:
                        message.note += 2
        path = directory / f"{song_id}.mid"
        midi.save(path)
        record = Pop909ClCorpusRecord(
            song_id=song_id, path=path, relative_path=f"POP909_processed/{song_id}.mid",
            corpus_relative_path=path.name, sha256=sha256(path.read_bytes()).hexdigest(),
            source_group_id=f"pending:{song_id}", lineage_group_id=f"pop909-lineage:{song_id}",
        )
        accepted.append(convert_pop909_cl_file(record))

    def index_for(dataset_id, inputs):
        index, _ = cache_canonical_corpus(
            inputs, cache_config=CorpusCacheConfig(tmp_path / "cache" / dataset_id),
            dataset_id=dataset_id, adapter_name="synthetic_transfer", adapter_version="1.0.0",
            adapter_config={}, source_identity=f"synthetic:{dataset_id}",
            source_fingerprint=sha256(dataset_id.encode()).hexdigest(),
        )
        path = tmp_path / f"{dataset_id}.index.json"
        dump_corpus_index(index, path)
        return index, bound(path)

    pop, pop_file = index_for("pop909_cl", [CanonicalCorpusInput(
        piece=row.piece, lineage_group_id=row.record.lineage_group_id,
        source_identity=row.record.song_id, source_relative_path=row.record.relative_path,
        source_sha256=row.record.sha256,
    ) for row in accepted])
    other_piece = replace(accepted[0].piece, dataset_name="synthetic_aux", piece_id="piece:aux-1",
                          source_group_id="aux:group", targets=(), annotations=())
    other, other_file = index_for("synthetic_aux", [CanonicalCorpusInput(
        piece=other_piece, lineage_group_id=accepted[0].record.lineage_group_id, source_identity="aux:1",
        source_relative_path="aux.json", source_sha256=sha256(b"aux").hexdigest(),
    )])
    assignments = {(row.dataset_id, row.piece_id): (
        "test" if row.source_identity == "003" else "validation"
    ) for index in (pop, other) for row in index.records}
    split = create_split_manifest((pop, other), assignments, seed=17)
    split_path = tmp_path / "global.split.json"
    dump_split_manifest(split, split_path)
    request = transfer.TransferRequest(
        "validation", (pop_file, other_file), bound(split_path), checkpoint, source,
        ("piece:pop909-cl-001", "piece:pop909-cl-002"),
    )
    return SimpleNamespace(root=tmp_path, request=request, pop=pop, other=other, split=split)


def planned(corpus, *, request=None, status="planned", binding_change=False, policy_change=None):
    request = request or corpus.request
    binding = transfer.preflight(request)
    policy = {"validation_allowed": True, "test_inputs_allowed": False, "test_targets_allowed": False,
              "test_metrics_allowed": False, "test_selection_allowed": False, "source": "synthetic test"}
    policy.update(policy_change or {})
    record = seal_record({
        "schema_version": "1.0.0", "experiment_id": "EXP-SYNTHETIC-TRANSFER", "kind": "engineering_gate",
        "phase": "WS4", "title": "Synthetic transfer test", "status": status,
        "question": "Does the explicit boundary hold?", "hypothesis": "Guarded synthetic evaluation works.",
        "decision_use": "Implementation mechanics only", "repository": {"git_commit": "a" * 40, "dirty": False},
        "seeds": [17], "budget": {"planned": 0},
        "protocol": {"pop909_transfer_binding_fingerprint": "a" * 64 if binding_change else binding["binding_fingerprint"]},
        "data_access": {"access_evidence_status": "not_applicable_planning_record" if status == "planned" else "pending_primary_evidence",
                        "validation_accessed": None, "test_inputs_accessed": None, "test_targets_accessed": None,
                        "test_metrics_computed": None, "test_used_for_selection": None, "declared_policy": policy},
        "results": {}, "validity": {"valid": None}, "claims": {"supported": [], "not_supported": ["scientific quality"]},
        "artifacts": [], "import_provenance": {"mode": "synthetic_fixture"},
    })
    path = corpus.root / "EXP-SYNTHETIC-TRANSFER.json"
    write_record_once(path, record)
    return path


def forbid_readers(monkeypatch):
    for name in ("load_model_checkpoint", "read_raw_source", "convert_pop909_cl_file"):
        monkeypatch.setattr(transfer, name, lambda *a, **kw: pytest.fail("payload reader called"))


def test_preflight_reads_only_complete_metadata_and_preserves_group_closure(corpus, monkeypatch):
    allowed = {item.path for item in corpus.request.indices} | {corpus.request.split_manifest.path}
    original = Path.open
    reads = []

    def checked(path, *args, **kwargs):
        assert path in allowed, f"unexpected source/checkpoint/cache read: {path}"
        reads.append(path)
        return original(path, *args, **kwargs)

    forbid_readers(monkeypatch)
    monkeypatch.setattr(Path, "open", checked)
    result = transfer.preflight(corpus.request)
    assert set(reads) == allowed
    assert result["selected_piece_count"] == result["available_split_piece_count"] == 2
    assert result["selected_component_count"] == 1
    assert len(result["index_fingerprints"]) == 2
    assert result["checkpoint_bytes_verified"] is result["source_bytes_verified"] is False
    with pytest.raises(ValueError, match="index_mismatch"):
        transfer.preflight(replace(corpus.request, indices=corpus.request.indices[:1]))


@pytest.mark.parametrize("split", ["test", "TEST"])
def test_test_guard_precedes_any_path_read(corpus, monkeypatch, split):
    monkeypatch.setattr(Path, "open", lambda *a, **kw: pytest.fail("TEST path opened"))
    with pytest.raises(ValueError, match="test_access_forbidden"):
        transfer.load_request(corpus.root / "missing.json", split=split)
    with pytest.raises(ValueError, match="test_access_forbidden"):
        transfer.evaluate_transfer(SimpleNamespace(split=split), experiment_record=corpus.root / "missing.json")


@pytest.mark.parametrize("pieces", [(), ("piece:pop909-cl-001", "piece:pop909-cl-001")])
def test_empty_or_duplicate_allowlist_rejected(corpus, pieces):
    with pytest.raises(ValueError, match="invalid_explicit_request"):
        replace(corpus.request, piece_ids=pieces)


@pytest.mark.parametrize("piece_id", ["piece:pop909-cl-003", "piece:pop909-cl-999"])
def test_nonmember_or_test_selection_fails_before_payload_readers(corpus, monkeypatch, piece_id):
    forbid_readers(monkeypatch)
    with pytest.raises(ValueError, match="selected_piece_not_in_requested_split"):
        transfer.evaluate_transfer(replace(corpus.request, piece_ids=(piece_id,)), experiment_record=corpus.root / "missing.json")


@pytest.mark.parametrize("field", ["index", "split_manifest"])
def test_bad_metadata_hash_rejected(corpus, monkeypatch, field):
    forbid_readers(monkeypatch)
    request = (replace(corpus.request, indices=(replace(corpus.request.indices[0], sha256="0" * 64), *corpus.request.indices[1:]))
               if field == "index" else replace(corpus.request, split_manifest=replace(corpus.request.split_manifest, sha256="0" * 64)))
    with pytest.raises(ValueError, match="artifact_binding_mismatch"):
        transfer.preflight(request)


@pytest.mark.parametrize("problem", ["binding", "running", "policy", "test_policy"])
def test_registration_binding_status_and_policy_precede_payload_readers(corpus, monkeypatch, problem):
    path = planned(corpus, binding_change=problem == "binding", status="running" if problem == "running" else "planned",
                   policy_change=({"validation_allowed": False} if problem == "policy" else
                                  {"test_inputs_allowed": True} if problem == "test_policy" else None))
    forbid_readers(monkeypatch)
    with pytest.raises(ValueError, match="matching_planned_record_required|registered_access_policy_mismatch"):
        transfer.evaluate_transfer(corpus.request, experiment_record=path)


def test_wrong_checkpoint_hash_fails_before_source(corpus, monkeypatch):
    request = replace(corpus.request, checkpoint=replace(corpus.request.checkpoint, sha256="0" * 64))
    path = planned(corpus, request=request)
    monkeypatch.setattr(transfer, "read_raw_source", lambda *a, **kw: pytest.fail("source opened"))
    with pytest.raises(ValueError, match="sha256|fingerprint|hash"):
        transfer.evaluate_transfer(request, experiment_record=path)


def test_symlink_escape_rejected_before_source_read(corpus, monkeypatch):
    path = planned(corpus)
    source = corpus.request.source_root / "POP909_processed/001.mid"
    outside = corpus.root / "outside.mid"
    source.rename(outside)
    source.symlink_to(outside)
    monkeypatch.setattr(transfer, "read_raw_source", lambda *a, **kw: pytest.fail("escaped source opened"))
    with pytest.raises(ValueError, match="source_path_escapes_root"):
        transfer.evaluate_transfer(corpus.request, experiment_record=path)


def test_changed_source_hash_rejected(corpus):
    path = planned(corpus)
    source = corpus.request.source_root / "POP909_processed/001.mid"
    source.write_bytes(source.read_bytes() + b"changed")
    with pytest.raises(Pop909ClConversionError) as caught:
        transfer.evaluate_transfer(corpus.request, experiment_record=path)
    assert caught.value.category == "pop909_frozen.source_fingerprint_mismatch"


@pytest.mark.parametrize("split", ["train", "validation"])
def test_end_to_end_candidate_first_registered_provenance_and_no_updates(corpus, monkeypatch, split):
    request = corpus.request
    if split == "train":
        manifest = create_split_manifest((corpus.pop, corpus.other), {
            (row.dataset_id, row.piece_id): "train" if row.split == "validation" else row.split
            for row in corpus.split.assignments
        }, seed=17)
        manifest_path = corpus.root / "train.split.json"
        dump_split_manifest(manifest, manifest_path)
        request = replace(request, split="train", split_manifest=bound(manifest_path))
    path = planned(corpus, request=request)
    record_before = path.read_bytes()
    trace = []
    raw_reader, target_reader = transfer.read_raw_source, transfer.convert_pop909_cl_file
    predict = CorrectedAnalysisGNNModel.predict

    def raw(record, **kwargs):
        trace.append("raw:" + record.song_id)
        piece = raw_reader(record, **kwargs)
        assert piece.targets == piece.annotations == ()
        return piece

    def prediction(model, batch):
        trace.append("predict")
        return predict(model, batch)

    def targets(record, **kwargs):
        assert trace[-3:] == ["raw:" + record.song_id, "predict", "predict"]
        trace.append("targets:" + record.song_id)
        return target_reader(record, **kwargs)

    monkeypatch.setattr(transfer, "read_raw_source", raw)
    monkeypatch.setattr(transfer, "convert_pop909_cl_file", targets)
    monkeypatch.setattr(CorrectedAnalysisGNNModel, "predict", prediction)
    monkeypatch.setattr(torch.optim.AdamW, "__init__", lambda *a, **kw: pytest.fail("optimizer constructed"))
    report = transfer.evaluate_transfer(request, experiment_record=path)
    assert trace == ["raw:001", "predict", "predict", "targets:001", "raw:002", "predict", "predict", "targets:002"]
    assert report["runtime"]["applied_updates"] == 0
    assert report["planned_record_fingerprint"] == load_record(path)["record_fingerprint"]
    assert report["provenance"]["kind"] == "explicit_indexed_pop909_cl_source"
    assert report["access_observations"]["target_reader_calls"] == 2
    assert report["access_observations"]["validation_accessed"] is (split == "validation")
    assert all(report["access_observations"][key] is False for key in report["access_observations"] if key.startswith("test_"))
    assert report["model"]["checkpoint_evidence"]["artifact_sha256"] == corpus.request.checkpoint.sha256
    assert report["views"]["native"]["coverage"]["pieces"] == 2
    assert report["views"]["native"]["coverage"]["groups"] == 1
    assert report["report_fingerprint"] == fingerprint({k: v for k, v in report.items() if k != "report_fingerprint"})
    assert path.read_bytes() == record_before
    with pytest.raises(ValueError, match="future_registered"):
        evaluation.evaluate_frozen(object(), iter(()), split="validation")


def test_source_change_after_predictions_is_rechecked_before_target_extraction(corpus, monkeypatch):
    path = planned(corpus)
    source = corpus.request.source_root / "POP909_processed/001.mid"
    predict = CorrectedAnalysisGNNModel.predict
    calls = 0

    def prediction(model, batch):
        nonlocal calls
        result = predict(model, batch)
        calls += 1
        if calls == 2:
            source.write_bytes(source.read_bytes() + b"changed-after-prediction")
        return result

    monkeypatch.setattr(CorrectedAnalysisGNNModel, "predict", prediction)
    with pytest.raises(Pop909ClConversionError) as caught:
        transfer.evaluate_transfer(corpus.request, experiment_record=path)
    assert calls == 2
    assert caught.value.category == "pop909_cl.file_fingerprint_mismatch"


def request_file(corpus):
    request = corpus.request
    value = asdict(request)
    value["schema_version"] = transfer.REQUEST_SCHEMA
    path = corpus.root / "request.json"
    path.write_text(json.dumps(value, default=str))
    return path


def test_cli_test_denied_before_request_or_output_access(tmp_path, monkeypatch):
    monkeypatch.setattr(transfer_run, "load_request", lambda *a, **kw: pytest.fail("request opened"))
    monkeypatch.setattr(Path, "exists", lambda *a: pytest.fail("output accessed"))
    with pytest.raises(ValueError, match="test_access_forbidden"):
        transfer_run.main(["--preflight", "--split", "TEST", "--request", str(tmp_path / "input"), "--output", str(tmp_path / "output")])


def test_cli_existing_output_rejected_before_request_or_evaluation(tmp_path, monkeypatch):
    output = tmp_path / "existing.json"
    output.write_text("retained")
    monkeypatch.setattr(transfer_run, "load_request", lambda *a, **kw: pytest.fail("request opened"))
    with pytest.raises(SystemExit):
        transfer_run.main(["--evaluate", "--split", "validation", "--request", "missing",
                           "--experiment-record", "missing", "--output", str(output)])
    assert output.read_text() == "retained"


@pytest.mark.parametrize("failure", [None, ValueError("synthetic preflight failure"), KeyboardInterrupt("synthetic preflight failure")])
def test_cli_fresh_success_or_failure_receipt_never_overwritten(corpus, monkeypatch, failure):
    request = request_file(corpus)
    output = corpus.root / "receipt.json"
    args = ["--preflight", "--split", "validation", "--request", str(request), "--output", str(output)]
    if failure:
        def fail(_):
            assert output.is_file()
            raise failure
        monkeypatch.setattr(transfer_run, "preflight", fail)
        with pytest.raises(type(failure), match="synthetic preflight failure"):
            transfer_run.main(args)
    else:
        assert transfer_run.main(args) == 0
    content = output.read_bytes()
    receipt = json.loads(content)
    assert receipt["schema_version"] == ("Pop909ClTransferFailure@1.0.0" if failure else transfer.PREFLIGHT_SCHEMA)
    with pytest.raises(SystemExit):
        transfer_run.main(args)
    assert output.read_bytes() == content
