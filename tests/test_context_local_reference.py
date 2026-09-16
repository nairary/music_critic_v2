"""Tiny generated-checkpoint/fixture mechanics only; no historical weights or data."""
from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace

import pytest
import torch

from music_critic.data import RationalTime
from music_critic.experiments.analysisgnn.corrected_model import (
    CorrectedAnalysisGNNModel, model_state_fingerprint,
)
from music_critic.experiments.context_diagnostics import FileBinding, raw_candidates
from music_critic.experiments.context_diagnostics_pairs import build_context_pair_suite
from music_critic.experiments import context_local_reference as module
from music_critic.experiments.context_local_reference import (
    ALLOWED_TASKS, FrozenC2Reference, TASK_LEVELS, build_reference_view, ordered_classes,
)
from music_critic.experiments.pop909_cl_frozen.checkpoint import create_model_only_payload
from music_critic.graph import MANDATORY_NODE_TYPES, model_input_fingerprint
from music_critic.models.heads import TaskPrediction
from music_critic.models.contracts import TASK_PREDICTION_CONTRACT_VERSION
from music_critic.tasks import ENTITY_NODE_TYPE_TO_CODE


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        payload = create_model_only_payload(CorrectedAnalysisGNNModel().eval())
    path = tmp_path_factory.mktemp("context-reference") / "random-fixture.pt"
    torch.save(payload, path)
    raw = path.read_bytes()
    return (FileBinding(str(path), sha256(raw).hexdigest(), len(raw)),
            payload["model_contract_fingerprint"], payload["model_state_fingerprint"])


@pytest.fixture
def piece():
    return replace(build_context_pair_suite()[0].left, split="validation")


@pytest.fixture
def reference(checkpoint):
    return FrozenC2Reference(*checkpoint)


def test_real_tiny_checkpoint_native_forward_all_requested_heads(reference, piece):
    before = model_state_fingerprint(reference._model)
    original_rng = torch.get_rng_state().clone()
    result = reference.predict(piece, ALLOWED_TASKS)
    assert result.output is not None
    assert len(result.output.predictions) == 18
    assert result.input_fingerprint == model_input_fingerprint(result.graph)
    assert result.evidence["checkpoint"]["model_state_fingerprint"] == before
    assert result.evidence["view"] == "native_full_raw"
    assert reference.forwards_started == reference.forwards_completed == 1
    assert reference.forwards_unknown == 0
    assert torch.equal(torch.get_rng_state(), original_rng)
    assert model_state_fingerprint(reference._model) == before
    assert not reference._model.training
    assert not any(parameter.requires_grad for parameter in reference._model.parameters())
    for task, level in TASK_LEVELS.items():
        expected = {candidate.candidate_id for candidate in raw_candidates(piece, level)}
        assert set(result.predictions[task]) == set(result.logits[task]) == expected
        for identity, logits in result.logits[task].items():
            assert len(logits) == len(ordered_classes((task,))[task])
            assert torch.isfinite(torch.tensor(logits)).all()
            assert result.predictions[task][identity] == max(range(len(logits)), key=logits.__getitem__)


def test_raw_train_view_never_loads_model_and_strips_target_payload(piece, monkeypatch):
    monkeypatch.setattr(module, "load_model_checkpoint", lambda *a, **kw: pytest.fail("no checkpoint read"))
    untrusted_targets = replace(piece, split="train", annotations=(object(),), targets=(object(),))
    view = build_reference_view(untrusted_targets, ALLOWED_TASKS)
    plain = build_reference_view(replace(piece, split="train"), ALLOWED_TASKS)
    assert view.input_fingerprint == plain.input_fingerprint
    assert view.output is None and view.predictions == view.logits == {}
    assert view.evidence["annotations_or_targets_used"] is False
    assert untrusted_targets.annotations and untrusted_targets.targets


@pytest.mark.parametrize("split", ["train", "test", "TEST", None, "VALIDATION"])
def test_forward_split_guard_precedes_graph_and_model(reference, piece, monkeypatch, split):
    monkeypatch.setattr(module, "build_reference_view", lambda *a: pytest.fail("split must precede graph"))
    monkeypatch.setattr(reference._model, "predict", lambda *a: pytest.fail("no forbidden forward"))
    with pytest.raises(ValueError, match="VALIDATION_only"):
        reference.predict(replace(piece, split=split), ALLOWED_TASKS)
    assert reference.forwards_started == reference.forwards_completed == 0


@pytest.mark.parametrize("split", ["test", "TEST", None])
def test_raw_view_keeps_test_closed(piece, split):
    with pytest.raises(ValueError, match="TEST_closed"):
        build_reference_view(replace(piece, split=split), ALLOWED_TASKS)


@pytest.mark.parametrize("tasks", [(), ("quality", "quality"), ("harmonic_rhythm",),
                                   ("staff",), ["quality"]])
def test_task_scope_and_fixed_classes(tasks):
    with pytest.raises(ValueError, match="fixed_unique_tasks"):
        ordered_classes(tasks)
    assert len(ordered_classes(("quality",))["quality"]) == 17


def test_checkpoint_sha_is_checked(checkpoint):
    binding, contract, state = checkpoint
    with pytest.raises(ValueError, match="sha256_mismatch"):
        FrozenC2Reference(replace(binding, sha256="0" * 64), contract, state)


@pytest.mark.parametrize("which", ["contract", "state"])
def test_expected_model_binding_is_independently_required(checkpoint, which):
    binding, contract, state = checkpoint
    with pytest.raises(ValueError, match="expected_model_binding_mismatch"):
        FrozenC2Reference(binding, "0" * 64 if which == "contract" else contract,
                          "0" * 64 if which == "state" else state)


def test_invalid_device_and_oversized_checkpoint_precede_read(checkpoint, monkeypatch):
    monkeypatch.setattr(module, "load_model_checkpoint", lambda *a, **kw: pytest.fail("must not read"))
    binding, contract, state = checkpoint
    with pytest.raises(ValueError, match="explicit_cpu_or_cuda0"):
        FrozenC2Reference(binding, contract, state, device="cuda:1")
    with pytest.raises(ValueError, match="checkpoint_budget"):
        FrozenC2Reference(replace(binding, size_bytes=64 * 1024 * 1024 + 1), contract, state)


def _fixture_output(view, tasks):
    predictions = []
    for task in tasks:
        level = TASK_LEVELS[task]
        count = view.graph[level].num_nodes
        code = ENTITY_NODE_TYPE_TO_CODE[level]
        counts = torch.zeros(len(MANDATORY_NODE_TYPES), dtype=torch.long)
        offsets = torch.full_like(counts, -1)
        counts[code], offsets[code] = count, 0
        width = len(ordered_classes((task,))[task])
        predictions.append(TaskPrediction(
            contract_version=TASK_PREDICTION_CONTRACT_VERSION, task_id=task,
            source_adapter="dilemmadata.analysisgnn.b3-sidecar", allowed_node_types=(level,),
            candidate_node_type_codes=torch.full((count,), code, dtype=torch.long),
            global_entity_indices=torch.arange(count), sample_indices=torch.zeros(count, dtype=torch.long),
            candidate_offsets_by_node_type=offsets, candidate_counts_by_node_type=counts,
            logits=torch.arange(count * width, dtype=torch.float32).reshape(count, width),
        ))
    return SimpleNamespace(predictions=tuple(predictions))


def test_endpoint_graph_candidates_are_explicitly_excluded_from_local_maps(piece):
    endpoint_note = replace(piece.notes[0], note_id="note:endpoint-fixture", onset_qn=piece.duration_qn,
                            duration_qn=RationalTime(0), is_grace=True)
    piece = replace(piece, notes=piece.notes + (endpoint_note,))
    tasks = ("cadence", "metrical_strength")
    view = build_reference_view(piece, tasks)
    predictions, logits = module._prediction_maps(view, _fixture_output(view, tasks), tasks)
    onset_id = f"onset:{piece.duration_qn.num}_{piece.duration_qn.den}"
    assert endpoint_note.note_id in view.graph["note"].entity_id
    assert onset_id in view.graph["onset"].entity_id
    assert view.evidence["excluded_endpoint_candidates"] == {"onset": [onset_id], "note": [endpoint_note.note_id]}
    assert endpoint_note.note_id not in predictions["metrical_strength"]
    assert onset_id not in logits["cadence"]


@pytest.mark.parametrize("corruption", ["duplicate", "missing", "width", "sample", "node_type", "offset", "nan"])
def test_prediction_candidate_and_class_contracts_fail_closed(piece, corruption):
    tasks = ("quality",)
    view = build_reference_view(piece, tasks)
    prediction = _fixture_output(view, tasks).predictions[0]
    if corruption == "duplicate":
        indices = prediction.global_entity_indices.clone()
        indices[1] = indices[0]
        prediction = replace(prediction, global_entity_indices=indices)
    elif corruption == "missing":
        with pytest.raises(ValueError, match="missing_prediction_task"):
            module._prediction_maps(view, SimpleNamespace(predictions=()), tasks)
        return
    elif corruption == "width":
        prediction = replace(prediction, logits=prediction.logits[:, :-1])
    elif corruption == "sample":
        prediction = replace(prediction, sample_indices=torch.ones_like(prediction.sample_indices))
    elif corruption == "node_type":
        prediction = replace(prediction, allowed_node_types=("note",))
    elif corruption == "offset":
        prediction = replace(prediction, candidate_offsets_by_node_type=torch.zeros_like(prediction.candidate_offsets_by_node_type))
    else:
        bad = prediction.logits.clone()
        bad[0, 0] = float("nan")
        prediction = replace(prediction, logits=bad)
    with pytest.raises(ValueError, match="candidate_or_class_mismatch|nonfinite_logits"):
        module._prediction_maps(view, SimpleNamespace(predictions=(prediction,)), tasks)


def test_global_index_permutation_maps_by_identity_not_row_order(piece):
    tasks = ("quality",)
    view = build_reference_view(piece, tasks)
    output = _fixture_output(view, tasks)
    expected = module._prediction_maps(view, output, tasks)
    prediction = output.predictions[0]
    reverse = torch.arange(len(prediction.logits)-1, -1, -1)
    reordered = replace(prediction, logits=prediction.logits[reverse],
                        global_entity_indices=prediction.global_entity_indices[reverse])
    assert module._prediction_maps(view, SimpleNamespace(predictions=(reordered,)), tasks) == expected


def test_exact_32_native_forward_budget_and_duplicate_piece_guard(reference, piece, monkeypatch):
    output = _fixture_output(build_reference_view(piece, ALLOWED_TASKS), ALLOWED_TASKS)
    calls = []
    monkeypatch.setattr(reference._model, "predict", lambda batch: calls.append(batch) or output)
    for index in range(32):
        reference.predict(replace(piece, piece_id=f"piece:fixture-{index}"), ALLOWED_TASKS)
    assert len(calls) == reference.forwards_completed == reference.forwards_started == 32
    assert reference.forwards_unknown == 0
    with pytest.raises(ValueError, match="one_forward_per_piece"):
        reference.predict(replace(piece, piece_id="piece:fixture-0"), ALLOWED_TASKS)
    with pytest.raises(ValueError, match="forward_budget"):
        reference.predict(replace(piece, piece_id="piece:fixture-32"), ALLOWED_TASKS)
    assert len(calls) == 32


def test_callback_order_partial_failure_and_terminal_instance(checkpoint, piece, monkeypatch):
    events = []
    reference = FrozenC2Reference(*checkpoint, progress=lambda event, detail: events.append((event, detail)))
    output = _fixture_output(build_reference_view(piece, ALLOWED_TASKS), ALLOWED_TASKS)
    def predict(batch):
        assert events[-1][0] == "forward_started"
        assert events[-1][1]["forwards_started"] == reference.forwards_started
        if reference.forwards_started == 4:
            raise RuntimeError("injected fourth forward")
        return output
    monkeypatch.setattr(reference._model, "predict", predict)
    for index in range(3):
        reference.predict(replace(piece, piece_id=f"piece:fixture-{index}"), ALLOWED_TASKS)
    with pytest.raises(RuntimeError, match="fourth forward"):
        reference.predict(replace(piece, piece_id="piece:fixture-fail"), ALLOWED_TASKS)
    assert (reference.forwards_started, reference.forwards_completed, reference.forwards_unknown) == (4, 3, 1)
    assert events[-1][0] == "reference_failed"
    assert events[-1][1]["stage"] == reference.failure_stage == "predict"
    assert events[-1][1]["applied_updates"] == 0
    with pytest.raises(ValueError, match="failed_instance"):
        reference.predict(replace(piece, piece_id="piece:fixture-next"), ALLOWED_TASKS)


@pytest.mark.parametrize("sync_fails", [False, True])
def test_cuda_completion_waits_for_sync_without_using_cuda(reference, piece, monkeypatch, sync_fails):
    # Device transitions and the forward are CPU fixture substitutes; no CUDA call.
    output = _fixture_output(build_reference_view(piece, ALLOWED_TASKS), ALLOWED_TASKS)
    reference.device = "cuda:0"
    monkeypatch.setattr(module, "move_raw_graph_batch", lambda batch, device: batch)
    order = []
    reference.progress = lambda event, detail: order.append(event)
    monkeypatch.setattr(reference._model, "predict", lambda batch: order.append("predict") or output)
    def synchronize(index):
        assert index == 0 and reference.forwards_completed == 0
        order.append("synchronize")
        if sync_fails:
            raise RuntimeError("injected CUDA sync failure")
    monkeypatch.setattr(module.torch.cuda, "synchronize", synchronize)
    if sync_fails:
        with pytest.raises(RuntimeError, match="sync failure"):
            reference.predict(piece, ALLOWED_TASKS)
        assert (reference.forwards_started, reference.forwards_completed, reference.forwards_unknown) == (1, 0, 1)
        assert reference.failure_stage == "cuda_synchronize"
        assert order == ["forward_started", "predict", "synchronize", "reference_failed"]
    else:
        reference.predict(piece, ALLOWED_TASKS)
        assert (reference.forwards_started, reference.forwards_completed, reference.forwards_unknown) == (1, 1, 0)
        assert order == ["forward_started", "predict", "synchronize", "forward_completed"]


def test_all_graph_levels_are_bounded_before_graph_build(piece, monkeypatch):
    monkeypatch.setattr(module, "build_raw_graph", lambda *args: pytest.fail("budget precedes graph"))
    with pytest.raises(ValueError, match="note_budget"):
        build_reference_view(replace(piece, notes=(piece.notes[0],) * (module.MAX_NOTES + 1)), ("quality",))
    with pytest.raises(ValueError, match="candidate_budget"):
        build_reference_view(replace(piece, beats=(piece.beats[0],) * (module.MAX_CANDIDATES_PER_LEVEL + 1)), ("metrical_strength",))


def test_post_forward_mapping_failure_preserves_completed_count(reference, piece, monkeypatch):
    monkeypatch.setattr(reference._model, "predict", lambda batch: SimpleNamespace(predictions=()))
    with pytest.raises(ValueError, match="missing_prediction_task"):
        reference.predict(piece, ALLOWED_TASKS)
    assert (reference.forwards_started, reference.forwards_completed, reference.forwards_unknown) == (1, 1, 0)
    assert reference.failure_stage == "prediction_mapping"


def test_callback_failure_never_enters_forward_and_is_not_masked(checkpoint, piece, monkeypatch):
    def fail(event, detail):
        raise RuntimeError("injected journal write")
    reference = FrozenC2Reference(*checkpoint, progress=fail)
    monkeypatch.setattr(reference._model, "predict", lambda batch: pytest.fail("journal must precede predict"))
    with pytest.raises(RuntimeError, match="journal write"):
        reference.predict(piece, ALLOWED_TASKS)
    assert (reference.forwards_started, reference.forwards_completed, reference.forwards_unknown) == (1, 0, 1)


def test_frozen_model_check_rejects_training_mode_before_forward(reference, piece):
    reference._model.train()
    with pytest.raises(ValueError, match="frozen_eval"):
        reference.predict(piece, ALLOWED_TASKS)
    assert reference.forwards_started == reference.forwards_completed == 0
