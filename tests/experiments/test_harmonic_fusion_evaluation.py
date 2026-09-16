"""Synthetic CPU acceptance for the new binding, never musical evidence."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from music_critic.data import RationalTime
from music_critic.experiments.analysisgnn import harmonic_fusion_evaluation as fusion_eval
from music_critic.experiments.analysisgnn.corrected_model import (
    CorrectedAnalysisGNNModel, corrected_task_head_specs, model_state_fingerprint,
)
from music_critic.experiments.analysisgnn.harmonic_fusion_checkpoint import (
    FUSION_CHECKPOINT_SCHEMA, create_model_payload, load_fusion_checkpoint,
)
from music_critic.experiments.analysisgnn.harmonic_logit_fusion import HarmonicFusionModel
from music_critic.experiments.pop909_cl_frozen import transfer
from music_critic.experiments.pop909_cl_frozen.checkpoint import load_model_checkpoint
from music_critic.experiments.pop909_cl_frozen.contracts import canonical_json, evaluation_protocol
from music_critic.experiments.pop909_cl_frozen.evaluation import SourceTargets, evaluate_frozen
from music_critic.experiments.pop909_cl_frozen.projection import (
    quality_class_labels, quality_gold_class_id, quality_subset_ids, root_class_labels,
)
from music_critic.experiments.pop909_cl_frozen.run import synthetic_input
from music_critic.experiments.registry import load_record, seal_record, write_record_once
from tests.experiments.test_pop909_cl_frozen_evaluation import event
from tests.experiments.test_pop909_cl_transfer import bound, checkpoint, corpus, planned


@pytest.fixture
def model():
    with torch.random.fork_rng(devices=[]):
        base = CorrectedAnalysisGNNModel()
        with torch.no_grad():
            for value in base.state_dict().values():
                value.zero_()
            for index, spec in enumerate(corrected_task_head_specs()):
                bias = base.task_heads.heads[f"task_{index:02d}"][-1].bias
                if spec.task_id in {"root", "quality"}:
                    bias.fill_(-20)
                    bias[root_class_labels().index("C") if spec.task_id == "root"
                         else quality_gold_class_id("M")] = 20
                elif spec.task_id == "local_key":
                    bias[0] = 1
        return HarmonicFusionModel(base).eval().requires_grad_(False)


@pytest.fixture
def item(tmp_path):
    return synthetic_input(tmp_path)


def connect(model, task, class_id):
    fusion = model.fusion
    others = [name for name in fusion.harmonic_task_ids if name != task]
    offset = sum(fusion.class_counts[name] for name in others[:others.index("local_key")])
    with torch.no_grad():
        fusion.weights[task][class_id, offset] = 100


def evaluate(model, inputs, **kwargs):
    return fusion_eval.evaluate_fusion_synthetic(model, inputs, split="synthetic", **kwargs)


def save_fusion(model, path):
    torch.save(create_model_payload(model), path)
    artifact = bound(path)
    return artifact, load_fusion_checkpoint(path, expected_sha256=artifact.sha256,
                                           expected_size_bytes=artifact.size_bytes)


def test_zero_fusion_uses_same_complete_views_arithmetic_and_new_model_identity(model, item):
    base = evaluate_frozen(model.base, [item], split="synthetic")
    fused = evaluate(model, [item])
    assert canonical_json(base["views"]) == canonical_json(fused["views"])
    assert base["pieces"] == fused["pieces"]
    assert fused["model"]["contract"] != base["model"]["contract"]
    assert fused["protocol"]["frozen_metric_protocol"] == evaluation_protocol()
    assert fused["schema_version"] == fusion_eval.SYNTHETIC_REPORT_SCHEMA
    assert fused["views"]["native"]["role"] == "primary"
    assert fused["views"]["metadata_removed"]["role"] == "secondary"
    assert fused["access_observations"]["real_data_execution_enabled"] is False


def test_actual_nonzero_fusion_changes_predictions_metrics_and_checkpoint_roundtrip(model, item, tmp_path):
    inputs = [replace(item, read_targets=lambda: SourceTargets((event(),)))]
    original = evaluate(model, inputs)
    base_state = model_state_fingerprint(model.base)
    outside = next(i for i in range(len(quality_class_labels())) if i not in quality_subset_ids())
    connect(model, "root", root_class_labels().index("D"))
    connect(model, "quality", outside)
    artifact, loaded = save_fusion(model, tmp_path / "fusion.pt")
    changed = evaluate(loaded.model, inputs, checkpoint_evidence=loaded.evidence)
    assert model_state_fingerprint(model.base) == base_state
    assert changed["model"]["checkpoint_evidence"]["artifact_sha256"] == artifact.sha256
    for view in ("native", "metadata_removed"):
        before, after = original["views"][view], changed["views"][view]
        assert before["tasks"]["root"]["accuracy"]["value"] == 1
        assert before["tasks"]["quality"]["accuracy"]["value"] == 1
        assert after["tasks"]["root"]["accuracy"]["value"] == 0
        assert after["tasks"]["root"]["confusion"][0][2] == 1
        assert after["tasks"]["quality"]["confusion"][quality_gold_class_id("M")][outside] == 1
        assert after["tasks"]["quality"]["outside_subset_predictions"] == 1
        assert after["root_quality_joint_accuracy"]["value"] == 0
        assert before["input_bindings"][0]["model_input_fingerprint"] == after["input_bindings"][0]["model_input_fingerprint"]
        assert before["input_bindings"][0]["logits_fingerprints"] != after["input_bindings"][0]["logits_fingerprints"]
    with pytest.raises(ValueError, match="schema_invalid"):
        load_model_checkpoint(artifact.path, expected_sha256=artifact.sha256, expected_size_bytes=artifact.size_bytes)
    assert evaluate_frozen(model.base, inputs, split="synthetic")["views"] == original["views"]


def test_masks_conflicts_unsupported_quality_exact_alignment_and_empty_denominators(model, item):
    rows = (
        event("joint", RationalTime(1, 3)), event("conflict", RationalTime(2, 3), 2, "m"),
        event("root-only", RationalTime(1), quality=None), event("quality-only", RationalTime(2), root=None),
        event("unsupported", RationalTime(3), quality="sus2"), event("unaligned", RationalTime(4)),
    )
    inputs = [replace(item, read_targets=lambda: SourceTargets(rows, no_chord_span_count=2,
                                                               trailing_unannotated_span_count=1))]
    result = evaluate(model, inputs)
    # Equality to the old arithmetic covers every per-class, exclusion and coverage field.
    assert result["views"] == evaluate_frozen(model.base, inputs, split="synthetic")["views"]
    for view in result["views"].values():
        assert view["tasks"]["root"]["eligible_events"] == 4
        assert view["tasks"]["quality"]["eligible_events"] == 3
        assert view["tasks"]["quality"]["coverage"]["excluded_events"]["unsupported"] == 1
        assert view["root_quality_joint_accuracy"]["eligible_events"] == 2
        assert view["resolution"]["multiple_events_same_beat"] == 1
        assert view["resolution"]["conflicting_label_beats"] == {"root": 1, "quality": 1}
        assert view["coverage"]["alignment_failures"] == {"missing_containing_beat": 1}
    for inputs in ([], [replace(item, read_targets=lambda: SourceTargets(source_status="expected_missing"))]):
        assert evaluate(model, inputs)["views"] == evaluate_frozen(model.base, inputs, split="synthetic")["views"]


def test_targets_open_after_both_predictions_and_do_not_change_input_or_logits(model, item, monkeypatch):
    trace = []
    original = model.predict

    def predict(batch):
        trace.append("predict")
        return original(batch)

    monkeypatch.setattr(model, "predict", predict)

    def targets():
        assert trace == ["predict", "predict"]
        trace.append("targets")
        return SourceTargets((event(),))

    first = evaluate(model, [replace(item, read_targets=targets)])
    second = evaluate(model, [replace(item, read_targets=lambda: SourceTargets((event(root=2, quality="m"),)))])
    for view in ("native", "metadata_removed"):
        assert first["views"][view]["input_bindings"] == second["views"][view]["input_bindings"]
    assert first["pieces"][0]["target_fingerprint"] != second["pieces"][0]["target_fingerprint"]


def test_aligned_candidate_permutation_preserves_per_beat_results(model, item, monkeypatch):
    original = model.base.predict
    permute = False

    def predict(batch, **kwargs):
        result = original(batch, **kwargs)
        rows = []
        for row in result.predictions:
            if row.task_id == "root":
                logits = torch.full_like(row.logits, -20)
                for index, label in enumerate(("C", "D", "E", "F")):
                    logits[index, root_class_labels().index(label)] = 20
                row = replace(row, logits=logits)
            if permute and row.allowed_node_types == ("beat",):
                row = replace(row, **{field: getattr(row, field).flip(0) for field in (
                    "logits", "sample_indices", "global_entity_indices", "candidate_node_type_codes")})
            rows.append(row)
        return replace(result, predictions=tuple(rows))

    monkeypatch.setattr(model.base, "predict", predict)
    events = tuple(event(f"event-{index}", RationalTime(index), root)
                   for index, root in enumerate((0, 2, 4, 5)))
    inputs = [replace(item, read_targets=lambda: SourceTargets(events))]
    first = evaluate(model, inputs)
    permute = True
    second = evaluate(model, inputs)
    assert first == second
    assert first["views"]["native"]["tasks"]["root"]["accuracy"]["value"] == 1


def test_cross_head_candidate_mismatch_is_rejected_before_targets(model, item, monkeypatch):
    original = model.base.predict

    def predict(batch, **kwargs):
        output = original(batch, **kwargs)
        return replace(output, predictions=tuple(
            replace(row, global_entity_indices=row.global_entity_indices.flip(0)) if row.task_id == "quality"
            else row for row in output.predictions))

    monkeypatch.setattr(model.base, "predict", predict)
    with pytest.raises(ValueError, match="harmonic_candidate_alignment_mismatch"):
        evaluate(model, [replace(item, read_targets=lambda: pytest.fail("targets read"))])


@pytest.mark.parametrize("split", ["test", "TEST", "train", "validation"])
def test_synthetic_split_denied_before_any_reader(split):
    with pytest.raises(ValueError):
        fusion_eval.evaluate_fusion_from_readers(split=split,
            checkpoint_reader=lambda: pytest.fail("checkpoint read"), source_reader=lambda: pytest.fail("source read"))


@pytest.mark.parametrize("field,value", [
    ("artifact_sha256", "x" * 64), ("artifact_size_bytes", True),
    ("schema_version", "old"), ("checkpoint_schema", "old"),
    ("model_state_fingerprint", "0" * 64), ("model_contract", {}),
    ("parameters_require_grad", True), ("optimizer_restored", True),
])
def test_forged_checkpoint_binding_rejected_before_source_reader(model, tmp_path, field, value):
    _, loaded = save_fusion(model, tmp_path / "fusion.pt")
    evidence = deepcopy(loaded.evidence)
    evidence[field] = value
    with pytest.raises(ValueError, match="checkpoint_evidence_binding_mismatch"):
        fusion_eval.evaluate_fusion_from_readers(split="synthetic",
            checkpoint_reader=lambda: replace(loaded, evidence=evidence),
            source_reader=lambda: pytest.fail("source read"))


def test_model_mutation_via_target_reader_is_rejected(model, item):
    def targets():
        connect(model, "root", 0)
        return SourceTargets()

    with pytest.raises(ValueError, match="model_mutated_during_evaluation"):
        evaluate(model, [replace(item, read_targets=targets)])


def test_model_mode_mutation_via_target_reader_is_rejected(model, item):
    def targets():
        model.train()
        return SourceTargets()

    with pytest.raises(ValueError, match="model_mutated_during_evaluation"):
        evaluate(model, [replace(item, read_targets=targets)])


def test_reader_requires_checkpoint_evidence_before_source_callback(model):
    with pytest.raises(ValueError, match="checked_checkpoint_required"):
        fusion_eval.evaluate_fusion_from_readers(split="synthetic",
            checkpoint_reader=lambda: SimpleNamespace(model=model, evidence=None),
            source_reader=lambda: pytest.fail("source read"))


@pytest.mark.parametrize("kind", ["base", "disabled", "train"])
def test_wrong_model_contract_or_mode_rejected_before_iterator(model, kind):
    invalid = (model.base if kind == "base" else HarmonicFusionModel(model.base, enabled=False)
               if kind == "disabled" else model.train())

    def inputs():
        pytest.fail("source iterated")
        yield

    with pytest.raises(ValueError, match="fusion_model_required|cpu_eval_model_required"):
        evaluate(invalid, inputs())


def registered_fusion(corpus, request):
    baseline = load_record(planned(corpus, request=request))
    binding = fusion_eval.fusion_transfer_preflight(request)
    baseline["experiment_id"] = "EXP-SYNTHETIC-FUSION"
    baseline["protocol"]["pop909_transfer_binding_fingerprint"] = binding["binding_fingerprint"]
    path = corpus.root / "EXP-SYNTHETIC-FUSION.json"
    write_record_once(path, seal_record(baseline))
    return path


def test_transfer_reuses_bound_metadata_and_requires_distinct_registered_protocol(corpus, monkeypatch):
    allowed = {item.path for item in corpus.request.indices} | {corpus.request.split_manifest.path}
    original = Path.open

    def checked(path, *args, **kwargs):
        assert path in allowed
        return original(path, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(Path, "open", checked)
        binding = fusion_eval.fusion_transfer_preflight(corpus.request)
    assert binding["schema_version"] == fusion_eval.TRANSFER_PREFLIGHT_SCHEMA
    old_binding = transfer.preflight(corpus.request)
    assert binding["selected_records"] == old_binding["selected_records"]
    assert binding["binding_fingerprint"] != old_binding["binding_fingerprint"]
    path = planned(corpus)
    monkeypatch.setattr(fusion_eval, "load_fusion_checkpoint", lambda *a, **k: pytest.fail("checkpoint read"))
    with pytest.raises(ValueError, match="matching_planned_record_required"):
        fusion_eval.evaluate_fusion_transfer(corpus.request, experiment_record=path)


def test_registered_transfer_uses_actual_fusion_loader_and_shared_source_join(corpus, model, tmp_path):
    connect(model, "root", root_class_labels().index("D"))
    artifact, _ = save_fusion(model, tmp_path / "fusion.pt")
    request = replace(corpus.request, checkpoint=artifact)
    record = registered_fusion(corpus, request)
    result = fusion_eval.evaluate_fusion_transfer(request, experiment_record=record)
    assert result["schema_version"] == fusion_eval.TRANSFER_REPORT_SCHEMA
    assert result["model"]["checkpoint_evidence"]["checkpoint_schema"] == FUSION_CHECKPOINT_SCHEMA
    assert result["model"]["checkpoint_evidence"]["artifact_sha256"] == artifact.sha256
    assert result["provenance"]["independent_zero_shot_evidence"] is False
    assert result["runtime"]["applied_updates"] == 0
    assert result["access_observations"]["source_piece_count"] == 2
    assert result["access_observations"]["test_inputs_accessed"] is False
    for view in result["views"].values():
        assert all(row["root"]["prediction"] == 2 for row in view["event_observations"])


def test_transfer_rejects_baseline_checkpoint_before_source_reads(corpus, monkeypatch):
    record = registered_fusion(corpus, corpus.request)
    monkeypatch.setattr(transfer, "_inputs", lambda *a, **k: pytest.fail("source read"))
    with pytest.raises(ValueError, match="schema_invalid"):
        fusion_eval.evaluate_fusion_transfer(corpus.request, experiment_record=record)


@pytest.mark.parametrize("split", ["test", "TEST"])
def test_transfer_test_denied_before_request_metadata_or_checkpoint(split, monkeypatch):
    monkeypatch.setattr(Path, "open", lambda *a, **k: pytest.fail("file read"))
    with pytest.raises(ValueError, match="test_access_forbidden"):
        fusion_eval.evaluate_fusion_transfer(SimpleNamespace(split=split), experiment_record=Path("absent"))
