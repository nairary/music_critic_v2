"""Synthetic-only full-population callback acceptance, no corpus access."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from music_critic.experiments.analysisgnn import harmonic_fusion_pilot_evaluation as pilot_eval
from music_critic.experiments.analysisgnn.contracts import fingerprint
from music_critic.experiments.analysisgnn.corrected_model import CorrectedAnalysisGNNModel, model_state_fingerprint
from music_critic.experiments.analysisgnn.corrected_training import (
    build_source_free_fixture, combine_single_record_raw_batches, load_frozen_class_weights,
)
from music_critic.experiments.analysisgnn.harmonic_fusion_resume import capture_safe_rng
from music_critic.experiments.analysisgnn.harmonic_fusion_training import evaluate_validation
from music_critic.experiments.analysisgnn.harmonic_logit_fusion import HarmonicFusionModel
from music_critic.experiments.analysisgnn.multitask_contract import PAPER_DEFINED_JOINT_COMPONENTS
from music_critic.experiments.analysisgnn.performance_replay import recursive_state_fingerprint


@pytest.fixture
def data():
    batch, sidecar = build_source_free_fixture()
    weights = load_frozen_class_weights()
    assignment = lambda rid, split, component: {"record_id": rid, "split": split, "source_component_id": component}
    assignments = {
        "val:b": assignment("val:b", "validation", "component:b"),
        "train:a": assignment("train:a", "train", "component:train"),
        "val:a": assignment("val:a", "validation", "component:a"),
        "excluded:metadata-only": assignment("excluded:metadata-only", "test", "component:excluded"),
    }
    payloads = {}
    for rid in ("val:a", "val:b"):
        current = deepcopy(sidecar)
        current["record_id"] = rid
        current["source_component_id"] = assignments[rid]["source_component_id"]
        if rid == "val:b":
            current["entities"][0]["targets"]["root"].update(available=False, masked=True, canonical_value=None)
        current["fingerprint"] = fingerprint({key: value for key, value in current.items() if key != "fingerprint"})
        payloads[rid] = current
    seen = {tuple(sidecar["entities"][0]["targets"][task]["canonical_value"]
                  for task in PAPER_DEFINED_JOINT_COMPONENTS)}
    reads = []

    def reader(record_id, *, split):
        assert split == "validation" and record_id in payloads
        reads.append(record_id)
        return batch.raw_graph_batch, deepcopy(payloads[record_id])

    return SimpleNamespace(batch=batch, assignments=assignments, weights=weights, seen=seen,
                           reads=reads, reader=reader, payloads=payloads)


@pytest.fixture
def model():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        return HarmonicFusionModel(CorrectedAnalysisGNNModel()).train()


def callback(data, **kwargs):
    args = dict(assignments=data.assignments, reader=data.reader, class_weights=data.weights,
                train_seen_tuples=data.seen, binding_fingerprint="a" * 64, synthetic=True)
    args.update(kwargs)
    return pilot_eval.make_full_validation_callback(**args)


def test_full_callback_preserves_all_existing_arithmetic_masks_components_and_identity(data, model):
    observed = callback(data)(model, split="validation")
    assert data.reads == ["val:a", "val:b"]
    expected = evaluate_validation(model, records=("val:a", "val:b"), reader=data.reader,
                                  class_weights=data.weights, train_seen_tuples=data.seen)
    population = observed.pop("full_validation_population")
    assert observed == expected
    assert len(observed["per_head"]) == len(observed["confusions"]) == 18
    assert observed["per_head"]["root"]["support"] == 1
    assert observed["per_head"]["quality"]["support"] == 2
    assert observed["per_head"]["quality"]["component_support"] == 2
    assert {row["record_id"] for row in observed["component_predictions"]} == {"val:a", "val:b"}
    assert observed["seen_tuple_support"] > 0 and observed["unseen_tuple_support"] == 0
    assert observed["prediction_consistency"]["total"] == 4
    assert population["evaluated_record_ids"] == ["val:a", "val:b"]
    assert population["evaluated_component_count"] == population["evaluated_record_count"] == 2
    assert population["complete"] is True and population["shift_pc"] == 0
    assert population["contract"]["mode"] == "synthetic_fixture"


def test_each_repeated_boundary_evaluates_entire_same_population(data, model):
    cb = callback(data)
    reports = [cb(model, split="validation") for _ in (0, 256, 512)]
    assert data.reads == ["val:a", "val:b"] * 3
    assert all(report == reports[0] for report in reports)
    assert cb.contract["tuple_scope"] == "synthetic_fixture_not_verified_train"
    assert cb.contract["native_train_tuple_source_sha256"] is None
    assert cb.contract["source_authentication"] == "caller_supplied_synthetic_fixture"
    assert cb.contract["data_binding_fingerprint"] == "a" * 64


def test_progress_finishes_records_only_after_prediction_and_preserves_rng_and_modes(data, model):
    events = []

    def progress(event):
        events.append(event)
        torch.rand(1)  # Even the final progress callback is outside model state.

    cb = callback(data, progress=progress)
    modes = [module.training for module in model.modules()]
    rng = recursive_state_fingerprint(capture_safe_rng())
    state = model_state_fingerprint(model)
    cb(model)
    assert [(event["record_id"], event["status"]) for event in events] == [
        ("val:a", "started"), ("val:a", "completed"), ("val:b", "started"), ("val:b", "completed")]
    assert [event["record_index"] for event in events] == [1, 1, 2, 2]
    assert all(event["record_count"] == 2 for event in events)
    assert [module.training for module in model.modules()] == modes
    assert model_state_fingerprint(model) == state
    assert recursive_state_fingerprint(capture_safe_rng()) == rng


def test_no_gold_targets_enter_forward(data, model, monkeypatch):
    original = model.predict
    seen = []

    def predict(raw, **kwargs):
        # Model receives the raw graph object only, with no sidecar argument.
        seen.append(raw["beat"].entity_id)
        return original(raw, **kwargs)

    monkeypatch.setattr(model, "predict", predict)
    first = callback(data)(model)
    for sidecar in data.payloads.values():
        for entity in sidecar["entities"]:
            for target in entity.get("targets", {}).values():
                target.update(available=False, masked=True, canonical_value=None)
    second = callback(data)(model)
    assert len(seen) == 4 and seen[0] == seen[1] == seen[2] == seen[3]
    assert second["prediction_consistency"] == first["prediction_consistency"]
    assert all(row["support"] == 0 for row in second["per_head"].values())
    assert second["full_validation_population"]["evaluated_record_count"] == 2


def test_callback_detaches_assignments_tuples_weights_and_public_contract(data, model):
    cb = callback(data)
    before = cb(model)
    contract = cb.contract
    contract["records"].clear()
    data.assignments.clear()
    data.seen.clear()
    for mask in data.weights.supported.values():
        mask.zero_()
    after = cb(model)
    assert after == before
    assert cb.contract["records"] == ["val:a", "val:b"]


@pytest.mark.parametrize("split", ["test", "TEST", "train"])
def test_split_guard_runs_before_any_reader(data, model, split):
    with pytest.raises(ValueError):
        callback(data)(model, split=split)
    assert data.reads == []


@pytest.mark.parametrize("mutation", ["component", "record", "batch"])
def test_reader_must_return_exact_record_component_and_one_full_piece(data, model, mutation):
    def reader(record_id, *, split):
        raw, sidecar = data.reader(record_id, split=split)
        if mutation == "component":
            sidecar["source_component_id"] = "wrong"
        elif mutation == "record":
            sidecar["record_id"] = "wrong"
        else:
            raw = combine_single_record_raw_batches([raw, raw])
        return raw, sidecar

    with pytest.raises(ValueError):
        callback(data, reader=reader)(model)
    assert data.reads == ["val:a"]


def test_exception_never_finishes_current_record_or_returns_partial_metrics(data, model, monkeypatch):
    events = []
    before_modes = [module.training for module in model.modules()]

    def predict(*args, **kwargs):
        raise RuntimeError("synthetic prediction failure")

    monkeypatch.setattr(model, "predict", predict)
    with pytest.raises(RuntimeError, match="synthetic prediction failure"):
        callback(data, progress=events.append)(model)
    assert [(row["record_id"], row["status"]) for row in events] == [("val:a", "started")]
    assert [module.training for module in model.modules()] == before_modes


def test_evaluator_cannot_silently_omit_a_population_member(data, model, monkeypatch):
    original = pilot_eval.evaluate_validation

    def truncated(model, *, records, **kwargs):
        return original(model, records=records[:-1], **kwargs)

    monkeypatch.setattr(pilot_eval, "evaluate_validation", truncated)
    with pytest.raises(ValueError, match="validation_incomplete"):
        callback(data)(model)


def test_evaluator_cannot_duplicate_or_reorder_members(data, model, monkeypatch):
    original = pilot_eval.evaluate_validation

    def reordered(model, *, records, **kwargs):
        return original(model, records=tuple(reversed(records)), **kwargs)

    monkeypatch.setattr(pilot_eval, "evaluate_validation", reordered)
    with pytest.raises(ValueError, match="complete_ordered_membership_required"):
        callback(data)(model)
    assert not data.reads


@pytest.mark.parametrize("seen", [None, set(), {("too", "short")}, {("x",) * 4 + (0,)}])
def test_missing_or_malformed_native_train_tuple_inventory_is_not_all_unseen(data, seen):
    with pytest.raises(ValueError, match="bound_native_train_tuple_inventory_required"):
        callback(data, train_seen_tuples=seen)
    assert not data.reads


@pytest.mark.parametrize("seen", [None, set()])
def test_verified_source_hash_does_not_turn_unknown_inventory_into_all_unseen(data, seen):
    with pytest.raises(ValueError, match="bound_native_train_tuple_inventory_required"):
        callback(data, synthetic=False, train_seen_tuples=seen,
                 native_train_tuple_source_sha256=pilot_eval.FROZEN_JOINT_TUPLE_SHA256)
    assert not data.reads


@pytest.mark.parametrize("source", [None, "", "a" * 64, "unknown", 123])
def test_production_rejects_missing_or_wrong_native_train_source_before_payload_io(data, source):
    with pytest.raises(ValueError, match="verified_native_train_tuple_source_required"):
        callback(data, synthetic=False, native_train_tuple_source_sha256=source)
    assert not data.reads


def test_synthetic_source_hash_cannot_claim_verified_native_train_provenance(data):
    cb = callback(data, native_train_tuple_source_sha256=pilot_eval.FROZEN_JOINT_TUPLE_SHA256)
    assert cb.contract["tuple_scope"] == "synthetic_fixture_not_verified_train"
    assert cb.contract["source_authentication"] == "caller_supplied_synthetic_fixture"
    assert cb.contract["native_train_tuple_source_sha256"] == pilot_eval.FROZEN_JOINT_TUPLE_SHA256


@pytest.mark.parametrize("mutation", ["identity", "cross_split", "empty"])
def test_population_metadata_rejects_identity_leakage_and_empty_validation(data, mutation):
    if mutation == "identity":
        data.assignments["val:a"]["record_id"] = "wrong"
    elif mutation == "cross_split":
        data.assignments["val:a"]["source_component_id"] = "component:train"
    else:
        data.assignments = {key: value for key, value in data.assignments.items() if value["split"] != "validation"}
    with pytest.raises(ValueError):
        callback(data)


def test_production_requires_full_counts_but_does_no_payload_io(data):
    with pytest.raises(ValueError, match="complete_frozen_population_required"):
        callback(data, synthetic=False,
                 native_train_tuple_source_sha256=pilot_eval.FROZEN_JOINT_TUPLE_SHA256)
    assert not data.reads
    assignments = {}
    for split, records, components in (("train", 1295, 1209), ("validation", 162, 147), ("test", 162, 151)):
        for index in range(records):
            rid = f"metadata:{split}:{index:04d}"
            assignments[rid] = {"record_id": rid, "split": split,
                                "source_component_id": f"component:{split}:{index % components:04d}"}
    cb = callback(data, assignments=assignments, synthetic=False,
                  native_train_tuple_source_sha256=pilot_eval.FROZEN_JOINT_TUPLE_SHA256)
    assert cb.contract["record_count"] == 162 and cb.contract["component_count"] == 147
    assert cb.contract["mode"] == "frozen_dilemmadata_validation"
    assert cb.contract["native_train_tuple_source_sha256"] == pilot_eval.FROZEN_JOINT_TUPLE_SHA256
    assert cb.contract["tuple_scope"] == "verified_native_train_not_augmented_exposure"
    assert not data.reads  # Only synthetic assignment metadata was constructed.


@pytest.mark.parametrize("kwargs", [{"binding_fingerprint": "invalid"}, {"synthetic": 1}])
def test_binding_and_mode_are_explicit(data, kwargs):
    with pytest.raises(ValueError, match="binding_or_mode_invalid"):
        callback(data, **kwargs)
