"""Compatibility and access boundaries for the private WS4 evaluation core."""

from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace

import pytest
import torch

from music_critic.experiments.analysisgnn.corrected_model import CorrectedAnalysisGNNModel
from music_critic.experiments.pop909_cl_frozen import evaluation, run
from music_critic.experiments.pop909_cl_frozen.checkpoint import CHECKPOINT_EVIDENCE_SCHEMA
from music_critic.experiments.pop909_cl_frozen.contracts import canonical_json


@pytest.fixture
def model():
    # A zero state keeps the pre-refactor byte fixture independent of random
    # initialization and floating-point differences in nonzero model logits.
    with torch.random.fork_rng(devices=[]):
        result = CorrectedAnalysisGNNModel().eval().requires_grad_(False)
        with torch.no_grad():
            for tensor in result.state_dict().values():
                tensor.zero_()
    return result


@pytest.fixture
def item(tmp_path):
    return run.synthetic_input(tmp_path)


def forbidden_inputs():
    pytest.fail("source iterator accessed before guard")
    yield


def test_v1_payload_and_hash_match_observed_pre_refactor_fixture(model, item):
    report = evaluation.evaluate_frozen(model, [item], split="synthetic")
    encoded = canonical_json(report).encode()
    # Captured by executing evaluate_frozen before extracting _evaluate_core.
    assert len(encoded) == 36333
    assert sha256(encoded).hexdigest() == "0f7ff77ccf7e2a714b247da78139c9dd2ff8c8136e92581cad4203f5033899fe"
    assert report["report_fingerprint"] == "ca25d610e4b1398748e2bec9122aee3db944027ce5bcd3d280bc4a2b287be535"
    core = evaluation._evaluate_core(model, [item])
    assert set(core) == {"model", "pieces", "views"}
    assert canonical_json(core) == canonical_json({key: report[key] for key in core})


@pytest.mark.parametrize("split", ["test", "TEST", "train", "validation"])
def test_public_guard_precedes_core_model_and_iteration(monkeypatch, split):
    monkeypatch.setattr(evaluation, "_evaluate_core", lambda *args, **kwargs: pytest.fail("core called"))
    with pytest.raises(ValueError, match="forbidden|future_registered"):
        evaluation.evaluate_frozen(object(), forbidden_inputs(), split=split)


@pytest.mark.parametrize("training,device", [(True, "cpu"), (False, "cuda")])
def test_core_cpu_eval_guard_precedes_iteration(training, device):
    model = SimpleNamespace(
        modules=lambda: (SimpleNamespace(training=training),),
        parameters=lambda: (SimpleNamespace(device=SimpleNamespace(type=device)),),
    )
    with pytest.raises(ValueError, match="cpu_eval_model_required"):
        evaluation._evaluate_core(model, forbidden_inputs())


def test_core_predicts_both_views_before_target_reader(model, item, monkeypatch):
    trace = []
    predict = model.predict

    def observed_predict(batch):
        trace.append("predict")
        return predict(batch)

    def targets():
        assert trace == ["predict", "predict"]
        trace.append("targets")
        return item.read_targets()

    monkeypatch.setattr(model, "predict", observed_predict)
    core = evaluation._evaluate_core(model, [replace(item, read_targets=targets)])
    assert trace == ["predict", "predict", "targets"]
    assert set(core["views"]) == {"native", "metadata_removed"}
    assert core["pieces"][0]["event_count"] == 1


def test_core_rejects_test_piece_before_prediction_or_targets(model, item, monkeypatch):
    monkeypatch.setattr(evaluation, "_predict_piece", lambda *args: pytest.fail("TEST prediction"))
    test_item = replace(item, piece=replace(item.piece, split="TEST"),
                        read_targets=lambda: pytest.fail("TEST targets"))
    with pytest.raises(ValueError, match="test_access_forbidden"):
        evaluation._evaluate_core(model, [test_item])


def test_core_rejects_bad_checkpoint_binding_before_iteration(model):
    with pytest.raises(ValueError, match="checkpoint_evidence_binding_mismatch"):
        evaluation._evaluate_core(model, forbidden_inputs(), checkpoint_evidence={})


def test_core_preserves_checkpoint_evidence_snapshot(model, item):
    identity = evaluation._evaluate_core(model, [])["model"]
    evidence = {
        "schema_version": CHECKPOINT_EVIDENCE_SCHEMA,
        "model_contract": identity["contract"],
        "model_contract_fingerprint": identity["contract"]["fingerprint"],
        "model_state_fingerprint": identity["state_fingerprint"],
        "load_policy": "cpu_weights_only_strict_model_state",
        "optimizer_restored": False, "checkpoint_rng_applied": False,
        "caller_rng_preserved": True, "model_training": False, "parameters_require_grad": False,
        "artifact_sha256": "a" * 64, "artifact_size_bytes": 1,
    }
    expected = canonical_json(evidence)

    def targets():
        evidence["artifact_sha256"] = "b" * 64
        return item.read_targets()

    core = evaluation._evaluate_core(model, [replace(item, read_targets=targets)], checkpoint_evidence=evidence)
    assert canonical_json(core["model"]["checkpoint_evidence"]) == expected
    assert core["model"]["checkpoint_evidence"] is not evidence


def test_core_rejects_model_mutation_during_target_read(model, item):
    def targets():
        with torch.no_grad():
            next(model.parameters()).add_(1)
        return item.read_targets()

    with pytest.raises(ValueError, match="model_mutated_during_evaluation"):
        evaluation._evaluate_core(model, [replace(item, read_targets=targets)])
