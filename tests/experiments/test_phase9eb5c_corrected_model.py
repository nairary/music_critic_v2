from __future__ import annotations

from dataclasses import asdict

import pytest
import torch

from music_critic.experiments.analysisgnn.corrected_model import (
    CORRECTED_MODEL_ID,
    CORRECTED_MODEL_SCHEMA,
    CorrectedAnalysisGNNModel,
    CorrectedAnalysisGNNModelError,
    corrected_model_contract,
    corrected_parameter_inventory,
    corrected_routing_contract,
)
from music_critic.experiments.analysisgnn.multitask_contract import TASK_BY_ID
from music_critic.experiments.analysisgnn.training_policy import (
    AUXILIARY_HEADS,
    DEFERRED_HEADS,
    PRIMARY_HEADS,
)
from music_critic.tasks import collate_multisource_samples
from tests.adapters.test_dilemmadata import CORPUS
from tests.adapters.test_dilemmadata_targets import _sample, _target


def _batch():
    return collate_multisource_samples(
        (_sample(_target(CORPUS, "an:training:same")),)
    )


def test_exact_corrected_architecture_and_parameter_inventory() -> None:
    model = CorrectedAnalysisGNNModel()
    assert CORRECTED_MODEL_ID == "music-critic-v2-corrected-analysisgnn-18head-v1"
    assert CORRECTED_MODEL_SCHEMA == "CorrectedAnalysisGNNModel@1.0.0"
    assert tuple(row.task_id for row in model.task_specs) == (
        *PRIMARY_HEADS,
        *AUXILIARY_HEADS,
    )
    assert len(PRIMARY_HEADS) == 8
    assert len(AUXILIARY_HEADS) == 10
    assert len(DEFERRED_HEADS) == 2
    assert len(model.task_heads.heads) == 18
    assert not model.task_heads.node_type_embeddings

    dimensions = {row.task_id: row.output_dim for row in model.task_specs}
    assert dimensions == {
        task_id: TASK_BY_ID[task_id].class_count
        for task_id in (*PRIMARY_HEADS, *AUXILIARY_HEADS)
    }
    assert dimensions["quality"] == 17
    assert dimensions["roman_numeral"] == 184
    assert "staff" not in dimensions
    assert not any(
        task in name
        for name in model.state_dict()
        for task in (*DEFERRED_HEADS, "staff")
    )

    for index, spec in enumerate(model.task_specs):
        head = model.task_heads.heads[f"task_{index:02d}"]
        assert isinstance(head[0], torch.nn.Linear)
        assert (head[0].in_features, head[0].out_features) == (128, 128)
        assert isinstance(head[1], torch.nn.GELU)
        assert isinstance(head[2], torch.nn.Dropout) and head[2].p == 0.1
        assert isinstance(head[3], torch.nn.Linear)
        assert (head[3].in_features, head[3].out_features) == (128, spec.output_dim)

    inventory = corrected_parameter_inventory(model)
    assert inventory["trainable"] == inventory["total"] == 3_661_936
    assert inventory["frozen"] == 0
    assert sum(inventory["heads"].values()) == (
        inventory["primary_heads"] + inventory["auxiliary_heads"]
    )
    assert inventory["fingerprint"] == corrected_parameter_inventory(model)["fingerprint"]


def test_raw_only_predictions_have_exact_routes_dimensions_and_fp32() -> None:
    model = CorrectedAnalysisGNNModel().eval()
    batch = _batch()
    with torch.no_grad():
        output = model(batch.raw_graph_batch)
    routes = corrected_routing_contract()["head_routes"]
    assert output.schema_version == CORRECTED_MODEL_SCHEMA
    assert len(output.predictions) == 18
    for row in output.predictions:
        assert row.logits.dtype == torch.float32
        assert row.logits.shape[1] == TASK_BY_ID[row.task_id].class_count
        assert row.allowed_node_types == (routes[row.task_id],)
    assert {routes[task] for task in PRIMARY_HEADS} == {"beat"}
    assert routes["cadence"] == "onset"
    assert routes["metrical_strength"] == "note"
    assert corrected_model_contract(model)["logit_fusion"] is False
    assert corrected_model_contract(model)["staff_included"] is False


def test_deferred_logit_request_fails_closed_without_parameters() -> None:
    model = CorrectedAnalysisGNNModel()
    with pytest.raises(CorrectedAnalysisGNNModelError) as caught:
        model.request_logits("phrase", _batch().raw_graph_batch)
    assert caught.value.category == "analysisgnn.corrected.deferred_head_logits_forbidden"
    assert all(row.task_id not in DEFERRED_HEADS for row in model.task_specs)


def test_model_prediction_api_does_not_accept_targets() -> None:
    model = CorrectedAnalysisGNNModel()
    batch = _batch()
    with pytest.raises(TypeError):
        model.predict(batch.raw_graph_batch, targets={})  # type: ignore[call-arg]
