from __future__ import annotations

from pathlib import Path

import pytest
import torch

from music_critic.evaluation import evaluate_dilemmadata_model
from music_critic.evaluation.dilemmadata_run import _model as load_model
from music_critic.models import (
    DilemmadataHierarchicalConfig,
    DilemmadataHierarchicalModel,
    dilemmadata_model_contract_dict,
)
from tests.models.test_onset_bigru_decoder import _batch, _model


def _checkpoint(path: Path, kind: str):
    model = _model(kind)
    torch.save(
        {
            "metadata": {
                "model_contract": dilemmadata_model_contract_dict(model)
            },
            "model_state": model.state_dict(),
        },
        path,
    )
    return model


def _components(batch):
    return {
        (dataset_id, piece_id): f"component-{index}"
        for index, (dataset_id, piece_id) in enumerate(
            zip(batch.dataset_ids, batch.piece_ids, strict=True)
        )
    }


def test_real_mlp_checkpoint_preserves_logits_through_evaluation_loader(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "mlp.pt"
    original = _checkpoint(checkpoint, "mlp").eval()
    loaded = load_model(checkpoint, torch.device("cpu")).eval()
    batch = _batch(0, 1)
    with torch.no_grad():
        _, expected = original.predict(batch.raw_graph_batch)
        _, observed = loaded.predict(batch.raw_graph_batch)
    assert loaded.config.decoder.kind == "mlp"
    for left, right in zip(expected, observed, strict=True):
        torch.testing.assert_close(left.logits, right.logits, rtol=0, atol=0)


def test_real_onset_bigru_checkpoint_loads_and_validates_bounded_fixture(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "onset-bigru.pt"
    original = _checkpoint(checkpoint, "onset_bigru")
    contract = dilemmadata_model_contract_dict(original)
    old_config = dict(contract["config"])
    old_config["task_weights"] = tuple(
        tuple(row) for row in old_config["task_weights"]
    )
    old_model = DilemmadataHierarchicalModel(
        DilemmadataHierarchicalConfig(**old_config)
    )
    with pytest.raises(RuntimeError, match="Unexpected key.*sequence_decoder"):
        old_model.load_state_dict(original.state_dict(), strict=True)
    loaded = load_model(checkpoint, torch.device("cpu"))
    batch = _batch(0, 1)
    report = evaluate_dilemmadata_model(
        loaded,
        (batch,),
        component_by_identity=_components(batch),
        membership_fingerprint="a" * 64,
    )
    assert loaded.config.decoder.kind == "onset_bigru"
    assert report["split"] == "validation"
    assert report["counts"]["record_count"] == 2
    assert 0 < report["aggregate"]["task_count"] <= 4
