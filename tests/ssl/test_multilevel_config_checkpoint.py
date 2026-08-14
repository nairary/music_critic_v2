from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from hydra import compose, initialize
import pytest
import torch

from music_critic.models import HierarchicalBaselineConfig
from music_critic.ssl.checkpoint import (
    SSLCheckpointError,
    load_ssl_checkpoint,
    save_ssl_checkpoint,
    ssl_checkpoint_metadata,
)
from music_critic.ssl.config import register_ssl_configs
from music_critic.ssl.model import MaskedGraphSSLConfig, MaskedGraphSSLModel
from music_critic.ssl.multilevel import (
    BEAT_LATENT,
    ONSET_LATENT,
    Phase8BMultilevelSSLModel,
    Phase8BObjectiveConfig,
    build_phase8b_model_from_config,
)
from music_critic.ssl.multilevel_checkpoint import (
    PHASE8B_CHECKPOINT_BINDING_CONTRACT_VERSION,
    transfer_phase7a_checkpoint_to_phase8b,
)


_DATA_FINGERPRINTS = {
    "kind": "bounded",
    "bounded_fixture_fingerprint": "a" * 64,
    "split_fingerprint": "b" * 64,
    "train_composition_fingerprint": "c" * 64,
    "validation_composition_fingerprint": "d" * 64,
    "validation_membership_fingerprint": "e" * 64,
}
_JOURNAL = (
    {
        "metric_row_version": "1.2.0",
        "epoch": 0,
        "next_epoch": 1,
        "learning_rate_used": 0.01,
        "next_learning_rate": 0.01,
        "train": {"total_ssl_loss": 0.75},
        "validation": {"total_ssl_loss": 0.75},
        "gradient_coverage": None,
    },
)


def _encoder_config() -> HierarchicalBaselineConfig:
    return HierarchicalBaselineConfig(
        hidden_dim=8,
        local_gnn_layers=1,
        transformer_layers=1,
        attention_heads=2,
        ffn_multiplier=2,
        dropout=0.0,
    )


def _ssl_config() -> MaskedGraphSSLConfig:
    return MaskedGraphSSLConfig(
        mask_rate=0.3,
        decoder_views=1,
        decoder_remask_probability=0.0,
        decoder_hidden_dim=8,
        projector_hidden_dim=8,
    )


def _multilevel(mode: str) -> Phase8BMultilevelSSLModel:
    return Phase8BMultilevelSSLModel(
        _encoder_config(),
        _ssl_config(),
        Phase8BObjectiveConfig.for_mode(mode),
    )


def _checkpoint_objects(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    return optimizer, scaler


def _save(path: Path, model, *, mode: str) -> None:
    optimizer, scaler = _checkpoint_objects(model)
    save_ssl_checkpoint(
        path,
        model,
        optimizer,
        scheduler=None,
        scaler=scaler,
        next_epoch=1,
        best_validation_loss=0.75,
        epoch_journal=_JOURNAL,
        resolved_config={"phase8b_objective": mode},
        data_fingerprints=_DATA_FINGERPRINTS,
    )


def test_hydra_exposes_all_modes_with_inactive_phase7a_default() -> None:
    register_ssl_configs()
    expected = {
        "phase7a_control": (1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0),
        "onset_only": (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        "beat_only": (0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        "bar_only": (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        "track_only": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        "multilevel_equal_weight": (0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0),
    }
    fields = (
        "phase7a_note_reconstruction",
        "phase7a_bar_latent",
        "phase7a_song_latent",
        "onset_latent",
        "beat_latent",
        "hierarchy_bar_latent",
        "track_latent",
    )
    with initialize(version_base="1.3", config_path=None):
        legacy = compose(config_name="ssl_training")
        assert legacy.phase8b_objective is None
        for mode, weights in expected.items():
            configured = compose(
                config_name="ssl_training",
                overrides=[f"+phase8b_objective={mode}"],
            )
            assert configured.phase8b_objective.mode == mode
            assert tuple(
                float(configured.phase8b_objective[name]) for name in fields
            ) == weights
        overridden = compose(
            config_name="ssl_training",
            overrides=[
                "+phase8b_objective=multilevel_equal_weight",
                "phase8b_objective.onset_latent=0.25",
                "phase8b_objective.beat_latent=0",
            ],
        )
        materialized = Phase8BObjectiveConfig.from_hydra(
            overridden.phase8b_objective
        )
        assert materialized.weight(ONSET_LATENT) == 0.25
        assert materialized.weight(BEAT_LATENT) == 0.0
        built = build_phase8b_model_from_config(
            overridden.model,
            overridden.ssl,
            overridden.phase8b_objective,
        )
        assert type(built) is Phase8BMultilevelSSLModel
        assert built.phase8b_objective_config.fingerprint == (
            materialized.fingerprint
        )


def test_phase7a_checkpoint_transfer_loads_old_state_and_preserves_new_heads(
    tmp_path: Path,
) -> None:
    torch.manual_seed(31)
    old = MaskedGraphSSLModel(_encoder_config(), _ssl_config())
    checkpoint = tmp_path / "phase7a.pt"
    _save(checkpoint, old, mode="phase7a_control")
    torch.manual_seed(79)
    new = _multilevel("onset_only")
    head_before = {
        name: value.detach().clone()
        for name, value in new.state_dict().items()
        if name.startswith("phase8b_latent_heads.")
    }

    report = transfer_phase7a_checkpoint_to_phase8b(checkpoint, new)

    assert report.checkpoint_binding_contract_version == (
        PHASE8B_CHECKPOINT_BINDING_CONTRACT_VERSION
    )
    assert report.loaded_parameter_tensors == tuple(old.state_dict())
    assert report.separately_initialized_parameter_tensors == tuple(head_before)
    for name, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[name])
    for name, value in head_before.items():
        assert torch.equal(value, new.state_dict()[name])

    forged = tmp_path / "phase7a-forged.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["model_state"].pop(next(iter(payload["model_state"])))
    torch.save(payload, forged)
    before_rejection = deepcopy(new.state_dict())
    with pytest.raises(SSLCheckpointError, match="transfer_keys_mismatch"):
        transfer_phase7a_checkpoint_to_phase8b(forged, new)
    for name, value in before_rejection.items():
        assert torch.equal(value, new.state_dict()[name])


def test_phase8b_checkpoint_round_trip_binds_objective_fingerprint_and_is_atomic(
    tmp_path: Path,
) -> None:
    torch.manual_seed(41)
    source = _multilevel("onset_only")
    checkpoint = tmp_path / "phase8b.pt"
    _save(checkpoint, source, mode="onset_only")

    torch.manual_seed(83)
    restored = _multilevel("onset_only")
    optimizer, scaler = _checkpoint_objects(restored)
    resume = load_ssl_checkpoint(
        checkpoint,
        restored,
        optimizer,
        scheduler=None,
        scaler=scaler,
        maximum_next_epoch=1,
        resolved_config={"phase8b_objective": "onset_only"},
        data_fingerprints=_DATA_FINGERPRINTS,
    )
    assert resume.next_epoch == 1
    assert resume.epoch_journal == _JOURNAL
    for name, value in source.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])

    incompatible = Phase8BMultilevelSSLModel(
        _encoder_config(),
        _ssl_config(),
        Phase8BObjectiveConfig.create(
            "onset_only",
            weights={
                family: (0.5 if family == ONSET_LATENT else 0.0)
                for family, _weight in Phase8BObjectiveConfig.for_mode(
                    "onset_only"
                ).family_weights
            },
        ),
    )
    incompatible_optimizer, incompatible_scaler = _checkpoint_objects(incompatible)
    before = deepcopy(incompatible.state_dict())
    with pytest.raises(SSLCheckpointError, match="metadata_mismatch"):
        load_ssl_checkpoint(
            checkpoint,
            incompatible,
            incompatible_optimizer,
            scheduler=None,
            scaler=incompatible_scaler,
            maximum_next_epoch=1,
            resolved_config={"phase8b_objective": "onset_only"},
            data_fingerprints=_DATA_FINGERPRINTS,
        )
    for name, value in before.items():
        assert torch.equal(value, incompatible.state_dict()[name])


def test_old_phase8b_runtime_binding_is_rejected_but_phase7a_is_unchanged() -> None:
    old_runtime = {
        "engine_contract_version": "1.0.0",
        "execution_mode": "onset_only",
        "model_class": "Phase8BMultilevelSSLModel",
        "objective_registry_fingerprint": "1" * 64,
        "objective_config": {},
        "objective_config_fingerprint": "2" * 64,
        "active_objective_families": ["onset_latent"],
        "active_objective_weights": [["onset_latent", 1.0]],
        "masking_config": {},
        "masking_config_fingerprint": "3" * 64,
        "mask_policy_mixture_fingerprint": "4" * 64,
    }
    with pytest.raises(
        SSLCheckpointError,
        match="phase8b_runtime_binding_invalid",
    ):
        ssl_checkpoint_metadata(
            _multilevel("onset_only"),
            resolved_config={"phase8b_runtime": old_runtime},
            data_fingerprints=_DATA_FINGERPRINTS,
        )

    phase7a = MaskedGraphSSLModel(_encoder_config(), _ssl_config())
    metadata = ssl_checkpoint_metadata(
        phase7a,
        resolved_config={"phase8b_objective": None},
        data_fingerprints=_DATA_FINGERPRINTS,
    )
    assert "phase8b_binding" not in metadata
