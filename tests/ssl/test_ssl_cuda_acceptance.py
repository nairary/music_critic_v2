"""Optional CUDA+AMP acceptance for the Phase 7A SSL harness."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize
import pytest
import torch

from music_critic.ssl.config import register_ssl_configs
from music_critic.ssl.data import (
    SSLDataError,
    _validate_moved_batch,
    build_ssl_data_runtime,
    move_ssl_batch,
)
from music_critic.ssl.engine import run_ssl_training
from music_critic.ssl.masking import (
    move_ssl_batch_with_prepared_binding,
    prepare_mask_binding,
    validate_prepared_mask_binding,
)
from music_critic.training.config import DataConfig


def _assert_ssl_batch_device(batch, device: torch.device) -> None:
    for store in batch.raw_graph_batch.stores:
        for value in store.values():
            if isinstance(value, torch.Tensor):
                assert value.device == device


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Phase 7A CUDA+AMP acceptance requires CUDA",
)
def test_bounded_cuda_amp_smoke(tmp_path: Path) -> None:
    register_ssl_configs()
    with initialize(version_base="1.3", config_path=None):
        config = compose(
            config_name="ssl_training",
            overrides=[
                "experiment=one_batch",
                "experiment.steps=2",
                "model=hierarchical",
                "model.hidden_dim=8",
                "model.local_gnn_layers=1",
                "model.transformer_layers=1",
                "model.attention_heads=2",
                "model.ffn_multiplier=2",
                "model.dropout=0",
                "data=bounded",
                "device=cuda",
                "device.amp=true",
                "ssl.decoder_hidden_dim=8",
                "ssl.projector_hidden_dim=8",
                f"output_dir={tmp_path / 'cuda-amp'}",
            ],
        )

    report = run_ssl_training(config)

    assert report["device"]["resolved_device"] == (
        f"cuda:{torch.cuda.current_device()}"
    )
    assert report["device"]["cuda_available"] is True
    assert report["device"]["cuda_device_name"]
    assert report["device"]["deterministic_algorithms"] is True
    assert report["device"]["cudnn_benchmark"] is False
    assert report["device"]["cudnn_deterministic"] is True
    assert report["device"]["cublas_workspace_config"] in {
        ":4096:8",
        ":16:8",
    }
    assert report["device"]["peak_allocated_bytes"] > 0
    assert report["device"]["peak_reserved_bytes"] > 0
    assert report["amp_enabled"] is True
    assert report["scaler_enabled"] is True
    assert report["initial"]["total_ssl_loss"] is not None
    assert report["final"]["total_ssl_loss"] is not None
    assert report["checkpoint_reload"]["bit_exact"] is True
    assert all(report["deterministic_repeat"].values())
    assert report["no_leakage_mutation_evidence"]["passed"] is True


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA device canonicalization parity requires CUDA",
)
def test_abstract_and_explicit_cuda_zero_transfer_are_equivalent() -> None:
    cpu_batch = build_ssl_data_runtime(
        DataConfig(),
        seed=42,
    ).first_train_batch
    binding = prepare_mask_binding(
        cpu_batch,
        global_seed=42,
        epoch=0,
        requested_mask_rate=0.30,
        stage="train",
    )

    with torch.cuda.device(0):
        abstract_batch, abstract_binding = (
            move_ssl_batch_with_prepared_binding(
                cpu_batch,
                binding,
                "cuda",
            )
        )
        explicit_batch, explicit_binding = (
            move_ssl_batch_with_prepared_binding(
                cpu_batch,
                binding,
                "cuda:0",
            )
        )

    expected = torch.device("cuda:0")
    _assert_ssl_batch_device(abstract_batch, expected)
    _assert_ssl_batch_device(explicit_batch, expected)
    assert (
        abstract_binding.selected_global_note_indices_tensor.device
        == expected
    )
    assert (
        explicit_binding.selected_global_note_indices_tensor.device
        == expected
    )
    validate_prepared_mask_binding(
        abstract_batch,
        abstract_binding,
    )
    validate_prepared_mask_binding(
        explicit_batch,
        explicit_binding,
    )
    assert abstract_binding.fingerprint == explicit_binding.fingerprint
    for abstract_store, explicit_store in zip(
        abstract_batch.raw_graph_batch.stores,
        explicit_batch.raw_graph_batch.stores,
        strict=True,
    ):
        for name, abstract_value in abstract_store.items():
            explicit_value = explicit_store[name]
            if isinstance(abstract_value, torch.Tensor):
                assert torch.equal(abstract_value, explicit_value)
            else:
                assert abstract_value == explicit_value


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="wrong-device validation requires at least two CUDA devices",
)
def test_ssl_wrong_cuda_index_is_rejected_with_location() -> None:
    source = build_ssl_data_runtime(
        DataConfig(),
        seed=42,
    ).first_train_batch
    moved = move_ssl_batch(source, "cuda:1")
    moved.raw_graph_batch["note"].x_cont = moved.raw_graph_batch[
        "note"
    ].x_cont.to("cuda:0")

    with pytest.raises(
        SSLDataError,
        match=(
            r"^ssl\.data\.device_transfer_tensor_mismatch:"
            r"location=node:note:x_cont;"
            r"expected=cuda:1;actual=cuda:0$"
        ),
    ):
        _validate_moved_batch(
            moved,
            source=source,
            device=torch.device("cuda:1"),
        )
