from __future__ import annotations

import re

import pytest
import torch

from music_critic.device import RuntimeDeviceError
from music_critic.graph import MANDATORY_EDGE_TYPES
from music_critic.ssl.contracts import SSL_CONTRACT_VERSION
from music_critic.ssl.checkpoint import (
    SSL_CHECKPOINT_CONTRACT_VERSION,
    SSL_EPOCH_JOURNAL_CONTRACT_VERSION,
    SSL_METRIC_ROW_VERSION,
)
from music_critic.ssl.data import (
    SSLDataError,
    _require_ssl_tensor_device,
    _validate_moved_batch,
    build_ssl_data_runtime,
    move_ssl_batch,
)
from music_critic.ssl.engine import (
    NO_LEAKAGE_MUTATION_EVIDENCE_CONTRACT_VERSION,
    PITCH_SENSITIVE_RECONSTRUCTION_EVIDENCE_CONTRACT_VERSION,
    SSL_PERFORMANCE_ROW_VERSION,
    SSL_RUN_MANIFEST_VERSION,
    SSL_TRAINING_REPORT_VERSION,
    _resolve_device,
)
from music_critic.training.config import DataConfig


def _batch():
    return build_ssl_data_runtime(
        DataConfig(),
        seed=42,
    ).first_train_batch


def test_ssl_cuda_resolution_is_concrete_before_exact_validation(
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    resolved = _resolve_device({"device": {"name": "cuda"}})

    assert torch.device("cuda") != torch.device("cuda:0")
    assert resolved == torch.device("cuda:0")
    assert resolved.index == 0


@pytest.mark.parametrize("name", ("cuda:0", "auto"))
def test_ssl_runtime_accepts_concrete_or_auto_cuda_with_amp(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    resolved = _resolve_device(
        {"device": {"name": name, "amp": True}}
    )

    assert resolved == torch.device("cuda:0")


def test_ssl_runtime_rejects_cpu_amp() -> None:
    with pytest.raises(
        ValueError,
        match=r"^ssl\.training\.amp_requires_cuda$",
    ):
        _resolve_device(
            {"device": {"name": "cpu", "amp": True}}
        )


@pytest.mark.parametrize(
    ("name", "category"),
    (
        ("mps", "runtime.device.type_unsupported"),
        ("cuda:not-an-index", "runtime.device.request_invalid"),
    ),
)
def test_ssl_runtime_preserves_structured_device_error(
    name: str,
    category: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=rf"ssl\.training\.device_invalid:{category}:",
    ):
        _resolve_device(
            {"device": {"name": name, "amp": False}}
        )


def test_ssl_runtime_preserves_cuda_index_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    with pytest.raises(
        ValueError,
        match=(
            r"runtime\.device\.cuda_index_out_of_range:"
            r"requested=cuda:1;visible_device_count=1"
        ),
    ):
        _resolve_device(
            {"device": {"name": "cuda:1", "amp": False}}
        )


def test_ssl_evidence_remediation_versions_are_narrow() -> None:
    assert SSL_CONTRACT_VERSION == "1.2.2"
    assert SSL_TRAINING_REPORT_VERSION == "1.2.3"
    assert NO_LEAKAGE_MUTATION_EVIDENCE_CONTRACT_VERSION == "1.0.0"
    assert (
        PITCH_SENSITIVE_RECONSTRUCTION_EVIDENCE_CONTRACT_VERSION
        == "1.0.0"
    )
    assert SSL_CHECKPOINT_CONTRACT_VERSION == "1.2.0"
    assert SSL_EPOCH_JOURNAL_CONTRACT_VERSION == "1.2.0"
    assert SSL_METRIC_ROW_VERSION == "1.2.0"
    assert SSL_RUN_MANIFEST_VERSION == "1.2.0"
    assert SSL_PERFORMANCE_ROW_VERSION == "1.2.0"


def test_ssl_transfer_rejects_unavailable_cuda_structurally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(
        RuntimeDeviceError,
        match=r"^runtime\.device\.cuda_unavailable:requested=cuda$",
    ):
        move_ssl_batch(_batch(), "cuda")


@pytest.mark.parametrize(
    ("kind", "expected_location"),
    (
        ("global", "global:raw_only"),
        ("node", "node:note:x_cont"),
        (
            "edge",
            "edge:"
            + "|".join(MANDATORY_EDGE_TYPES[0])
            + ":edge_index",
        ),
    ),
)
def test_ssl_transfer_mismatch_reports_exact_graph_location(
    kind: str,
    expected_location: str,
) -> None:
    source = _batch()
    moved = move_ssl_batch(source, "cpu")
    graph = moved.raw_graph_batch
    if kind == "global":
        graph.raw_only = torch.empty_like(
            graph.raw_only,
            device="meta",
        )
    elif kind == "node":
        graph["note"].x_cont = torch.empty_like(
            graph["note"].x_cont,
            device="meta",
        )
    else:
        edge_type = MANDATORY_EDGE_TYPES[0]
        graph[edge_type].edge_index = torch.empty_like(
            graph[edge_type].edge_index,
            device="meta",
        )

    message = (
        "ssl.data.device_transfer_tensor_mismatch:"
        f"location={expected_location};expected=cpu;actual=meta"
    )
    with pytest.raises(
        SSLDataError,
        match=f"^{re.escape(message)}$",
    ):
        _validate_moved_batch(
            moved,
            source=source,
            device=torch.device("cpu"),
        )


def test_ssl_transfer_mismatch_reports_prepared_binding_field() -> None:
    with pytest.raises(
        SSLDataError,
        match=(
            r"^ssl\.data\.device_transfer_tensor_mismatch:"
            r"location=binding:selected_global_note_indices_tensor;"
            r"expected=cuda:1;actual=cpu$"
        ),
    ):
        _require_ssl_tensor_device(
            torch.tensor([0], dtype=torch.long),
            device=torch.device("cuda:1"),
            location="binding:selected_global_note_indices_tensor",
        )
