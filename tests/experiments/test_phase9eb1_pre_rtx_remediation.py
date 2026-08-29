from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch_geometric.data import HeteroData

from music_critic.experiments.analysisgnn import run as run_module
from music_critic.experiments.analysisgnn.model import _official_onset_pool


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_onset_pool_matches_official_scatter_mean_out_clone_formula() -> None:
    encoded = torch.arange(18, dtype=torch.float32).reshape(6, 3).requires_grad_()
    onset_edges = torch.tensor(
        [
            [1, 2, 3, 3, 4, 4, 5, 5],
            [2, 1, 4, 5, 3, 5, 3, 4],
        ],
        dtype=torch.long,
    )

    reference = encoded.clone()
    official_neighbor_count = torch.zeros(6, dtype=encoded.dtype)
    for target, neighbor in onset_edges.t().tolist():
        reference[target] += encoded[neighbor]
        official_neighbor_count[target] += 1
    reference /= official_neighbor_count.clamp_min(1).unsqueeze(-1)

    actual = _official_onset_pool(encoded, onset_edges)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    torch.testing.assert_close(actual[0], encoded[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], encoded[1] + encoded[2], rtol=0, atol=0)
    torch.testing.assert_close(
        actual[3],
        (encoded[3] + encoded[4] + encoded[5]) / 2,
        rtol=0,
        atol=0,
    )
    actual.sum().backward()
    assert encoded.grad is not None and bool(torch.isfinite(encoded.grad).all())


def test_real_graph_smoke_selector_reads_train_only_and_prefers_largest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = (
        SimpleNamespace(
            record_id="train-b",
            piece_id="piece-b",
            source_group_id="group-b",
            split="train",
        ),
        SimpleNamespace(
            record_id="validation-largest",
            piece_id="piece-validation",
            source_group_id="group-validation",
            split="validation",
        ),
        SimpleNamespace(
            record_id="train-a",
            piece_id="piece-a",
            source_group_id="group-a",
            split="train",
        ),
        SimpleNamespace(
            record_id="test-largest",
            piece_id="piece-test",
            source_group_id="group-test",
            split="test",
        ),
    )
    note_counts = {
        "piece-a": 12,
        "piece-b": 12,
        "piece-validation": 100,
        "piece-test": 200,
    }
    loaded: list[str] = []

    def fake_load(_cache_root: object, row: object) -> tuple[object, object, object]:
        piece_id = row.piece_id  # type: ignore[attr-defined]
        loaded.append(piece_id)
        return SimpleNamespace(notes=tuple(range(note_counts[piece_id]))), object(), object()

    monkeypatch.setattr(run_module, "load_common_record", fake_load)
    selected, piece, _targets, _projection = run_module._select_real_smoke_record(
        SimpleNamespace(records=rows),
        "/ignored/cache",
    )
    assert loaded == ["piece-a", "piece-b"]
    assert selected.record_id == "train-a"
    assert selected.split == "train"
    assert len(piece.notes) == 12


def _smoke_graph() -> HeteroData:
    graph = HeteroData()
    graph["note"].x = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    graph["note"].quality = torch.tensor([0, 1, 2])
    graph["note"].inversion = torch.tensor([0, 1, 2])
    graph["measure"].x = torch.zeros((1, 3))
    graph["beat"].x = torch.zeros((1, 3))
    graph["note", "onset", "note"].edge_index = torch.tensor(
        [[0, 1], [1, 0]], dtype=torch.long
    )
    return graph


def test_real_graph_smoke_builds_p1_and_runs_no_optimizer_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = SimpleNamespace(
        record_id="train-record",
        piece_id="train-piece",
        source_group_id="train-group",
        split="train",
    )
    piece = SimpleNamespace(notes=(object(), object(), object()))
    monkeypatch.setattr(
        run_module,
        "_select_real_smoke_record",
        lambda *_args, **_kwargs: (row, piece, object(), object()),
    )
    called: dict[str, object] = {}

    def fake_build(
        _piece: object,
        _targets: object,
        _projection: object,
        *,
        transposition: str,
    ) -> tuple[HeteroData, tuple[object, ...]]:
        called["transposition"] = transposition
        return _smoke_graph(), (object(), object())

    class FakeModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.output = nn.Linear(3, 54)

        def forward(self, graph: HeteroData) -> dict[str, torch.Tensor]:
            logits = self.output(graph["note"].x)
            return {"quality": logits[:, :50], "inversion": logits[:, 50:]}

        def architecture_manifest(self) -> dict[str, object]:
            return {"kind": "targeted-test-double"}

    monkeypatch.setattr(run_module, "build_analysisgnn_graph", fake_build)
    monkeypatch.setattr(run_module, "graph_fingerprint", lambda _graph: "a" * 64)
    monkeypatch.setattr(run_module, "AnalysisGNNCommonModel", FakeModel)
    monkeypatch.setattr(
        run_module, "environment_report", lambda: {"claim": "targeted-test"}
    )

    result = run_module.real_graph_smoke(
        SimpleNamespace(records=(row,)),
        "/ignored/cache",
        device_name="cpu",
    )
    assert called == {"transposition": "P1"}
    assert result["acceptance"] is True
    assert result["record_id"] == "train-record"
    assert result["split"] == "train"
    assert result["graph_sha256"] == "a" * 64
    assert result["logit_shapes"] == {"quality": [3, 50], "inversion": [3, 4]}
    assert result["finite_gradients"] is True
    assert result["optimizer_step"] is False


def test_real_graph_smoke_rejects_non_train_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = SimpleNamespace(
        record_id="validation-record",
        piece_id="validation-piece",
        source_group_id="validation-group",
        split="validation",
    )
    monkeypatch.setattr(
        run_module,
        "_select_real_smoke_record",
        lambda *_args, **_kwargs: (row, SimpleNamespace(notes=(object(),)), object(), object()),
    )
    with pytest.raises(AssertionError, match="validation/test"):
        run_module.real_graph_smoke(
            SimpleNamespace(records=(row,)),
            "/ignored/cache",
            device_name="cpu",
        )


def test_runbook_and_runtime_policy_freeze_real_smoke_stop_order() -> None:
    runbook = (
        REPOSITORY_ROOT / "docs" / "PHASE9EB1_ANALYSISGNN_RUNBOOK.md"
    ).read_text(encoding="utf-8")
    environment = runbook.index("scripts/run_phase9eb1_analysisgnn.py environment")
    prepare = runbook.index("scripts/run_phase9eb1_analysisgnn.py prepare-data")
    preflight = runbook.index(
        "scripts/run_phase9eb1_analysisgnn.py label-binding-preflight"
    )
    cpu = runbook.index("--device cpu --output outputs/phase9eb1/smoke/real-train-cpu.json")
    cuda = runbook.index(
        "--device cuda --output outputs/phase9eb1/smoke/real-train-cuda.json"
    )
    stop = runbook.index("**STOP.")
    training = runbook.index("scripts/run_phase9eb1_analysisgnn.py train")
    assert environment < prepare < preflight < cpu < cuda < stop < training

    policy = json.loads(
        (REPOSITORY_ROOT / "configs" / "phase9eb1" / "runtime_policy.json").read_text(
            encoding="utf-8"
        )
    )
    assert policy["remote_rtx_required"][:4] == [
        "conflict_free_all_719_label_binding_preflight",
        "real_train_graphmuse_cpu_smoke",
        "same_real_train_graphmuse_cuda_smoke",
        "smoke_artifact_identity_review_and_stop_gate",
    ]
    assert policy["cuda_environment_required"] == {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8"
    }
