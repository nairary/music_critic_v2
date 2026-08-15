from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from music_critic.experiments.phase8b2 import artifacts
from music_critic.ssl import engine as ssl_engine
from music_critic.ssl import phase8b_engine
from music_critic.training import engine as training_engine


class _BoundaryReached(RuntimeError):
    pass


def _raise_boundary(*_args: object, **_kwargs: object) -> None:
    raise _BoundaryReached


def test_cuda_evidence_apis_receive_logical_integers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, list[int]] = {
        "name": [],
        "allocated": [],
        "reserved": [],
    }

    def _name(index: int) -> str:
        assert type(index) is int
        observed["name"].append(index)
        return "mock-cuda-device"

    def _allocated(index: int) -> int:
        assert type(index) is int
        observed["allocated"].append(index)
        return 1024

    def _reserved(index: int) -> int:
        assert type(index) is int
        observed["reserved"].append(index)
        return 2048

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", _name)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", _allocated)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", _reserved)
    for module in (ssl_engine, phase8b_engine, training_engine, artifacts):
        monkeypatch.setattr(
            module,
            "resolve_cuda_device_index",
            lambda _device: 1,
        )

    ssl_evidence = ssl_engine._device_evidence(torch.device("cuda:1"))
    peak_evidence = phase8b_engine._cuda_peak_memory(
        torch.device("cuda:1")
    )
    training_evidence = training_engine._device_evidence(
        torch.device("cuda:1")
    )
    environment = artifacts.environment_evidence(torch.device("cuda:1"))

    assert observed == {
        "name": [1, 1, 1],
        "allocated": [1, 1, 1],
        "reserved": [1, 1, 1],
    }
    for evidence in (
        ssl_evidence,
        peak_evidence,
        training_evidence,
        environment,
    ):
        assert evidence["cuda_logical_device_index"] == 1


def test_all_three_training_preflight_resets_receive_logical_integer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[int] = []

    def _reset(index: int) -> None:
        assert type(index) is int
        observed.append(index)

    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", _reset)

    monkeypatch.setattr(ssl_engine, "_validate_config", lambda _config: None)
    monkeypatch.setattr(ssl_engine, "_set_determinism", lambda _seed: None)
    monkeypatch.setattr(
        ssl_engine,
        "_resolve_device",
        lambda _config: torch.device("cuda:1"),
    )
    monkeypatch.setattr(
        ssl_engine, "resolve_cuda_device_index", lambda _device: 1
    )
    monkeypatch.setattr(ssl_engine, "build_ssl_data_runtime", _raise_boundary)
    with pytest.raises(_BoundaryReached):
        ssl_engine._prepare({"seed": 17, "data": {}})

    monkeypatch.setattr(
        phase8b_engine,
        "_materialize",
        lambda _config: (
            {"seed": 17, "data": {}},
            None,
            None,
            None,
            None,
        ),
    )
    monkeypatch.setattr(
        phase8b_engine, "resolve_cuda_device_index", lambda _device: 1
    )
    monkeypatch.setattr(
        phase8b_engine, "build_ssl_data_runtime", _raise_boundary
    )
    with pytest.raises(_BoundaryReached):
        phase8b_engine._prepare({})

    monkeypatch.setattr(
        training_engine, "_validate_config", lambda _config: None
    )
    monkeypatch.setattr(
        training_engine, "_set_determinism", lambda _seed: None
    )
    monkeypatch.setattr(
        training_engine,
        "_resolve_device",
        lambda _config: torch.device("cuda:1"),
    )
    monkeypatch.setattr(
        training_engine, "resolve_cuda_device_index", lambda _device: 1
    )
    monkeypatch.setattr(
        training_engine, "build_data_runtime", _raise_boundary
    )
    with pytest.raises(_BoundaryReached):
        training_engine._prepare(
            {
                "seed": 17,
                "transfer": {
                    "comparison_protocol_fingerprint": "",
                    "downstream_data_order_seed": 17,
                },
                "data": {},
            }
        )

    assert observed == [1, 1, 1]


def test_runtime_device_cuda_api_source_audit_has_no_device_objects() -> None:
    targets = {
        "reset_peak_memory_stats",
        "max_memory_allocated",
        "max_memory_reserved",
        "memory_allocated",
        "memory_reserved",
        "synchronize",
        "get_device_name",
        "get_device_properties",
        "get_device_capability",
    }
    files = (
        Path("src/music_critic/ssl/engine.py"),
        Path("src/music_critic/ssl/phase8b_engine.py"),
        Path("src/music_critic/training/engine.py"),
        Path("src/music_critic/ssl/phase8a_cuda_acceptance.py"),
        Path("src/music_critic/ssl/phase8b_cuda_acceptance.py"),
        Path("src/music_critic/experiments/phase8b2/artifacts.py"),
    )
    forbidden_names = {"device", "resolved_device"}

    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(
                node.func, ast.Attribute
            ):
                continue
            if node.func.attr not in targets or not node.args:
                continue
            argument = node.args[0]
            assert not (
                isinstance(argument, ast.Name)
                and argument.id in forbidden_names
            ), f"{path}:{node.lineno}:{node.func.attr} received {argument.id}"
