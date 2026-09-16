"""CPU-only emulation of modern and Torch 2.2 autocast query contracts."""
from types import SimpleNamespace

import pytest
import torch

from music_critic.ssl import autocast_runtime as query
from music_critic.ssl.group_masking_pilot import GroupMaskingPilot, _runtime_binding


def api(monkeypatch, family, *, enabled=False):
    state = {"cpu": {"enabled": enabled, "dtype": torch.bfloat16},
             "cuda": {"enabled": enabled, "dtype": torch.float16}}
    calls = []
    def modern_enabled(device):
        calls.append(("enabled", device))
        return state[device]["enabled"]
    def modern_dtype(device):
        calls.append(("dtype", device))
        return state[device]["dtype"]
    def legacy_enabled():
        calls.append(("enabled", "cuda"))
        return state["cuda"]["enabled"]
    def cpu_enabled():
        calls.append(("enabled", "cpu"))
        return state["cpu"]["enabled"]
    def cpu_dtype():
        calls.append(("dtype", "cpu"))
        return state["cpu"]["dtype"]
    def gpu_dtype():
        calls.append(("dtype", "cuda"))
        return state["cuda"]["dtype"]
    if family == "modern":
        monkeypatch.setattr(torch, "is_autocast_enabled", modern_enabled)
        monkeypatch.setattr(torch, "get_autocast_dtype", modern_dtype, raising=False)
        # A modern query must not silently consult a legacy getter.
        def forbidden():
            pytest.fail("legacy getter called on modern API")
        for name in ("is_autocast_cpu_enabled", "get_autocast_cpu_dtype", "get_autocast_gpu_dtype"):
            monkeypatch.setattr(torch, name, forbidden, raising=False)
    else:
        monkeypatch.delattr(torch, "get_autocast_dtype", raising=False)
        monkeypatch.setattr(torch, "is_autocast_enabled", legacy_enabled)
        monkeypatch.setattr(torch, "is_autocast_cpu_enabled", cpu_enabled, raising=False)
        monkeypatch.setattr(torch, "get_autocast_cpu_dtype", cpu_dtype, raising=False)
        monkeypatch.setattr(torch, "get_autocast_gpu_dtype", gpu_dtype, raising=False)
    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda: pytest.fail("CUDA must not initialize"))
    return state, calls


@pytest.mark.parametrize("family", ["modern", "legacy"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("enabled", [False, True])
def test_both_api_families_read_actual_selected_device_without_cuda(monkeypatch, family, device, enabled):
    state, calls = api(monkeypatch, family, enabled=enabled)
    # Different state on the opposite device catches using the CUDA default
    # no-argument getter as an accidental CPU getter.
    other = "cuda" if device == "cpu" else "cpu"
    state[other]["enabled"] = not enabled
    assert query.autocast_runtime_binding(device) == {
        "autocast_enabled": enabled, "autocast_dtype": str(state[device]["dtype"])}
    assert calls == [("enabled", device), ("dtype", device)]


@pytest.mark.parametrize("family", ["modern", "legacy"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("field", ["enabled", "dtype"])
def test_actual_state_change_remains_runtime_drift(monkeypatch, family, device, field):
    state, _ = api(monkeypatch, family)
    selected = torch.device("cpu" if device == "cpu" else "cuda:0")
    # This invokes the actual reviewed drift validator without allocating a
    # model/GPU; its input is the real complete runtime binding function.
    guard = SimpleNamespace(device=selected, binding={"runtime": _runtime_binding(selected)})
    GroupMaskingPilot._validate_runtime(guard)
    state[device][field] = True if field == "enabled" else torch.float32
    with pytest.raises(ValueError, match="runtime_drift"):
        GroupMaskingPilot._validate_runtime(guard)


@pytest.mark.parametrize("device,getter", [("cuda", "get_autocast_gpu_dtype"),
                                            ("cpu", "get_autocast_cpu_dtype"),
                                            ("cpu", "is_autocast_cpu_enabled")])
def test_missing_legacy_state_query_fails_closed(monkeypatch, device, getter):
    api(monkeypatch, "legacy")
    monkeypatch.delattr(torch, getter)
    with pytest.raises(ValueError, match="autocast_runtime_query_unavailable"):
        query.autocast_runtime_binding(device)


def test_available_modern_getter_error_is_not_hidden_by_fallback(monkeypatch):
    api(monkeypatch, "modern")
    def invalid(device):
        raise TypeError("synthetic modern query failure")
    monkeypatch.setattr(torch, "is_autocast_enabled", invalid)
    with pytest.raises(TypeError, match="synthetic modern query failure"):
        query.autocast_runtime_binding("cuda")


@pytest.mark.parametrize("field,value", [("enabled", 0), ("dtype", "torch.float16")])
def test_malformed_query_cannot_weaken_runtime_binding(monkeypatch, field, value):
    state, _ = api(monkeypatch, "legacy")
    state["cuda"][field] = value
    with pytest.raises(ValueError, match="autocast_runtime_query_invalid"):
        query.autocast_runtime_binding("cuda")


def test_modern_and_legacy_fields_are_semantically_identical(monkeypatch):
    with monkeypatch.context() as modern:
        api(modern, "modern", enabled=True)
        expected = {device: query.autocast_runtime_binding(device) for device in ("cpu", "cuda")}
    with monkeypatch.context() as legacy:
        api(legacy, "legacy", enabled=True)
        assert {device: query.autocast_runtime_binding(device) for device in ("cpu", "cuda")} == expected


def test_actual_current_torch_cpu_autocast_context_is_observed():
    before = query.autocast_runtime_binding("cpu")
    with torch.autocast("cpu", enabled=not before["autocast_enabled"], dtype=torch.bfloat16):
        inside = query.autocast_runtime_binding("cpu")
        assert inside["autocast_enabled"] is not before["autocast_enabled"]
        assert inside["autocast_dtype"] == "torch.bfloat16"
    assert query.autocast_runtime_binding("cpu") == before
