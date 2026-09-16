"""Read actual autocast state across supported Torch API generations.

Torch 2.2 has separate CPU/CUDA dtype and CPU-enabled getters. Later releases
accept a device type in the unified getters. Querying either family changes no
runtime flags and does not initialize CUDA; missing APIs never mean disabled.
"""
from __future__ import annotations

import torch


def _required_getter(name):
    getter = getattr(torch, name, None)
    if not callable(getter):
        raise ValueError("group_pilot.autocast_runtime_query_unavailable:" + name)
    return getter


def autocast_runtime_binding(device_type: str) -> dict:
    """Return the same numerical-runtime fields on modern and legacy Torch.

    Presence of the unified dtype getter identifies the modern API family.
    Errors from an available getter propagate: a broken state query cannot be
    converted into a successful legacy fallback or an assumed disabled state.
    """
    if device_type not in {"cpu", "cuda"}:
        raise ValueError("group_pilot.autocast_runtime_device_invalid")
    if getattr(torch, "get_autocast_dtype", None) is not None:
        enabled = _required_getter("is_autocast_enabled")(device_type)
        dtype = _required_getter("get_autocast_dtype")(device_type)
    elif device_type == "cuda":
        enabled = _required_getter("is_autocast_enabled")()
        dtype = _required_getter("get_autocast_gpu_dtype")()
    else:
        enabled = _required_getter("is_autocast_cpu_enabled")()
        dtype = _required_getter("get_autocast_cpu_dtype")()
    if type(enabled) is not bool or not isinstance(dtype, torch.dtype):
        raise ValueError("group_pilot.autocast_runtime_query_invalid")
    return {"autocast_enabled": enabled, "autocast_dtype": str(dtype)}
