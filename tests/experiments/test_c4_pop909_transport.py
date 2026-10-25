"""Synthetic C4 private IPC boundaries; no corpus or checkpoint is read."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import pickle

import numpy as np
import pytest
import torch
from torch_geometric.data import HeteroData

from music_critic.experiments.analysisgnn.performance_prefetch import PrefetchError
from music_critic.experiments.analysisgnn.temporal_slices.cache import (
    decode_temporal_graph, encode_temporal_graph, transpose_temporal_historical,
)
from music_critic.experiments.analysisgnn.temporal_slices.graph import build_temporal_batch
from music_critic.experiments.c4_pop909.transport import (
    MAGIC, _tensor_from_bytes, assert_private_cpu_payload,
    pack_private_cpu_payload, unpack_private_cpu_payload,
)
from test_temporal_slice_cache import fixture

CAP = 10**7


@pytest.mark.parametrize("shift", range(12))
def test_packed_graph_roundtrip_preserves_all_shifts_and_exact_time(fixture, shift):
    first, second, sidecar = fixture
    graph = transpose_temporal_historical(build_temporal_batch((first, second)),
                                          (shift, (shift + 5) % 12))
    body = {"graph": encode_temporal_graph(graph), "sidecars": [sidecar],
            "source_fingerprint": "synthetic-source", "record_ids": ("a", "b")}
    before = torch.get_rng_state().clone()
    assert_private_cpu_payload(body)
    payload = pack_private_cpu_payload(body, cap=CAP)
    decoded = unpack_private_cpu_payload(payload, cap=CAP)
    assert_private_cpu_payload(decoded)
    assert decode_temporal_graph(decoded["graph"]).fingerprint() == graph.fingerprint()
    assert decoded["sidecars"] == body["sidecars"]
    assert decoded["source_fingerprint"] == body["source_fingerprint"]
    assert decoded["record_ids"] == body["record_ids"]
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize("dtype", [torch.bool, torch.uint8, torch.int8, torch.int16,
    torch.int32, torch.int64, torch.float16, torch.bfloat16, torch.float32,
    torch.float64, torch.complex64, torch.complex128])
def test_decoded_tensors_own_exact_values_for_all_supported_dtypes(dtype):
    tensor = torch.arange(12).reshape(3, 4).to(dtype).t()
    source = [tensor, tensor, tensor[1:3], torch.empty((2, 0, 3), dtype=dtype),
              torch.tensor(1, dtype=dtype)]
    payload = pack_private_cpu_payload(source, cap=CAP)
    decoded = unpack_private_cpu_payload(payload, cap=CAP)
    for expected, actual in zip(source, decoded):
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert actual.device.type == "cpu"
        assert torch.equal(expected.contiguous().reshape(-1).view(torch.uint8),
                           actual.reshape(-1).view(torch.uint8))
    assert decoded[0] is decoded[1]
    assert decoded[0].is_contiguous()
    assert decoded[0].data_ptr() != tensor.data_ptr()
    independent_view = decoded[2].clone()
    decoded[0].zero_()
    assert torch.equal(decoded[2], independent_view)
    assert torch.equal(unpack_private_cpu_payload(payload, cap=CAP)[0], tensor)


def test_float_special_values_retain_their_bits():
    source = torch.tensor([0., -0., float("inf"), -float("inf"), float("nan")])
    decoded = unpack_private_cpu_payload(pack_private_cpu_payload(source, cap=CAP), cap=CAP)
    assert torch.equal(source.view(torch.uint8), decoded.view(torch.uint8))


@pytest.mark.parametrize("store_name", ["first", "last", "_slice_dict", "_inc_dict"])
def test_cpu_guard_checks_every_graph_store_and_batch_metadata(store_name):
    graph = HeteroData()
    for name in ("first", "last"):
        graph[name].x = torch.ones(2, 1)
    graph._slice_dict = {"index": torch.ones(2)}
    graph._inc_dict = {"index": torch.ones(2)}
    assert_private_cpu_payload(graph)
    if store_name.startswith("_"):
        getattr(graph, store_name)["bad"] = torch.empty(1, device="meta")
    else:
        graph[store_name].x = torch.empty(2, 1, device="meta")
    with pytest.raises(PrefetchError, match="cpu_tensor_required"):
        assert_private_cpu_payload(graph)


@pytest.mark.parametrize("value", [object(), {"bad": set()},
    {"bad": np.array([object()], dtype=object)}, {"bad": torch.empty(1, device="meta")}])
def test_cpu_guard_rejects_unsupported_or_non_cpu_leaves(value):
    with pytest.raises(PrefetchError):
        assert_private_cpu_payload(value)


def test_cpu_guard_rechecks_mutated_recursive_payloads():
    payload = []
    payload.append(payload)
    assert_private_cpu_payload(payload)
    payload.append(torch.empty(1, device="meta"))
    with pytest.raises(PrefetchError, match="cpu_tensor_required"):
        assert_private_cpu_payload(payload)


@pytest.mark.parametrize("value", [torch.ones(2, requires_grad=True),
    torch.sparse_coo_tensor([[0]], [1.], (2,)), torch.empty(2, device="meta")])
def test_nonplain_cpu_tensor_cannot_be_packed(value):
    with pytest.raises(ValueError, match="plain_cpu_input_tensor"):
        pack_private_cpu_payload(value, cap=CAP)


def _token_payload(token):
    class MalformedTensor:
        def __reduce__(self):
            return (_tensor_from_bytes, token)
    return MAGIC + pickle.dumps(MalformedTensor(), protocol=5)


@pytest.mark.parametrize("token", [(), ("unknown",),
    ("torch.int64", (1,), b"\0" * 8, "extra"),
    ("torch.bad", (1,), b"\0" * 8), ("torch.int64", (True,), b"\0" * 8),
    ("torch.int64", (-1,), b""), ("torch.int64", (10**8,), b""),
    ("torch.int64", (1,), b"\0"), ("torch.int64", (1,), bytearray(8))])
def test_malformed_tensor_metadata_is_rejected_before_allocation(token, monkeypatch):
    monkeypatch.setattr(torch, "frombuffer", lambda *args, **kwargs:
                        pytest.fail("allocated malformed tensor"))
    with pytest.raises(ValueError):
        unpack_private_cpu_payload(_token_payload(token), cap=CAP)


def test_byte_caps_and_corruption_fail_closed():
    with pytest.raises(ValueError, match="tensor_bytes_exceed_cap"):
        pack_private_cpu_payload(torch.zeros(100), cap=100)
    with pytest.raises(PrefetchError, match="result_exceeds_reservation"):
        pack_private_cpu_payload({"metadata": "x" * 1000}, cap=100)
    payload = pack_private_cpu_payload(torch.ones(3), cap=CAP)
    with pytest.raises(ValueError, match="invalid_payload"):
        unpack_private_cpu_payload(payload, cap=len(payload) - 1)
    with pytest.raises(ValueError, match="trailing_payload_bytes"):
        unpack_private_cpu_payload(payload + b"x", cap=CAP)
    with pytest.raises((EOFError, pickle.UnpicklingError)):
        unpack_private_cpu_payload(payload[:-1], cap=CAP)
    with pytest.raises(ValueError, match="private_decoder_required"):
        pickle.loads(payload[len(MAGIC):])


def _spawned_decode(payload):
    decoded = unpack_private_cpu_payload(payload, cap=CAP)
    graph = decode_temporal_graph(decoded)
    return graph.fingerprint(), [len(part["note"].x_cat) for part in graph.base.to_data_list()]


def test_dynamic_graph_batch_survives_spawn_process(fixture):
    first, second, _ = fixture
    graph = build_temporal_batch((first, second))
    payload = pack_private_cpu_payload(encode_temporal_graph(graph), cap=CAP)
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
        actual = pool.submit(_spawned_decode, payload).result(timeout=30)
    assert actual == (graph.fingerprint(), [len(part["note"].x_cat) for part in graph.base.to_data_list()])
