"""C4 private CPU IPC adapted from the admitted C3 pipeline8 implementation.

Source: V2 commit 6e90163444a528154b38e1813f35e345b5668747,
analysisgnn/temporal_slices/transport.py. Arithmetic and byte format are
preserved; only the Python package import boundary is changed.

This is NOT an artifact/cache reader. Only a source-bound spawned worker may
produce its input, as for the reference pickle transport. Ordinary pickle keeps
PyG's object graph, batch slice/inc dictionaries and Python metadata intact;
tensor reducers bypass Torch's per-storage save/load machinery. Tensor
values/dtypes/shapes are exact. Strides become contiguous and distinct tensor
views do not retain storage aliasing: decoded inputs own private CPU memory.
Repeated references to the same tensor object remain repeated references.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import io
import math
import pickle

import numpy as np
import torch
from torch_geometric.data import HeteroData

from ..analysisgnn.performance_prefetch import PrefetchError, _LimitedBuffer

MAGIC = b"MCTV2PT1"
SCHEMA = "TemporalPackedTensorIPC@1.0.0"
_DTYPES = {str(dtype): dtype for dtype in (
    torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
    torch.float16, torch.bfloat16, torch.float32, torch.float64,
    torch.complex64, torch.complex128,
)}
_ELEMENT_BYTES = {name: torch.empty((), dtype=dtype).element_size()
                  for name, dtype in _DTYPES.items()}
_PRIMITIVE_TYPES = frozenset((type(None), bool, int, float, str, bytes))


def assert_private_cpu_payload(value):
    """Same allowed leaves as the reference checker, without scalar recursion.

    Theory sidecars contain millions of primitive fields. They need no tensor,
    cycle or dataclass inspection: exact scalar types are skipped while their
    parent is visited. Containers, tensors, NumPy values and all PyG stores and
    batch metadata still receive the original checks. No new object is cached
    or trusted across calls; this guard cannot be used as a reusable seal.
    """
    pending, seen = [value], set()
    while pending:
        item = pending.pop()
        if type(item) in _PRIMITIVE_TYPES:
            continue
        identity = id(item)
        if identity in seen:
            continue
        seen.add(identity)
        if isinstance(item, torch.Tensor):
            if item.device.type != "cpu":
                raise PrefetchError("prefetch.cpu_tensor_required")
        elif isinstance(item, (np.ndarray, np.generic)):
            if item.dtype.hasobject:
                raise PrefetchError("prefetch.numpy_object_payload_forbidden")
        elif isinstance(item, Mapping):
            for key, child in item.items():
                if type(key) not in _PRIMITIVE_TYPES:
                    pending.append(key)
                if type(child) not in _PRIMITIVE_TYPES:
                    pending.append(child)
        elif isinstance(item, (list, tuple)):
            pending.extend(child for child in item if type(child) not in _PRIMITIVE_TYPES)
        elif is_dataclass(item) and not isinstance(item, type):
            pending.extend(child for field in fields(item)
                           if type(child := getattr(item, field.name)) not in _PRIMITIVE_TYPES)
        elif isinstance(item, HeteroData):
            # Do not create temporary dictionaries: their recycled IDs must not
            # enter the traversal's seen set or hide another store's contents.
            for store in item.stores:
                for key, child in store.items():
                    if type(key) not in _PRIMITIVE_TYPES:
                        pending.append(key)
                    if type(child) not in _PRIMITIVE_TYPES:
                        pending.append(child)
            for name in ("_slice_dict", "_inc_dict"):
                if hasattr(item, name):
                    pending.append(getattr(item, name))
        else:
            raise PrefetchError("prefetch.unsupported_cpu_payload:" + type(item).__name__)


def _require(condition, reason):
    if not condition:
        raise ValueError("temporal_transport." + reason)


class _TensorPickler(pickle.Pickler):
    def __init__(self, stream, cap):
        super().__init__(stream, protocol=5)
        self.cap = cap
        self.tensor_bytes = 0

    def reducer_override(self, value):
        # Unlike persistent_id, this hook is skipped for exact builtin types;
        # large sidecars keep the fast C pickle path for ints/lists/dicts/etc.
        if not isinstance(value, torch.Tensor):
            return NotImplemented
        _require(type(value) is torch.Tensor and value.device.type == "cpu"
                 and value.layout == torch.strided and not value.requires_grad
                 and not value.is_conj() and not value.is_neg(),
                 "plain_cpu_input_tensor_required")
        dtype = str(value.dtype)
        _require(dtype in _DTYPES, "unsupported_dtype")
        size = value.numel() * value.element_size()
        _require(self.tensor_bytes + size <= self.cap, "tensor_bytes_exceed_cap")
        self.tensor_bytes += size
        # Flatten before byte view: scalar and zero-sized tensors are supported.
        raw = value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        return (_tensor_from_bytes, (dtype, tuple(value.shape), raw))


def _tensor_from_bytes(*args):
    # A bare pickle.loads must not accidentally accept this private format.
    raise ValueError("temporal_transport.private_decoder_required")


class _TensorUnpickler(pickle.Unpickler):
    def __init__(self, stream, cap):
        super().__init__(stream)
        self.cap = cap
        self.tensor_bytes = 0

    def find_class(self, module, name):
        if module == __name__ and name == "_tensor_from_bytes":
            return self._restore_tensor
        return super().find_class(module, name)

    def _restore_tensor(self, *args):
        _require(len(args) == 3, "invalid_tensor_token")
        dtype, shape, raw = args
        _require(type(dtype) is str and dtype in _DTYPES, "unsupported_dtype")
        _require(type(shape) is tuple and len(shape) <= 64
                 and all(type(n) is int and 0 <= n <= self.cap for n in shape),
                 "invalid_tensor_shape")
        size = math.prod(shape) * _ELEMENT_BYTES[dtype]
        _require(type(raw) is bytes and len(raw) == size, "tensor_byte_count_mismatch")
        _require(self.tensor_bytes + size <= self.cap, "tensor_bytes_exceed_cap")
        self.tensor_bytes += size
        if size:
            # Own a writable buffer; do not alias pickle bytes or another payload.
            tensor = torch.frombuffer(bytearray(raw), dtype=_DTYPES[dtype]).reshape(shape)
        else:
            tensor = torch.empty(shape, dtype=_DTYPES[dtype])
        return tensor


def pack_private_cpu_payload(value, *, cap):
    """Encode one private worker value under its serialized-byte reservation."""
    _require(type(cap) is int and cap > len(MAGIC), "invalid_cap")
    stream = _LimitedBuffer(cap)
    stream.write(MAGIC)
    _TensorPickler(stream, cap - len(MAGIC)).dump(value)
    return stream.getvalue()


def unpack_private_cpu_payload(payload, *, cap):
    """Decode only bounded bytes supplied by our worker; never call on files."""
    _require(type(cap) is int and cap > len(MAGIC), "invalid_cap")
    _require(type(payload) is bytes and len(payload) <= cap
             and payload.startswith(MAGIC), "invalid_payload")
    stream = io.BytesIO(payload)
    stream.seek(len(MAGIC))
    value = _TensorUnpickler(stream, cap - len(MAGIC)).load()
    _require(stream.read(1) == b"", "trailing_payload_bytes")
    return value
