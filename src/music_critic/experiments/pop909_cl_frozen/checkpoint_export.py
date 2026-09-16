"""Bounded, non-executing extraction of the historical model pickle prefix.

This is a separate export boundary, not a fallback in the frozen loader. It
supports only the protocol-2 layout written by corrected_training. A symbolic
stack identifies five complete root entries before optimizer_state. Nothing
after that boundary is interpreted or deserialized. The reduced ZIP and final
export both pass the unchanged CPU weights_only strict model loader.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import math
from pathlib import Path, PurePosixPath
import pickletools
import re
from tempfile import TemporaryDirectory
import zipfile

import torch

from .checkpoint import (
    CORRECTED_TRAINING_CHECKPOINT_SCHEMA,
    FrozenCheckpointError,
    _checked_artifact,
    create_model_only_payload,
    load_model_checkpoint,
)


EXPORT_SCHEMA = "CorrectedAnalysisGNNPrefixExportEvidence@1.0.0"
MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
MAX_PICKLE_BYTES = 4 * 1024 * 1024
MAX_MEMBER_BYTES = 64 * 1024 * 1024
MAX_MEMBERS = 4096
MAX_OPCODES = 200_000
MAX_MEMO = 20_000
_FIELDS = (
    "schema_version", "resolved_config", "model_contract_fingerprint",
    "model_state", "model_state_fingerprint",
)
_GLOBALS = frozenset({
    "collections OrderedDict", "torch._utils _rebuild_tensor_v2", "torch FloatStorage",
})
_MARK = object()


def _fail(reason: str) -> None:
    raise FrozenCheckpointError("checkpoint.export_" + reason, "unsupported historical prefix/archive")


@dataclass(eq=False)
class _Container:
    kind: str
    items: object


@dataclass(frozen=True)
class _Global:
    name: str


@dataclass(frozen=True)
class _Storage:
    key: str
    elements: int


@dataclass(frozen=True)
class _Tensor:
    storage: _Storage


def _primitive(value: object) -> bool:
    # Pickle memo references can encode exponentially many traversal paths in
    # a tiny DAG. Visit each completed container once, and reject cycles/depth.
    active: set[int] = set()
    completed: dict[int, bool] = {}

    def visit(item: object, depth: int) -> bool:
        if depth > 64:
            return False
        if item is None or type(item) in (str, bool, int):
            return True
        if type(item) is float:
            return math.isfinite(item)
        if not isinstance(item, _Container) or item.kind not in ("dict", "list", "tuple", "ordered"):
            return False
        identity = id(item)
        if identity in active:
            return False
        if identity in completed:
            return completed[identity]
        if len(active) + len(completed) >= MAX_MEMO:
            return False
        active.add(identity)
        values = item.items
        result = (
            all(type(k) in (str, int) and visit(v, depth + 1) for k, v in values.items())
            if isinstance(values, dict) else all(visit(v, depth + 1) for v in values)
        )
        active.remove(identity)
        completed[identity] = result
        return result

    return visit(value, 0)


def _extract_prefix(data: bytes, storage_sizes: dict[str, int]) -> tuple[bytes, dict[str, object]]:
    """Interpret only safe symbolic structure; never invoke pickle callables."""

    if len(data) > MAX_PICKLE_BYTES or not data.startswith(b"\x80\x02"):
        _fail("pickle_protocol_or_size")
    stack: list[object] = []
    memo: dict[int, object] = {}
    root: _Container | None = None
    referenced: dict[str, int] = {}
    globals_seen: set[str] = set()

    def pop() -> object:
        if not stack or stack[-1] is _MARK:
            _fail("pickle_stack")
        return stack.pop()

    def marked() -> list[object]:
        for index in range(len(stack) - 1, -1, -1):
            if stack[index] is _MARK:
                result = stack[index + 1:]
                del stack[index:]
                return result
        _fail("pickle_mark")

    def pairs(target: object, values: list[object]) -> None:
        if not isinstance(target, _Container) or target.kind not in ("dict", "ordered"):
            _fail("pickle_dictionary")
        if target is root or len(values) % 2:
            _fail("root_layout")
        for key, value in zip(values[::2], values[1::2]):
            if type(key) not in (str, int) or key in target.items:
                _fail("duplicate_or_invalid_key")
            target.items[key] = value

    try:
        for count, (opcode, arg, position) in enumerate(pickletools.genops(data), 1):
            if count > MAX_OPCODES:
                _fail("opcode_limit")
            name = opcode.name
            if name == "PROTO":
                if position != 0 or arg != 2:
                    _fail("pickle_protocol")
            elif name == "MARK":
                stack.append(_MARK)
            elif name in ("BINUNICODE", "SHORT_BINSTRING", "BININT", "BININT1", "BININT2", "LONG1", "LONG4", "BINFLOAT"):
                # BINSTRING is intentionally not supported: Py2 bytes semantics
                # are outside the historical Python3 contract.
                if name == "SHORT_BINSTRING":
                    _fail("pickle_opcode")
                stack.append(arg)
            elif name in ("NONE", "NEWTRUE", "NEWFALSE"):
                stack.append({"NONE": None, "NEWTRUE": True, "NEWFALSE": False}[name])
            elif name in ("EMPTY_DICT", "EMPTY_LIST", "EMPTY_TUPLE"):
                kind = {"EMPTY_DICT": "dict", "EMPTY_LIST": "list", "EMPTY_TUPLE": "tuple"}[name]
                value = _Container(kind, {} if kind == "dict" else [])
                if root is None:
                    if kind != "dict" or stack or position != 2:
                        _fail("root_layout")
                    root = value
                stack.append(value)
            elif name in ("BINPUT", "LONG_BINPUT"):
                if type(arg) is not int or arg < 0 or arg >= MAX_MEMO or arg in memo or not stack or stack[-1] is _MARK:
                    _fail("pickle_memo")
                memo[arg] = stack[-1]
            elif name in ("BINGET", "LONG_BINGET"):
                if arg not in memo:
                    _fail("pickle_memo")
                stack.append(memo[arg])
            elif name == "GLOBAL":
                if arg not in _GLOBALS:
                    _fail("forbidden_global")
                globals_seen.add(arg)
                stack.append(_Global(arg))
            elif name in ("TUPLE", "TUPLE1", "TUPLE2", "TUPLE3"):
                if name == "TUPLE":
                    values = marked()
                else:
                    values = [pop() for _ in range(int(name[-1]))][::-1]
                stack.append(_Container("tuple", values))
            elif name == "BINPERSID":
                value = pop()
                if not isinstance(value, _Container) or value.kind != "tuple" or len(value.items) != 5:
                    _fail("storage_reference")
                tag, dtype, key, location, elements = value.items
                if (tag != "storage" or dtype != _Global("torch FloatStorage")
                    or type(key) is not str or not re.fullmatch(r"[0-9]+", key)
                    or type(location) is not str or not re.fullmatch(r"cpu|cuda:[0-9]+", location)
                    or type(elements) is not int or elements < 0
                    or elements * 4 != storage_sizes.get(key)
                    or elements * 4 > MAX_MEMBER_BYTES):
                    _fail("storage_reference")
                referenced[key] = elements * 4
                stack.append(_Storage(key, elements))
            elif name == "REDUCE":
                args, function = pop(), pop()
                if not isinstance(function, _Global) or not isinstance(args, _Container) or args.kind != "tuple":
                    _fail("reduce_shape")
                if function.name == "collections OrderedDict" and not args.items:
                    stack.append(_Container("ordered", {}))
                elif function.name == "torch._utils _rebuild_tensor_v2":
                    values = args.items
                    if len(values) not in (6, 7):
                        _fail("tensor_reference")
                    storage, offset, shape, stride, grad, hooks = values[:6]
                    if (not isinstance(storage, _Storage) or type(offset) is not int or offset < 0
                        or not isinstance(shape, _Container) or shape.kind != "tuple"
                        or not isinstance(stride, _Container) or stride.kind != "tuple"
                        or len(shape.items) != len(stride.items) or len(shape.items) > 8
                        or grad is not False or not isinstance(hooks, _Container)
                        or hooks.kind != "ordered" or hooks.items
                        or (len(values) == 7 and values[6] is not None)):
                        _fail("tensor_reference")
                    expected_stride, elements = 1, 1
                    for dimension, actual_stride in zip(reversed(shape.items), reversed(stride.items)):
                        if (type(dimension) is not int or dimension <= 0
                            or type(actual_stride) is not int or actual_stride != expected_stride):
                            _fail("tensor_layout")
                        elements *= dimension
                        expected_stride *= dimension
                        if elements > MAX_MEMBER_BYTES // 4:
                            _fail("tensor_size")
                    if offset + elements > storage.elements:
                        _fail("tensor_storage_extent")
                    stack.append(_Tensor(storage))
                else:
                    _fail("forbidden_reduce")
            elif name in ("SETITEMS", "SETITEM"):
                values = marked() if name == "SETITEMS" else [pop(), pop()][::-1]
                if not stack:
                    _fail("pickle_stack")
                pairs(stack[-1], values)
            elif name in ("APPENDS", "APPEND"):
                values = marked() if name == "APPENDS" else [pop()]
                if not stack or not isinstance(stack[-1], _Container) or stack[-1].kind != "list":
                    _fail("pickle_list")
                stack[-1].items.extend(values)
            elif name == "BUILD":
                state = pop()
                if (not stack or not isinstance(stack[-1], _Container) or stack[-1].kind != "ordered"
                    or not isinstance(state, _Container) or state.kind != "dict"
                    or set(state.items) != {"_metadata"} or not _primitive(state)):
                    _fail("ordered_metadata")
            else:
                _fail("pickle_opcode")

            # Six root keys would leave root, MARK, five pairs, next key on
            # the stack. The next key opcode itself is excluded from output.
            if (len(stack) == 13 and stack[0] is root and stack[1] is _MARK
                and tuple(stack[2:12:2]) == _FIELDS):
                if stack[-1] != "optimizer_state":
                    _fail("root_field_order")
                values = stack[3:12:2]
                schema, config, contract_hash, state, state_hash = values
                if (schema != CORRECTED_TRAINING_CHECKPOINT_SCHEMA
                    or not isinstance(config, _Container) or config.kind != "dict" or not _primitive(config)
                    or any(type(x) is not str or not re.fullmatch(r"[0-9a-f]{64}", x) for x in (contract_hash, state_hash))
                    or not isinstance(state, _Container) or state.kind != "ordered" or not state.items
                    or any(type(k) is not str or not isinstance(v, _Tensor) for k, v in state.items.items())):
                    _fail("root_field_types")
                return data[:position] + b"u.", {
                    "prefix_cutoff_bytes": position,
                    "source_pickle_sha256": sha256(data).hexdigest(),
                    "prefix_fields": list(_FIELDS),
                    "source_model_contract_fingerprint": contract_hash,
                    "source_model_state_fingerprint": state_hash,
                    "referenced_storage_sizes": dict(sorted(referenced.items())),
                    "prefix_globals": sorted(globals_seen),
                    "tail_interpreted": False,
                }
        _fail("prefix_boundary_missing")
    except FrozenCheckpointError:
        raise
    except (ValueError, IndexError, KeyError, TypeError, OverflowError, RecursionError) as exc:
        raise FrozenCheckpointError("checkpoint.export_pickle_malformed", "invalid symbolic pickle structure") from exc


def _reduce_archive(content: bytes) -> tuple[bytes, dict[str, object]]:
    """Copy all non-pickle members exactly; bound and validate before reads."""

    if len(content) > MAX_ARCHIVE_BYTES:
        _fail("archive_size")
    try:
        with zipfile.ZipFile(BytesIO(content)) as source:
            infos = source.infolist()
            names = [item.filename for item in infos]
            if (not infos or len(infos) > MAX_MEMBERS or len(set(names)) != len(names)
                or sum(item.file_size for item in infos) > MAX_ARCHIVE_BYTES):
                _fail("archive_inventory")
            prefixes = set()
            for item in infos:
                path = PurePosixPath(item.filename)
                if (path.is_absolute() or ".." in path.parts or len(path.parts) < 2
                    or str(path) != item.filename or "\\" in item.filename or item.is_dir()
                    or item.flag_bits & 1 or item.file_size > MAX_MEMBER_BYTES
                    or item.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                    or (item.external_attr >> 16) & 0o170000 == 0o120000):
                    _fail("archive_member")
                prefixes.add(path.parts[0])
            if len(prefixes) != 1:
                _fail("archive_root")
            prefix = next(iter(prefixes)) + "/"
            pickle_name = prefix + "data.pkl"
            if pickle_name not in names or source.getinfo(pickle_name).file_size > MAX_PICKLE_BYTES:
                _fail("archive_pickle")
            allowed_meta = {"data.pkl", "byteorder", "version", ".data/serialization_id", ".format_version", ".storage_alignment"}
            storage_sizes = {}
            for item in infos:
                relative = item.filename[len(prefix):]
                if re.fullmatch(r"data/[0-9]+", relative):
                    storage_sizes[relative[5:]] = item.file_size
                elif relative not in allowed_meta:
                    _fail("archive_unknown_member")
            source_pickle = source.read(pickle_name)
            reduced_pickle, evidence = _extract_prefix(source_pickle, storage_sizes)
            copied = {}
            buffer = BytesIO()
            with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as output:
                for item in infos:
                    member = reduced_pickle if item.filename == pickle_name else source.read(item)
                    output.writestr(item.filename, member)
                    if item.filename != pickle_name:
                        copied[item.filename] = {"size_bytes": len(member), "sha256": sha256(member).hexdigest()}
            evidence.update({
                "prefix_pickle_sha256": sha256(reduced_pickle).hexdigest(),
                "copied_members": copied,
                "source_archive_members": len(infos),
            })
            return buffer.getvalue(), evidence
    except FrozenCheckpointError:
        raise
    except (zipfile.BadZipFile, RuntimeError, ValueError, OSError, EOFError) as exc:
        raise FrozenCheckpointError("checkpoint.export_archive_malformed", "invalid bounded ZIP archive") from exc


def export_model_only_checkpoint(
    source: str | Path,
    destination: str | Path,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
) -> dict[str, object]:
    """Write a new verified model-only artifact; preserve the immutable source.

    The caller owns experiment registration and selection/update provenance.
    No corpus access, optimizer restoration, inference or training occurs here.
    The destination is created exclusively after both strict loads succeed.
    """

    if type(expected_size_bytes) is not int or not 0 < expected_size_bytes <= MAX_ARCHIVE_BYTES:
        _fail("expected_size")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    content = _checked_artifact(source, expected_sha256, expected_size_bytes)
    reduced, evidence = _reduce_archive(content)
    with TemporaryDirectory(prefix="music-critic-model-export-") as temporary:
        prefix_path = Path(temporary) / "prefix.pt"
        prefix_path.write_bytes(reduced)
        loaded = load_model_checkpoint(prefix_path, expected_sha256=sha256(reduced).hexdigest(), expected_size_bytes=len(reduced))
        payload = create_model_only_payload(loaded.model)
        buffer = BytesIO()
        torch.save(payload, buffer, pickle_protocol=2)
        final_bytes = buffer.getvalue()
        final_path = Path(temporary) / "model-only.pt"
        final_path.write_bytes(final_bytes)
        verified = load_model_checkpoint(final_path, expected_sha256=sha256(final_bytes).hexdigest(), expected_size_bytes=len(final_bytes))
        if (loaded.evidence["model_state_fingerprint"] != verified.evidence["model_state_fingerprint"]
            or verified.evidence["model_state_fingerprint"] != evidence["source_model_state_fingerprint"]):
            _fail("export_state_mismatch")
    with destination.open("xb") as output:
        output.write(final_bytes)
    return {
        "schema_version": EXPORT_SCHEMA,
        "source_sha256": expected_sha256,
        "source_size_bytes": expected_size_bytes,
        "export_sha256": sha256(final_bytes).hexdigest(),
        "export_size_bytes": len(final_bytes),
        "reduced_archive_sha256": sha256(reduced).hexdigest(),
        "reduced_archive_size_bytes": len(reduced),
        "extraction": evidence,
        "source_verification": loaded.evidence,
        "export_verification": verified.evidence,
        "selection_update_fields_extracted": False,
        "optimizer_restored": False,
        "checkpoint_rng_applied": False,
    }
