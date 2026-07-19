"""Bounded-memory JSON object and JSONL readers for production adapters."""

from __future__ import annotations

from decimal import Decimal
import json
from os import PathLike
from pathlib import Path
from typing import Any, Iterator


__all__ = ["JSONStreamError", "find_object_record", "iter_jsonl", "iter_object_records"]


class JSONStreamError(ValueError):
    """Raised when a streamed JSON source violates the supported shape."""


class _IncrementalJSON:
    def __init__(self, path: Path, *, chunk_size: int = 1024 * 1024) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.path = path
        self.handle = path.open("r", encoding="utf-8")
        self.chunk_size = chunk_size
        self.buffer = ""
        self.position = 0
        self.eof = False
        self.decoder = json.JSONDecoder(parse_float=Decimal)

    def close(self) -> None:
        self.handle.close()

    def _fill(self) -> bool:
        if self.eof:
            return False
        if self.position:
            self.buffer = self.buffer[self.position :]
            self.position = 0
        chunk = self.handle.read(self.chunk_size)
        if chunk:
            self.buffer += chunk
            return True
        self.eof = True
        return False

    def skip_space(self) -> None:
        while True:
            while self.position < len(self.buffer) and self.buffer[self.position].isspace():
                self.position += 1
            if self.position < len(self.buffer) or not self._fill():
                return

    def peek(self) -> str:
        self.skip_space()
        return self.buffer[self.position] if self.position < len(self.buffer) else ""

    def consume(self, expected: str) -> None:
        actual = self.peek()
        if actual != expected:
            raise JSONStreamError(
                f"{self.path}: expected {expected!r}, found {actual or '<eof>'!r}"
            )
        self.position += 1

    def value(self) -> Any:
        self.skip_space()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.position)
            except json.JSONDecodeError as exc:
                if self._fill():
                    continue
                raise JSONStreamError(f"{self.path}: malformed JSON: {exc}") from exc
            self.position = end
            return value


def _path(value: str | PathLike[str]) -> Path:
    try:
        path = Path(value)
    except (TypeError, ValueError) as exc:
        raise JSONStreamError("JSON source path must be string-like") from exc
    if not path.is_file():
        raise JSONStreamError(f"JSON source is not a readable file: {path}")
    return path


def iter_object_records(
    path: str | PathLike[str], *, chunk_size: int = 1024 * 1024
) -> Iterator[tuple[str, Any]]:
    """Yield a complete top-level object or legacy object fragment incrementally."""

    source = _path(path)
    stream = _IncrementalJSON(source, chunk_size=chunk_size)
    try:
        has_braces = stream.peek() == "{"
        if has_braces:
            stream.consume("{")
        while True:
            marker = stream.peek()
            if not marker:
                if has_braces:
                    raise JSONStreamError(f"{source}: unterminated top-level object")
                break
            if has_braces and marker == "}":
                stream.consume("}")
                if stream.peek():
                    raise JSONStreamError(f"{source}: trailing content after top-level object")
                break
            key = stream.value()
            if not isinstance(key, str):
                raise JSONStreamError(f"{source}: top-level record key must be a string")
            stream.consume(":")
            yield key, stream.value()
            marker = stream.peek()
            if marker == ",":
                stream.consume(",")
                if not stream.peek() or (has_braces and stream.peek() == "}"):
                    raise JSONStreamError(f"{source}: trailing comma in top-level object")
                continue
            if has_braces and marker == "}":
                continue
            if not has_braces and not marker:
                break
            raise JSONStreamError(
                f"{source}: unexpected top-level delimiter {marker or '<eof>'!r}"
            )
    except OSError as exc:
        raise JSONStreamError(f"{source}: failed while reading JSON: {exc}") from exc
    finally:
        stream.close()


def find_object_record(path: str | PathLike[str], record_id: str) -> Any:
    """Return one requested record while detecting duplicate requested IDs."""

    if not isinstance(record_id, str) or not record_id:
        raise JSONStreamError("record_id must be a non-empty string")
    found: Any = None
    count = 0
    for key, record in iter_object_records(path):
        if key == record_id:
            count += 1
            if count == 1:
                found = record
    if count == 0:
        raise JSONStreamError(f"record {record_id!r} was not found in {Path(path)}")
    if count > 1:
        raise JSONStreamError(
            f"record {record_id!r} occurs {count} times in {Path(path)}"
        )
    return found


def iter_jsonl(path: str | PathLike[str]) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield JSONL objects with Decimal-preserving numeric parsing."""

    source = _path(path)
    decoder = json.JSONDecoder(parse_float=Decimal)
    try:
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = decoder.decode(line)
                except json.JSONDecodeError as exc:
                    raise JSONStreamError(
                        f"{source}:{line_number}: malformed JSON: {exc}"
                    ) from exc
                if not isinstance(value, dict):
                    raise JSONStreamError(
                        f"{source}:{line_number}: JSONL row must be an object"
                    )
                yield line_number, value
    except OSError as exc:
        raise JSONStreamError(f"{source}: failed while reading JSONL: {exc}") from exc
