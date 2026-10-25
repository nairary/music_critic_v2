"""Selective, streaming JSON metadata reader; no corpus access at import time.

This reads a top-level object keyed by clip IDs. Only explicitly allowed scalar
paths are materialized. Other values are lexically validated/skipped, never
passed to a JSON object decoder. Physical file bytes (including skipped music)
are necessarily read; this is not a claim of zero byte access to those fields.
Schema discovery reports keys/types, not values, and never enters music trees.
Exhaust the iterator to validate the whole document; closing a preview iterator
validates only its consumed prefix. No model, adapter, or legacy imports occur.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Iterable, Iterator, TextIO


VERSION = "music-overlap-metadata@1.0.0"
SCALAR_FIELDS = frozenset({
    "id", "uid", "hash", "clip_id", "song_id", "title", "name", "artist",
    "artist_name", "song_name", "composer", "composer_name", "subtitle",
    "url", "uri", "link", "permalink", "slug", "youtube_id", "YouTubeID",
    "youtubeId", "video_id", "ori_uid", "audio_path", "audio_tag", "split",
})
CONTAINERS = frozenset({"hooktheory", "youtube", "metadata", "meta"})
# Normalized names are rejected even in caller-supplied extension paths.
_MUSIC_KEYS = frozenset({
    "json", "jsonapi", "xml", "xmldata", "musicxml", "midi", "score", "notation",
    "notes", "chords", "annotations", "targets", "melody",
    "harmony", "harmonies", "encoded", "teacherencoded", "keys", "key",
    "meters", "tempos", "beats", "structure", "sections", "phrases", "alignment",
})
_STRING_SPECIAL = re.compile(r'["\\\x00-\x1f]')
_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\Z")


class MetadataJSONError(ValueError):
    """Malformed JSON, an unsafe projection path, or an explicit resource bound."""


@dataclass(frozen=True)
class MetadataRecord:
    clip_id: str
    metadata: dict[str, object]
    schema: dict[str, str]


def _music_key(key: str) -> bool:
    return re.sub(r"[^a-z0-9]", "", key.casefold()) in _MUSIC_KEYS


def _paths(values: Iterable[tuple[str, ...]]) -> set[tuple[str, ...]]:
    result = set()
    for value in values:
        if (not isinstance(value, tuple) or not value
                or any(not isinstance(key, str) or not key or _music_key(key)
                       for key in value)):
            raise MetadataJSONError("unsafe or malformed metadata path")
        result.add(value)
    return result


def _pointer(path: tuple[str, ...]) -> str:
    return "/" + "/".join(key.replace("~", "~0").replace("/", "~1") for key in path)


class _Lexer:
    def __init__(self, source: TextIO, chunk_size: int, max_scalar_chars: int):
        self.source = source
        self.chunk_size = chunk_size
        self.max_scalar_chars = max_scalar_chars
        self.buffer = ""
        self.position = 0
        self.eof = False

    def _fill(self) -> bool:
        if self.position < len(self.buffer):
            return True
        if self.eof:
            return False
        self.buffer = self.source.read(self.chunk_size)
        if not isinstance(self.buffer, str):
            raise MetadataJSONError("source must be a text stream")
        self.position = 0
        self.eof = not self.buffer
        return not self.eof

    def char(self) -> str:
        return self.buffer[self.position] if self._fill() else ""

    def take(self) -> str:
        value = self.char()
        if not value:
            raise MetadataJSONError("unexpected end of JSON")
        self.position += 1
        return value

    def whitespace(self) -> None:
        while self.char() and self.char() in " \r\n\t":
            self.position += 1

    def expect(self, value: str) -> None:
        self.whitespace()
        if self.take() != value:
            raise MetadataJSONError("unexpected JSON delimiter")

    def kind(self) -> str:
        self.whitespace()
        char = self.char()
        if char in {'"': 1, "{": 1, "[": 1}:
            return {'"': "string", "{": "object", "[": "array"}[char]
        if char in ("t", "f"):
            return "boolean"
        if char == "n":
            return "null"
        if char and char in "-0123456789":
            return "number"
        raise MetadataJSONError("invalid JSON value")

    def string(self, capture: bool) -> str | None:
        self.expect('"')
        parts = ['"'] if capture else None
        length = 0

        def append(value: str) -> None:
            nonlocal length
            if parts is not None:
                length += len(value)
                if length > self.max_scalar_chars:
                    raise MetadataJSONError("metadata string exceeds configured bound")
                parts.append(value)

        while self._fill():
            match = _STRING_SPECIAL.search(self.buffer, self.position)
            end = match.start() if match else len(self.buffer)
            append(self.buffer[self.position:end])
            self.position = end
            if not match:
                continue
            char = self.take()
            append(char)
            if char == '"':
                return json.loads("".join(parts)) if parts is not None else None
            if char != "\\":
                raise MetadataJSONError("unescaped control character in JSON string")
            escape = self.take()
            append(escape)
            if escape == "u":
                for _ in range(4):
                    digit = self.take()
                    if digit not in "0123456789abcdefABCDEF":
                        raise MetadataJSONError("invalid JSON unicode escape")
                    append(digit)
            elif escape not in '"\\/bfnrt':
                raise MetadataJSONError("invalid JSON string escape")
        raise MetadataJSONError("unterminated JSON string")

    def scalar(self, capture: bool) -> object:
        kind = self.kind()
        if kind == "string":
            return self.string(capture)
        if kind in ("object", "array"):
            raise MetadataJSONError("expected scalar")
        token = []
        while self.char() and self.char() not in " \r\n\t,]}":
            token.append(self.take())
            if len(token) > 1024:
                raise MetadataJSONError("JSON primitive exceeds 1024 characters")
        raw = "".join(token)
        if raw not in ("true", "false", "null") and not _NUMBER.fullmatch(raw):
            raise MetadataJSONError("invalid JSON primitive")
        if not capture:
            return None
        value = json.loads(raw)
        if isinstance(value, float) and not math.isfinite(value):
            raise MetadataJSONError("non-finite projected metadata number")
        return value

    def skip(self, depth: int = 0) -> None:
        if depth > 128:
            raise MetadataJSONError("JSON nesting exceeds 128 levels")
        kind = self.kind()
        if kind not in ("object", "array"):
            self.scalar(False)
            return
        opening, closing = ("{", "}") if kind == "object" else ("[", "]")
        self.expect(opening)
        self.whitespace()
        if self.char() == closing:
            self.take()
            return
        while True:
            if kind == "object":
                self.string(False)
                self.expect(":")
            self.skip(depth + 1)
            self.whitespace()
            char = self.take()
            if char == closing:
                return
            if char != ",":
                raise MetadataJSONError("invalid JSON container delimiter")


def iter_metadata_records(
    source: str | Path | TextIO,
    *,
    include_schema: bool = False,
    extra_scalar_paths: Iterable[tuple[str, ...]] = (),
    schema_paths: Iterable[tuple[str, ...]] = (),
    chunk_size: int = 65536,
    max_scalar_chars: int = 1048576,
) -> Iterator[MetadataRecord]:
    """Project clip-keyed JSON without constructing any excluded subtree.

    ``metadata`` preserves the nesting of allowed scalar paths. By default these
    are SCALAR_FIELDS at record level and directly inside CONTAINERS. Explicit
    extra_scalar_paths add exact scalar paths, never object/array values.
    ``schema`` maps JSON pointers to shallow JSON types, with no field values.
    It covers the record and default containers; schema_paths explicitly enables
    shallow key discovery in additional object paths. Music paths are forbidden.
    No split is silently filtered: callers bind current split metadata separately.
    Duplicate clip IDs or keys in projected/discovered objects fail closed;
    excluded subtrees are syntax-checked but not duplicate-key audited.
    Caller-owned streams stay open. Files opened here close with the iterator.
    """
    if type(chunk_size) is not int or chunk_size < 1:
        raise MetadataJSONError("chunk_size must be a positive integer")
    if type(max_scalar_chars) is not int or max_scalar_chars < 1:
        raise MetadataJSONError("max_scalar_chars must be a positive integer")
    scalar_paths = {(name,) for name in SCALAR_FIELDS}
    scalar_paths |= {(container, name) for container in CONTAINERS for name in SCALAR_FIELDS}
    scalar_paths |= _paths(extra_scalar_paths)
    discover = {(), *((name,) for name in CONTAINERS)} | _paths(schema_paths)
    needed = scalar_paths | (discover if include_schema else set())
    descend = {path[:length] for path in needed for length in range(1, len(path))}
    if include_schema:
        descend |= discover

    def project(lexer: _Lexer, prefix: tuple[str, ...], schema: dict[str, str]) -> dict:
        if len(prefix) > 128:
            raise MetadataJSONError("projection nesting exceeds 128 levels")
        lexer.expect("{")
        result = {}
        seen = set()
        lexer.whitespace()
        if lexer.char() == "}":
            lexer.take()
            return result
        while True:
            key = lexer.string(True)
            if key in seen:
                raise MetadataJSONError("duplicate key in metadata object")
            seen.add(key)
            lexer.expect(":")
            path = (*prefix, key)
            kind = lexer.kind()
            if _music_key(key):
                lexer.skip()
            else:
                if include_schema and prefix in discover:
                    schema[_pointer(path)] = kind
                if path in scalar_paths and kind not in ("object", "array"):
                    result[key] = lexer.scalar(True)
                elif path in descend and kind == "object":
                    child = project(lexer, path, schema)
                    if child:
                        result[key] = child
                else:
                    lexer.skip()
            lexer.whitespace()
            char = lexer.take()
            if char == "}":
                return result
            if char != ",":
                raise MetadataJSONError("invalid metadata object delimiter")

    context = nullcontext(source) if hasattr(source, "read") else open(source, encoding="utf-8")
    with context as handle:
        lexer = _Lexer(handle, chunk_size, max_scalar_chars)
        lexer.expect("{")
        seen_ids = set()
        lexer.whitespace()
        if lexer.char() != "}":
            while True:
                clip_id = lexer.string(True)
                if clip_id in seen_ids:
                    raise MetadataJSONError("duplicate clip ID")
                seen_ids.add(clip_id)
                lexer.expect(":")
                schema = {}
                metadata = project(lexer, (), schema)
                yield MetadataRecord(clip_id, metadata, schema)
                lexer.whitespace()
                char = lexer.take()
                if char == "}":
                    break
                if char != ",":
                    raise MetadataJSONError("invalid top-level object delimiter")
        else:
            lexer.take()
        lexer.whitespace()
        if lexer.char():
            raise MetadataJSONError("trailing JSON content")
