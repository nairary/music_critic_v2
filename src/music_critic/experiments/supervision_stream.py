"""Read an authenticated JSON object's large arrays one element at a time.

Authentication and admission belong to the caller. Exhaust the iterator to
validate the complete document: early rows may precede a malformed later row.
Only selected top-level arrays stream; each other value is decoded in full.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, TextIO


_DEFAULT_ARRAYS = frozenset({'notes', 'harmonies', 'note_contexts', 'objects', 'relations'})
_WHITESPACE = ' \t\r\n'


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON object key: {key!r}')
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f'non-JSON constant: {value}')


class _Reader:
    def __init__(self, handle: TextIO, chunk_size: int):
        self.handle = handle
        self.chunk_size = chunk_size
        self.buffer = ''
        self.position = 0
        self.eof = False

    def _fill(self):
        if self.eof:
            return False
        chunk = self.handle.read(self.chunk_size)
        self.buffer += chunk
        self.eof = not chunk
        return bool(chunk)

    def _compact(self):
        self.buffer = self.buffer[self.position:]
        self.position = 0

    def peek(self):
        """Next non-whitespace character, or the empty string at EOF."""
        while True:
            while self.position < len(self.buffer):
                char = self.buffer[self.position]
                if char not in _WHITESPACE:
                    return char
                self.position += 1
            self._compact()
            if not self._fill():
                return ''

    def expect(self, char):
        if self.peek() != char:
            raise ValueError(f'expected JSON punctuation {char!r}')
        self.position += 1

    def value(self):
        first = self.peek()
        if not first:
            raise ValueError('truncated JSON value')
        self._compact()
        index = 0
        depth = 0
        quoted = False
        escaped = False
        structured = first in '[{'
        string = first == '"'
        # Find a complete lexical boundary before decoding. In particular, a
        # number ending at a chunk edge may continue with digits, '.' or 'e'.
        while True:
            if index == len(self.buffer):
                if self._fill():
                    continue
                if structured or string or index == 0:
                    raise ValueError('truncated JSON value')
                break
            char = self.buffer[index]
            if not structured and not string:
                if char in _WHITESPACE + ',]}':
                    break
                index += 1
                continue
            index += 1
            if quoted:
                if escaped:
                    escaped = False
                elif char == '\\':
                    escaped = True
                elif char == '"':
                    quoted = False
                    if string:
                        break
            elif char == '"':
                quoted = True
            elif char in '[{':
                depth += 1
            elif char in ']}':
                depth -= 1
                if depth == 0:
                    break
        value = json.loads(self.buffer[:index], object_pairs_hook=_unique_object,
                           parse_constant=_reject_constant)
        self.position = index
        return value


def iter_top_object(path, streamed_arrays=_DEFAULT_ARRAYS, chunk_size=65536) -> Iterator[tuple[str, object]]:
    """Yield ``(key, value)`` pairs, flattening selected top-level arrays.

    Empty selected arrays yield no rows. All object keys, including nested
    keys, must be unique. Buffer storage is bounded by a single decoded value
    plus read chunks; top-level whitespace and consumed rows are discarded.
    UTF-8 input is opened lazily and closed on exhaustion or generator.close().
    """
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError('chunk_size must be a positive integer')
    streamed_arrays = frozenset(streamed_arrays)
    if any(not isinstance(key, str) for key in streamed_arrays):
        raise ValueError('streamed array keys must be strings')
    with Path(path).open('r', encoding='utf-8') as handle:
        reader = _Reader(handle, chunk_size)
        reader.expect('{')
        seen = set()
        if reader.peek() != '}':
            while True:
                if reader.peek() != '"':
                    raise ValueError('JSON object key must be a string')
                key = reader.value()
                if key in seen:
                    raise ValueError(f'duplicate top-level JSON key: {key!r}')
                seen.add(key)
                reader.expect(':')
                if key in streamed_arrays:
                    reader.expect('[')
                    if reader.peek() != ']':
                        while True:
                            yield key, reader.value()
                            if reader.peek() == ']':
                                break
                            reader.expect(',')
                    reader.expect(']')
                else:
                    yield key, reader.value()
                if reader.peek() == '}':
                    break
                reader.expect(',')
        reader.expect('}')
        if reader.peek():
            raise ValueError('trailing data after JSON object')
