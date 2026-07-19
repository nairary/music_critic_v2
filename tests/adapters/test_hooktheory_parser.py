from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from music_critic.adapters._json_stream import (
    JSONStreamError,
    find_object_record,
    iter_jsonl,
    iter_object_records,
)


def test_complete_object_preserves_decimal_lexemes_with_tiny_chunks(tmp_path: Path) -> None:
    path = tmp_path / "records.json"
    path.write_text('{"a":{"beat":4.50},"b":{"beat":1}}', encoding="utf-8")
    records = dict(iter_object_records(path, chunk_size=3))
    assert records["a"]["beat"] == Decimal("4.50")
    assert str(records["a"]["beat"]) == "4.50"
    assert records["b"]["beat"] == 1


def test_legacy_object_fragment_is_supported(tmp_path: Path) -> None:
    path = tmp_path / "fragment.json"
    path.write_text('"a":{"value":1.25},\n"b":{"value":2}', encoding="utf-8")
    assert [key for key, _ in iter_object_records(path, chunk_size=2)] == ["a", "b"]


def test_lookup_reads_deterministically_and_rejects_duplicate_requested_id(
    tmp_path: Path,
) -> None:
    path = tmp_path / "fragment.json"
    path.write_text('"x":{"n":1},"x":{"n":2}', encoding="utf-8")
    with pytest.raises(JSONStreamError, match="occurs 2 times"):
        find_object_record(path, "x")


def test_lookup_reports_missing_record(tmp_path: Path) -> None:
    path = tmp_path / "records.json"
    path.write_text('{"a":1}', encoding="utf-8")
    with pytest.raises(JSONStreamError, match="was not found"):
        find_object_record(path, "missing")


@pytest.mark.parametrize(
    "payload",
    [
        '{"a":1',
        '{"a":1,}',
        '{"a":1} trailing',
        '"a":1,',
        '[1,2,3]',
    ],
)
def test_malformed_sources_raise_clear_errors(tmp_path: Path, payload: str) -> None:
    path = tmp_path / "bad.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(JSONStreamError):
        list(iter_object_records(path, chunk_size=2))


def test_jsonl_preserves_decimal_and_rejects_non_objects(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"duration":0.10}\n\n[1]\n', encoding="utf-8")
    rows = iter_jsonl(path)
    line, row = next(rows)
    assert line == 1
    assert row["duration"] == Decimal("0.10")
    with pytest.raises(JSONStreamError, match="row must be an object"):
        next(rows)
