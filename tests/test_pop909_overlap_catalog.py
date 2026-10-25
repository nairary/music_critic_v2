"""Synthetic XLSX fixtures only; no real corpus reads."""

import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from xml.sax.saxutils import escape
from zipfile import ZipFile

import pytest

spec = importlib.util.spec_from_file_location(
    "pop909_overlap_catalog", Path(__file__).parents[1] / "scripts/pop909_overlap_catalog.py"
)
catalog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(catalog)


def workbook(path, data_rows, *, shared=False, extra_members=None):
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    strings = ["song_id", "name", "artist", "modify_times"]
    if shared:
        header = "".join(f'<c r="{col}1" t="s"><v>{i}</v></c>' for i, col in enumerate("ABCD"))
    else:
        header = "".join(
            f'<c r="{col}1" t="inlineStr"><is><t>{text}</t></is></c>'
            for col, text in zip("ABCD", strings)
        )
    members = {
        "xl/workbook.xml": f'<workbook xmlns="{ns}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="POP909" sheetId="1" r:id="rId1"/></sheets></workbook>',
        "xl/_rels/workbook.xml.rels": '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Target="/xl/worksheets/sheet1.xml" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"/></Relationships>',
        "xl/worksheets/sheet1.xml": f'<worksheet xmlns="{ns}"><sheetData><row r="1">{header}</row>{data_rows}</sheetData></worksheet>',
    }
    if shared:
        members["xl/sharedStrings.xml"] = f'<sst xmlns="{ns}">' + "".join(f"<si><t>{x}</t></si>" for x in strings) + "</sst>"
    members.update(extra_members or {})
    with ZipFile(path, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return path


def row(number=2, song_id="1", title="合成标题", artist="Synthetic artist"):
    return (
        f'<row r="{number}"><c r="A{number}"><v>{song_id}</v></c>'
        f'<c r="B{number}" t="inlineStr"><is><t>{escape(title)}</t></is></c>'
        f'<c r="C{number}" t="inlineStr"><is><t>{escape(artist)}</t></is></c></row>'
    )


def test_projection_preserves_unicode_raw_metadata_and_row_provenance(tmp_path):
    path = workbook(tmp_path / "synthetic.xlsx", row())
    result = catalog.read_catalog(path, source_url="https://example.invalid/catalog", upstream_commit="fixture")
    item = result["rows"][0]
    assert item["id"] == "001"
    assert item["dataset"] == "pop909"
    assert item["titles"] == [{"field": "name", "value": "合成标题"}]
    assert item["creators"] == [{"field": "artist", "value": "Synthetic artist"}]
    assert item["source_metadata"] == {"song_id": "1", "name": "合成标题", "artist": "Synthetic artist", "modify_times": None}
    assert item["provenance"]["row"] == 2
    assert item["provenance"]["sheet"] == "POP909"
    assert item["provenance"]["source_url"] == "https://example.invalid/catalog"


def test_shared_strings_and_missing_creator_remain_distinct(tmp_path):
    path = workbook(tmp_path / "shared.xlsx", row(song_id="007.0", artist=""), shared=True)
    result = catalog.read_catalog(path, source_url="fixture", upstream_commit="fixture")
    assert result["rows"][0]["id"] == "007"
    assert result["rows"][0]["creators"] == []
    assert result["rows"][0]["source_metadata"]["artist"] == ""
    assert result["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("song_id", ["0", "910", "-1", "1.5", "../../1"])
def test_invalid_identifiers_are_not_silently_normalized(tmp_path, song_id):
    path = workbook(tmp_path / "invalid.xlsx", row(song_id=song_id))
    with pytest.raises(ValueError, match="song ID"):
        catalog.read_catalog(path, source_url="fixture", upstream_commit="fixture")


def test_duplicate_ids_and_absent_titles_fail_closed(tmp_path):
    duplicate = workbook(tmp_path / "duplicate.xlsx", row() + row(number=3, song_id="001"))
    with pytest.raises(ValueError, match="Duplicate POP909 song ID"):
        catalog.read_catalog(duplicate, source_url="fixture", upstream_commit="fixture")
    empty_title = workbook(tmp_path / "empty.xlsx", row(title=""))
    with pytest.raises(ValueError, match="Missing POP909 title"):
        catalog.read_catalog(empty_title, source_url="fixture", upstream_commit="fixture")


def test_formulas_and_unsafe_archive_paths_are_rejected(tmp_path):
    formula = workbook(tmp_path / "formula.xlsx", row().replace("<v>1</v>", "<f>1+1</f><v>2</v>"))
    with pytest.raises(ValueError, match="Formula"):
        catalog.read_catalog(formula, source_url="fixture", upstream_commit="fixture")
    unsafe = workbook(tmp_path / "unsafe.xlsx", row(), extra_members={"../escape": "ignored"})
    with pytest.raises(ValueError, match="Unsafe workbook member"):
        catalog.read_catalog(unsafe, source_url="fixture", upstream_commit="fixture")


def test_cli_binds_source_bytes_and_does_not_overwrite(tmp_path):
    path = workbook(tmp_path / "source.xlsx", row())
    output, receipt = tmp_path / "rows.jsonl", tmp_path / "receipt.json"
    command = [
        sys.executable, str(Path(catalog.__file__)), "--input", str(path),
        "--output", str(output), "--receipt", str(receipt),
        "--source-url", "https://example.invalid/synthetic", "--upstream-commit", "synthetic",
        "--expected-sha256", "0" * 64,
    ]
    failure = subprocess.run(command, capture_output=True, text=True)
    assert failure.returncode != 0 and "SHA256 mismatch" in failure.stderr
    assert not output.exists() and not receipt.exists()
    command[-1] = hashlib.sha256(path.read_bytes()).hexdigest()
    success = subprocess.run(command, capture_output=True, text=True)
    assert success.returncode == 0, success.stderr
    original_output = output.read_bytes()
    saved = json.loads(receipt.read_text())
    assert saved["row_count"] == 1
    assert saved["output_sha256"] == hashlib.sha256(original_output).hexdigest()
    assert saved["musical_payload_read"] is False
    retry = subprocess.run(command, capture_output=True, text=True)
    assert retry.returncode != 0 and "Refuse to overwrite" in retry.stderr
    assert output.read_bytes() == original_output
