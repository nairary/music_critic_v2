"""Synthetic identity metadata only; no real corpus payload reads."""

import importlib.util
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from xml.sax.saxutils import escape
from zipfile import ZipFile


spec = importlib.util.spec_from_file_location(
    "project_jazz_catalogs", Path(__file__).parents[1] / "scripts/project_jazz_catalogs.py"
)
catalog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(catalog)


class WjdProjectionTests(unittest.TestCase):
    def test_cli_rejects_changed_source_before_publishing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "source.json"
            path.write_text(json.dumps([{"melid": 1, "compid": 2, "title": "Synthetic", "performer": None}]))
            command = [sys.executable, str(Path(catalog.__file__)), "wjd", "--snapshot", str(path),
                       "--output-root", str(root / "result"), "--expected-snapshot-sha256", "0" * 64]
            failed = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("SHA256 mismatch", failed.stderr)
            self.assertFalse((root / "result" / "catalog.jsonl").exists())
            command[-1] = hashlib.sha256(path.read_bytes()).hexdigest()
            success = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(success.returncode, 0, success.stderr)
            repeated = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn("overwrite", repeated.stderr)

    def test_authorizer_denies_music_columns_and_tables(self):
        self.assertEqual(catalog.wjd_authorizer(sqlite3.SQLITE_READ, "melody", "pitch", "main", None), sqlite3.SQLITE_DENY)
        self.assertEqual(catalog.wjd_authorizer(sqlite3.SQLITE_READ, "composition_info", "form", "main", None), sqlite3.SQLITE_DENY)
        self.assertEqual(catalog.wjd_authorizer(sqlite3.SQLITE_READ, "composition_info", "composer", "main", None), sqlite3.SQLITE_OK)
        self.assertEqual(catalog.wjd_authorizer(sqlite3.SQLITE_UPDATE, "composition_info", "composer", "main", None), sqlite3.SQLITE_DENY)

    def test_missing_title_and_creator_are_preserved_and_duplicate_ids_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.json"
            row = {"melid": 1, "compid": 2, "title": None, "performer": None}
            path.write_text(json.dumps([row]))
            result = catalog.project_wjd(path)
            self.assertEqual(result["rows"][0]["titles"], [])
            self.assertEqual(result["rows"][0]["creators"], [])
            path.write_text(json.dumps([row, row]))
            with self.assertRaisesRegex(ValueError, "Duplicate WJD melid"):
                catalog.project_wjd(path)

    def test_composer_and_performer_are_distinct_and_solos_remain_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshot.json"
            rows = [
                {"melid": 1, "compid": 10, "title": "Été", "performer": "Soloist"},
                {"melid": 2, "compid": 10, "title": "Été", "performer": None},
            ]
            snapshot.write_text(json.dumps(rows), encoding="utf-8")
            database = root / "fixture.db"
            with sqlite3.connect(database) as connection:
                connection.executescript(
                    "CREATE TABLE solo_info(melid, compid, title, performer);"
                    "CREATE TABLE composition_info(compid, title, composer, form);"
                    "CREATE TABLE melody(pitch);"
                    "INSERT INTO composition_info VALUES(10, 'Été', 'Composer', 'UNREAD');"
                )
                connection.executemany(
                    "INSERT INTO solo_info VALUES(?, ?, ?, ?)",
                    [(x["melid"], x["compid"], x["title"], x["performer"]) for x in rows],
                )
            result = catalog.project_wjd(snapshot, sqlite_path=database)
            self.assertEqual([x["id"] for x in result["rows"]], ["1", "2"])
            self.assertEqual(result["rows"][0]["creators"], [
                {"field": "performer", "value": "Soloist"},
                {"field": "composer", "value": "Composer"},
            ])
            self.assertEqual(result["rows"][1]["creators"], [{"field": "composer", "value": "Composer"}])
            self.assertTrue(all("melody" not in q for q in result["sqlite_queries"]))
            self.assertNotIn("UNREAD", json.dumps(result))


def synthetic_workbook(path, *, duplicate=False, formula=False, extra_member=False):
    strings = ["ID", "Piece", "Album", "Composer", "Arranger", "Note payload", "合成标题", "Collection", "Composer", "Arranger"]
    shared = "".join(f"<si><t>{escape(value)}</t></si>" for value in strings)
    # Invalid XML character reference proves the unselected shared string is not decoded.
    shared += "<si><t>&#0;</t></si>"
    header = "".join(f'<c r="{column}1" t="s"><v>{i}</v></c>' for i, column in enumerate("ABCDEF"))
    def row(number, value):
        return f'<row r="{number}"><c r="A{number}"><v>{value}</v></c>' + "".join(
            f'<c r="{column}{number}" t="s">' + ("<f>1+1</f>" if formula and column == "B" else "") + f'<v>{index}</v></c>'
            for column, index in zip("BCDEF", [6, 7, 8, 9, 10])
        ) + "</row>"
    contents = {
        "xl/workbook.xml": '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Catalog" sheetId="1" r:id="rId1"/></sheets></workbook>',
        "xl/_rels/workbook.xml.rels": '<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>',
        "xl/sharedStrings.xml": f'<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">{shared}</sst>',
        "xl/worksheets/sheet1.xml": f'<worksheet><sheetData><row r="1">{header}</row>{row(2, 1)}{row(3, 1) if duplicate else ""}</sheetData></worksheet>',
    }
    if extra_member:
        contents["../unsafe.xml"] = "ignored"
    with ZipFile(path, "w") as archive:
        for name, text in contents.items():
            archive.writestr(name, text)
    return path


class ChordWorkbookTests(unittest.TestCase):
    def test_writer_retains_integrity_receipt_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = {"rows": [{"dataset": "fixture", "id": "1", "titles": [], "creators": []}], "sources": {}, "musical_payload_read": False}
            catalog.write_catalog(result, root, command=["synthetic"])
            original = (root / "catalog.jsonl").read_bytes()
            receipt = json.loads((root / "receipt.json").read_text())
            self.assertEqual(receipt["row_count"], 1)
            self.assertEqual(receipt["catalog"]["sha256"], hashlib.sha256(original).hexdigest())
            with self.assertRaisesRegex(ValueError, "overwrite"):
                catalog.write_catalog(result, root, command=["synthetic"])
            self.assertEqual((root / "catalog.jsonl").read_bytes(), original)

    def test_projection_requires_complete_tree_and_binds_workbook_git_blob(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workbook = synthetic_workbook(root / "synthetic.xlsx")
            raw = workbook.read_bytes()
            sha = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
            tree = {"sha": "b" * 40, "truncated": False, "tree": [
                {"path": "Chord_Jazzification_Dataset/1.xlsx", "type": "blob", "sha": "c" * 40, "size": 100},
                {"path": "Chord_Jazzification_Dataset/corpus.xlsx", "type": "blob", "sha": sha, "size": len(raw)},
            ]}
            tree_path = root / "tree.json"
            tree_path.write_text(json.dumps(tree))
            result = catalog.project_chord(tree_path, workbook, upstream_commit="a" * 40, expected_count=1)
            item = result["rows"][0]
            self.assertEqual(item["id"], "1")
            self.assertEqual(item["titles"], [{"field": "Piece", "value": "合成标题"}])
            self.assertEqual(item["creators"], [{"field": "composer", "value": "Composer"}, {"field": "arranger", "value": "Arranger"}])
            self.assertEqual(item["source_metadata"]["identity_cells"], ["A2", "B2", "C2", "D2", "E2"])
            with self.assertRaisesRegex(ValueError, "complete expected work IDs"):
                catalog.project_chord(tree_path, workbook, upstream_commit="a" * 40, expected_count=2)
            tree["truncated"] = True
            tree_path.write_text(json.dumps(tree))
            with self.assertRaisesRegex(ValueError, "truncated"):
                catalog.project_chord(tree_path, workbook, upstream_commit="a" * 40, expected_count=1)
            tree["truncated"] = False
            tree["tree"][1]["sha"] = "0" * 40
            tree_path.write_text(json.dumps(tree))
            with self.assertRaisesRegex(ValueError, "Git blob"):
                catalog.project_chord(tree_path, workbook, upstream_commit="a" * 40, expected_count=1)

    def test_only_identity_columns_and_referenced_strings_are_decoded(self):
        with tempfile.TemporaryDirectory() as directory:
            source = synthetic_workbook(Path(directory) / "synthetic.xlsx")
            result = catalog.read_chord_workbook(source)
            self.assertEqual(result["rows"], [{"ID": "1", "Piece": "合成标题", "Album": "Collection", "Composer": "Composer", "Arranger": "Arranger", "row_number": 2}])
            self.assertEqual(result["skipped_columns"], ["F"])
            self.assertNotIn(10, result["decoded_shared_string_ids"])

    def test_duplicate_ids_formulas_and_unsafe_paths_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            for setting, error in [("duplicate", "Duplicate"), ("formula", "Formula"), ("extra_member", "Unsafe")]:
                source = synthetic_workbook(Path(directory) / f"{setting}.xlsx", **{setting: True})
                with self.subTest(setting=setting), self.assertRaisesRegex(ValueError, error):
                    catalog.read_chord_workbook(source)


if __name__ == "__main__":
    unittest.main()
