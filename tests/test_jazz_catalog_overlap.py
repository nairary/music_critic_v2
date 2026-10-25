"""Synthetic metadata fixtures test retrieval mechanics, not work identity."""

import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "jazz_overlap", Path(__file__).parents[1] / "scripts/audit_jazz_catalog_overlap.py"
)
overlap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(overlap)


def record(identity, title, creators=(), **extra):
    return {
        "dataset": "synthetic", "id": identity,
        "titles": [{"field": "title", "value": title}],
        "creators": [{"field": field, "value": value} for field, value in creators],
        "source_metadata": {"source_url": "https://example.invalid/synthetic"},
        **extra,
    }


def catalog(root, name, rows):
    path = root / (name + ".jsonl")
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return path


class JazzCatalogOverlapTests(unittest.TestCase):
    def test_projected_plural_creator_field_keeps_composer_role(self):
        from scripts.project_jazz_web_metadata import catalog_row, iter_jht_metadata
        projection = next(iter_jht_metadata(io.StringIO(
            '[{"title":"Song","composers":"Writer","trees":[]}]')))
        left = catalog_row("jht", "1", projection, {})
        pairs = overlap.match_catalogs([left], [record("r", "Song", [("composer", "Writer")])])
        evidence = pairs[0]["creator_field_evidence"][0]
        self.assertEqual(evidence["left_field"], "/composers")
        self.assertEqual(evidence["left_role"], "composer")
        self.assertTrue(evidence["same_role"])
        self.assertEqual(overlap._summarize(pairs, searchable=True)["exact_same_role_creator"]["pairs"], 1)

    def test_variants_have_explicit_operations_and_exact_lane_stays_separate(self):
        cases = [
            ("Girl from Ipanema, The", "The Girl from Ipanema", "move_trailing_article"),
            ("The Nearness of You", "Nearness of You", "remove_leading_article"),
            ("Café", "Cafe", "accent_fold_latin"),
            ("Autumn Leaves (Piano Arrangement)", "Autumn Leaves", "remove_bracketed_marker"),
            ("The Café [Live Version]", "Cafe", "remove_bracketed_marker"),
        ]
        for left_title, right_title, operation in cases:
            with self.subTest(left=left_title, right=right_title):
                pairs = overlap.match_catalogs([record("l", left_title)], [record("r", right_title)])
                self.assertEqual(len(pairs), 1)
                self.assertEqual(pairs[0]["retrieval_lanes"], ["deterministic_title_variant"])
                self.assertTrue(any(operation in e["left_operations"] + e["right_operations"]
                                    for e in pairs[0]["title_evidence"]))

    def test_variants_do_not_create_substrings_or_erase_composition_markers(self):
        for title in ("Autumn Leaves Forever", "Autum Leaves", "My Autumn Leaves",
                      "Autumn Leaves (Part II)", "Autumn Leaves (From the Film)",
                      "Autumn Leaves (Live)", "Autumn Leaves (Version of You)"):
            with self.subTest(title=title):
                self.assertEqual(overlap.match_catalogs([record("l", "Autumn Leaves")],
                                                        [record("r", title)]), [])
        for left_title, right_title in (("नाना", "निना"), ("夜曲", "晴天")):
            self.assertEqual(overlap.match_catalogs([record("l", left_title)],
                                                    [record("r", right_title)]), [])

    def test_missing_names_never_generate_candidates(self):
        for missing in (None, "", "NA", "N/A", "!!!"):
            with self.subTest(missing=missing):
                self.assertEqual(overlap.match_catalogs([record("l", missing)],
                                                        [record("r", missing)]), [])
        self.assertEqual(overlap.normalize("café"), overlap.normalize("cafe\u0301"))

    def test_exact_pairs_preserve_all_source_fields_and_ambiguity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left_row = record("solo", "Ａｕｔｕｍｎ—Leaves", [("composer", "Writer")],
                              membership={"native": "TEST", "current": None},
                              availability={"midi": "unavailable"}, clip_ids=["c1", "c2"])
            left = catalog(root, "jazz", [left_row, record("missing", None)])
            right_rows = [record("arr1", "autumn leaves", [("performer", "Writer")]),
                          record("arr2", "Autumn Leaves", [("composer", "Other")])]
            right = catalog(root, "local", right_rows)
            summary = overlap.run({"jazz": left}, {"local": right}, root / "out")
            pairs = overlap.read_catalog(root / "out/jazz__local.jsonl")
            self.assertEqual(len(pairs), 2)
            self.assertEqual(pairs[0]["left"], left_row)
            self.assertEqual(pairs[0]["right"], right_rows[0])
            self.assertEqual(pairs[0]["work_identity"], "unverified")
            self.assertEqual(pairs[0]["music_alignment"], "not_attempted")
            counts = summary["pairs"]["jazz__local"]["lanes"]["exact_normalized_title"]
            self.assertEqual(counts, {"pairs": 2, "left_records": 1, "right_records": 2})
            self.assertFalse(pairs[0]["creator_field_evidence"][0]["same_role"])
            self.assertEqual(pairs[1]["creator_evidence"], "no_exact_creator_agreement")

    def test_complete_matrix_and_unavailable_catalogs_keep_null_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            named = catalog(root, "named", [record("a", "One"), record("b", "Unmatched")])
            other = catalog(root, "other", [record("c", "ONE"), record("d", "Different")])
            unnamed = catalog(root, "unnamed", [record("missing", None)])
            empty = catalog(root, "empty", [])
            left = {"j1": named, "j2": named, "j3": unnamed, "j4": None}
            right = {"r1": other, "r2": other, "r3": empty, "r4": root / "missing.jsonl"}
            summary = overlap.run(left, right, root / "out")
            self.assertEqual(len(summary["pairs"]), 16)
            for name, status in summary["pairs"].items():
                self.assertTrue((root / "out" / (name + ".jsonl")).is_file())
                self.assertTrue((root / "out" / (name + ".csv")).is_file())
                self.assertTrue((root / "out" / (name + ".unmatched_left.jsonl")).is_file())
                if status["status"] == "not_assessed":
                    self.assertIsNone(status["lanes"]["all_title_candidates"]["pairs"])
                    self.assertIsNone(status["unmatched"]["left"])
            unmatched = overlap.read_catalog(root / "out/j1__r1.unmatched_left.jsonl")
            self.assertEqual([(r["record"]["id"], r["reason"]) for r in unmatched],
                             [("b", "no_title_candidate")])
            self.assertEqual(summary["catalogs"]["left"]["j3"]["status"], "unsearchable")
            self.assertEqual(summary["catalogs"]["right"]["r4"]["status"], "unavailable")
            self.assertEqual(summary["catalogs"]["left"]["j1"]["rows"], 2)

    def test_duplicate_ids_and_unsafe_output_names_fail_before_output_created(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good = catalog(root, "good", [record("a", "One")])
            duplicate = catalog(root, "duplicate", [record("same", "One"), record("same", "Two")])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                overlap.run({"left": good}, {"right": duplicate}, root / "out")
            self.assertFalse((root / "out").exists())
            for bad in ("../escape", "/absolute", "a/b", "a\\b", "a__b", "."):
                with self.subTest(name=bad), self.assertRaisesRegex(ValueError, "Unsafe"):
                    overlap.run({bad: good}, {"right": good}, root / "out")
            self.assertFalse((root / "out").exists())

    def test_csv_formula_cells_and_record_json_are_preserved_safely(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = record("\t=1+1", " =SUM(A1:A2)", [("composer", "+Composer")])
            source = catalog(root, "source", [row])
            overlap.run({"left": source}, {"right": source}, root / "out")
            with (root / "out/left__right.csv").open(encoding="utf-8-sig", newline="") as handle:
                exported = list(csv.DictReader(handle))[0]
            self.assertEqual(exported["left_id"], "'\t=1+1")
            self.assertEqual(exported["left_titles"], "'=SUM(A1:A2)")
            self.assertEqual(exported["left_creators"], "'+Composer")
            self.assertEqual(json.loads(exported["left_record"]), row)
            self.assertEqual(overlap.read_catalog(root / "out/left__right.jsonl")[0]["left"], row)
            with self.assertRaises(FileExistsError):
                overlap.run({"left": source}, {"right": source}, root / "out")

    def test_safe_file_availability_only_stats_and_rejects_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            music = root / "music"
            music.mkdir()
            (music / "piece.mid").write_bytes(b"synthetic bytes must not be decoded")
            (music / "escape.mid").symlink_to(root / "outside.mid")
            with patch.object(Path, "open", side_effect=AssertionError("no payload reads")):
                available = overlap.safe_file_availability(music, "./piece.mid")
                self.assertEqual(available["status"], "available")
                self.assertEqual(available["size_bytes"], 35)
                self.assertEqual(overlap.safe_file_availability(music, "absent.mid")["status"], "missing")
                self.assertEqual(overlap.safe_file_availability(music, None)["status"], "unavailable")
                for path in ("../outside.mid", "/absolute.mid", "C:\\outside.mid", "escape.mid"):
                    with self.subTest(path=path), self.assertRaises(ValueError):
                        overlap.safe_file_availability(music, path)

    def test_creator_missing_and_unknown_roles_do_not_imply_agreement(self):
        left = record("l", "Song", [("composer_name", "Person")])
        for creator in ([], [{"field": "performer", "value": None}]):
            right = record("r", "Song")
            right["creators"] = creator
            pair = overlap.match_catalogs([left], [right])[0]
            self.assertEqual(pair["creator_evidence"], "creator_missing")
        pairs = overlap.match_catalogs([left], [record("r", "Song", [("composer", "PERSON")])])
        self.assertTrue(pairs[0]["creator_field_evidence"][0]["same_role"])
        self.assertEqual(pairs[0]["creator_field_evidence"][0]["right_role"], "composer")
        pairs = overlap.match_catalogs([record("l", "Song", [("credit", "Person")])],
                                       [record("r", "Song", [("credit", "Person")])])
        self.assertIsNone(pairs[0]["creator_field_evidence"][0]["same_role"])
        self.assertEqual(pairs[0]["creator_field_evidence"][0]["role_comparison"], "unknown_role")
        counts = overlap._summarize(pairs, searchable=True)
        self.assertEqual(counts["exact_same_role_creator"]["pairs"], 0)
        self.assertEqual(counts["exact_cross_role_creator_only"]["pairs"], 0)
        self.assertEqual(counts["exact_unknown_role_creator"]["pairs"], 1)

    def test_multiple_fields_keep_all_evidence_without_duplicate_pairs(self):
        left = record("l", "The Song")
        left["titles"].append({"field": "documented_alternate_title", "value": "Song"})
        right = record("r", "Song")
        right["titles"].append({"field": "other_title", "value": "ＳＯＮＧ"})
        pairs = overlap.match_catalogs([left], [right])
        self.assertEqual(len(pairs), 1)
        exact = [e for e in pairs[0]["title_evidence"] if e["lane"] == "exact_normalized_title"]
        self.assertEqual(len(exact), 1)
        self.assertEqual(len(exact[0]["right_fields"]), 2)
        self.assertEqual(pairs[0]["primary_retrieval_lane"], "exact_normalized_title")


if __name__ == "__main__":
    unittest.main()
