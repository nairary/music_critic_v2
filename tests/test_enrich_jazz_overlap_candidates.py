"""Synthetic stat-only enrichment; no musical payload may be opened."""

import importlib.util
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "enrich_jazz", Path(__file__).parents[1] / "scripts/enrich_jazz_overlap_candidates.py")
enrich = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(enrich)


def candidate(dataset, identity, source_metadata=None, membership=None, availability=None):
    return {
        "left": {"dataset": "wjd", "id": "solo:1", "titles": [{"field": "title", "value": "Song"}],
                 "membership": {"native_split": None, "current_split": None}},
        "right": {"dataset": dataset, "id": identity,
                  "titles": [{"field": "title", "value": "Song"}],
                  "source_metadata": source_metadata or {}, "membership": membership or {},
                  "availability": availability or {}},
        "title_evidence": [{"normalized_key": "song", "lane": "exact_normalized_title"}],
        "retrieval_lanes": ["exact_normalized_title"],
        "creator_field_evidence": [], "work_identity": "unverified", "music_alignment": "not_attempted",
    }


def emit_pair(matches, name, rows):
    path = matches / (name + ".jsonl")
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


class EnrichmentTests(unittest.TestCase):
    def test_availability_counts_unique_source_ids_across_dependent_catalogs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matches = root / "matches"
            matches.mkdir()
            first = candidate("pdmx", "arr1", {"path": "missing.json"})
            second = candidate("pdmx", "arr1", {"path": "missing.json"})
            second["left"]["dataset"] = "jaah"
            third = candidate("pdmx", "arr2", {"path": "missing.json"})
            emit_pair(matches, "wjd__pdmx", [first, third])
            emit_pair(matches, "jaah__pdmx", [second])
            emit_pair(matches, "wjd__dilemmadata", [])
            summary = enrich.run(matches, root, root / "out")
            self.assertEqual(summary["exported_candidates"], 3)
            self.assertEqual(summary["corpora"]["pdmx"]["unique_right_records"], 2)
            counts = summary["corpora"]["pdmx"]["availability"]["pdmx_symbolic_json"]["missing"]
            self.assertEqual(counts, {"file_observations": 2, "right_records": 2})
            self.assertEqual(summary["pairs"]["wjd__dilemmadata"]["exported_candidates"], 0)
            self.assertEqual(summary["pairs"]["wjd__dilemmadata"]["source_overlap_status"], "not_reassessed")
            self.assertEqual(summary["corpora"]["dilemmadata"]["unique_right_records"], 0)
            self.assertTrue((root / "out/wjd__dilemmadata.csv").is_file())

    def test_unsafe_music_and_input_catalog_paths_fail_without_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matches = root / "matches"
            matches.mkdir()
            (root / "data/pdmx").mkdir(parents=True)
            (root / "data/pdmx/escape.mid").symlink_to(root / "outside.mid")
            for relative in ("../outside.mid", "/absolute.mid", "escape.mid", "C:\\outside.mid"):
                with self.subTest(relative=relative):
                    emit_pair(matches, "wjd__pdmx", [candidate("pdmx", "a", {"mid": relative})])
                    with self.assertRaises(ValueError):
                        enrich.run(matches, root, root / "out")
                    self.assertFalse((root / "out").exists())
            (matches / "wjd__pdmx.jsonl").unlink()
            (matches / "wjd__pdmx.jsonl").symlink_to(root / "outside.jsonl")
            with self.assertRaisesRegex(ValueError, "Unsafe pair"):
                enrich.run(matches, root, root / "out")

    def test_duplicate_pairs_and_inconsistent_right_metadata_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matches = root / "matches"
            matches.mkdir()
            row = candidate("pdmx", "arr1", {"path": "missing.json"})
            emit_pair(matches, "wjd__pdmx", [row, row])
            with self.assertRaisesRegex(ValueError, "Duplicate candidate"):
                enrich.run(matches, root, root / "out")
            emit_pair(matches, "wjd__pdmx", [row])
            second = candidate("pdmx", "arr1", {"path": "different.json"})
            second["left"]["dataset"] = "jaah"
            emit_pair(matches, "jaah__pdmx", [second])
            with self.assertRaisesRegex(ValueError, "Conflicting source-record"):
                enrich.run(matches, root, root / "out")
            self.assertFalse((root / "out").exists())

    def test_fresh_output_and_csv_formula_suppression_preserve_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matches = root / "matches"
            matches.mkdir()
            row = candidate("pdmx", "\t=2+2")
            row["right"]["titles"][0]["value"] = "+Unsafe"
            emit_pair(matches, "wjd__pdmx", [row])
            enrich.run(matches, root, root / "out")
            with (root / "out/candidates.csv").open(encoding="utf-8-sig", newline="") as handle:
                exported = next(csv.DictReader(handle))
            self.assertEqual(exported["right_id"], "'\t=2+2")
            self.assertEqual(exported["right_titles"], "'+Unsafe")
            self.assertEqual(json.loads(exported["candidate_json"])["right"], row["right"])
            with patch.object(Path, "open", side_effect=AssertionError("must reject output before reads")):
                with self.assertRaises(FileExistsError):
                    enrich.run(matches, root, root / "out")

    def test_split_conflict_flags_do_not_replace_the_supplied_evidence(self):
        row = {"id": "r", "membership": {"native_split": "VAL", "current_split": "validation",
                 "current_global_split": "train", "saved_profile_split": "TEST",
                 "native_current_conflict": True}}
        before = json.loads(json.dumps(row))
        evidence = enrich.split_evidence(row)[0]
        self.assertFalse(evidence["native_current_conflict"])
        self.assertTrue(evidence["reported_native_current_conflict"])
        self.assertTrue(evidence["conflict_evidence_inconsistent"])
        self.assertTrue(evidence["current_fields_conflict"])
        self.assertEqual(row, before)

    def test_cache_and_tsv_files_have_precise_scopes_and_split_conflicts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matches = root / "matches"
            matches.mkdir()
            hook_root = root / "data/cache/hooktheory/multisource-cache-v1.0.0"
            pop_root = root / "data/cache/pop909_cl/multisource-cache-v1.0.0"
            dilemma_root = root / "data/dilemmadata/dilemmadata-v1.0/johentsch-dilemmadata-d60ee75"
            for path in (hook_root / "clip.json", pop_root / "001.json", dilemma_root / "piece.tsv"):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("payload must not be read")
            hook = candidate("hooktheory", "group", membership={"records": [
                {"source_id": "clip1", "native_split": "TEST", "current_split": "train",
                 "native_current_conflict": True,
                 "canonical_availability": {"path": "clip.json", "status": "present"}},
                {"source_id": "clip2", "native_split": "TRAIN", "current_split": None,
                 "canonical_availability": {"path": None, "status": "unspecified"}}]})
            pop = candidate("pop909", "001", availability={"path": "001.json", "status": "present"},
                            membership={"native_split": None, "current_split": "test"})
            dilemma = candidate("dilemmadata", "an:train:piece", availability={"path": "piece.tsv", "status": "present"},
                                membership={"native_split": "train", "current_global_split": None,
                                            "saved_profile_split": "validation", "split_evidence": "archived.jsonl"})
            paths = {emit_pair(matches, name, [row]).resolve() for name, row in
                     (("wjd__hooktheory", hook), ("wjd__pop909", pop), ("wjd__dilemmadata", dilemma))}
            original_open = Path.open

            def guarded_open(path, mode="r", *args, **kwargs):
                if "r" in mode and path.resolve() not in paths:
                    raise AssertionError("Unexpected read: " + str(path))
                return original_open(path, mode, *args, **kwargs)

            with patch.object(Path, "open", guarded_open):
                summary = enrich.run(matches, root, root / "out")
            rows = {row["right"]["dataset"]: row for row in
                    (json.loads(line) for line in (root / "out/candidates.jsonl").read_text().splitlines())}
            h = rows["hooktheory"]["local_metadata_enrichment"]
            self.assertEqual([file["kind"] for file in h["right_files"]], ["canonical_cache_only"] * 2)
            self.assertEqual([file["status"] for file in h["right_files"]], ["available", "unavailable"])
            self.assertTrue(h["right_splits"][0]["native_current_conflict"])
            self.assertIsNone(h["right_splits"][1]["native_current_conflict"])
            self.assertIsNone(h["right_splits"][1]["current_split"])
            self.assertEqual(rows["pop909"]["local_metadata_enrichment"]["right_files"][0]["kind"], "canonical_cache_only")
            d = rows["dilemmadata"]["local_metadata_enrichment"]
            self.assertEqual(d["right_files"][0]["kind"], "dilemmadata_raw_tsv")
            self.assertEqual(d["right_splits"][0]["archive_split"], "validation")
            self.assertIsNone(d["right_splits"][0]["current_split"])
            self.assertTrue(d["right_splits"][0]["native_archive_conflict"])
            with (root / "out/candidates.csv").open(encoding="utf-8-sig", newline="") as handle:
                csv_rows = {row["right_dataset"]: row for row in csv.DictReader(handle)}
            self.assertEqual(json.loads(csv_rows["dilemmadata"]["right_archive_splits"])[0]["split"], "validation")
            self.assertEqual(json.loads(csv_rows["hooktheory"]["right_current_splits"])[1]["split"], None)
            self.assertEqual(summary["corpora"]["hooktheory"]["unique_right_records"], 1)

    def test_pdmx_stats_three_locators_without_payload_reads_or_source_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matches = root / "matches"
            matches.mkdir()
            (root / "data/pdmx/json").mkdir(parents=True)
            payload = root / "data/pdmx/json/piece.json"
            payload.write_text("must not decode this musical payload")
            row = candidate("pdmx", "arrangement:1", {"path": "./json/piece.json",
                                                       "mid": "mid/absent.mid", "mxl": None})
            source = emit_pair(matches, "wjd__pdmx", [row])
            (matches / "wjd__pdmx.unmatched_right.jsonl").write_text("not JSON; must never open")
            before = source.read_bytes()
            original_open = Path.open

            def guarded_open(path, mode="r", *args, **kwargs):
                if "r" in mode and path.resolve() != source.resolve():
                    raise AssertionError("Unexpected read: " + str(path))
                return original_open(path, mode, *args, **kwargs)

            with patch.object(Path, "open", guarded_open):
                summary = enrich.run(matches, root, root / "out")
            rows = [json.loads(line) for line in (root / "out/wjd__pdmx.jsonl").read_text().splitlines()]
            exported = rows[0]
            for key, value in row.items():
                self.assertEqual(exported[key], value)
            files = exported["local_metadata_enrichment"]["right_files"]
            self.assertEqual([(f["source_field"], f["status"]) for f in files],
                             [("source_metadata.path", "available"),
                              ("source_metadata.mid", "missing"),
                              ("source_metadata.mxl", "unavailable")])
            self.assertTrue(all(f["method"] == "stat_only_no_content_read" for f in files))
            self.assertEqual(summary["corpora"]["pdmx"]["unique_right_records"], 1)
            self.assertEqual(source.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
