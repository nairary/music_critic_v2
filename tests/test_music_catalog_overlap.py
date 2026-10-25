"""Synthetic checks of retrieval boundaries, not corpus-quality tests."""
import importlib.util
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

SPEC = importlib.util.spec_from_file_location("overlap", Path(__file__).parents[1] / "scripts/audit_music_catalog_overlap.py")
overlap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(overlap)


def record(identity, titles, creators=()):
    return {"id": identity, "titles": [{"field": "title", "value": s} for s in titles],
            "creators": [{"field": "artist", "value": s} for s in creators]}


class CatalogOverlapTests(unittest.TestCase):
    def test_unicode_preserves_cjk_and_accents(self):
        self.assertEqual(overlap.normalize("夜曲"), "夜曲")
        self.assertNotEqual(overlap.normalize("夜曲"), overlap.normalize("晴天"))
        self.assertEqual(overlap.normalize("ＡＢＣ—Ｄ"), "abc d")
        self.assertNotEqual(overlap.normalize("été"), overlap.normalize("ete"))
        self.assertNotEqual(overlap.normalize("नाना"), overlap.normalize("निना"))
        self.assertEqual(overlap.normalize("café"), overlap.normalize("cafe\u0301"))

    def test_apostrophes_and_empty_sentinels(self):
        self.assertEqual(overlap.normalize("Don't Stop"), overlap.normalize("dont-stop"))
        index = overlap.catalog_index([record("a", [None, "NA", "!!!"])])
        self.assertEqual(overlap.candidates(index, record("b", ["N/A", ""])), [])

    def test_same_title_conflicting_creators_remains_unverified(self):
        index = overlap.catalog_index([record("a", ["Home"], ["Artist A"])])
        pair = overlap.candidates(index, record("b", ["Home"], ["Artist B"]))[0]
        self.assertEqual(pair["creator_evidence"], "no_exact_creator_agreement")
        self.assertEqual(pair["work_identity"], "unverified")

    def test_multiple_matching_fields_are_one_pair(self):
        index = overlap.catalog_index([record("a", ["Song", "SONG"], ["Artist"] )])
        pairs = overlap.candidates(index, record("b", ["song", "ＳＯＮＧ"], ["ARTIST"]))
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["creator_evidence"], "exact_normalized_creator")
        self.assertEqual(len(pairs[0]["title_evidence"][0]["right_fields"]), 2)

    def test_missing_creator_is_not_negative_label(self):
        index = overlap.catalog_index([record("a", ["Song"], ["Artist"])])
        pair = overlap.candidates(index, record("b", ["Song"]))[0]
        self.assertEqual(pair["creator_evidence"], "creator_missing")

    def test_no_substring_or_decoration_removal(self):
        index = overlap.catalog_index([record("a", ["Song"])])
        for title in ("Song of Life", "Song (Live)", "My Song"):
            self.assertEqual(overlap.candidates(index, record("b", [title])), [])

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            overlap.catalog_index([record("a", ["One"]), record("a", ["Two"])])

    def test_cross_role_creator_match_is_separate(self):
        index = overlap.catalog_index([record("a", ["Song"], ["Artist"])])
        right = record("b", ["Song"])
        right["creators"] = [{"field": "composer_name", "value": "Artist"}]
        pairs = overlap.candidates(index, right)
        self.assertFalse(pairs[0]["creator_field_evidence"][0]["same_role"])
        counts = overlap.summarize(pairs)
        self.assertEqual(counts["exact_cross_role_creator_only"]["pairs"], 1)
        self.assertEqual(counts["exact_same_role_creator"]["pairs"], 0)

    def test_arrangements_and_clips_counted_separately(self):
        left = record("a", ["Song"], ["Artist"])
        left["clip_ids"] = ["c1", "c2"]
        index = overlap.catalog_index([left])
        pairs = sum((overlap.candidates(index, record(b, ["Song"], ["Artist"]))
                     for b in ("b1", "b2")), [])
        counts = overlap.summarize(pairs)["all_title_candidates"]
        self.assertEqual((counts["pairs"], counts["left_records"], counts["right_records"], counts["left_clip_ids"]), (2, 1, 2, 2))

    def test_file_pipeline_keeps_ambiguous_pairs_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hook, pop = root / "hook.jsonl", root / "pop.jsonl"
            ht = record("ht", ["Song"], ["Artist"])
            ht["clip_ids"] = ["clip1", "clip2"]
            hook.write_text(json.dumps(ht) + "\n")
            pop.write_text(json.dumps(record("001", ["Song"], ["Artist"])) + "\n")
            pdmx = root / "pdmx.csv"
            with pdmx.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=overlap.PDMX_FIELDS)
                writer.writeheader()
                for identity, creator, conflict in (("a", "Artist", "False"), ("b", "Other", "True")):
                    writer.writerow({"path": identity, "song_name": "Song", "title": "SONG",
                                     "artist_name": creator, "license_conflict": conflict,
                                     "best_path": "a", "mid": identity + ".mid"})
                writer.writerow({"path": "c", "title": "Unmatched"})
            summary = overlap.run(hook, pop, pdmx, root / "output")
            counts = summary["pairs"]["hooktheory_pdmx"]
            self.assertEqual(summary["catalogs"]["pdmx"]["rows"], 3)
            self.assertEqual(counts["all_title_candidates"]["pairs"], 2)
            self.assertEqual(counts["all_title_candidates"]["pdmx_best_paths"], 1)
            self.assertEqual(counts["exact_normalized_creator"]["pairs"], 1)
            self.assertEqual(counts["all_title_candidates"]["right_license_conflict_rows"], 1)
            rows = overlap.read_catalog(root / "output/hooktheory_pdmx.jsonl")
            self.assertEqual(rows[1]["right"]["source_metadata"]["mid"], "b.mid")
            self.assertEqual(summary["pdmx_rows_with_title_candidates_in_both"], ["a", "b"])
            with self.assertRaises(FileExistsError):
                overlap.run(hook, pop, pdmx, root / "output")

    def test_hooktheory_grouping_keeps_distinct_artists_and_missing_titles(self):
        records = [SimpleNamespace(clip_id=str(i), metadata={"split": "TRAIN", "hooktheory": {
            "artist": artist, "song": song}}) for i, (artist, song) in enumerate(
                [("a", "song"), ("a", "song"), ("b", "song"), ("a", "")])]
        grouped, missing = overlap.group_hooktheory(records)
        self.assertEqual(len(grouped), 2)
        self.assertEqual(grouped[0]["clip_ids"], ["0", "1"])
        self.assertEqual(missing, [{"clip_id": "3", "reason": "missing_song_metadata"}])


if __name__ == "__main__":
    unittest.main()
