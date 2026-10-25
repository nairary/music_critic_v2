"""Synthetic-only checks for the JHT/JAAH identity projection boundary."""

import io
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import music_overlap_metadata as lexer_module
from scripts.project_jazz_web_metadata import (
    MetadataJSONError,
    catalog_row,
    fetch_catalog_sources,
    iter_jht_metadata,
    project_catalog,
    project_jaah_metadata,
    write_population_views,
)


class JazzIdentityProjectionTests(unittest.TestCase):
    def test_jht_array_keeps_distinct_records_and_composer_role(self):
        source = '[{"title":"Été","composers":"A; B","trees":[{"title":"bad"}]},' \
                 '{"title":"Été","composers":null,"chords":["bad"]}]'
        rows = list(iter_jht_metadata(io.StringIO(source), chunk_size=1))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].metadata, {"/title": "Été", "/composers": "A; B"})
        self.assertEqual(rows[1].metadata, {"/title": "Été", "/composers": None})
        row = catalog_row("jht", "index:0", rows[0], {"version": "abc"})
        self.assertEqual(row["creators"], [{"field": "/composers", "value": "A; B"}])
        self.assertEqual(row["source_metadata"]["version"], "abc")

    def test_jaah_preserves_native_and_jams_identity_paths(self):
        source = json.dumps({
            "title": "Take", "artist": "Performer", "composer": "Composer",
            "mbid": "recording-id", "year": 1950,
            "file_metadata": {"title": "Track", "artist": "Artist", "duration": 5,
                              "identifiers": {"musicbrainz": "id2"}},
            "sandbox": {"composer": "C", "recording_id": "R", "key": "forbidden"},
            "annotations": [{"title": "forbidden"}],
        })
        result = project_jaah_metadata(io.StringIO(source), chunk_size=2)
        row = catalog_row("jaah", "track.json", result, {})
        self.assertEqual([x["value"] for x in row["titles"]], ["Take", "Track"])
        self.assertEqual({x["field"] for x in row["creators"]},
                         {"/artist", "/composer", "/file_metadata/artist", "/sandbox/composer"})
        self.assertEqual(row["source_metadata"]["identity_metadata"]["/mbid"], "recording-id")
        self.assertNotIn("/file_metadata/duration", result.metadata)
        self.assertNotIn("/sandbox/key", result.metadata)

    def test_music_and_unknown_payloads_never_reach_decoder(self):
        source = '{"title":"safe","parts":[{"title":"MUSIC_SECRET"}],' \
                 '"chords":{"a":"CHORD_SECRET"},"sandbox":{"unknown":"UNKNOWN_SECRET"},' \
                 '"trees":{"name":"TREE_SECRET"},"composer":["ARRAY_SECRET"]}'
        decoder = json.loads
        tokens = []

        def spy(value, *args, **kwargs):
            tokens.append(value)
            return decoder(value, *args, **kwargs)

        with patch.object(lexer_module.json, "loads", side_effect=spy):
            result = project_jaah_metadata(io.StringIO(source), chunk_size=1)
        self.assertEqual(result.metadata, {"/title": "safe"})
        self.assertFalse(any("SECRET" in token for token in tokens))
        self.assertFalse(any(token.startswith(("{", "[")) for token in tokens))
        self.assertEqual(result.schema["/composer"], "array")

    def test_missing_empty_and_nontext_identity_fields_are_not_invented(self):
        result = project_jaah_metadata(io.StringIO('{"title":null,"artist":"","composer":17}'))
        row = catalog_row("jaah", "empty", result, {})
        self.assertEqual(row["titles"], [])
        self.assertEqual(row["creators"], [])
        self.assertEqual(row["source_metadata"]["identity_metadata"]["/composer"], 17)

    def test_duplicates_trailing_data_and_malformed_skipped_payload_fail(self):
        for source in ('{"title":"a","title":"b"}', '{"parts":[1,]}',
                       '{"title":"a"} {}', '{"parts":{"x":"bad\\q"}}',
                       '{"sandbox":{"artist":"a","artist":"b"}}', '[]'):
            with self.subTest(source=source):
                with self.assertRaises(MetadataJSONError):
                    project_jaah_metadata(io.StringIO(source), chunk_size=1)
        for source in ('[{}] {}', '[{},]', '{}', '[null]'):
            with self.subTest(source=source):
                with self.assertRaises(MetadataJSONError):
                    list(iter_jht_metadata(io.StringIO(source), chunk_size=1))

    def test_empty_array_and_caller_stream_lifetime(self):
        source = io.StringIO(' [] ')
        self.assertEqual(list(iter_jht_metadata(source)), [])
        self.assertFalse(source.closed)

    def test_fetch_pins_verifies_and_preserves_all_duplicate_title_records(self):
        bodies = {
            "annotations/a.json": b'{"title":"Same","artist":"First","parts":[]}',
            "annotations/b.json": b'{"title":"Same","artist":"Second","parts":[]}',
            "README.md": b"Synthetic catalog documentation",
        }
        tree = {"truncated": False, "tree": [
            {"type": "blob", "path": path, "size": len(body),
             "sha": hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest()}
            for path, body in bodies.items()
        ]}

        class Response(io.BytesIO):
            status = 200
            headers = {"Content-Type": "application/octet-stream"}

        def urlopen(request, **kwargs):
            if "/git/trees/" in request.full_url:
                return Response(json.dumps(tree).encode())
            self.assertIn("/" + "a" * 40 + "/", request.full_url)
            name = request.full_url.split("/" + "a" * 40 + "/", 1)[1]
            return Response(bodies[name])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("scripts.project_jazz_web_metadata.urlopen", side_effect=urlopen):
                self.assertEqual(fetch_catalog_sources("jaah", "a" * 40, root),
                                 {"selected_files": 3, "failed_fetches": 0})
            summary = project_catalog(root)
            self.assertEqual(summary["rows"], 2)
            rows = [json.loads(line) for line in (root / "catalog.jsonl").read_text().splitlines()]
            self.assertEqual({row["id"] for row in rows}, {"annotations/a.json", "annotations/b.json"})
            self.assertTrue(all(row["source_metadata"]["version"] == "a" * 40 for row in rows))
            with self.assertRaises(FileExistsError):
                project_catalog(root)

    def test_population_views_use_field_presence_without_reading_tree_values(self):
        rows = [catalog_row("jht", str(index), projection, {}) for index, projection in enumerate(
            iter_jht_metadata(io.StringIO('[{"title":"One","trees":[]},{"title":"Two"}]')))]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = "".join(json.dumps(row) + "\n" for row in rows)
            (root / "catalog.jsonl").write_text(raw)
            summary = write_population_views(root)
            self.assertEqual(summary["tree_field_present_rows"], 1)
            self.assertEqual((root / "catalog.jsonl").read_text(), raw)
            subset = [json.loads(line) for line in (root / "catalog-annotated.jsonl").read_text().splitlines()]
            self.assertEqual([row["id"] for row in subset], ["0"])
            self.assertEqual(subset[0]["membership"]["tree_value_inspected"], False)


if __name__ == "__main__":
    unittest.main()
