"""Synthetic-only checks for the metadata projection boundary."""

import io
import json
import unittest
from unittest.mock import patch

from scripts import music_overlap_metadata as reader
from scripts.music_overlap_metadata import MetadataJSONError, iter_metadata_records


class MetadataProjectionTests(unittest.TestCase):
    def test_selects_metadata_without_descending_into_music(self):
        source = io.StringIO(
            '{"clip":{"title":"Song","artist":"Artist",'
            '"json":{"title":"NOT METADATA","notes":[{"pitch":60}]},'
            '"json_api":{"artist":"NOT METADATA"},"score":{"title":"NOT METADATA"},'
            '"metadata":{"title":"Source title","notes":[1,2]},'
            '"unknown":{"artist":"NOT METADATA"}}}'
        )
        row, = iter_metadata_records(source, include_schema=True, chunk_size=3)
        self.assertEqual(row.clip_id, "clip")
        self.assertEqual(row.metadata, {
            "title": "Song", "artist": "Artist", "metadata": {"title": "Source title"}
        })
        self.assertEqual(row.schema, {
            "/title": "string", "/artist": "string", "/metadata": "object",
            "/metadata/title": "string", "/unknown": "object",
        })

    def test_escaped_strings_and_chunk_boundaries(self):
        title = 'quote" slash\\ tab\t newline\n snowman \u2603'
        source = json.dumps({
            "clip\\a": {"title": title, "notes": [
                {"text": '}{["\\', "nested": [True, False, None, -1.25e9]},
            ], "youtube": {"id": "video"}},
            "clip-b": {"title": None, "id": 42},
        }, ensure_ascii=True)
        for size in (1, 2, 3, 7, 64, 65536):
            with self.subTest(chunk_size=size):
                rows = list(iter_metadata_records(io.StringIO(source), chunk_size=size))
                self.assertEqual([row.clip_id for row in rows], ["clip\\a", "clip-b"])
                self.assertEqual(rows[0].metadata, {"title": title, "youtube": {"id": "video"}})
                self.assertEqual(rows[1].metadata, {"title": None, "id": 42})
                self.assertEqual(rows[0].schema, {})

    def test_excluded_strings_and_objects_never_reach_decoder(self):
        source = '{"c":{"title":"safe","json":{"title":"MUSIC_SECRET"},' \
                 '"other":{"artist":"UNKNOWN_SECRET"},"xmlData":"XML_SECRET"}}'
        decode = json.loads
        decoded_tokens = []

        def spy(token, *args, **kwargs):
            decoded_tokens.append(token)
            return decode(token, *args, **kwargs)

        with patch.object(reader.json, "loads", side_effect=spy):
            row, = iter_metadata_records(io.StringIO(source), chunk_size=2)
        self.assertEqual(row.metadata, {"title": "safe"})
        self.assertFalse(any("SECRET" in token for token in decoded_tokens))
        self.assertFalse(any(token.startswith(("{", "[")) for token in decoded_tokens))

    def test_explicit_schema_descent_reports_only_types(self):
        source = json.dumps({"c": {"unknown": {"name": "secret", "notes": [60],
                                                 "deep": {"title": "secret"}},
                                   "meta": {"a/b~": None}}})
        row, = iter_metadata_records(
            io.StringIO(source), include_schema=True, schema_paths=(("unknown",),),
        )
        self.assertEqual(row.metadata, {})
        self.assertEqual(row.schema, {
            "/unknown": "object", "/unknown/name": "string", "/unknown/deep": "object",
            "/meta": "object", "/meta/a~1b~0": "null",
        })

    def test_extra_paths_allow_scalars_only_and_require_exact_path(self):
        source = '{"c":{"catalog":{"artist":{"name":"A"},"title":"T"},' \
                 '"other":{"name":"not selected"},"name":["not scalar"]}}'
        row, = iter_metadata_records(
            io.StringIO(source), extra_scalar_paths=(("catalog", "artist", "name"),),
        )
        self.assertEqual(row.metadata, {"catalog": {"artist": {"name": "A"}}})

    def test_forbidden_extra_and_schema_paths_reject_before_source_read(self):
        class NoRead(io.StringIO):
            def read(self, *args):
                raise AssertionError("unsafe path must reject before reading source")

        for path in (("json", "title"), ("meta", "XML_Data", "title"), ("json_api", "title"),
                     ("Annotations",), ("notes",), (), ("meta", "")):
            for option in ("extra_scalar_paths", "schema_paths"):
                with self.subTest(path=path, option=option):
                    with self.assertRaises(MetadataJSONError):
                        list(iter_metadata_records(NoRead("{}"), **{option: (path,)}))

    def test_malformed_skipped_payload_fails_without_decoding(self):
        for payload in ('[1,]', '{"x":}', '"bad\\q"', '"bad\\u12xy"',
                        '"line\nfeed"', '01', 'NaN', '[1 2]', 'trueX', '{',
                        '"unterminated', '1.', '+1', '--2'):
            with self.subTest(payload=repr(payload)):
                with self.assertRaises(MetadataJSONError):
                    list(iter_metadata_records(io.StringIO('{"c":{"json":' + payload + '}}'),
                                               chunk_size=1))

    def test_rejects_duplicates_in_projected_objects_and_top_level(self):
        for source in ('{"c":{"title":"a","title":"b"}}',
                       '{"c":{},"c":{}}', '{"c":{"meta":{"title":1,"title":2}}}'):
            with self.subTest(source=source):
                with self.assertRaises(MetadataJSONError):
                    list(iter_metadata_records(io.StringIO(source)))

    def test_excluded_duplicate_keys_are_not_interpreted_as_metadata(self):
        row, = iter_metadata_records(io.StringIO('{"c":{"json":{"title":1,"title":2}}}'))
        self.assertEqual(row.metadata, {})

    def test_document_shapes_and_trailing_content(self):
        self.assertEqual(list(iter_metadata_records(io.StringIO(' { } \n'))), [])
        for source in ('[]', '{"c":[]}', '{"c":null}', '{"c":{}} trailing',
                       '{"c":{},}', '{"c":{"title":true,}}', ''):
            with self.subTest(source=source):
                with self.assertRaises(MetadataJSONError):
                    list(iter_metadata_records(io.StringIO(source)))

    def test_large_skipped_string_is_not_subject_to_metadata_scalar_bound(self):
        source = io.StringIO(json.dumps({"c": {"title": "ok", "xmlData": "x" * 100000}}))
        row, = iter_metadata_records(source, max_scalar_chars=32, chunk_size=17)
        self.assertEqual(row.metadata, {"title": "ok"})
        self.assertFalse(source.closed)
        with self.assertRaises(MetadataJSONError):
            list(iter_metadata_records(io.StringIO('{"c":{"title":"' + 'x' * 100 + '"}}'),
                                       max_scalar_chars=32))

    def test_streaming_reads_are_bounded_and_preview_is_lazy(self):
        class BoundedStream(io.StringIO):
            def read(self, size=-1):
                if not 0 < size <= 7:
                    raise AssertionError("unbounded read")
                return super().read(size)

        source = BoundedStream('{"c":{"title":"A"},"later": MALFORMED}')
        iterator = iter_metadata_records(source, chunk_size=7)
        self.assertEqual(next(iterator).metadata, {"title": "A"})
        iterator.close()
        self.assertFalse(source.closed)
        with self.assertRaises(MetadataJSONError):
            list(iter_metadata_records(io.StringIO(source.getvalue()), chunk_size=7))

    def test_explicit_resource_bounds_and_nonfinite_selected_number(self):
        for option in ({"chunk_size": 0}, {"chunk_size": True}, {"max_scalar_chars": -1}):
            with self.assertRaises(MetadataJSONError):
                list(iter_metadata_records(io.StringIO('{}'), **option))
        with self.assertRaises(MetadataJSONError):
            list(iter_metadata_records(io.StringIO('{"c":{"id":1e9999}}')))
        deep = '[' * 130 + '0' + ']' * 130
        with self.assertRaises(MetadataJSONError):
            list(iter_metadata_records(io.StringIO('{"c":{"json":' + deep + '}}')))


if __name__ == "__main__":
    unittest.main()
