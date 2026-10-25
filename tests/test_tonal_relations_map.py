"""Synthetic relation-map mechanics; no musical gold or corpus examples."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts import tonal_relations_map as review


def example():
    levels = ("note", "sonority", "sequence", "phrase", "section", "part", "piece")
    return {
        "schema": "tonal-relations-map@1", "title": "Synthetic relation map",
        "cases": [{
            "id": "synthetic", "title": "Synthetic only", "genres": ["pop"],
            "membership": "train", "time_unit": "quarter_note", "time_origin": "source zero",
            "limitations": ["Synthetic; no musical interpretation"],
            "assets": [{"locator": "source.json", "sha256": "a" * 64}],
            "events": [
                {"id": "held", "start": "-1/1", "duration": "3/1", "pitch": 60,
                 "voice": "source part 1/voice 1", "source_ref": "row 1"},
                {"id": "attack", "start": "1/1", "duration": "1/1", "pitch": 64,
                 "voice": "source part 2/voice 1", "source_ref": "row 2"},
                {"id": "context", "start": "3/1", "duration": "1/1", "pitch": 61,
                 "voice": "source part 2/voice 1", "source_ref": "row 3"},
            ],
            "claims": [{
                "id": "claim", "source_url": "https://example.org/analysis",
                "source_anchor": "Synthetic example 1", "statement": "Synthetic held-note relation",
                "relation": "held_note", "status": "partial",
                "participants": [{"role": "held", "event_ids": ["held"]}],
                "assessment": "Only source binding tested", "alternatives": ["Another reading"],
                "missing": ["No independent interpretation"],
            }],
            "objects": [{"id": "vertical", "kind": "sonority", "label": "Synthetic sonority",
                         "start": "1/1", "end": "2/1", "event_ids": ["held", "attack"],
                         "basis": "Both source events sound throughout this interval"}],
            "relations": [{
                "id": "relation", "kind": "held_across_attack", "level": "cross_level",
                "claim_id": "claim", "participants": [
                    {"role": "held", "event_ids": ["held"], "object_ids": []},
                    {"role": "sonority", "event_ids": [], "object_ids": ["vertical"]}],
                "context_event_ids": ["context"], "alignment": "exact_event_ids",
                "confidence": None, "target_available": False,
            }],
            "coverage": {level: {"status": "observed" if level in {"note", "sonority"} else "unavailable",
                                  "reason": "Synthetic observed events" if level in {"note", "sonority"}
                                  else "No source evidence for this level"} for level in levels},
        }],
    }


class MapTests(unittest.TestCase):
    def test_v1_preserved_and_exact_sustained_object_validates_without_mutation(self):
        data = example()
        before = copy.deepcopy(data)
        result = review.validate(data)
        self.assertEqual(result["cases"], 1)
        self.assertEqual(result["events"], 3)
        self.assertEqual(result["objects"], 1)
        self.assertEqual(result["relations"], 1)
        self.assertEqual(result["claim_statuses"], {"partial": 1})
        self.assertEqual(data, before)

    def test_object_intervals_membership_and_ids_are_exact(self):
        for field, value in (("start", 1.0), ("start", "2/2"), ("end", "1/1"),
                             ("end", "3/1"), ("start", "0/1"), ("event_ids", []),
                             ("event_ids", ["context"]), ("event_ids", ["held", "held"]),
                             ("event_ids", ["unknown"]), ("id", "held"), ("basis", "")):
            with self.subTest(field=field, value=value):
                data = example()
                data["cases"][0]["objects"][0][field] = value
                with self.assertRaises(review.ReviewError):
                    review.validate(data)
        for duration in ("0/1", "1/1"):
            data = example()
            data["cases"][0]["events"][0]["duration"] = duration
            with self.assertRaises(review.ReviewError):
                review.validate(data)

    def test_overlapping_groups_are_not_forced_into_a_tree(self):
        data = example()
        case = data["cases"][0]
        case["objects"].append({"id": "episode", "kind": "phrase", "label": "Source phrase",
                                "start": "1/1", "end": "4/1", "event_ids": ["attack", "context"],
                                "basis": "Synthetic source boundary"})
        case["coverage"]["phrase"] = {"status": "partial", "reason": "One fragment"}
        self.assertEqual(review.validate(data)["objects"], 2)

    def test_relations_reject_dangling_refs_and_never_enable_targets(self):
        for field, value in (("claim_id", "missing"), ("alignment", "probable"),
                             ("confidence", 0.99), ("target_available", True),
                             ("target_available", 0), ("context_event_ids", ["missing"]),
                             ("context_event_ids", ["held", "held"]), ("level", "unknown"),
                             ("kind", ""), ("participants", [])):
            with self.subTest(field=field, value=value):
                data = example()
                data["cases"][0]["relations"][0][field] = value
                with self.assertRaises(review.ReviewError):
                    review.validate(data)
        for field, value in (("event_ids", ["vertical"]), ("object_ids", ["held"]),
                             ("event_ids", ["held", "held"])):
            with self.subTest(field=field):
                data = example()
                data["cases"][0]["relations"][0]["participants"][0][field] = value
                with self.assertRaises(review.ReviewError):
                    review.validate(data)

    def test_unavailable_relation_keeps_absence_and_claim_status(self):
        data = example()
        relation = data["cases"][0]["relations"][0]
        relation.update(alignment="unavailable", participants=[])
        self.assertEqual(review.validate(data)["relations"], 1)
        self.assertEqual(data["cases"][0]["claims"][0]["status"], "partial")
        relation["participants"] = [{"role": "missing", "event_ids": [], "object_ids": []}]
        with self.assertRaises(review.ReviewError):
            review.validate(data)

    def test_coverage_does_not_invent_absent_levels(self):
        for level, status in (("part", "observed"), ("note", "unavailable"),
                              ("sonority", "unavailable"), ("phrase", "predicted")):
            with self.subTest(level=level, status=status):
                data = example()
                data["cases"][0]["coverage"][level]["status"] = status
                with self.assertRaises(review.ReviewError):
                    review.validate(data)
        data = example()
        del data["cases"][0]["coverage"]["phrase"]
        with self.assertRaises(review.ReviewError):
            review.validate(data)
        data = example()
        data["cases"][0]["coverage"]["phrase"]["reason"] = " "
        with self.assertRaises(review.ReviewError):
            review.validate(data)

    def test_excluded_case_has_only_rejected_claims_and_missing_coverage(self):
        data = example()
        case = data["cases"][0]
        case.update(membership="excluded", events=[], objects=[], relations=[])
        case["claims"][0].update(status="rejected", participants=[])
        for coverage in case["coverage"].values():
            coverage.update(status="unavailable", reason="Excluded before payload access")
        self.assertEqual(review.validate(data)["objects"], 0)
        case["relations"] = [{"id": "not-admitted", "kind": "source_claim", "level": "piece",
                              "claim_id": "claim", "participants": [], "context_event_ids": [],
                              "alignment": "unavailable", "confidence": None, "target_available": False}]
        with self.assertRaises(review.ReviewError):
            review.validate(data)

    def test_html_preserves_payload_escapes_text_and_exposes_inspection_controls(self):
        data = example()
        data["title"] = '__DATA__ </script><img src=x onerror="alert(1)">'
        data["cases"][0]["claims"][0]["statement"] = "__TITLE__ </script>"
        rendered = review.render_html(data)
        self.assertNotIn("<img src=x", rendered)
        payload = rendered.split('<script id="evidence" type="application/json">', 1)[1].split('</script>', 1)[0]
        self.assertEqual(json.loads(payload), data)
        for control in ('id="case"', 'id="relations"', 'id="objects"', 'id="map"',
                        'id="piano"', 'id="coverage"', 'id="source"', 'id="scope"'):
            self.assertIn(control, rendered)
        self.assertIn("не оценка напряжения", rendered)
        self.assertIn("Участок связи", rendered)
        self.assertIn("Весь доступный контекст", rendered)

    def test_new_output_is_written_once_using_path_first_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "map.html"
            review.write_html_once(target, example())
            original = target.read_bytes()
            with self.assertRaises(FileExistsError):
                review.write_html_once(target, example())
            self.assertEqual(target.read_bytes(), original)

    def test_v1_protections_remain_active(self):
        for field, value in (("membership", "test"), ("time_unit", "mixed")):
            data = example()
            data["cases"][0][field] = value
            with self.assertRaises(review.ReviewError):
                review.validate(data)
        data = example()
        data["cases"][0]["claims"][0]["source_url"] = "javascript:alert(1)"
        with self.assertRaises(review.ReviewError):
            review.validate(data)


if __name__ == "__main__":
    unittest.main()
