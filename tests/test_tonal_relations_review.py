"""Synthetic review-sidecar mechanics; these are not musical gold examples."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location(
    "tonal_review", Path(__file__).parents[1] / "scripts/tonal_relations_review.py"
)
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


def example():
    return {"schema": "tonal-relations-review@1", "title": "Synthetic",
            "cases": [{"id": "synthetic", "title": "Synthetic only", "genres": ["pop"],
                       "membership": "train", "time_unit": "quarter_note",
                       "time_origin": "source zero", "limitations": ["Synthetic"],
                       "assets": [{"locator": "source.json", "sha256": "a" * 64}],
                       "events": [{"id": "n1", "start": "1/3", "duration": "2/3",
                                   "pitch": 60, "voice": "part1/voice2", "source_ref": "row 7"}],
                       "claims": [{"id": "c1", "source_url": "https://example.org/paper",
                                   "source_anchor": "Example 1", "statement": "Synthetic assertion",
                                   "relation": "held_note", "status": "partial",
                                   "participants": [{"role": "held", "event_ids": ["n1"]}],
                                   "assessment": "Source notes bound; interpretation not checked",
                                   "alternatives": [], "missing": ["Independent interpretation"]}]}]}


class ReviewTests(unittest.TestCase):
    def test_exact_event_references_and_missing_evidence(self):
        data = example()
        result = review.validate(data)
        self.assertEqual(result["events"], 1)
        self.assertEqual(result["claim_statuses"], {"partial": 1})
        self.assertEqual(data["cases"][0]["claims"][0]["missing"], ["Independent interpretation"])

    def test_dangling_or_duplicate_event_ids_fail(self):
        for change in ("duplicate", "dangling"):
            with self.subTest(change=change):
                data = example()
                case = data["cases"][0]
                if change == "duplicate":
                    case["events"].append(copy.deepcopy(case["events"][0]))
                else:
                    case["claims"][0]["participants"][0]["event_ids"] = ["unknown"]
                with self.assertRaises(review.ReviewError):
                    review.validate(data)

    def test_no_float_time_or_negative_duration_or_boolean_pitch(self):
        for field, value in (("start", 0.5), ("start", "2/6"), ("duration", "-1/1"),
                             ("duration", "1/0"), ("pitch", True)):
            with self.subTest(field=field, value=value):
                data = example()
                data["cases"][0]["events"][0][field] = value
                with self.assertRaises(review.ReviewError):
                    review.validate(data)

    def test_pickup_origin_is_not_silently_shifted(self):
        data = example()
        data["cases"][0]["events"][0]["start"] = "-1/3"
        self.assertEqual(review.validate(data)["events"], 1)
        self.assertEqual(data["cases"][0]["events"][0]["start"], "-1/3")

    def test_context_references_preserve_nonselected_notes_and_must_exist(self):
        data = example()
        case = data["cases"][0]
        case["events"].append({"id": "n2", "start": "3/1", "duration": "1/1",
                               "pitch": 61, "voice": "part1/voice2", "source_ref": "row 8"})
        claim = case["claims"][0]
        claim["context_event_ids"] = ["n1", "n2"]
        self.assertEqual(review.validate(data)["events"], 2)
        self.assertEqual(claim["participants"][0]["event_ids"], ["n1"])
        for field in ("context_event_ids", "source_window_event_ids"):
            with self.subTest(field=field):
                bad = copy.deepcopy(data)
                bad["cases"][0]["claims"][0][field] = ["missing-note"]
                with self.assertRaises(review.ReviewError):
                    review.validate(bad)

    def test_missing_evidence_cannot_be_verified(self):
        data = example()
        claim = data["cases"][0]["claims"][0]
        claim.update(status="verified", participants=[])
        with self.assertRaises(review.ReviewError):
            review.validate(data)
        claim.update(status="partial", missing=[])
        with self.assertRaises(review.ReviewError):
            review.validate(data)

    def test_excluded_candidate_retained_without_events(self):
        data = example()
        case = data["cases"][0]
        case["membership"] = "excluded"
        with self.assertRaises(review.ReviewError):
            review.validate(data)
        case["events"] = []
        case["claims"][0].update(status="rejected", participants=[], missing=["Known holdout"])
        self.assertEqual(review.validate(data)["claim_statuses"], {"rejected": 1})

    def test_forbidden_memberships_and_unsafe_provenance_fail(self):
        for field, value in (("membership", "test"), ("membership", "validation"),
                             ("time_unit", "inferred_beats")):
            data = example()
            data["cases"][0][field] = value
            with self.assertRaises(review.ReviewError):
                review.validate(data)
        for path in ("/source.json", "../source.json", "C:\\source.json"):
            data = example()
            data["cases"][0]["assets"][0]["locator"] = path
            with self.assertRaises(review.ReviewError):
                review.validate(data)
        data = example()
        data["cases"][0]["claims"][0]["source_url"] = "javascript:alert(1)"
        with self.assertRaises(review.ReviewError):
            review.validate(data)

    def test_html_escapes_untrusted_text_and_preserves_source_values(self):
        data = example()
        data["title"] = '</script><img src=x onerror="alert(1)">'
        rendered = review.render_html(data)
        self.assertNotIn('<img src=x', rendered)
        payload = rendered.split('<script id="evidence" type="application/json">', 1)[1].split('</script>', 1)[0]
        self.assertEqual(json.loads(payload), data)
        self.assertIn("не прогноз модели", rendered)

    def test_output_is_created_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "review.html"
            review.write_html_once(example(), path)
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                review.write_html_once(example(), path)
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
