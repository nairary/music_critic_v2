"""Synthetic cache geometry/provenance checks, not corpus-quality evidence."""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from music_critic.data import load_piece, dumps_piece
from music_critic.pll.contracts import ContextProfile, MaskQuery
from music_critic.pll.graph import build_tile_graph
from music_critic.pll.masking import build_observed_tiles
from music_critic.pll.observed_view import ObservedMusicView
from music_critic.relation_probe.data import sanitize
from music_critic.experiments.harmonic_event_reasoner.cache import (
    BASE_NODES, EXTRA_EDGES, LIMITS, UnsupportedHGeometry,
    assemble_tile, build_cache, edge_schema,
)
from music_critic.experiments.harmonic_event_reasoner.evidence import binding


def fixture():
    return sanitize(load_piece(Path(__file__).parent / "fixtures/data/canonical_piece_v2.json"))


def encoded_states(graph):
    result = {}
    for index, node in enumerate(BASE_NODES):
        count = graph.slice_features.shape[0] if node == "slice" else graph.base[node].num_nodes
        result[node] = torch.arange(count * 128, dtype=torch.float32).reshape(count, 128) / 1000 + index
    return result


def harmony():
    return dict(id="H:0", start=[0, 1], end=[2, 1], view_id="view:0", labels=[
        dict(note_id="note:melody-000", start=[0, 1], end=[2, 1], value=True, origin="synthetic", confidence=1),
        dict(note_id="note:melody-001", start=[1, 1], end=[2, 1], value=False, origin="synthetic", confidence=1),
    ])


class CacheGeometryTests(unittest.TestCase):
    def setUp(self):
        self.piece = fixture()
        start = self.piece.bars[0].start_qn
        end = self.piece.bars[-1].start_qn + self.piece.bars[-1].duration_qn
        view = ObservedMusicView(self.piece, MaskQuery(start, end, tuple(t.track_id for t in self.piece.tracks)),
                                 tuple(range(len(self.piece.bars))))
        self.tile = build_observed_tiles(view, ContextProfile())[0]
        self.graph = build_tile_graph(self.tile)
        self.states = encoded_states(self.graph)

    def assemble(self, hs):
        return assemble_tile(self.piece, self.tile, self.graph, self.states, hs)

    def test_sidecar_changes_neither_candidate_universe_nor_features_or_edges(self):
        first = harmony()
        unknown = deepcopy(first)
        unknown["labels"] = []
        a, b = self.assemble([first]), self.assemble([unknown])
        for key in ("x", "node_type", "edge_index", "edge_type"):
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        qa, qb = a["queries"][0], b["queries"][0]
        self.assertEqual(qa["candidate_geometry"], qb["candidate_geometry"])
        self.assertEqual(a["oracle_geometry_sha256"], b["oracle_geometry_sha256"])
        self.assertEqual(len(qa["candidates"]), 3)
        self.assertEqual(qa["mask"].tolist(), [True, True, True])
        self.assertEqual(qb["mask"].tolist(), [False, False, False])
        self.assertEqual(qa["y"].tolist(), [1.0, 1.0, 0.0])
        # H pooling includes the explicitly nonmember candidate as well.
        torch.testing.assert_close(a["x"][qa["source"]], a["x"][qa["candidates"]].mean(0))

    def test_fractional_H_bounds_split_contexts_without_changing_raw_slices(self):
        h = harmony()
        h.update(start=[1, 2], end=[3, 2])
        for label in h["labels"]:
            label["start"] = [1, 2] if label["note_id"].endswith("000") else [1, 1]
            label["end"] = [3, 2]
        raw_times = tuple(self.graph.slots[0].time_slots)
        result = self.assemble([h])
        contexts = result["queries"][0]["candidate_geometry"]
        self.assertEqual([(x["start"], x["end"]) for x in contexts], [([1, 2], [1, 1]), ([1, 1], [3, 2]), ([1, 1], [3, 2])])
        self.assertEqual(tuple(self.graph.slots[0].time_slots), raw_times)
        self.assertNotIn(Fraction(1, 2), [t for t, _ in raw_times])

    def test_geometric_H_edges_include_known_nonmember_and_unknown(self):
        h = harmony()
        h["labels"] = []
        result = self.assemble([h])
        query = result["queries"][0]
        tag = edge_schema().index(EXTRA_EDGES[4])
        edges = result["edge_index"][:, result["edge_type"] == tag]
        self.assertEqual(edges[0].tolist(), [query["source"]] * 3)
        self.assertEqual(edges[1].tolist(), query["candidates"].tolist())
        self.assertTrue((result["node_type"][query["candidates"]] == 7).all())
        self.assertEqual(result["node_type"][query["source"]].item(), 8)

    def test_identical_raw_queries_mask_conflicting_views(self):
        a, b = harmony(), harmony()
        b.update(id="H:alternative", view_id="other")
        for label in b["labels"]:
            label["value"] = not label["value"]
        result = self.assemble([a, b])
        qa, qb = result["queries"]
        self.assertNotEqual(qa["view_id"], qb["view_id"])
        torch.testing.assert_close(qa["candidates"], qb["candidates"])
        torch.testing.assert_close(result["x"][qa["source"]], result["x"][qb["source"]])
        torch.testing.assert_close(qa["y"], 1 - qb["y"])
        self.assertFalse(qa['mask'].any())
        self.assertFalse(qb['mask'].any())
        self.assertEqual(qa['equivalent_view_conflicts'], 3)

    def test_whole_event_must_fit_tile_and_scope_is_not_inferred_from_labels(self):
        h = harmony()
        h["end"] = [6, 1]
        result = self.assemble([h])
        self.assertFalse(result["queries"])
        self.assertEqual(result["exclusions"][0]["reason"], "whole_H_does_not_fit_tile")
        invalid = harmony()
        invalid["scope_track_ids"] = ["track:unknown"]
        self.assertEqual(self.assemble([invalid])["exclusions"][0]["reason"], "unsupported_explicit_H_track_scope")

    def test_candidate_limit_excludes_instead_of_truncating(self):
        with patch.dict(LIMITS, max_contexts_per_tile=1):
            with self.assertRaisesRegex(UnsupportedHGeometry, "context_count_limit"):
                self.assemble([harmony()])


class CacheBuildTests(unittest.TestCase):
    def dataset(self, root):
        raw = root / "piece.json"
        raw.write_text(dumps_piece(fixture()))
        target = root / "targets.json"
        target.write_text(json.dumps([harmony()]))
        row = dict(id="work:fixture", family="family:fixture", split="train",
                   raw=binding(raw), targets=binding(target))
        (root / "manifest.json").write_text(json.dumps(dict(works=[row])))
        return row

    def test_builder_freezes_encoder_and_saves_bound_cpu_cache(self):
        class TestEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))
                self.calls = 0

            def _encode_graph(self, graph, flags):
                assert not self.training and not self.weight.requires_grad
                assert not torch.is_grad_enabled()
                assert flags.shape == (graph.base["note"].num_nodes, 2)
                self.calls += 1
                return encoded_states(graph)

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            row = self.dataset(root)
            encoder = TestEncoder()
            with patch("music_critic.pll.encoder.PLLContextEncoder.from_pretrained", return_value=encoder) as load:
                result = build_cache(root, root / "cache", root / "checkpoint", "a" * 64, 17, device="cpu")
            self.assertEqual(result["status"], "completed")
            self.assertEqual(encoder.calls, 1)
            self.assertEqual(result["counts"]["train_mixed_h"], 1)
            self.assertEqual(result["counts"]["train_positive"], 2)
            self.assertEqual(result["counts"]["train_negative"], 1)
            self.assertLessEqual(len(result["edge_schema"]), 64)
            cached = result["tiles"][0]
            path = root / "cache" / cached["path"]
            self.assertEqual(binding(path)["sha256"], cached["sha256"])
            payload = torch.load(path, weights_only=True)
            self.assertEqual(payload["x"].device.type, "cpu")
            self.assertEqual(payload["parent_raw_sha256"], row["raw"]["sha256"])
            self.assertEqual(load.call_args.kwargs["expected_sha256"], "a" * 64)
            self.assertEqual(len({q["h_id"] for q in payload["queries"]}), len(payload["queries"]))

    def test_tampered_admitted_payload_fails_with_partial_manifest(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self.dataset(root)
            (root / "targets.json").write_text("[]")
            with patch("music_critic.pll.encoder.PLLContextEncoder.from_pretrained", return_value=torch.nn.Linear(1, 1)):
                with self.assertRaisesRegex(ValueError, "admission size mismatch"):
                    build_cache(root, root / "cache", root / "checkpoint", "a" * 64, 17, device="cpu")
            manifest = json.loads((root / "cache" / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["errors"][0]["type"], "ValueError")


if __name__ == "__main__":
    unittest.main()
