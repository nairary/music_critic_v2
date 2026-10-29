"""Independent cache contract regressions using only canonical synthetic data.

The pretrained checkpoint acquisition is replaced with a tiny deterministic
nn.Module; canonical loading, sanitization, tiling, graph construction, exact
fragment assembly, serialization and cache authentication are real. This tests
cache contracts, not pretrained C3 quality or native hardware admission.
"""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from hashlib import sha256
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from music_critic.data import load_piece
from music_critic.graph import GraphBuildError
from music_critic.experiments.hiergraph_export import file_binding, fingerprint
from music_critic.experiments.hierarchical_graph_analyzer import cache
from music_critic.pll.contracts import ContextProfile, MaskQuery
from music_critic.pll.graph import build_tile_graph
from music_critic.pll.masking import build_observed_tiles
from music_critic.pll.observed_view import ObservedMusicView
from music_critic.relation_probe.data import sanitize

FIXTURE = Path(__file__).parents[1] / 'fixtures/data/canonical_piece_v2.json'


class _SyntheticEncoder(nn.Module):
    def __init__(self, *, nonfinite=False):
        super().__init__()
        self.calls = 0
        self.nonfinite = nonfinite

    def _encode_graph(self, graph, flags):
        self.calls += 1
        states = {name: torch.full((graph.slice_features.shape[0] if name == 'slice'
                                   else graph.base[name].num_nodes, 128), float(i))
                  for i, name in enumerate(cache.BASE_NODES)}
        if self.nonfinite:
            states['note'][0, 0] = float('nan')
        return states


class _NoTargetAccessDataset:
    manifest = {'fingerprint': 'synthetic-target-export'}

    def __init__(self, target_marker='unused'):
        self.works = {'work-fixture': {
            'work_id': 'work-fixture', 'family_id': 'family-fixture', 'split': 'fit',
            'source_bindings': {'canonical_piece': dict(file_binding(FIXTURE),
                role='canonical_piece', work_id='work-fixture')},
            # A deliberate trap: this nonexisting target location must never be read.
            'numeric': {'path': target_marker},
        }}

    def load_work(self, *args, **kwargs):
        raise AssertionError('raw cache attempted a target read')

    def targets_for_tile(self, *args, **kwargs):
        raise AssertionError('raw cache attempted a target read')


class CacheSpecificationReviewTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.piece = sanitize(load_piece(FIXTURE))

    def _build(self, output, *, encoder=None, target_marker='unused'):
        encoder = encoder or _SyntheticEncoder()
        with patch('music_critic.pll.encoder.PLLContextEncoder.from_pretrained', return_value=encoder), \
                redirect_stdout(io.StringIO()):
            manifest = cache.build_raw_cache(_NoTargetAccessDataset(target_marker), FIXTURE.parent,
                output, 'synthetic-checkpoint-not-opened', 'a' * 64, 1, device='cpu', wall_seconds=30)
        return manifest, encoder

    def _local_payload(self):
        start = self.piece.bars[0].start_qn
        end = self.piece.bars[-1].start_qn + self.piece.bars[-1].duration_qn
        view = ObservedMusicView(self.piece, MaskQuery(start, end, tuple(t.track_id for t in self.piece.tracks)),
                                 tuple(range(len(self.piece.bars))))
        tile = build_observed_tiles(view, ContextProfile())[0]
        graph = build_tile_graph(tile)
        state = _SyntheticEncoder()._encode_graph(graph, None)
        return cache.assemble_raw_tile(self.piece, tile, graph, state)

    def _context(self):
        return dict(ids=[b.bar_id for b in self.piece.bars],
                    states=torch.zeros(len(self.piece.bars), 128),
                    spans=[[cache.pair(b.start_qn.to_fraction()),
                            cache.pair((b.start_qn + b.duration_qn).to_fraction())]
                           for b in self.piece.bars])

    def test_all_raw_tiles_are_cached_without_gold_target_access(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(cache.LIMITS, core_bars=1, halo_bars=0):
            directory = Path(directory)
            first, encoder = self._build(directory / 'first', target_marker='no-labels-exist')
            second, _ = self._build(directory / 'second', target_marker='different-impossible-gold-path')
            one, two = first['works'][0], second['works'][0]
            self.assertEqual(encoder.calls, 2)
            self.assertEqual(len(one['tiles']), 2)
            self.assertEqual(one['exclusions'], [])
            self.assertEqual(one['context_coverage'], {'observed_bars': 2, 'encoded_bars': 2})
            self.assertEqual([(r['metadata']['core_start'], r['metadata']['core_end']) for r in one['tiles']],
                             [([0, 1], [1, 1]), ([1, 1], [5, 1])])
            for a, b in zip(one['tiles'], two['tiles']):
                left = torch.load(directory / 'first' / a['path'], weights_only=True)
                right = torch.load(directory / 'second' / b['path'], weights_only=True)
                self.assertEqual(set(left['raw']), set(cache.RAW_KEYS))
                self.assertEqual(left['metadata'], right['metadata'])
                for name in cache.RAW_KEYS:
                    torch.testing.assert_close(left['raw'][name], right['raw'][name], rtol=0, atol=0)

    def test_production_binding_already_contains_work_and_role(self):
        binding = _NoTargetAccessDataset().works['work-fixture']['source_bindings']['canonical_piece']
        self.assertEqual(binding['work_id'], 'work-fixture')
        self.assertEqual(binding['role'], 'canonical_piece')
        with tempfile.TemporaryDirectory() as directory:
            manifest, _ = self._build(Path(directory) / 'output')
        self.assertEqual(manifest['status'], 'completed')
        self.assertEqual(manifest['source_bindings'], [binding])
        self.assertTrue(manifest['works'][0]['tiles'])

    def test_excluded_tile_keeps_exact_core_and_halo_metadata(self):
        original = cache.assemble_raw_tile
        def capacity_failure(piece, tile, graph, encoded):
            if tile.core_bar_ids == (piece.bars[0].bar_id,):
                raise cache.UnsupportedRawGeometry('synthetic_fragment_capacity')
            return original(piece, tile, graph, encoded)
        with tempfile.TemporaryDirectory() as directory, patch.dict(cache.LIMITS, core_bars=1, halo_bars=1), \
                patch.object(cache, 'assemble_raw_tile', side_effect=capacity_failure):
            manifest, _ = self._build(Path(directory) / 'output')
        row = manifest['works'][0]
        self.assertEqual(len(row['tiles']), 1)
        self.assertEqual(row['exclusions'], [dict(tile=0, reason='synthetic_fragment_capacity',
            domain='raw_tile', core_start=[0, 1], core_end=[1, 1], halo_start=[0, 1], halo_end=[5, 1])])
        self.assertEqual(row['tiles'][0]['metadata']['core_start'], [1, 1])

    def test_whole_work_capacity_exclusion_keeps_full_exact_domain(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(cache.LIMITS, max_notes_per_tile=1):
            manifest, _ = self._build(Path(directory) / 'output')
        row = manifest['works'][0]
        self.assertEqual(row['tiles'], [])
        self.assertEqual(len(row['exclusions']), 1)
        exclusion = row['exclusions'][0]
        self.assertEqual(exclusion['domain'], 'whole_work')
        self.assertEqual((exclusion['core_start'], exclusion['core_end']), ([0, 1], [5, 1]))
        self.assertEqual((exclusion['halo_start'], exclusion['halo_end']), ([0, 1], [5, 1]))

    def test_validator_geometry_error_retains_exact_exclusion_and_continues(self):
        reason = 'build_raw_graph requires a validator-clean CanonicalPiece: BAR_INVALID@/bars/3'
        def graph_with_unsupported_first_tile(tile, profile):
            if tile.core_bar_ids == (self.piece.bars[0].bar_id,):
                raise GraphBuildError(reason)
            return build_tile_graph(tile, profile)
        with tempfile.TemporaryDirectory() as directory, patch.dict(cache.LIMITS, core_bars=1, halo_bars=0), \
                patch('music_critic.pll.graph.build_tile_graph', side_effect=graph_with_unsupported_first_tile):
            manifest, encoder = self._build(Path(directory) / 'output')
        row = manifest['works'][0]
        self.assertEqual(manifest['status'], 'completed')
        self.assertEqual(encoder.calls, 1)
        self.assertEqual(len(row['tiles']), 1)
        self.assertEqual(row['context_coverage'], {'observed_bars': 2, 'encoded_bars': 1})
        self.assertEqual(row['exclusions'], [dict(tile=0, reason=reason, domain='raw_tile',
            core_start=[0, 1], core_end=[1, 1], halo_start=[0, 1], halo_end=[1, 1])])
        self.assertEqual(row['tiles'][0]['metadata']['core_start'], [1, 1])

    def test_other_graph_build_error_remains_fatal(self):
        reason = 'feature registry contains a raw-inference-unsafe field'
        with tempfile.TemporaryDirectory() as directory, \
                patch('music_critic.pll.graph.build_tile_graph', side_effect=GraphBuildError(reason)):
            output = Path(directory) / 'output'
            with self.assertRaisesRegex(GraphBuildError, reason):
                self._build(output)
            receipt = json.loads((output / 'manifest.json').read_text())
        self.assertEqual(receipt['status'], 'failed')
        self.assertEqual(receipt['error'], {'type': 'GraphBuildError', 'reason': reason})
        self.assertEqual(receipt['works'][0]['exclusions'], [])

    def test_encoder_error_is_not_caught_as_graph_geometry(self):
        reason = 'build_raw_graph requires a validator-clean CanonicalPiece: synthetic encoder failure'
        class BrokenEncoder(_SyntheticEncoder):
            def _encode_graph(self, graph, flags):
                raise GraphBuildError(reason)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'output'
            with self.assertRaisesRegex(GraphBuildError, 'synthetic encoder failure'):
                self._build(output, encoder=BrokenEncoder())
            receipt = json.loads((output / 'manifest.json').read_text())
        self.assertEqual(receipt['status'], 'failed')
        self.assertEqual(receipt['works'][0]['exclusions'], [])

    def test_nonfinite_raw_state_is_failure_not_silent_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'output'
            with self.assertRaisesRegex(ValueError, 'finite aligned128d'):
                self._build(output, encoder=_SyntheticEncoder(nonfinite=True))
            receipt = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(receipt['status'], 'failed')
            self.assertEqual(receipt['error']['type'], 'ValueError')
            self.assertEqual(receipt['works'][0]['exclusions'], [])

    def test_final_edge_cap_includes_added_global_context_edges(self):
        payload = self._local_payload()
        existing_edges = payload['raw']['edge_type'].numel()
        context = self._context()
        before = {name: value.clone() for name, value in payload['raw'].items()}
        self.assertGreater(cache.attach_global_bars(payload, context)['edge_type'].numel(), existing_edges)
        with patch.dict(cache.LIMITS, max_flat_edges=existing_edges):
            with self.assertRaisesRegex(cache.UnsupportedRawGeometry, 'final_raw_edge_capacity'):
                cache.attach_global_bars(payload, context)
        for name in cache.RAW_KEYS:
            torch.testing.assert_close(payload['raw'][name], before[name], rtol=0, atol=0)

    def test_raw_profile_supports_17_and_64_tracks_through_real_c3_seam(self):
        from music_critic.pll.encoder import PLLContextEncoder
        from music_critic.pll.contracts import CodecProfile
        profile = cache.RawGraphProfile(max_tracks=64)
        # The existing conditional codec and public PLL profile stay bounded16.
        with self.assertRaises(ValueError):
            ContextProfile(max_tracks=17)
        with self.assertRaises(ValueError):
            CodecProfile(max_tracks=17)
        torch.manual_seed(17)
        encoder = PLLContextEncoder().eval().requires_grad_(False)
        for count in (17, 64):
            with self.subTest(tracks=count):
                tracks = tuple(replace(self.piece.tracks[0], track_id=f'track:{i}', source_track_index=i)
                               for i in range(count))
                notes = tuple(replace(self.piece.notes[0], note_id=f'note:synthetic-{i:03d}', track_id=f'track:{i}')
                              for i in range(count))
                piece = replace(self.piece, tracks=tracks, notes=notes)
                start = piece.bars[0].start_qn
                end = piece.bars[-1].start_qn + piece.bars[-1].duration_qn
                view = ObservedMusicView(piece, MaskQuery(start, end, tuple(t.track_id for t in tracks)),
                                         tuple(range(len(piece.bars))))
                tiles = build_observed_tiles(view, profile)
                self.assertEqual(len(tiles), 1)
                self.assertEqual(len(tiles[0].piece.tracks), count)
                graph = build_tile_graph(tiles[0], profile)
                flags = torch.tensor(tiles[0].boundary_flags, dtype=torch.float32).reshape(-1, 2)
                with torch.no_grad():
                    states = encoder._encode_graph(graph, flags)
                self.assertEqual(states['track'].shape, (count, 128))
                self.assertEqual(states['note'].shape, (count, 128))
                self.assertTrue(all(torch.isfinite(value).all() for value in states.values()))
                payload = cache.assemble_raw_tile(piece, tiles[0], graph, states)
                self.assertEqual(int((payload['raw']['node_type'] == 4).sum()), count)
                self.assertEqual({row['note_id'] for row in payload['metadata']['fragments']},
                                 {note.note_id for note in notes})

    def test_raw_profile_validates_own_finite_geometry_caps(self):
        for values in ({'max_tracks': 65}, {'max_tracks': True}, {'max_tracks': 0},
                       {'core_bars': 33, 'context_bars': 32}, {'halo_bars': -1},
                       {'max_notes_per_tile': 0}, {'max_edges_per_tile': 1.5},
                       {'max_slices_per_tile': False}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                cache.RawGraphProfile(**values)

    def test_valid_sha_cannot_license_raw_schema_drift(self):
        baseline = dict(schema='HiergraphRawCache@1', status='completed', works=[],
            dataset_fingerprint='dataset', node_schema=list(cache.NODE_TYPES),
            edge_schema=[list(edge) for edge in cache.edge_schema()],
            feature_contract='raw_C3_frozen_all_tiles_full_work_bars',
            limits=deepcopy(cache.LIMITS), test_locked=True)
        changes = {
            'node_schema': list(reversed(cache.NODE_TYPES)),
            'edge_schema': list(reversed(baseline['edge_schema'])),
            'feature_contract': 'old_semantically_different_features',
            'limits': dict(cache.LIMITS, max_times=999),
            'test_locked': False,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            def write(row):
                row = deepcopy(row)
                row['fingerprint'] = fingerprint(row)
                raw = json.dumps(row, sort_keys=True).encode()
                path.write_bytes(raw)
                return sha256(raw).hexdigest()
            cache.RawCache(directory, write(baseline), dataset_fingerprint='dataset')
            for name, value in changes.items():
                with self.subTest(field=name), self.assertRaisesRegex(ValueError, 'identity mismatch'):
                    cache.RawCache(directory, write(dict(baseline, **{name: value})), dataset_fingerprint='dataset')


if __name__ == '__main__':
    unittest.main()
