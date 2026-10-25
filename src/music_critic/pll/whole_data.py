"""Opt-in complete-source canonical TRAIN preparation and whole-piece sampling.

Complete-source means that no window is extracted from the supplied canonical
piece. Whether that source itself is a full composition, an excerpt or unknown
is a separate, evidence-backed declaration. This module never joins clips into
an invented song and never resolves corpus membership from file names.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import math
from pathlib import Path
import time
from typing import Mapping

from music_critic.data import dumps_piece, loads_piece
from music_critic.data.serialization import canonical_json_sha256
from .data import DataManifest, MaskSampler, SampledExample, seal_manifest
from .prepare import SourceCandidate, _json_once


SOURCES = frozenset(('dilemmadata', 'hooktheory', 'pop909_cl', 'pdmx', 'synthetic'))
COVERAGE = frozenset(('full', 'excerpt', 'unknown'))


@dataclass(frozen=True)
class WholeSourceCandidate(SourceCandidate):
    """Metadata only; paths are relative to an explicit source root."""
    native_split: str | None = None
    coverage: str = 'unknown'
    coverage_evidence: str = 'Source completeness has not been established'


def _safe_path(root: Path, relative: str) -> Path:
    if (not isinstance(relative, str) or not relative or '\\' in relative or '\0' in relative
            or Path(relative).is_absolute() or '..' in Path(relative).parts):
        raise ValueError('pll.whole_relative_path_required')
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('pll.whole_path_escape')
    return path


def _preflight(candidates, root, weights, provenance):
    if not isinstance(provenance, str) or not provenance.strip() or not candidates:
        raise ValueError('pll.whole_provenance_and_candidates_required')
    if len({c.record_id for c in candidates}) != len(candidates):
        raise ValueError('pll.whole_duplicate_record')
    if not weights or any(s not in SOURCES or isinstance(w, bool) or not isinstance(w, (int, float))
                         or not math.isfinite(w) or w <= 0 for s, w in weights.items()):
        raise ValueError('pll.whole_source_weights')
    group_splits, source_groups, hash_works = {}, {}, {}
    for c in candidates:
        if any(not isinstance(v, str) or not v.strip() for v in
               (c.record_id, c.source, c.piece_id, c.source_group_id, c.work_id, c.coverage_evidence)):
            raise ValueError('pll.whole_identity_required')
        if (c.source not in SOURCES or c.coverage not in COVERAGE or
                c.split not in ('train', 'validation', 'test') or
                c.native_split not in (None, 'train', 'validation', 'test')):
            raise ValueError('pll.whole_source_split_or_coverage')
        if (type(c.size_bytes) is not int or c.size_bytes < 1 or not isinstance(c.sha256, str)
                or len(c.sha256) != 64 or any(v not in '0123456789abcdef' for v in c.sha256)):
            raise ValueError('pll.whole_hash_required')
        _safe_path(root, c.path)
        for key in (('work', c.work_id), ('source', c.source, c.source_group_id), ('hash', c.sha256)):
            group_splits.setdefault(key, set()).add(c.split)
        key = c.source, c.source_group_id
        source_groups.setdefault(key, set()).add(c.work_id)
        hash_works.setdefault(c.sha256, set()).add(c.work_id)
    if any(len(values) != 1 for values in group_splits.values()):
        raise ValueError('pll.whole_group_crosses_split')
    if any(len(values) != 1 for values in source_groups.values()):
        raise ValueError('pll.whole_source_group_crosses_work_components')
    if any(len(values) != 1 for values in hash_works.values()):
        raise ValueError('pll.whole_identical_content_crosses_work_components')
    whole_keys = [(c.source, c.work_id) for c in candidates if c.split == 'train']
    if len(set(whole_keys)) != len(whole_keys):
        raise ValueError('pll.whole_one_record_per_source_work')
    # A native held-out hint closes its whole declared cross-source component.
    forbidden = {c.work_id for c in candidates if c.native_split in ('validation', 'test')}
    return tuple(c for c in candidates if c.split == 'train' and c.work_id not in forbidden)


class WholeDataManifest(DataManifest):
    """Existing v1 records plus a fingerprint-bound whole-piece sidecar."""

    def __init__(self, root: Path, payload: dict):
        super().__init__(root, payload)
        extra = payload.get('whole_piece')
        if (not isinstance(extra, dict) or extra.get('schema') != 'PLLWholePieceCoverage@1'
                or extra.get('sampling_unit') != 'complete_supplied_canonical_piece'
                or extra.get('clips_joined') is not False or
                not isinstance(extra.get('records'), dict) or set(extra['records']) != set(self.by_id)):
            raise ValueError('pll.whole_coverage_required')
        self.coverage = extra['records']
        allowed = extra.get('allowed_coverage')
        if (not isinstance(allowed, list) or not allowed or len(set(allowed)) != len(allowed)
                or not set(allowed) <= COVERAGE):
            raise ValueError('pll.whole_coverage_policy')
        pairs = [(r.source, r.work_id) for r in self.records]
        if len(set(pairs)) != len(pairs):
            raise ValueError('pll.whole_one_record_per_source_work')
        for record_id, row in self.coverage.items():
            digest = row.get('source_sha256') if isinstance(row, dict) else None
            if (not isinstance(row, dict) or row.get('coverage') not in allowed
                    or not isinstance(row.get('coverage_evidence'), str)
                    or not row['coverage_evidence'].strip() or row.get('source_piece_id') != self.by_id[record_id].piece_id
                    or row.get('source_complete') is not True or not isinstance(digest, str)
                    or len(digest) != 64 or any(v not in '0123456789abcdef' for v in digest)
                    or type(row.get('source_size_bytes')) is not int or row['source_size_bytes'] < 1
                    or type(row.get('bars')) is not int or row['bars'] < 1
                    or type(row.get('notes')) is not int or row['notes'] < 0):
                raise ValueError('pll.whole_coverage_invalid')

    def load_piece(self, record_id):
        piece = super().load_piece(record_id)
        row = self.coverage[record_id]
        end = piece.bars[-1].start_qn + piece.bars[-1].duration_qn
        if (len(piece.bars) != row['bars'] or len(piece.notes) != row['notes']
                or row.get('start_qn') != [piece.bars[0].start_qn.num, piece.bars[0].start_qn.den]
                or row.get('end_qn') != [end.num, end.den]):
            raise ValueError('pll.whole_source_boundary_mismatch')
        return piece


class WholeMaskSampler(MaskSampler):
    """Source/work-balanced queries whose predictor context is the entire piece."""

    def __init__(self, manifest: WholeDataManifest, *, seed: int, context_profile=None,
                 whole_config=None, length_weights=None, single_track_probability=.5):
        from .whole_contracts import WholePieceConfig
        if not isinstance(manifest, WholeDataManifest):
            raise ValueError('pll.whole_manifest_required')
        self.whole_config = whole_config or WholePieceConfig()
        super().__init__(manifest, seed=seed, context_profile=context_profile,
            length_weights=length_weights, single_track_probability=single_track_probability)
        self.identity = canonical_json_sha256(dict(schema='PLLWholeMaskSampler@1',
            local_sampler_identity=self.identity, whole_config=asdict(self.whole_config)))

    def draw(self) -> SampledExample:
        from .contracts import MaskQuery
        from .whole_contracts import split_whole_piece
        sources = tuple(sorted(self.groups))
        source = self.rng.choices(sources, [self.manifest.source_weights[s] for s in sources], k=1)[0]
        work = self.rng.choice(tuple(sorted(self.groups[source])))
        record_id = self.rng.choice(self.groups[source][work])
        piece = self.manifest.load_piece(record_id)
        if not piece.bars or not piece.tracks:
            raise ValueError('pll.metric_geometry_and_tracks_required')
        lengths = tuple(sorted(k for k in self.length_weights if k <= len(piece.bars)))
        if not lengths:
            raise ValueError('pll.no_declared_mask_length_fits')
        length = self.rng.choices(lengths, [self.length_weights[k] for k in lengths], k=1)[0]
        start = self.rng.randrange(len(piece.bars) - length + 1)
        first, last = piece.bars[start], piece.bars[start + length - 1]
        tracks = tuple(t.track_id for t in piece.tracks)
        if self.rng.random() < self.single_track_probability:
            tracks = (self.rng.choice(tracks),)
        query = MaskQuery(first.start_qn, last.start_qn + last.duration_qn, tracks)
        example = split_whole_piece(piece, query, whole_config=self.whole_config, context_profile=self.profile)
        result = SampledExample(record_id, source, work, self.draws, example)
        self.draws += 1
        return result


def prepare_whole_sources(candidates, output, *, source_root, source_weights: Mapping[str, float],
                          split_provenance: str, input_bindings=(), allowed_coverage=('full',),
                          whole_config=None):
    """Write whole canonical sources once; real runs require a registered caller."""
    candidates, source_root, output = tuple(candidates), Path(source_root).resolve(), Path(output)
    from .whole_contracts import WholePieceConfig
    whole_config = whole_config or WholePieceConfig()
    admitted = _preflight(candidates, source_root, source_weights, split_provenance)
    if (not allowed_coverage or len(set(allowed_coverage)) != len(allowed_coverage)
            or not set(allowed_coverage) <= COVERAGE):
        raise ValueError('pll.whole_coverage_policy')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'pieces').mkdir()
    started = time.monotonic()
    report = dict(schema='PLLWholePreparationReceipt@1', status='running', input_bindings=list(input_bindings),
        sampling_unit='complete_supplied_canonical_piece', clips_joined=False,
        allowed_coverage=list(allowed_coverage), resource_profile=asdict(whole_config),
        admission_scope='Canonical validity and conservative complete-piece bar/note bounds; '
                        'per-query codec, tile graph and native runtime admission remain separate',
        metadata_excluded_nontrain=len(candidates) - len(admitted), accesses=[], admitted=[], exclusions=[],
        verified_train_reads=0, test_payloads_read=0, validation_payloads_read=0,
        cross_source_near_duplicate_coverage='unknown', full_corpus_claim=False)
    records, coverage = [], {}
    try:
        for c in sorted(admitted, key=lambda c: c.record_id):
            if c.coverage not in allowed_coverage:
                report['exclusions'].append(dict(record_id=c.record_id, reason='coverage_policy', coverage=c.coverage))
                continue
            access = dict(record_id=c.record_id, declared_split=c.split, verified=False)
            report['accesses'].append(access)
            raw = _safe_path(source_root, c.path).read_bytes()
            if len(raw) != c.size_bytes or sha256(raw).hexdigest() != c.sha256:
                raise ValueError('pll.whole_source_hash_mismatch')
            piece = loads_piece(raw.decode('utf-8'))
            access['observed_split'] = piece.split
            access['observed_target_arrays'] = len(piece.targets)
            if piece.split not in (None, 'train'):
                raise ValueError('pll.whole_embedded_nontrain')
            if (piece.piece_id, piece.dataset_name, piece.source_group_id) != (c.piece_id, c.source, c.source_group_id):
                raise ValueError('pll.whole_source_identity_mismatch')
            if c.source == 'pop909_cl' and (any(n.channel == 1 for n in piece.notes)
                    or any(t.channel == 1 for t in piece.tracks)):
                raise ValueError('pll.whole_pop909_annotation_channel_forbidden')
            access['verified'] = True
            report['verified_train_reads'] += 1
            if not piece.bars or not piece.tracks:
                report['exclusions'].append(dict(record_id=c.record_id, reason='metric_geometry_and_tracks_required'))
                continue
            reason = ('whole_piece_bar_limit' if len(piece.bars) > whole_config.max_piece_bars else
                      'whole_piece_note_limit' if len(piece.notes) > whole_config.max_piece_notes else None)
            if reason:
                report['exclusions'].append(dict(record_id=c.record_id, reason=reason))
                continue
            # Retain all exact events and source geometry, no cropping/rebasing.
            piece = replace(piece, split='train', targets=(), annotations=(), key_signature_events=(), quality_flags=())
            data = dumps_piece(piece).encode('utf-8')
            digest = sha256(data).hexdigest()
            relative = 'pieces/' + digest + '.json'
            target = output / relative
            if not target.exists():
                with target.open('xb') as stream:
                    stream.write(data)
            records.append(dict(record_id=c.record_id, source=c.source, work_id=c.work_id,
                source_group_id=c.source_group_id, piece_id=c.piece_id, split='train', path=relative,
                sha256=digest, size_bytes=len(data)))
            coverage[c.record_id] = dict(coverage=c.coverage, coverage_evidence=c.coverage_evidence,
                source_complete=True, source_piece_id=c.piece_id, source_sha256=c.sha256,
                source_size_bytes=c.size_bytes, bars=len(piece.bars), notes=len(piece.notes),
                start_qn=[piece.bars[0].start_qn.num, piece.bars[0].start_qn.den],
                end_qn=[(piece.bars[-1].start_qn + piece.bars[-1].duration_qn).num,
                        (piece.bars[-1].start_qn + piece.bars[-1].duration_qn).den])
            report['admitted'].append(dict(record_id=c.record_id, **coverage[c.record_id]))
        if not records or set(source_weights) != {r['source'] for r in records}:
            raise ValueError('pll.whole_all_requested_sources_required')
        manifest = seal_manifest(dict(schema='PLLDataManifest@1', records=records,
            source_weights=dict(source_weights), group_before_window=True,
            cross_source_components_declared=True, split_provenance=split_provenance,
            whole_piece=dict(schema='PLLWholePieceCoverage@1', sampling_unit='complete_supplied_canonical_piece',
                             clips_joined=False, allowed_coverage=list(allowed_coverage),
                             preparation_profile=asdict(whole_config), records=coverage)))
        WholeDataManifest(output, manifest)
        _json_once(output / 'manifest.json', manifest)
        report.update(status='completed', manifest_fingerprint=manifest['fingerprint'], records=len(records),
                      source_counts=dict(Counter(r['source'] for r in records)),
                      coverage_counts=dict(Counter(r['coverage'] for r in coverage.values())))
        return report
    except BaseException as exc:
        report.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
        raise
    finally:
        unverified = [a for a in report['accesses'] if not a['verified']]
        report['unverified_reads'] = unverified
        for name in ('test', 'validation'):
            report[name + '_payloads_read'] = (None if any('observed_split' not in a for a in unverified)
                else sum(a.get('observed_split') == name for a in report['accesses']))
        report['wall_seconds'] = time.monotonic() - started
        _json_once(output / 'prepare-receipt.json', report)
