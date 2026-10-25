"""Hash-bound TRAIN manifests and source/work-balanced metric mask sampling.

A manifest records a caller's cross-source work grouping; its hashes verify
identity and internal split consistency, not the scientific completeness of
that grouping. No duplicate-work detector or TEST loader is hidden here.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import random
from typing import Mapping

from music_critic.data.serialization import canonical_json_sha256, loads_piece
from .contracts import ContextProfile, MaskQuery
from .masking import split_masked_piece
from .observed_view import MaskedExample


def strict_json(path: Path):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('pll.duplicate_json_key')
            result[key] = value
        return result
    return json.loads(path.read_text(), object_pairs_hook=object_pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('pll.nonfinite_json')))


def _split_identity(records):
    keys = ('record_id', 'source', 'work_id', 'source_group_id', 'piece_id', 'split', 'sha256')
    return canonical_json_sha256(sorted(
        [{key: record[key] for key in keys} for record in records], key=lambda row: row['record_id']))


def seal_manifest(payload: Mapping) -> dict:
    """Seal explicit caller-provided identities, without inventing provenance."""
    result = dict(payload)
    result.pop('fingerprint', None)
    result['split_group_fingerprint'] = _split_identity(result['records'])
    result['fingerprint'] = canonical_json_sha256(result)
    return result


@dataclass(frozen=True)
class DataRecord:
    record_id: str
    source: str
    work_id: str
    source_group_id: str
    piece_id: str
    split: str
    path: str
    sha256: str
    size_bytes: int


class DataManifest:
    def __init__(self, root: Path, payload: dict):
        if payload.get('schema') != 'PLLDataManifest@1':
            raise ValueError('pll.manifest_schema')
        if payload.get('fingerprint') != seal_manifest(payload)['fingerprint']:
            raise ValueError('pll.manifest_fingerprint')
        if payload.get('split_group_fingerprint') != _split_identity(payload['records']):
            raise ValueError('pll.split_group_fingerprint')
        if (payload.get('group_before_window') is not True or
                payload.get('cross_source_components_declared') is not True or
                not isinstance(payload.get('split_provenance'), str) or not payload['split_provenance'].strip()):
            raise ValueError('pll.split_provenance_required')
        self.root = root.resolve()
        self.fingerprint = payload['fingerprint']
        self.split_group_fingerprint = payload['split_group_fingerprint']
        self.records = tuple(DataRecord(**row) for row in payload['records'])
        if not self.records or len({r.record_id for r in self.records}) != len(self.records):
            raise ValueError('pll.empty_or_duplicate_records')
        # Check the complete manifest before opening any canonical file.
        if any(record.split != 'train' for record in self.records):
            raise ValueError('pll.train_only_manifest_TEST_closed')
        group_work, hash_work = {}, {}
        for record in self.records:
            for value in (record.record_id,record.source,record.work_id,record.source_group_id,record.piece_id):
                if not isinstance(value,str) or not value.strip():
                    raise ValueError('pll.record_identity_required')
            if (type(record.size_bytes) is not int or record.size_bytes < 1 or
                    not isinstance(record.sha256,str) or len(record.sha256)!=64 or
                    any(c not in '0123456789abcdef' for c in record.sha256)):
                raise ValueError('pll.record_hash_required')
            group = record.source, record.source_group_id
            if group in group_work and group_work[group] != record.work_id:
                raise ValueError('pll.source_group_crosses_work_components')
            if record.sha256 in hash_work and hash_work[record.sha256] != record.work_id:
                raise ValueError('pll.identical_content_crosses_work_components')
            group_work[group], hash_work[record.sha256] = record.work_id, record.work_id
            self._path(record)
        self.source_weights = dict(payload['source_weights'])
        if set(self.source_weights) != {r.source for r in self.records} or any(
            isinstance(w,bool) or not isinstance(w,(int,float)) or not math.isfinite(w) or w<=0
            for w in self.source_weights.values()
        ):
            raise ValueError('pll.source_weights_must_cover_sources')
        self.by_id = {r.record_id:r for r in self.records}
        self.verified_train_reads = 0
        self.unverified_reads: list[dict] = []

    @classmethod
    def load(cls, path: str | Path):
        path = Path(path)
        return cls(path.parent, strict_json(path))

    def _path(self, record):
        path = Path(record.path)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('pll.record_path_must_be_relative')
        target = (self.root/path).resolve()
        if not target.is_relative_to(self.root):
            raise ValueError('pll.record_path_escape')
        return target

    def load_piece(self, record_id: str):
        record = self.by_id[record_id]
        observed_split = None
        try:
            raw = self._path(record).read_bytes()
            if len(raw)!=record.size_bytes or sha256(raw).hexdigest()!=record.sha256:
                raise ValueError('pll.canonical_bytes_changed')
            piece = loads_piece(raw.decode('utf-8'))
            observed_split = piece.split
            if (piece.piece_id != record.piece_id or piece.dataset_name != record.source or
                    piece.source_group_id != record.source_group_id or piece.split != record.split):
                raise ValueError('pll.canonical_identity_or_split_mismatch')
        except Exception as exc:
            # A mismatched/unparseable payload cannot substantiate a claim of
            # verified TRAIN-only access, even though its manifest said TRAIN.
            self.unverified_reads.append(dict(record_id=record_id,declared_split=record.split,
                observed_split_claim=observed_split,error=type(exc).__name__+': '+str(exc)))
            raise
        self.verified_train_reads += 1
        return piece


@dataclass(frozen=True)
class SampledExample:
    record_id: str
    source: str
    work_id: str
    draw_index: int
    example: MaskedExample

    def receipt(self):
        query = self.example.context.query
        return dict(record_id=self.record_id,source=self.source,work_id=self.work_id,
            draw_index=self.draw_index, start=[query.start.num,query.start.den],
            end=[query.end.num,query.end.den],tracks=list(self.example.tracks))


def _tuples(value):
    return tuple(_tuples(v) for v in value) if isinstance(value,(list,tuple)) else value


class MaskSampler:
    def __init__(self, manifest: DataManifest, *, seed: int,
                 context_profile: ContextProfile | None = None,
                 length_weights: Mapping[int,float] | None = None,
                 single_track_probability: float = .5):
        if type(seed) is not int:
            raise ValueError('pll.sampler_seed')
        self.manifest = manifest
        self.profile = context_profile or ContextProfile()
        self.length_weights = {int(k):v for k,v in (length_weights or {1:.5,2:.3,4:.2}).items()}
        if (not self.length_weights or any(k not in (1,2,4) for k in self.length_weights) or
                any(isinstance(w,bool) or not isinstance(w,(int,float)) or not math.isfinite(w) or w<=0 for w in self.length_weights.values()) or
                not 0<=single_track_probability<=1):
            raise ValueError('pll.sampler_weights')
        self.single_track_probability = single_track_probability
        from dataclasses import asdict
        self.identity = canonical_json_sha256(dict(manifest=manifest.fingerprint,
            seed=seed,context=asdict(self.profile),lengths={str(k):v for k,v in self.length_weights.items()},
            single_track_probability=single_track_probability))
        groups=defaultdict(lambda:defaultdict(list))
        for record in manifest.records:
            groups[record.source][record.work_id].append(record.record_id)
        self.groups={source:{work:tuple(sorted(records)) for work,records in works.items()}
                     for source,works in groups.items()}
        self.rng=random.Random(seed)
        self.draws=0

    def state_dict(self):
        return dict(schema='PLLMaskSamplerState@1',identity=self.identity,
                    draws=self.draws,rng_state=self.rng.getstate())

    def load_state_dict(self, state):
        if (state.get('schema')!='PLLMaskSamplerState@1' or state.get('identity')!=self.identity or
                type(state.get('draws')) is not int or state['draws']<0):
            raise ValueError('pll.sampler_state_mismatch')
        rng=random.Random()
        rng.setstate(_tuples(state['rng_state']))
        self.rng,self.draws=rng,state['draws']

    def draw(self) -> SampledExample:
        sources=tuple(sorted(self.groups))
        source=self.rng.choices(sources,[self.manifest.source_weights[s] for s in sources],k=1)[0]
        work=self.rng.choice(tuple(sorted(self.groups[source])))
        record_id=self.rng.choice(self.groups[source][work])
        piece=self.manifest.load_piece(record_id)
        if not piece.bars or not piece.tracks:
            raise ValueError('pll.metric_geometry_and_tracks_required')
        lengths=tuple(sorted(k for k in self.length_weights if k<=len(piece.bars) and k<=self.profile.context_bars))
        if not lengths:
            raise ValueError('pll.no_declared_mask_length_fits')
        length=self.rng.choices(lengths,[self.length_weights[k] for k in lengths],k=1)[0]
        start=self.rng.randrange(len(piece.bars)-length+1)
        first,last=piece.bars[start],piece.bars[start+length-1]
        tracks=tuple(t.track_id for t in piece.tracks)
        if self.rng.random()<self.single_track_probability:
            tracks=(self.rng.choice(tracks),)
        query=MaskQuery(first.start_qn,last.start_qn+last.duration_qn,tracks)
        example=split_masked_piece(piece,query,self.profile)
        result=SampledExample(record_id,source,work,self.draws,example)
        self.draws+=1
        return result
