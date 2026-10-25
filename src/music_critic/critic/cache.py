"""Individually bound frozen feature entries using weights-only deserialization."""
from __future__ import annotations

from dataclasses import asdict
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import re
import tempfile

import torch

from .contracts import (CriticFeaturePyramid, FeatureBinding, FeatureSample,
                        RawInterval, RawNote, SampleGeometry)
from .features import FEATURE_LEVELS, expected_stages, geometry_payload

CACHE_SCHEMA = 'FrozenCriticFeatureCache@1.0.0'
# FP32 encoder batching may change reduction roundoff. Existing bytes stay immutable.
FEATURE_ATOL = 1e-5
FEATURE_RTOL = 1e-5


def _sha(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _is_sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _validate_binding(binding):
    if not isinstance(binding, FeatureBinding) or not _is_sha(binding.model_state_id):
        raise ValueError('cache requires known model state binding')
    if binding.dtype != 'float32':
        raise ValueError('unsupported cache dtype binding')
    expected_stages(binding.adapter_version)


def _validate_sample(sample, binding):
    _validate_binding(binding)
    if not _is_sha(sample.raw_hash):
        raise ValueError('cache requires a raw hash')
    if dict(sample.stages) != expected_stages(binding.adapter_version):
        raise ValueError('cache stage binding mismatch')
    if set(sample.embeddings) != set(FEATURE_LEVELS):
        raise ValueError('cache feature level mismatch')
    geometry = sample.geometry
    counts = {'note_local': len(geometry.notes), 'note': len(geometry.notes),
              'track': len(geometry.tracks), 'song': 1}
    counts.update({level: len(geometry.intervals[level]) for level in ('onset', 'beat', 'bar', 'slice')})
    for level, value in sample.embeddings.items():
        if (not isinstance(value, torch.Tensor) or value.requires_grad or value.grad_fn is not None
                or value.dtype != torch.float32 or tuple(value.shape) != (counts[level], 128)
                or not torch.isfinite(value).all()):
            raise ValueError('cache requires finite detached aligned float32 features')


def _metadata(sample, binding):
    return {'schema': CACHE_SCHEMA, 'binding': asdict(binding), 'raw_hash': sample.raw_hash,
            'stages': dict(sample.stages), 'geometry': geometry_payload(sample.geometry)}


def _entry_key(metadata):
    return _sha(metadata)


def _content_hash(metadata, embeddings):
    digest = sha256(_sha(metadata).encode())
    for level in FEATURE_LEVELS:
        value = embeddings[level].detach().cpu().contiguous()
        digest.update(level.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _geometry(payload):
    return SampleGeometry(payload['sample_id'], tuple(
        RawNote(n[0], n[1], Fraction(*n[2]), Fraction(*n[3]), n[4], n[5]) for n in payload['notes']),
        tuple(payload['tracks']),
        {level: tuple(RawInterval(i[0], Fraction(*i[1]), Fraction(*i[2]), i[3]) for i in rows)
         for level, rows in payload['intervals'].items()}, Fraction(*payload['duration']))


class FrozenFeatureCache:
    """Each raw original/variant has its own content-addressed file.

    The caller supplies expected raw and backbone bindings on every read. A
    checksum detects accidental corruption; this is not an artifact signature.
    """
    def __init__(self, root):
        self.root = Path(root)

    def put(self, pyramid: CriticFeaturePyramid) -> tuple[str, ...]:
        _validate_binding(pyramid.binding)
        for sample in pyramid.samples:
            _validate_sample(sample, pyramid.binding)
        self.root.mkdir(parents=True, exist_ok=True)
        keys = []
        for sample in pyramid.samples:
            metadata = _metadata(sample, pyramid.binding)
            embeddings = {level: value.detach().cpu().clone() for level, value in sample.embeddings.items()}
            key = _entry_key(metadata)
            content_hash = _content_hash(metadata, embeddings)
            destination = self.root / (key + '.pt')
            if destination.exists():
                previous = self.get(key, pyramid.binding, sample.raw_hash)
                if any(not torch.allclose(previous.embeddings[level], embeddings[level],
                                          atol=FEATURE_ATOL, rtol=FEATURE_RTOL)
                       for level in FEATURE_LEVELS):
                    raise ValueError('conflicting features for the same frozen cache binding')
            else:
                with tempfile.NamedTemporaryFile(dir=self.root, suffix='.tmp', delete=False) as stream:
                    temporary = Path(stream.name)
                try:
                    torch.save({'metadata': metadata, 'embeddings': embeddings,
                                'content_hash': content_hash}, temporary)
                    temporary.replace(destination)
                finally:
                    temporary.unlink(missing_ok=True)
            keys.append(key)
        return tuple(keys)

    def get(self, key: str, expected_binding: FeatureBinding, expected_raw_hash: str) -> FeatureSample:
        _validate_binding(expected_binding)
        if not _is_sha(key) or not _is_sha(expected_raw_hash):
            raise ValueError('invalid cache key or expected raw hash')
        # Loading inside inference_mode must still yield normal detached constants
        # which autograd can save when training the new critic.
        with torch.inference_mode(False):
            payload = torch.load(self.root / (key + '.pt'), map_location='cpu', weights_only=True)
        try:
            metadata = payload['metadata']
            if metadata['schema'] != CACHE_SCHEMA:
                raise ValueError('cache schema mismatch')
            if metadata['binding'] != asdict(expected_binding):
                raise ValueError('cache binding mismatch')
            if metadata['raw_hash'] != expected_raw_hash:
                raise ValueError('cache raw hash mismatch')
            if _entry_key(metadata) != key:
                raise ValueError('cache metadata integrity mismatch')
            sample = FeatureSample(_geometry(metadata['geometry']), payload['embeddings'],
                                   metadata['raw_hash'], metadata['stages'])
            _validate_sample(sample, expected_binding)
            if _content_hash(metadata, sample.embeddings) != payload['content_hash']:
                raise ValueError('cache content integrity mismatch')
        except (KeyError, TypeError, IndexError, ZeroDivisionError) as error:
            raise ValueError('invalid cache payload') from error
        return sample
