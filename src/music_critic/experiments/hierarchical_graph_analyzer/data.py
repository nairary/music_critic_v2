"""Torch-free, target-only HIERGRAPH runtime adapter.

Analytic views, exact source geometry and label availability live on this side
of the model boundary. This module constructs no encoder inputs or candidates.
"""
from collections import Counter
from copy import deepcopy
from fractions import Fraction
from hashlib import sha256
from pathlib import Path

from music_critic.experiments.hiergraph_export import (
    checked_read, _json_bytes, _require, fingerprint, read_gzip_json,
)


def exact_time(value):
    """Accept exact rational encodings only; float geometry cannot be restored."""
    if isinstance(value, Fraction):
        return value
    if type(value) is int:
        return Fraction(value)
    if isinstance(value, dict) and set(value) == {'num', 'den'}:
        numerator, denominator = value['num'], value['den']
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        numerator, denominator = value
    else:
        raise ValueError('exact rational time required')
    _require(type(numerator) is int and type(denominator) is int and denominator > 0,
             'integer rational with positive denominator required')
    return Fraction(numerator, denominator)


def _harmonies(numeric):
    _require(numeric.get('schema') == 'HiergraphNumericTargets@3', 'V3 numeric targets required')
    rows = numeric['targets']['harmonies']; count = len(rows['ids'])
    _require(all(isinstance(i, str) and i for i in rows['ids']) and len(set(rows['ids'])) == count,
             'unique harmony identities required')
    _require(len(rows['intervals']) == len(rows['view_ids']) == count, 'geometry array length mismatch')
    _require(all(v is None or isinstance(v, str) for v in rows['view_ids']), 'scalar analytic view required')
    _require(set(rows['values']) == set(rows['masks']), 'field mask/value identity mismatch')
    for field, values in rows['values'].items():
        masks = rows['masks'][field]
        _require(len(values) == len(masks) == count, 'field array length mismatch')
        _require(all(type(m) is bool and type(v) is int and ((m and v >= 0) or (not m and v == -1))
                     for v, m in zip(values, masks)), 'unavailable values require sentinel -1')
    for span in rows['intervals']:
        _require(len(span) == 2 and exact_time(span[0]) <= exact_time(span[1]), 'invalid H interval')
    return rows


def targets_for_tile(numeric, *, active_fields, core_start, core_end, halo_start, halo_end):
    """Partition target sets by analytic view with exact unique start ownership.

    Every original event is counted per view. Point events, nonowned starts and
    owned events extending beyond the halo remain explicit coverage exclusions.
    Boundaries are never clipped to create a different supervised H instance.
    """
    core_start, core_end, halo_start, halo_end = map(exact_time,
        (core_start, core_end, halo_start, halo_end))
    _require(halo_start <= core_start < core_end <= halo_end, 'positive nested core/halo required')
    rows = _harmonies(numeric)
    fields = tuple(sorted(active_fields))
    _require(all(f.startswith('H.') for f in fields), 'H-only runtime target heads required')
    _require(all(f in rows['values'] for f in fields), 'active field absent in numeric export')
    groups = {}
    for index, (identity, interval, view) in enumerate(zip(rows['ids'], rows['intervals'], rows['view_ids'])):
        if view not in groups:
            groups[view] = dict(indices=[], coverage=Counter(), outside=[], outside_known=Counter())
        group = groups[view]; coverage = group['coverage']; coverage['total_events'] += 1
        start, end = map(exact_time, interval)
        if start == end:
            coverage['point_events'] += 1
            coverage['point_events_owned'] += int(core_start <= start < core_end)
        elif not core_start <= start < core_end:
            coverage['not_owned'] += 1
        elif start < halo_start or end > halo_end:
            coverage['outside_halo'] += 1; group['outside'].append(identity)
            group['outside_known'].update(f for f in fields if rows['masks'][f][index])
        else:
            group['indices'].append(index); coverage['admitted_events'] += 1
    results = []
    duration = halo_end - halo_start
    for view, group in groups.items():
        indices = group['indices']
        spans = [tuple(map(exact_time, rows['intervals'][i])) for i in indices]
        coverage = {name: group['coverage'][name] for name in
                    ('total_events', 'point_events', 'point_events_owned', 'not_owned', 'outside_halo', 'admitted_events')}
        coverage['outside_halo_ids'] = group['outside']
        coverage['outside_halo_known_fields'] = {f:group['outside_known'][f] for f in fields}
        result = dict(ids=[rows['ids'][i] for i in indices], exact_spans=spans,
            spans=[[float((start-halo_start)/duration), float((end-halo_start)/duration)] for start,end in spans],
            fields={f:dict(values=[rows['values'][f][i] for i in indices],
                           mask=[rows['masks'][f][i] for i in indices]) for f in fields},
            view_id=view, scope_id='harmonies', known_negative_spans=[], ambiguous_spans=[],
            core_span=(core_start,core_end), halo_span=(halo_start,halo_end),coverage=coverage,
            evidence={rows['ids'][i]:deepcopy(numeric.get('evidence',{}).get(rows['ids'][i],{})) for i in indices})
        results.append(result)
    return results


class HiergraphTargetDataset:
    """Authenticated manifest/vocabulary plus a one-work bounded numeric cache."""
    def __init__(self, root, manifest, vocabulary, heads, *, max_work_bytes):
        self.root = Path(root).resolve()
        self.manifest, self.vocabulary, self.heads = manifest, vocabulary, heads
        self.works = {w['work_id']:w for w in manifest['works']}
        _require(len(self.works) == len(manifest['works']), 'duplicate manifest work')
        family_roles = {}
        for work in self.works.values():
            _require(work['split'] in ('fit','selection','dev'), 'global evaluation split forbidden')
            family = work['family_id']; role = work['split']
            _require(family not in family_roles or family_roles[family] == role, 'family split leakage')
            family_roles[family] = role
        self.value_maps = {name:deepcopy(values) for scope in vocabulary['event']['fields'].values()
                           for name,values in scope.items()}
        self.object_value_maps = deepcopy(vocabulary['objects']['fields'])
        process_maps = self.object_value_maps.get('harmonic_process', {})
        for name in vocabulary.get('active_fields', []):
            self.value_maps.setdefault(name, deepcopy(process_maps.get(name, [])))
        self.ontology_value_maps = {name:[deepcopy(row['value']) for row in rows]
                                    for name,rows in vocabulary.get('fit_support', {}).items()}
        self.active_fields = tuple(sorted(name for name,row in heads['fields'].items()
                                          if name.startswith('H.') and row['active']))
        _require(all(self.value_maps.get(name) for name in self.active_fields), 'active head missing fitted values')
        self.head_sizes = {name:len(self.value_maps[name]) for name in self.active_fields}
        self.max_work_bytes = max_work_bytes
        self._cached_id, self._cached_numeric = None, None

    def load_work(self, work_id):
        if self._cached_id == work_id:
            return self._cached_numeric
        work = self.works[work_id]
        # Release the previous shard before bounded decompression of another.
        self._cached_id, self._cached_numeric = None, None
        numeric = read_gzip_json(self.root, work['numeric'], self.max_work_bytes)
        _require(numeric.get('vocabulary_fingerprint') == self.vocabulary['fingerprint'],
                 'numeric/vocabulary fingerprint mismatch')
        _require(numeric['targets'].get('piece_id') == work['piece_id'], 'numeric piece identity mismatch')
        rows = _harmonies(numeric)
        for field,values in rows['values'].items():
            size = len(self.value_maps.get(field, []))
            _require(all(not mask or value < size for value,mask in zip(values,rows['masks'][field])),
                     'numeric ID outside fitted vocabulary')
        self._cached_id, self._cached_numeric = work_id, numeric
        return numeric

    def targets_for_tile(self, work_id, *, core_start, core_end, halo_start, halo_end):
        work = self.works[work_id]
        rows = targets_for_tile(self.load_work(work_id), active_fields=self.active_fields,
            core_start=core_start,core_end=core_end,halo_start=halo_start,halo_end=halo_end)
        for row in rows:
            row.update(work_id=work_id,piece_id=work['piece_id'],family_id=work['family_id'],split=work['split'],
                       vocabulary_fingerprint=self.vocabulary['fingerprint'])
        return rows


def load_target_dataset(root, *, manifest_sha256, max_work_bytes=1024**3):
    """Authenticate external SHA before metadata decode, then every bound file."""
    root=Path(root).resolve();path=root/'manifest.json'
    _require(not path.is_symlink() and path.is_file() and path.stat().st_size <= 64*1024**2,
             'bounded regular manifest required')
    raw=path.read_bytes();_require(sha256(raw).hexdigest()==manifest_sha256,'manifest SHA mismatch')
    manifest=_json_bytes(raw)
    _require(manifest.get('schema')=='HiergraphDataset@1' and manifest.get('status')=='completed'
             and manifest.get('fingerprint')==fingerprint(manifest), 'completed authenticated export required')
    _require(manifest.get('test_locked') is True,'TEST must remain locked')
    vocabulary=_json_bytes(checked_read(root,manifest['vocabulary'],64*1024**2))
    heads=_json_bytes(checked_read(root,manifest['heads'],4*1024**2))
    _require(vocabulary.get('schema')=='HiergraphVocabulary@3'
             and vocabulary.get('fingerprint')==fingerprint(vocabulary),'V3 vocabulary fingerprint required')
    _require(heads.get('schema')=='HiergraphTrainableHeads@1' and
             heads.get('fingerprint')==fingerprint(heads),'heads fingerprint mismatch')
    _require(heads.get('vocabulary_fingerprint')==vocabulary['fingerprint'],'heads/vocabulary mismatch')
    _require(manifest['contract_fingerprint']==vocabulary['contract_fingerprint']==heads['contract_fingerprint'],
             'projection contract mismatch')
    _require(manifest['split_fingerprint']==vocabulary['split_fingerprint'],'fit split mismatch')
    return HiergraphTargetDataset(root,manifest,vocabulary,heads,max_work_bytes=max_work_bytes)
