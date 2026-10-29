"""Whole-piece raw routing and separate source-scoped H/h/P target adapter.

The raw assembler accepts only a sanitized canonical piece and the old frozen
raw cache. Numeric theory exports are consumed exclusively by targets_for_work.
No physical note incidence is interpreted as chord membership or process role.
"""
from collections import Counter
from copy import deepcopy
from itertools import islice

from .data import exact_time


def _require(test, message):
    if not test:
        raise ValueError(message)


def _empty_rows():
    return dict(ids=[], intervals=[], view_ids=[], values={}, masks={})


def _rows(value, name):
    """Validate native parallel arrays, including masked integer sentinels."""
    value = _empty_rows() if value is None else value
    count = len(value['ids'])
    _require(all(isinstance(x, str) and x for x in value['ids'])
             and len(set(value['ids'])) == count, 'unique ' + name + ' IDs required')
    _require(len(value['intervals']) == len(value['view_ids']) == count,
             name + ' geometry array length mismatch')
    _require(all(v is None or isinstance(v, str) for v in value['view_ids']),
             'scalar analytic view required')
    _require(set(value['values']) == set(value['masks']), 'field mask/value identity mismatch')
    for field, vals in value['values'].items():
        masks = value['masks'][field]
        _require(len(vals) == len(masks) == count, 'field array length mismatch')
        _require(all(type(v) is int and type(m) is bool and
                     ((m and v >= 0) or (not m and v == -1)) for v, m in zip(vals, masks)),
                 'unavailable values require sentinel -1')
    for span in value['intervals']:
        _require(len(span) == 2 and exact_time(span[0]) <= exact_time(span[1]),
                 'invalid ' + name + ' interval')
    for field, vals in value.get('states', {}).items():
        _require(len(vals) == count, 'field states array length mismatch')
    return value


def full_head_sizes(dataset):
    """Lift only the first-slice P phase gate; keep fitted-support/PU gates."""
    result = {'H': {}, 'h': {}, 'P': {}}
    maps = dict(dataset.value_maps)
    maps.update(dataset.object_value_maps.get('harmonic_process', {}))
    for name, row in sorted(dataset.heads['fields'].items()):
        scope = name.split('.', 1)[0]
        if scope not in result:
            continue
        reasons = [r for r in row.get('deferred_reasons', [])
                   if r != 'P_evidence_only_initial_slice']
        # Unknown inactive reasons must never be promoted merely by omission.
        eligible = row.get('active') is True or (
            scope == 'P' and 'P_evidence_only_initial_slice' in row.get('deferred_reasons', [])
            and not reasons)
        if eligible and not reasons:
            _require(bool(maps.get(name)), 'active head missing fitted values: ' + name)
            result[scope][name] = len(maps[name])
    return result


def targets_for_work(numeric, *, active_fields, origin, duration):
    """Adapt native V3 export without merging views or inventing absent labels.

    Exact point objects and out-of-work intervals are retained with coverage
    counters. Unsupported relation/participant addresses retain their identity
    and known-positive status so metrics can expose unrecoverable support.
    """
    _require(numeric.get('schema') == 'HiergraphNumericTargets@3', 'V3 numeric targets required')
    origin, duration = exact_time(origin), exact_time(duration)
    _require(duration > 0, 'positive whole-piece duration required')
    fields = tuple(sorted(set(active_fields)))
    _require(all(f.split('.', 1)[0] in ('H', 'h', 'P') for f in fields),
             'H/h/P fields required')
    native = numeric['targets']
    H = _rows(native.get('harmonies'), 'H')
    h = _rows(native.get('note_contexts'), 'h')
    P = _rows(numeric.get('objects', {}).get('groups', {}).get('harmonic_process'), 'P')
    for key in ('note_indices', 'harmony_ids'):
        _require(len(h.get(key, [])) == len(h['ids']), 'h ' + key + ' array length mismatch')
    _require(all(identity is None or isinstance(identity, str) for identity in h.get('harmony_ids', [])),
             'h harmony identities required')
    if 'applicability' in h:
        _require(len(h['applicability']) == len(h['ids']), 'h applicability array length mismatch')
    _require(all(type(i) is int and i >= 0 for i in h.get('note_indices', [])),
             'h canonical note indices required')
    _require(len(P.get('member_ids', [])) == len(P['ids']), 'P member array length mismatch')
    for members in P.get('member_ids', []):
        _require(isinstance(members, (tuple, list)) and all(isinstance(i, str) for i in members)
                 and len(set(members)) == len(members), 'unique P member IDs required')
    groups = {'H': H, 'h': h, 'P': P}
    for level, rows in groups.items():
        _require(all(f in rows['values'] for f in fields if f.startswith(level + '.') and rows['ids']),
                 'active field absent in numeric export: ' + level)
    note_ids = {}
    for row in native.get('participants', []):
        if row.get('kind') == 'note':
            identity, index = row['id'], row['canonical_note_index']
            _require(type(index) is int and index >= 0 and identity not in note_ids,
                     'unique canonical note participants required')
            note_ids[identity] = index
    _require(set(note_ids.values()) == set(range(len(note_ids))), 'contiguous canonical note indices required')
    _require(all(i in set(note_ids.values()) for i in h.get('note_indices', [])),
             'h canonical note missing from participants')
    # View insertion order is stable even when the native null view is present.
    views = dict.fromkeys(v for rows in groups.values() for v in rows['view_ids'])
    relations = numeric.get('addressed_relations', [])
    for row in relations:
        views.setdefault(row.get('view_id', row.get('harmony_view')), None)
    if not views:
        views[None] = None
    results = []
    evidence = numeric.get('evidence', {})
    global_ids = {}
    for level, rows in groups.items():
        for identity, view in zip(rows['ids'], rows['view_ids']):
            _require(identity not in global_ids and identity not in note_ids, 'duplicate target identity')
            global_ids[identity] = (level, view)
    for view in views:
        output = {'view_id': view, **{key: numeric[key] for key in
            ('source_fingerprint', 'projection_fingerprint', 'vocabulary_fingerprint') if key in numeric}}
        index_maps = {}
        coverage = Counter()
        for level, rows in groups.items():
            indices = [i for i, v in enumerate(rows['view_ids']) if v == view]
            spans = [tuple(map(exact_time, rows['intervals'][i])) for i in indices]
            index_maps[level] = {rows['ids'][i]: j for j, i in enumerate(indices)}
            field_names = [f for f in fields if f.startswith(level + '.')]
            target = dict(ids=[rows['ids'][i] for i in indices], exact_spans=spans,
                geometry_eligible=[origin <= a < b <= origin+duration for a, b in spans],
                spans=[[float((a-origin)/duration), float((b-origin)/duration)] for a, b in spans],
                fields={f: dict(values=[rows['values'][f][i] for i in indices],
                                mask=[rows['masks'][f][i] for i in indices]) for f in field_names},
                known_negative_spans=[], ambiguous_spans=[],
                evidence={rows['ids'][i]: deepcopy(evidence.get(rows['ids'][i], {})) for i in indices})
            if rows.get('states'):
                target['states'] = {f: [rows['states'][f][i] for i in indices] for f in rows['states']}
            coverage[level + '_events'] = len(indices)
            coverage[level + '_point_events'] = sum(a == b for a, b in spans)
            coverage[level + '_outside_work'] = sum(a < origin or b > origin + duration for a, b in spans)
            if level == 'h':
                target['note_index'] = [rows['note_indices'][i] for i in indices]
                target['H_index'] = [index_maps['H'].get(rows['harmony_ids'][i], -1) for i in indices]
                target['harmony_ids'] = [rows['harmony_ids'][i] for i in indices]
                target['applicability'] = [rows.get('applicability', ['note_harmonic_context'] * len(rows['ids']))[i] for i in indices]
                coverage['h_without_H'] = target['H_index'].count(-1)
            if level == 'P':
                target['member_ids'] = [list(rows['member_ids'][i]) for i in indices]
            output[level] = target
        def endpoint(identity):
            if identity in note_ids:
                return dict(kind='n', index=note_ids[identity], id=identity)
            for level, mapping in index_maps.items():
                if identity in mapping:
                    return dict(kind=level, index=mapping[identity], id=identity)
            original = global_ids.get(identity)
            return dict(kind=original[0] if original else 'unsupported', index=-1, id=identity,
                        reason='other_view' if original else 'unsupported_address')
        output['participants'] = []
        for pi, members in enumerate(output['P']['member_ids']):
            for identity in members:
                ep = endpoint(identity)
                output['participants'].append(dict(P_index=pi, endpoint=ep, value=True, mask=True))
                coverage['participants'] += 1
                coverage['participants_without_H_h'] += int(ep['kind'] not in ('H', 'h') or ep['index'] < 0)
        output['relations'] = []
        for relation in relations:
            if relation.get('view_id', relation.get('harmony_view')) != view:
                continue
            value = relation.get('value')
            _require(value is None or type(value) is bool, 'boolean/unknown relation value required')
            _require(type(relation.get('available', False)) is bool, 'boolean relation availability required')
            result = deepcopy(relation)
            for key in ('source_ids', 'target_ids'):
                _require(isinstance(relation.get(key), (list, tuple)) and relation[key]
                         and all(isinstance(i, str) and i for i in relation[key])
                         and len(set(relation[key])) == len(relation[key]), 'nonempty unique relation endpoint set required')
            result.update(source=[endpoint(i) for i in relation['source_ids']],
                          target=[endpoint(i) for i in relation['target_ids']],
                          value=value, mask=bool(relation.get('available', False) and value is not None))
            if 'start_qn' in relation and 'end_qn' in relation:
                a, b = exact_time(relation['start_qn']), exact_time(relation['end_qn'])
                _require(a <= b, 'invalid relation interval')
                result['exact_span'] = (a, b)
            output['relations'].append(result)
            coverage['relations'] += 1
            coverage['relations_masked'] += int(not result['mask'])
            coverage['relations_unsupported_endpoints'] += int(any(e['index'] < 0 for e in result['source'] + result['target']))
        output['coverage'] = dict(coverage)
        results.append(output)
    return results


def assemble_work_raw(piece, tiles, *, max_tiles=2048, max_notes=200000):
    """Join old target-blind tiles using canonical physical note identities.

    Frozen cache rows are not averaged across overlapping windows. The model
    receives each detailed tile, its uniquely owned core, and its mapping into
    canonical notes. This function contains no labels or target-dependent gates.
    """
    import torch
    from .cache import NODE_TYPES, RAW_KEYS, UnsupportedRawGeometry, edge_schema
    _require(type(max_tiles) is int and max_tiles > 0 and type(max_notes) is int and max_notes > 0,
             'positive integer whole-work capacities required')
    if len(piece.notes) > max_notes:
        raise UnsupportedRawGeometry('whole_work_note_capacity')
    # One extra item is enough to detect an oversized lazy cache without
    # decompressing and retaining every tile before enforcing the budget.
    tiles = tuple(islice(iter(tiles), max_tiles + 1))
    if not tiles or len(tiles) > max_tiles:
        raise UnsupportedRawGeometry('whole_work_tile_capacity')
    _require(bool(piece.bars), 'raw piece needs observed meter bars')
    origin = piece.bars[0].start_qn.to_fraction()
    end = (piece.bars[-1].start_qn + piece.bars[-1].duration_qn).to_fraction()
    duration = end - origin
    _require(duration > 0, 'positive whole-piece duration required')
    notes = piece.notes
    note_ids = tuple(n.note_id for n in notes)
    _require(len(set(note_ids)) == len(note_ids), 'unique canonical physical notes required')
    track_index = {t.track_id: i for i, t in enumerate(piece.tracks)}
    note_spans = [(n.onset_qn.to_fraction(), (n.onset_qn+n.duration_qn).to_fraction()) for n in notes]
    # build_observed_tiles canonicalizes track names/order before clipping.
    _require(tuple(t.track_id for t in piece.tracks) == tuple(f'track:{i}' for i in range(len(piece.tracks))),
             'sanitized canonical track ordering required')
    track_order = track_index
    spans, cores, mappings, raw_tiles, owners = [], [], [], [], []
    coverage = Counter()
    represented = set()
    previous_end = origin
    for raw, metadata in tiles:
        _require(set(raw) == set(RAW_KEYS), 'fixed old raw tile schema required')
        a, b, ca, cb = [exact_time(metadata[k]) for k in ('halo_start', 'halo_end', 'core_start', 'core_end')]
        _require(origin <= a <= ca < cb <= b <= end, 'positive nested raw tile geometry required')
        if ca != previous_end:
            raise UnsupportedRawGeometry('whole_work_missing_or_overlapping_core')
        previous_end = cb
        times = tuple(map(exact_time, metadata['raw_times']))
        _require(len(times) == len(raw['time_positions']) and len(times) >= 2
                 and times[0] == a and times[-1] == b
                 and all(u < v for u, v in zip(times, times[1:])), 'exact aligned raw time grid required')
        positions = torch.tensor([float((t-a)/(b-a)) for t in times], dtype=raw['time_positions'].dtype)
        _require(torch.equal(raw['time_positions'].cpu(), positions), 'cache/exact raw time grid mismatch')
        owners.append(torch.tensor([i for i, t in enumerate(times) if ca <= t < cb], dtype=torch.long))
        selected = [i for i, (u, v) in enumerate(note_spans)
                    if (a <= u < b if u == v else u < b and v > a)]
        selected.sort(key=lambda i: (max(note_spans[i][0], a)-a,
             track_order[notes[i].track_id], notes[i].pitch,
             min(note_spans[i][1], b)-max(note_spans[i][0], a), notes[i].note_id))
        nodes = (raw['node_type'] == NODE_TYPES.index('note')).nonzero().flatten()
        _require(len(nodes) == len(selected), 'cache/canonical local note count mismatch')
        _require(torch.equal(nodes.cpu(), torch.arange(len(selected))), 'contiguous first note rows required')
        expected_times = torch.tensor([[float((note_spans[i][0]-a)/(b-a)),
                                        float((note_spans[i][1]-a)/(b-a))] for i in selected],
                                       dtype=raw['node_time'].dtype).reshape(-1, 2)
        _require(torch.equal(raw['node_time'][nodes].cpu(), expected_times),
                 'cache/canonical exact note geometry mismatch')
        # Exact metadata plus fragment->note edges independently authenticate the
        # reconstructed row ordering for all positive pitched notes.
        tag = edge_schema().index(('fragment', 'of_note', 'note'))
        links = raw['edge_index'][:, raw['edge_type'] == tag].cpu()
        linked = {}
        for source, target in links.T.tolist():
            _require(source not in linked, 'duplicate fragment note link')
            linked[source] = target
        for fragment in metadata.get('fragments', []):
            ni = linked.get(fragment['node'])
            _require(ni is not None and 0 <= ni < len(selected), 'fragment missing canonical note edge')
            _require(note_ids[selected[ni]] == fragment['note_id'], 'fragment canonical note identity mismatch')
            u, v = exact_time(fragment['start']), exact_time(fragment['end'])
            na, nb = note_spans[selected[ni]]
            _require(max(a, na) <= u < v <= min(b, nb), 'fragment outside physical note/halo')
        _require(len(linked) == len(metadata.get('fragments', [])), 'fragment metadata count mismatch')
        represented.update(selected)
        mappings.append(torch.tensor(selected, dtype=torch.long))
        raw_tiles.append(raw)
        spans.append((a, b)); cores.append((ca, cb))
        coverage['local_note_occurrences'] += len(selected)
        coverage['fragments'] += len(linked)
    if previous_end != end:
        raise UnsupportedRawGeometry('whole_work_missing_or_overlapping_core')
    if len(represented) != len(notes):
        raise UnsupportedRawGeometry('whole_work_unrepresented_canonical_notes')
    def normalized(intervals):
        return torch.tensor([[float((a-origin)/duration), float((b-origin)/duration)]
                             for a, b in intervals], dtype=torch.float32).reshape(-1, 2)
    bars = [(b.start_qn.to_fraction(), (b.start_qn+b.duration_qn).to_fraction()) for b in piece.bars]
    coverage.update(tiles=len(tiles), canonical_notes=len(notes), represented_notes=len(represented),
                    overlapping_note_occurrences=sum(len(x) for x in mappings)-len(notes), bars=len(bars))
    return dict(tiles=tuple(raw_tiles), tile_spans=normalized(spans), core_spans=normalized(cores),
                note_index_by_tile=tuple(mappings), owned_time_indices=tuple(owners), note_ids=note_ids, origin=origin, duration=duration,
                notes=dict(spans=normalized(note_spans), pitch=torch.tensor([n.pitch for n in notes], dtype=torch.long),
                           track_index=torch.tensor([track_index[n.track_id] for n in notes], dtype=torch.long),
                           is_percussion=torch.tensor([n.is_percussion for n in notes], dtype=torch.bool),
                           is_grace=torch.tensor([n.is_grace for n in notes], dtype=torch.bool)),
                bar_spans=normalized(bars), coverage=dict(coverage))
