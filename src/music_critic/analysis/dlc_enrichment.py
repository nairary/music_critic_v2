"""Native DLC note annotations with exact row lineage, without invented harmony spans.

Pure functions: callers admit TRAIN metadata and authenticate bytes before use.
Source annotations have their original, limited note-row applicability. They do
not assert that a harmony remains active throughout a sustained canonical note.
"""
from collections import Counter, defaultdict
from fractions import Fraction
import json

from music_critic.analysis.target_binding import bind_source_events


HARMONIC_FIELDS = (
    'label', 'alt_label', 'globalkey', 'localkey', 'chord', 'special', 'numeral',
    'form', 'figbass', 'changes', 'relativeroot', 'chord_type', 'globalkey_mode',
    'localkey_mode', 'localkey_resolved', 'effective_localkey', 'applied_to_numeral',
    'root_tpc', 'bass_note_tpc', 'localkey_tpc', 'tonicized_tpc', 'chord_tones',
    'added_tones', 'a_simpleNumeral', 'a_degree1', 'a_degree2', 'a_root', 'a_bass',
    'a_localKey', 'a_tonicizedKey', 'a_quality', 'a_inversion', 'a_isOnset',
)
NOTE_FIELDS = ('note_degree', 'tpc_is_in_label', 'tpc_is_root', 'tpc_is_bass')
OTHER_FIELDS = ('cadence', 'cadence_type', 'phraseend', 'a_phraseend', 'pedal',
                'pedal_resolved', 'section_start')
BOOL_FIELDS = {'tpc_is_in_label','tpc_is_root','tpc_is_bass','a_isOnset','section_start','a_phraseend'}
EMPTY_MEANINGFUL = {'changes','alt_label','special','form','figbass','relativeroot','added_tones'}


def rational(value):
    q=Fraction(value)
    return {'num':q.numerator,'den':q.denominator}


def _time(value):
    q=Fraction(value)
    return f'{q.numerator}/{q.denominator}'


def _field(value, available, evidence, reason=None):
    return dict(value=value if available else None, available=bool(available),
                origin='source_native' if available else 'unknown', confidence=None,
                evidence=evidence, unknown_reason=None if available else reason,
                semantic_level='source_note_annotation')


def native_fields(row, *, usable, point, evidence):
    result={}
    for name in (*HARMONIC_FIELDS,*NOTE_FIELDS,*OTHER_FIELDS):
        raw=row.get(name)
        available=raw is not None and (raw != '' or name in EMPTY_MEANINGFUL)
        reason=None if available else 'source_missing'
        validity=('valid_cadence_label' if name in ('cadence','cadence_type') else
                  'valid_phrase_label' if name in ('phraseend','a_phraseend') else
                  'valid_pedal_point_label' if name in ('pedal','pedal_resolved') else
                  'valid_section_start_label' if name=='section_start' else 'valid_chord_label')
        if row.get(validity) != 'True': available=False; reason='source_validity_unavailable'
        if not usable: available=False; reason='source_binding_masked'
        if point and name in NOTE_FIELDS: available=False; reason='zero_duration_note'
        value=raw
        if name in BOOL_FIELDS and raw not in (None,''):
            if raw in ('True','False'): value=raw=='True'
            else: available=False; reason='invalid_source_boolean'
        result['dlc.'+name]=_field(value,available,evidence,reason)
    return result


def build_dlc_enrichment(piece, alignment, rows, metadata):
    if metadata.get('current_split')!='train' or piece.split not in (None,'train'):
        raise ValueError('TRAIN admission required')
    rid=metadata['record_id']; source_sha=metadata['source_sha256']
    version=f'{rid}@sha256:{source_sha}'
    events=[]
    for line,row in enumerate(rows,2):
        a=Fraction(row['quarterbeats_playthrough']); d=4*Fraction(row['duration'])
        events.append(dict(id=f'{rid}:source-row:{line}',source_version_id=version,
            source_row_id=f'{rid}#line={line}',source_row_number=line,start=_time(a),
            duration=_time(d),end=_time(a+d),pitch=int(row['pitch']),
            source_staff=row.get('staff'),source_voice=row.get('voice'),part_id=None,
            is_note_onset=True if row.get('is_note_onset')=='True' else False if row.get('is_note_onset')=='False' else None,
            time_unit='quarter_note'))
    bindings=bind_source_events(piece,alignment,dict(source_version_id=version,current_split='train',
        source_table={'sha256':source_sha},events=events),record_id=rid,source_sha256=source_sha,
        current_split='train',source_to_canonical_offset_qn=metadata.get('source_to_canonical_offset_qn','0/1'))
    contexts=[]; groups={}; omitted=[]; seen=set(); notes={n.note_id:(i,n) for i,n in enumerate(piece.notes)}
    for row,b in zip(rows,bindings):
        gid=row.get('unfolded_harmony_index','')
        raw={k:v for k,v in row.items() if k in (*HARMONIC_FIELDS,*NOTE_FIELDS,*OTHER_FIELDS) or k.startswith('valid_')}
        group=groups.setdefault(gid,dict(id=gid,raw_fields={},field_alternatives={},source_rows=[]))
        group['source_rows'].append(b.source_row_number)
        for key,value in raw.items():
            if key in NOTE_FIELDS: continue
            if key not in group['raw_fields']: group['raw_fields'][key]=value
            elif group['raw_fields'][key]!=value:
                group['field_alternatives'].setdefault(key,[group['raw_fields'][key]])
                if value not in group['field_alternatives'][key]:group['field_alternatives'][key].append(value)
        if b.canonical_note_id is None:
            omitted.append(dict(source_row_number=b.source_row_number,reason=b.mask_reason));continue
        seen.add(b.canonical_note_id); idx,note=notes[b.canonical_note_id]
        contexts.append(dict(id=b.source_event_id,note_id=b.canonical_note_id,canonical_note_index=idx,
            start_qn=rational(b.canonical_start_qn),end_qn=rational(b.canonical_end_qn),harmony_id=None,
            applicability='source_note_annotation',source_row_number=b.source_row_number,
            source_harmony_group_id=gid,
            source_staff=b.source_staff,source_voice=b.source_voice,tie_continuation=b.tie_continuation,
            binding_available=b.binding_available,binding_unknown_reason=b.mask_reason,
            fields=native_fields(row,usable=b.binding_available,point=b.canonical_start_qn==b.canonical_end_qn,
                                 evidence=[f'source-row:{b.source_row_number}'])))
    intervals=defaultdict(list)
    for c in contexts:
        intervals[c['note_id']].append((Fraction(c['start_qn']['num'],c['start_qn']['den']),
                                        Fraction(c['end_qn']['num'],c['end_qn']['den'])))
    for note_id,(index,note) in notes.items():
        a=note.onset_qn.to_fraction();z=a+note.duration_qn.to_fraction();cursor=a;gaps=[]
        for left,right in sorted(intervals[note_id]):
            if left>cursor:gaps.append((cursor,left))
            cursor=max(cursor,right)
        if cursor<z:gaps.append((cursor,z))
        if a==z and note_id not in seen:gaps.append((a,z))
        for ordinal,(left,right) in enumerate(gaps):
            contexts.append(dict(id=f'{rid}:unknown-note:{index}:{ordinal}',note_id=note_id,canonical_note_index=index,
                start_qn=rational(left),end_qn=rational(right),harmony_id=None,applicability='unavailable',
                fields=native_fields({},usable=False,point=left==right,evidence=[])))
    counts=Counter();unknown=Counter()
    for c in contexts:
        for k,f in c['fields'].items():
            if f['available']:counts[k]+=1
            else:unknown[f['unknown_reason']]+=1
    return dict(schema='EventTheoryEnrichment@1',dataset_id=piece.dataset_name,piece_id=piece.piece_id,
        record_id=rid,current_split='train',canonical_piece=metadata['canonical_piece'],
        source_bindings=metadata['source_bindings'],harmonies=[],note_contexts=contexts,relations=[],
        source_annotation_groups=list(groups.values()),
        coverage=dict(canonical_notes=len(piece.notes),context_rows=len(contexts),
            canonical_notes_accounted=len({c['note_id'] for c in contexts}),native_source_rows=len(rows),
            source_annotation_groups=len(groups),exact_harmonic_spans=0,
            semantic_relations=0,available_by_field=dict(counts),unknown_by_reason=dict(unknown),
            omitted_source_rows=omitted),
        provenance=dict(source_version_id=version,origin='source_native',confidence=None,
            applicability='source note annotation only; no inferred harmonic interval or semantic relation',
            source_row_identity='exact TSV line; original source binding and repair masks retained',
            geometry='source note segment mapped to canonical note; not an assertion of harmony duration',
            source_annotation_origin='expert dataset annotations; deterministic source-to-canonical projection'))


def numerical_targets(bundle):
    fields=sorted({k for c in bundle['note_contexts'] for k in c['fields']})
    vocab={}; values={};masks={}
    for key in fields:
        encoded={json.dumps(c['fields'][key]['value'],sort_keys=True,ensure_ascii=False):c['fields'][key]['value']
                 for c in bundle['note_contexts'] if c['fields'].get(key,{}).get('available')}
        keys=sorted(encoded);vocab[key]=[encoded[k] for k in keys]; lookup={k:i for i,k in enumerate(keys)}
        masks[key]=[c['fields'].get(key,{}).get('available',False) for c in bundle['note_contexts']]
        values[key]=[lookup[json.dumps(c['fields'][key]['value'],sort_keys=True,ensure_ascii=False)]
                     if c['fields'].get(key,{}).get('available') else -1 for c in bundle['note_contexts']]
    return dict(schema='EventTheoryNumeric@1',scope='per-piece vocabularies; use shared loader for dataset vocabulary',
        context_ids=[c['id'] for c in bundle['note_contexts']],
        canonical_note_indices=[c['canonical_note_index'] for c in bundle['note_contexts']],
        field_vocabularies=vocab,values=values,masks=masks)
