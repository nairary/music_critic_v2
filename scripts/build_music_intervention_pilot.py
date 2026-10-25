#!/usr/bin/env python3
"""Bounded, source-conditioned TRAIN pilot; hypotheses, not quality labels.

Uses a previously authenticated extraction receipt. Never searches a corpus,
loads a model or reads VALIDATION/TEST. Root/quality are source context only;
the exact observed-note postconditions are the independently checked effects.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import importlib.util
import sys

import mido

from music_critic.data import load_piece, dump_piece, RationalTime as Q
from music_critic.data.validation import validate_or_raise
from music_critic.experiments.music_interventions import (
    NotePatch, Intervention, apply_interventions, check_compatibility,
)
from music_critic.exporters.midi import MidiRenderConfig, piece_to_midi_bytes
from render_intervention_preview import render_preview

# The package initializer imports the complete SSL/training stack. Reuse the
# existing pure exact-window module without requiring that unrelated runtime.
_window_spec = importlib.util.spec_from_file_location(
    '_music_intervention_raw_windows',
    Path(__file__).resolve().parents[1] / 'src/music_critic/pdmx_pipeline/windowing.py',
)
_windows = importlib.util.module_from_spec(_window_spec)
sys.modules[_window_spec.name] = _windows
_window_spec.loader.exec_module(_windows)
WindowConfig, iter_bar_windows = _windows.WindowConfig, _windows.iter_bar_windows


VERSION = 'MusicalInterventionPilot@1.0.0'
WORKS = {
    'dlc:corelli:op01n01a': ('Корелли · op. 1 № 1, часть a', 0, 14),
    'dlc:schubert_winterreise:n02': ('Шуберт · Winterreise, № 2', 5, 20),
    'dlc:schumann_kinderszenen:n01': ('Шуман · Kinderszenen, № 1', 0, 30),
}
TRIADS = {'major triad': (0, 4, 7), 'minor triad': (0, 3, 7),
          'diminished triad': (0, 3, 6)}
ZERO = Q(0)


def rat(value):
    return [value.num, value.den]


def qfloat(value):
    return value.num / value.den


def scope(note):
    return note.track_id, note.staff, note.voice


def overlap(a, b):
    return max(ZERO, min(a.onset_qn+a.duration_qn, b.onset_qn+b.duration_qn)
               - max(a.onset_qn, b.onset_qn))


def pitch_name(pitch):
    return ['C', 'C♯', 'D', 'E♭', 'E', 'F', 'F♯', 'G', 'A♭', 'A', 'B♭', 'B'][pitch % 12] + str(pitch//12-1)


def parse_root(value):
    if not isinstance(value, str) or not value or value[0] not in 'CDEFGAB':
        return None
    if any(x not in '#b-' for x in value[1:]):
        return None
    return ({'C':0, 'D':2, 'E':4, 'F':5, 'G':7, 'A':9, 'B':11}[value[0]]
            +value[1:].count('#')-value[1:].count('b')-value[1:].count('-')) % 12


def harmonic_context(sidecar):
    by_time = defaultdict(list)
    for entity in sidecar['entities']:
        if entity['entity_type'] != 'harmonic_event':
            continue
        t = Q(**entity['onset_qn'])
        values = {}
        for key in ['local_key', 'root', 'quality', 'roman_numeral']:
            state = entity['targets'][key]
            values[key] = state['canonical_value'] if state['available'] and not state['masked'] else None
        by_time[t].append((values, entity['source_entity_id']))
    result = {}
    for t, entries in by_time.items():
        signatures = {tuple(v[k] for k in ['local_key','root','quality','roman_numeral']) for v, _ in entries}
        if len(signatures) == 1:
            result[t] = dict(entries[0][0], source_entities=[e for _, e in entries])
    return result


def spaced(candidates, limit=5, minimum=Q(5)):
    """Choose dispersed positions, with an explicit applicability ceiling."""
    chosen = []
    for c in sorted(candidates, key=lambda c:(c['onset'], c['key'])):
        if all(abs((c['onset']-old['onset']).to_fraction()) >= minimum.to_fraction() for old in chosen):
            chosen.append(c)
        if len(chosen) == limit:
            break
    return chosen


def build_candidates(piece, sidecar, offset):
    if len(piece.tracks)!=1:
        raise ValueError('pilot staff-based selection requires one verified raw track')
    harmony, rhythm, melody = [], [], []
    context = harmonic_context(sidecar)
    notes = list(piece.notes)
    by_id = {n.note_id:n for n in notes}
    eligible = lambda n: Q(4) <= n.onset_qn and n.onset_qn+n.duration_qn <= piece.duration_qn-Q(4) and n.duration_qn >= Q(1,4)
    # Harmonic intervention: a sustained chromatic clash against an unchanged
    # note. Root/quality narrows applicability; it is not a fresh chord label.
    for n in notes:
        if not eligible(n) or n.staff is None or n.staff == 1:
            continue
        label = context.get(n.onset_qn+offset)
        if not label or label['quality'] not in TRIADS:
            continue
        root = parse_root(label['root'])
        if root is None:
            continue
        pcs = {(root+x)%12 for x in TRIADS[label['quality']]}
        if n.pitch%12 not in pcs:
            continue
        sounding = [m for m in notes if m.note_id != n.note_id and overlap(n,m) >= Q(1,4)]
        for delta in (1,-1):
            new = n.pitch+delta
            if not 24 <= new <= 100 or new%12 in pcs:
                continue
            witnesses = [m for m in sounding if (new-m.pitch)%12 in (1,11)
                         and (n.pitch-m.pitch)%12 not in (1,11)]
            if not witnesses:
                continue
            witness = max(witnesses,key=lambda m:qfloat(overlap(n,m)))
            harmony.append(dict(onset=n.onset_qn,key=n.note_id,
                patches=(NotePatch(n.note_id,pitch=new),),
                context=tuple(sorted({n.note_id,*(m.note_id for m in sounding)})),
                description=f'{pitch_name(n.pitch)} → {pitch_name(new)} на стане {n.staff}; с {pitch_name(witness.pitch)} возникает интервал {abs(new-witness.pitch)} полутонов (класс интервала 1: секунда/септима). Исходная разметка: {label["root"]}, {label["quality"]}, {label["roman_numeral"]}.',
                check=dict(kind='new_semitone_clash',note=n.note_id,witness=witness.note_id,
                           before_pitch=n.pitch,after_pitch=new,minimum_overlap=rat(Q(1,4))),
                source_harmony=label))
            break
    # Shift one notated staff's complete attack group. This is rhythmic
    # desynchronization, not a claim to isolate rhythm from vertical harmony.
    groups=defaultdict(list)
    for n in notes:
        if n.staff is not None and n.staff != 1:
            groups[(n.staff,n.onset_qn)].append(n)
    for (staff,t), group in groups.items():
        if not all(eligible(n) for n in group):
            continue
        refs=[m for m in notes if m.staff != staff and m.onset_qn==t and m.duration_qn>ZERO]
        if not refs:
            continue
        shift=Q(1,4)
        if any(n.onset_qn+shift+n.duration_qn>piece.duration_qn-Q(2) for n in group):
            continue
        # Reject newly overlapping repeated pitch in that same notated voice.
        if any(scope(n)==scope(m) and n.pitch==m.pitch and n.note_id!=m.note_id
               and overlap(replace(n,onset_qn=n.onset_qn+shift),m)>ZERO
               for n in group for m in notes if m not in group):
            continue
        around=[m.note_id for m in notes if t-Q(1)<=m.onset_qn<=t+Q(2)]
        rhythm.append(dict(onset=t,key=str(staff),
            patches=tuple(NotePatch(n.note_id,onset_qn=t+shift) for n in group),
            context=tuple(sorted(set(around))),
            description=f'Вступление стана {staff} задержано на ¼ четверти (125 мс при 120 BPM), {len(group)} нот. Высоты и длительности сохранены; другие станы вступают вовремя.',
            check=dict(kind='onset_desynchronization',notes=[n.note_id for n in group],
                       witnesses=[m.note_id for m in refs],original_onset=rat(t),shift=rat(shift))))
    # Notated staff1/voice1, with unique attacks and continuous neighbors.
    # This is a source-scoped line; it is not an automatic melody extractor.
    line=[n for n in notes if n.staff==1 and n.voice==1 and n.duration_qn>ZERO]
    counts=Counter(n.onset_qn for n in line)
    for previous,n,following in zip(line,line[1:],line[2:]):
        if not eligible(n) or any(counts[m.onset_qn]!=1 for m in (previous,n,following)):
            continue
        if previous.onset_qn+previous.duration_qn != n.onset_qn or n.onset_qn+n.duration_qn != following.onset_qn:
            continue
        old=max(abs(n.pitch-previous.pitch),abs(following.pitch-n.pitch))
        if old>5:
            continue
        options=[n.pitch+d for d in (12,-12) if 36<=n.pitch+d<=100]
        if not options:
            continue
        new=max(options,key=lambda p:min(abs(p-previous.pitch),abs(following.pitch-p)))
        if min(abs(new-previous.pitch),abs(following.pitch-new))<8:
            continue
        melody.append(dict(onset=n.onset_qn,key=n.note_id,
            patches=(NotePatch(n.note_id,pitch=new),),
            context=(previous.note_id,n.note_id,following.note_id),
            description=f'В линии стан 1 / голос 1: {pitch_name(previous.pitch)}–{pitch_name(n.pitch)}–{pitch_name(following.pitch)} → {pitch_name(previous.pitch)}–{pitch_name(new)}–{pitch_name(following.pitch)}. Октавный излом; классы высот и ритм сохранены.',
            check=dict(kind='octave_line_break',note=n.note_id,previous=previous.note_id,
                       following=following.note_id,before_pitch=n.pitch,after_pitch=new)))
    result=[]
    for aspect, prefix, title, pool in [('harmony','H','Хроматическое столкновение',harmony),
                                       ('rhythm','R','Рассинхронизация вступления',rhythm),
                                       ('melody','M','Октавный излом линии',melody)]:
        for i,candidate in enumerate(spaced(pool),1):
            result.append(dict(candidate,id=f'{prefix}{i}',aspect=aspect,title=f'{prefix}{i} · {title}'))
    if harmony:
        h=next(x for x in result if x['aspect']=='harmony');note=by_id[h['patches'][0].note_id]
        result.append(dict(id='D1',aspect='deletion',title='D1 · Удаление ноты (пример конфликта)',
            onset=note.onset_qn,key=note.note_id,patches=(NotePatch(note.note_id,delete=True),),
            context=(note.note_id,),description=f'Удалить {pitch_name(note.pitch)} из места H1. H1 после удаления этой ноты применить нельзя.',
            check=dict(kind='note_deleted',note=note.note_id)))
    return result, {k:len(v) for k,v in [('harmony',harmony),('rhythm',rhythm),('melody',melody)]}


def verify_effect(piece, item):
    """Recompute a limited observable condition on the COMPLETE result."""
    ns={n.note_id:n for n in piece.notes};c=item['check'];kind=c['kind']
    ids=[c[k] for k in ('note','witness','previous','following') if k in c]
    if kind=='note_deleted':
        return dict(id=item['id'],passed=c['note'] not in ns,condition=kind)
    if any(x not in ns for x in ids):
        return dict(id=item['id'],passed=False,condition=kind,reason='required_note_deleted')
    if kind=='new_semitone_clash':
        a,b=ns[c['note']],ns[c['witness']]
        passed=(a.pitch==c['after_pitch'] and (a.pitch-b.pitch)%12 in (1,11)
                and overlap(a,b)>=Q(*c['minimum_overlap']))
    elif kind=='onset_desynchronization':
        t,delta=Q(*c['original_onset']),Q(*c['shift'])
        passed=all(x in ns and ns[x].onset_qn==t+delta for x in c['notes']) and any(
            x in ns and ns[x].onset_qn==t for x in c['witnesses'])
    elif kind=='octave_line_break':
        a,b,d=ns[c['previous']],ns[c['note']],ns[c['following']]
        passed=(b.pitch==c['after_pitch'] and b.pitch%12==c['before_pitch']%12
                and min(abs(b.pitch-a.pitch),abs(d.pitch-b.pitch))>=8
                and a.onset_qn+a.duration_qn==b.onset_qn
                and b.onset_qn+b.duration_qn==d.onset_qn)
    else:
        raise ValueError('unknown postcondition')
    return dict(id=item['id'],passed=passed,condition=kind,
                quality_preference='unreviewed')


def routed_midi(piece):
    """Preview-only channels preserve independent same-pitch note lifetimes."""
    omitted=[n for n in piece.notes if n.duration_qn==ZERO]
    if any(not n.is_grace for n in omitted):
        raise ValueError('cannot omit a zero-duration ordinary note')
    sounding=[n for n in piece.notes if n.duration_qn>ZERO]
    channels=[x for x in range(16) if x!=9]
    scopes=sorted({scope(n) for n in sounding},key=str)
    preferred={s:channels[i%len(channels)] for i,s in enumerate(scopes)}
    active={};notes=[]
    for n in sounding:
        active={key:end for key,end in active.items() if end>n.onset_qn}
        order=[preferred[scope(n)],*(c for c in channels if c!=preferred[scope(n)])]
        channel=next((c for c in order if (c,n.pitch) not in active),None)
        if channel is None:
            raise ValueError('MIDI same-pitch channel capacity exceeded')
        active[(channel,n.pitch)]=n.onset_qn+n.duration_qn
        notes.append(replace(n,channel=channel,program=0,velocity=80))
    view=replace(piece,notes=tuple(notes),quality_flags=())
    payload,report=piece_to_midi_bytes(view,config=MidiRenderConfig(
        include_click_track=False,include_target_markers=False,melody_velocity=80))
    # Independent SMF note-on/off pairing, before retaining the artifact.
    from io import BytesIO
    file=mido.MidiFile(file=BytesIO(payload));observed=[]
    for track in file.tracks:
        t=0;active={}
        for event in track:
            t+=event.time
            if event.type=='note_on' and event.velocity:
                key=event.channel,event.note
                if key in active:raise ValueError('overlapping same-pitch MIDI')
                active[key]=t
            elif event.type=='note_off' or (event.type=='note_on' and not event.velocity):
                key=event.channel,event.note;start=active.pop(key)
                observed.append((event.note,Q(start,file.ticks_per_beat),Q(t-start,file.ticks_per_beat)))
        if active:raise ValueError('hanging MIDI notes')
    expected=[(n.pitch,n.onset_qn,n.duration_qn) for n in sounding]
    if Counter(observed)!=Counter(expected):raise ValueError('MIDI note roundtrip mismatch')
    return payload,dict(exact_timing=report.exact_timing,note_roundtrip=True,
                        notes=len(expected),preview_only_channel_allocation=True,
                        omitted_zero_duration_grace_ids=[n.note_id for n in omitted],
                        omission_scope='MIDI/audio preview only; canonical retains every grace note')


def event_rows(piece):
    # Pilot intentionally admits only constant, explicitly documented120 BPM.
    if len(piece.tempo_events)!=1 or piece.tempo_events[0].microseconds_per_quarter!=500000:
        raise ValueError('pilot audio requires the declared constant120BPM preview')
    return [dict(id=n.note_id,pitch=n.pitch,onset_qn=rat(n.onset_qn),duration_qn=rat(n.duration_qn),
                 onset_seconds=qfloat(n.onset_qn)*.5,duration_seconds=qfloat(n.duration_qn)*.5,
                 staff=n.staff,voice=n.voice,velocity=80) for n in piece.notes]


def verify_transposition(source,changed,delta):
    old={n.note_id:n for n in source.notes};new={n.note_id:n for n in changed.notes}
    return old.keys()==new.keys() and all(
        new[k].pitch==n.pitch+delta and new[k].onset_qn==n.onset_qn
        and new[k].duration_qn==n.duration_qn for k,n in old.items())


def candidate_json(c):
    patches=[]
    for p in c['patches']:
        d={'note_id':p.note_id}
        if p.pitch is not None:d['pitch']=p.pitch
        if p.onset_qn is not None:d['onset_qn']=rat(p.onset_qn)
        if p.duration_qn is not None:d['duration_qn']=rat(p.duration_qn)
        if p.delete:d['delete']=True
        patches.append(d)
    return {**{k:c[k] for k in ['id','aspect','title','description','check']},
            'patches':patches,'context_note_ids':list(c['context']),
            'location_seconds':qfloat(c['onset'])*.5,
            'source_harmony':c.get('source_harmony'),
            'quality_preference':'unreviewed','primary_aspect_is_hypothesis':True}


def build_pilot(source_root,output):
    output.mkdir(parents=True,exist_ok=False)
    receipt=json.loads((source_root/'source-receipt.json').read_text())
    if receipt['validation_payload_read'] is not False or receipt['test_payload_read'] is not False:
        raise ValueError('only TRAIN payloads are allowed')
    if receipt['manifest_sha256']!='48a5301dcf83c12eb6ba5f5a00e658e3d0972221bc9db5eea5fa1e62f384dd38':
        raise ValueError('unknown frozen source manifest')
    grouped=defaultdict(dict)
    source_components={}
    for row in receipt['files']:
        if row['split']!='train' or row['record_id'] not in WORKS:
            raise ValueError('source is outside the selected TRAIN scope')
        path=Path(row['path']);raw=path.read_bytes()
        if len(raw)!=row['size_bytes'] or sha256(raw).hexdigest()!=row['sha256']:
            raise ValueError('source hash/size mismatch')
        if row['kind'] in grouped[row['record_id']]:
            raise ValueError('duplicate source binding')
        grouped[row['record_id']][row['kind']]=path
        old=source_components.setdefault(row['record_id'],row['source_component_id'])
        if old!=row['source_component_id']:
            raise ValueError('inconsistent source component')
    if set(grouped)!=set(WORKS):
        raise ValueError('selected source set is incomplete')
    works=[];all_reports=[]
    for work_index,(record_id,paths) in enumerate(sorted(grouped.items()),1):
        source=load_piece(paths['piece'])
        if source.split not in (None,'train'):
            raise ValueError('canonical split is not TRAIN')
        source=replace(source,split='train') # frozen manifest owns membership
        sidecar=json.loads(paths['sidecar'].read_text())
        if sidecar['record_id']!=record_id or sidecar['source_component_id']!=source_components[record_id]:
            raise ValueError('sidecar identity mismatch')
        title,start,bar_count=WORKS[record_id]
        # Existing raw windowing preserves source-note identity for clipped ties.
        window=next(w for w in iter_bar_windows(source,WindowConfig(bar_count=bar_count,stride=1))
                    if w.source_bar_start==start)
        piece=window.piece
        if piece is None:
            raise ValueError('selected window unavailable')
        duration=qfloat(piece.duration_qn)*.5
        candidates,pool_counts=build_candidates(piece,sidecar,window.global_offset_qn)
        counts=Counter(c['aspect'] for c in candidates)
        if any(counts[a]==0 for a in ['harmony','rhythm','melody']):
            raise ValueError(f'no applicable examples for an aspect: {record_id} {counts}')
        interventions={c['id']:Intervention(c['id'],c['patches'],c['context']) for c in candidates}
        by_id={c['id']:c for c in candidates}
        pairwise=[dict(left=a,right=b,status=(r:=check_compatibility(interventions[a],interventions[b])).status,
                       reasons=list(r.reasons)) for a,b in combinations(interventions,2)]
        presets=[('original','Исходник',[])]
        presets.extend((c['id'],c['title'],[c['id']]) for c in candidates)
        for aspect,prefix in [('harmony','H'),('rhythm','R'),('melody','M')]:
            ids=[c['id'] for c in candidates if c['aspect']==aspect]
            for count in (3,5):
                if len(ids)>=count:presets.append((f'{prefix}_dose_{count}',f'{aspect}: {count} места',ids[:count]))
        combined=[c['id'] for c in candidates if c['aspect']!='deletion']
        presets.append(('combined','Все H + R + M',combined))
        safe=list(combined)
        while safe:
            trial=apply_interventions(piece,tuple(interventions[i] for i in safe)).piece
            failed={i for i in safe if not verify_effect(trial,by_id[i])['passed']}
            if not failed:break
            safe=[i for i in safe if i not in failed]
        if safe and safe!=combined:
            presets.append(('combined_checked','Комбинация с сохранёнными проверяемыми эффектами',safe))
        # Benign-control hypothesis: complete exact transposition with unchanged
        # intervals/timing. No preference label is manufactured.
        control=Intervention('T2',tuple(NotePatch(n.note_id,pitch=n.pitch+2) for n in piece.notes))
        workdir=output/f'work{work_index}';workdir.mkdir()
        materialized=[]
        for stem,name,ids in presets:
            result=apply_interventions(piece,tuple(interventions[i] for i in ids)) if ids else None
            changed=result.piece if result else piece
            checks=[verify_effect(changed,by_id[i]) for i in ids]
            if len(ids)==1 and any(not x['passed'] for x in checks):
                raise ValueError(f'single edit postcondition failed: {record_id} {ids}')
            materialized.append((stem,name,ids,changed,checks,result.compatibility.status if result else 'original'))
        control_piece=apply_interventions(piece,(control,)).piece
        control_ok=verify_transposition(piece,control_piece,2)
        if not control_ok:raise ValueError('transposition invariant failed')
        materialized.append(('control_transpose_2','Контроль: весь фрагмент +2 полутона',[],control_piece,
                             [dict(condition='all_note_intervals_and_timing_preserved',passed=control_ok)],'control'))
        # One conservative fixed gain over every rendered variant, incl. releases.
        max_active=1
        for _,_,_,changed,_,_ in materialized:
            events=[]
            for n in changed.notes:
                events.extend([(qfloat(n.onset_qn)*.5,1),
                               (qfloat(n.onset_qn+n.duration_qn)*.5+.08,-1)])
            active=0
            for _,delta in sorted(events):active+=delta;max_active=max(max_active,active)
        gain=.8/(1.555*max_active)
        kept=[]
        for stem,name,ids,changed,checks,status in materialized:
            dump_piece(changed,workdir/f'{stem}.json')
            midi,midi_report=routed_midi(changed);(workdir/f'{stem}.mid').write_bytes(midi)
            wav_report=render_preview(event_rows(changed),workdir/f'{stem}.wav',duration_seconds=duration,gain=gain)
            wav_report['omitted_zero_duration_grace_ids']=midi_report['omitted_zero_duration_grace_ids']
            kept.append(dict(name=name,ids=ids,file_stem=f'work{work_index}/{stem}',
                             compatibility=status,observable_checks=checks,
                             all_observable_conditions_hold=all(x['passed'] for x in checks),
                             musical_preference='unreviewed'))
            all_reports.append(dict(work=record_id,variant=stem,midi=midi_report,audio=wav_report))
        dump_piece(piece,workdir/'source_clip.json')
        works.append(dict(id=f'work{work_index}',title=title,source_id=record_id,
            duration_seconds=duration,seconds_per_quarter=.5,preview_gain=gain,
            tempo_note=f'120 BPM — технический темп корпуса; исходный темп неизвестен. Единый синтетический тембр. Форшлаги без длительности не звучат: {sum(n.duration_qn==ZERO for n in piece.notes)}; в canonical JSON они сохранены.',
            original_wav=f'work{work_index}/original.wav',original_midi=f'work{work_index}/original.mid',
            source_bars=[window.source_bar_start+1,window.source_bar_stop],
            global_offset_qn=rat(window.global_offset_qn),clipped_note_count=window.clipped_note_count,
            note_identity_map=dict(zip((n.note_id for n in piece.notes),window.original_note_ids)),
            notes=event_rows(piece),interventions=[candidate_json(c) for c in candidates],
            pairwise=pairwise,presets=kept,applicable_pool_counts=pool_counts,
            retained_counts=dict(counts),status='constructed_not_listener_validated'))
        print(json.dumps(dict(record=record_id,seconds=duration,counts=dict(counts),variants=len(kept)),ensure_ascii=False),flush=True)
    payload=dict(version=VERSION,works=works,quality_labels='unreviewed',
                 split='train',independent_source_works=len(works),test_accessed=False)
    (output/'pilot.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    (output/'render-verification.json').write_text(json.dumps(all_reports,indent=2)+'\n')
    return payload


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    build_pilot(args.source_root,args.output)


if __name__=='__main__':main()
