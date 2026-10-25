#!/usr/bin/env python3
"""Auditable whole-accompaniment alternatives on two fixed TRAIN excerpts.

Source annotations describe original context only; replacements are listening
hypotheses, not inferred gold labels or listener preferences. No training.
"""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path

from build_music_intervention_pilot import (
    Q, ZERO, load_piece, dump_piece, harmonic_context, parse_root, pitch_name,
    rat, qfloat, overlap, event_rows, routed_midi, candidate_json,
    NotePatch, Intervention, apply_interventions, check_compatibility,
)
from build_intervention_player import build_player
from render_intervention_preview import render_preview

SPECS = {
 'work3': dict(spans=[(Q(14),Q(16)),(Q(30),Q(32)),(Q(58),Q(60))],
               freeze=(Q(8),Q(16)),tonic=7,third=4,dominant=2,
               expected_key='G',roman='I',extension_from=11,extension_to=10,
               extension_name='Сопровождение G–B–D превращается в G–B♭–D, но в мелодии остаётся B: конфликт малой и большой терций'),
 'work2': dict(spans=[(Q(27),Q(29))],freeze=(Q(18),Q(29)),
               tonic=4,third=3,dominant=11,expected_key='E',roman='i',
               extension_from=11,extension_to=3,
               extension_name='Сопровождение E–G–B превращается в E–G–D♯: Em(maj7) без квинты, против удержанной E в мелодии'),
}


def nearest_pitch(pitch, pc):
    return min((p for p in range(24,101) if p%12==pc),key=lambda p:(abs(p-pitch),p))


def inside_notes(piece, start, end):
    """Require complete accompaniment notes inside exact boundaries."""
    notes=[n for n in piece.notes if n.staff!=1 and n.duration_qn>ZERO
           and n.onset_qn<end and n.onset_qn+n.duration_qn>start]
    if not notes or any(n.staff is None or n.onset_qn<start or n.onset_qn+n.duration_qn>end for n in notes):
        raise ValueError('accompaniment boundary spill or missing staff')
    return notes


def cadence_evidence(piece, context, start, end, spec):
    rows=sorted(context.items())
    at=[(t,h) for t,h in rows if t==start]
    previous=[(t,h) for t,h in rows if t<start]
    if len(at)!=1 or not previous:raise ValueError('missing exact arrival or previous annotation')
    prev_t,prev=previous[-1];_,arrival=at[0]
    tonic=spec['tonic'];dom=(tonic+7)%12
    if not (arrival['local_key']==prev['local_key']==spec['expected_key']
            and arrival['roman_numeral']==spec['roman']
            and parse_root(arrival['root'])==tonic
            and parse_root(prev['root'])==dom
            and prev['roman_numeral'] in ('V','V7')
            and prev['quality'] in ('major triad','dominant seventh chord')
            and arrival['quality']==('major triad' if spec['third']==4 else 'minor triad')):
        raise ValueError('source does not establish the claimed local dominant to tonic')
    # Every available timestamp inside the chosen tonic must agree; unknowns
    # are barriers, not forward-filled context.
    for t,h in rows:
        if start<=t<end and any(h[k]!=arrival[k] for k in ('local_key','root','quality','roman_numeral')):
            raise ValueError('ambiguous or changed harmonic context inside tonic')
    notes=inside_notes(piece,start,end)
    expected={tonic,(tonic+spec['third'])%12,(tonic+7)%12}
    if {n.pitch%12 for n in notes}!=expected:raise ValueError('observed accompaniment is not the stated complete triad')
    held=[n for n in piece.notes if n.staff==1 and n.voice==1 and n.onset_qn<=start and n.onset_qn+n.duration_qn>=end]
    if len(held)!=1:raise ValueError('need one explicitly preserved held melody note')
    return dict(start=rat(start),end=rat(end),previous_onset=rat(prev_t),previous=prev,
                arrival=arrival,accompaniment_ids=[n.note_id for n in notes],
                melody_id=held[0].note_id,melody_pitch=held[0].pitch)


def map_chord(notes, tonic, third, destination, destination_third=4):
    mapping={tonic:destination,(tonic+third)%12:(destination+destination_third)%12,(tonic+7)%12:(destination+7)%12}
    if any(n.pitch%12 not in mapping for n in notes):raise ValueError('non-chord note in replacement')
    return {n.note_id:nearest_pitch(n.pitch,mapping[n.pitch%12]) for n in notes}


def frozen_chord(notes, root):
    """Keep attacks/register nearby; source bass roles receive the root.

    Other pitches are mapped to the nearest member of the chosen seventh chord.
    The result is labelled a revoicing, not isolated functional replacement.
    """
    pcs={root,(root+4)%12,(root+7)%12,(root+10)%12}
    result={};bass_ids=set()
    for n in notes:
        is_bass=(n.staff==2 and n.voice==2) or (n.staff==3 and n.pitch==min(m.pitch for m in notes if m.staff==3 and m.onset_qn==n.onset_qn))
        if is_bass:bass_ids.add(n.note_id)
        options=[nearest_pitch(n.pitch,pc) for pc in (pcs if not is_bass else {root})]
        result[n.note_id]=min(options,key=lambda p:(abs(p-n.pitch),p))
    for pc in sorted(pcs-set(p%12 for p in result.values())):
        counts=Counter(p%12 for p in result.values())
        eligible=[n for n in notes if n.note_id not in bass_ids and counts[result[n.note_id]%12]>1]
        if not eligible:raise ValueError('insufficient independent notes for seventh-chord voicing')
        n=min(eligible,key=lambda n:(abs(nearest_pitch(n.pitch,pc)-n.pitch)-abs(result[n.note_id]-n.pitch),n.note_id))
        result[n.note_id]=nearest_pitch(n.pitch,pc)
    if set(p%12 for p in result.values())!=pcs:raise ValueError('fixed seventh chord lacks a chord member')
    return result


def candidate(piece, ident, title, description, mapping, spans, expected_sets, evidence=(), kind='replacement'):
    by_id={n.note_id:n for n in piece.notes}
    patches=tuple(NotePatch(k,pitch=p) for k,p in sorted(mapping.items()) if by_id[k].pitch!=p)
    if len(patches)<2:raise ValueError('harmonic replacement must change multiple notes')
    start=min(a for a,b in spans);end=max(b for a,b in spans)
    context=tuple(n.note_id for n in piece.notes if n.onset_qn<end and n.onset_qn+n.duration_qn>max(ZERO,start-Q(4)))
    return dict(id=ident,aspect='harmony',title=ident+' · '+title,description=description,
                onset=start,patches=patches,context=context,source_harmony=list(evidence),
                check=dict(kind=kind,spans=[dict(start=rat(a),end=rat(b),expected_pcs=sorted(pcs)) for (a,b),pcs in zip(spans,expected_sets)],
                           planned_pitches={p.note_id:p.pitch for p in patches},
                           note_count_changed=len(patches)))


def verify_variant(original,changed,item):
    before={n.note_id:n for n in original.notes};after={n.note_id:n for n in changed.notes}
    planned=item['check']['planned_pitches']
    if before.keys()!=after.keys():raise ValueError('note inventory changed')
    for k,n in before.items():
        m=after[k]
        if m.pitch!=planned.get(k,n.pitch):raise ValueError('unexpected or missing pitch edit')
        if (m.onset_qn,m.duration_qn,m.staff,m.voice,m.track_id,m.velocity)!=(n.onset_qn,n.duration_qn,n.staff,n.voice,n.track_id,n.velocity):
            raise ValueError('non-pitch field changed')
        if n.staff==1 and m.pitch!=n.pitch:raise ValueError('preserved notated melody was changed')
    details=[]
    for span in item['check']['spans']:
        a,b=Q(*span['start']),Q(*span['end'])
        notes=inside_notes(changed,a,b)
        observed={n.pitch%12 for n in notes}
        if observed!=set(span['expected_pcs']):raise ValueError('replacement chord pitch classes do not match')
        details.append(dict(**span,observed_pcs=sorted(observed)))
    clashes=[]
    for k in planned:
        old,n=before[k],after[k]
        for m in changed.notes:
            if m.staff==1 and m.duration_qn>ZERO and overlap(n,m)>ZERO and (n.pitch-m.pitch)%12 in (1,11) and (old.pitch-m.pitch)%12 not in (1,11):
                clashes.append(dict(note=k,melody=m.note_id,interval_semitones=abs(n.pitch-m.pitch),overlap_qn=rat(overlap(n,m))))
    if item['check']['kind']=='melody_clash' and len(clashes)<2:raise ValueError('no repeated new melody clash')
    return dict(id=item['id'],passed=True,condition=item['check']['kind'],spans=details,
                changed_notes=len(planned),preserved_staff1=True,exact_rhythm_preserved=True,
                new_melody_clashes=clashes,quality_preference='unreviewed')


def make_candidates(piece,context,spec):
    spans=spec['spans'];evidence=[cadence_evidence(piece,context,a,b,spec) for a,b in spans]
    tonic=spec['tonic'];triad={tonic,(tonic+spec['third'])%12,(tonic+7)%12}
    a,b=spans[0];notes=inside_notes(piece,a,b)
    shifted={(p+1)%12 for p in triad}
    results=[candidate(piece,'H1','Целый чужой аккорд в месте разрешения',
        f'После доминанты вместо исходной тоники весь аккорд сопровождения сдвинут на полутон вверх. Меняются {len(notes)} нот; сохранены ритм и линия стана 1. Было: {" / ".join(pitch_name(p+60)[:-1] for p in sorted(triad))}; стало: {" / ".join(pitch_name(p+60)[:-1] for p in sorted(shifted))}. Это подмена аккорда целиком, а не одна неверная нота.',
        {n.note_id:n.pitch+1 for n in notes},[(a,b)],[shifted],evidence[:1])]
    mapping={}
    for a,b in spans:mapping.update(map_chord(inside_notes(piece,a,b),tonic,spec['third'],spec['dominant']))
    dompcs={spec['dominant'],(spec['dominant']+4)%12,(spec['dominant']+7)%12}
    results.append(candidate(piece,'H2',f'Тоника заменена доминантой: {len(spans)} мест',
        f'В {len(spans)} проверенных местах разрешения V→{spec["roman"]} сопровождение остаётся на мажорной доминанте, тогда как нотированная мелодия приходит в исходную тонику. Это проверка отсутствия гармонического прихода; само сочетание может быть допустимым в другом контексте.',mapping,spans,[dompcs]*len(spans),evidence))
    a,b=spec['freeze'];notes=inside_notes(piece,a,b);root=spec['dominant'];pcs={root,(root+4)%12,(root+7)%12,(root+10)%12}
    mapping=frozen_chord(notes,root)
    progression=[]
    for t,h in sorted(context.items()):
        if a<=t<b:
            label=f'{h["root"]} {h["roman_numeral"]}'
            if not progression or label!=progression[-1]:progression.append(label)
    results.append(candidate(piece,'H3','Один аккорд вместо всей последовательности',
        f'С {qfloat(a)*.5:g} до {qfloat(b)*.5:g} с сопровождение удерживает только звуки доминантсептаккорда, хотя мелодия и её ритм продолжаются. Исходная цепочка: {" → ".join(progression)}. Сохранены атаки и длительности, но изменены также бас и расположение аккордовых звуков.',mapping,[(a,b)],[pcs],kind='frozen_harmony'))
    notes=[n for n in piece.notes if n.staff!=1]
    mapping={n.note_id:n.pitch+1 for n in notes}
    a,b=ZERO,piece.duration_qn;pcs={(n.pitch+1)%12 for n in notes if n.duration_qn>ZERO}
    results.append(candidate(piece,'H4','Сопровождение постоянно в другой тональности',
        'На протяжении всего фрагмента всё сопровождение сдвинуто на полутон вверх, а стан 1 остаётся исходным. Сопровождение согласовано само с собой, но движется в другой тональности относительно сохранённой нотированной мелодии. Ритм и настройки громкости сохранены.',mapping,[(a,b)],[pcs],kind='accompaniment_key_mismatch'))
    mapping={};expected=[]
    for a,b in spans:
        ns=inside_notes(piece,a,b)
        mapping.update({n.note_id:nearest_pitch(n.pitch,spec['extension_to']) for n in ns if n.pitch%12==spec['extension_from']})
        expected.append((triad-{spec['extension_from']})|{spec['extension_to']})
    results.append(candidate(piece,'H5','Повторяющийся конфликт аккорда с мелодией',
        spec['extension_name']+'. Повторяющиеся звуки сопровождения заменены при неизменной удержанной ноте мелодии. Возникают новые секунды/септимы/ноны; это не универсальный запрет на такие надстройки. Часть исходного аккордового тона заменяется, новые атаки не добавляются.',mapping,spans,expected,evidence,kind='melody_clash'))
    return results


def checked_json(path,sha,size=None):
    raw=path.read_bytes()
    if sha256(raw).hexdigest()!=sha or (size is not None and len(raw)!=size):raise ValueError(f'hash mismatch: {path}')
    return json.loads(raw)


def build(source_root,output):
    output.mkdir(parents=True,exist_ok=False)
    manifest=checked_json(source_root/'artifact-manifest.json','e77d07fe262ad741dfb76bda01918e7e4ac0ac327fb65cae152e6c859c2465a8')
    bindings={r['path']:r for r in manifest}
    original_root=source_root/'attempt003'
    def old_json(relative):
        row=bindings[relative];return checked_json(original_root/relative,row['sha256'],row['size_bytes'])
    payload=old_json('pilot.json')
    if payload['split']!='train' or payload['test_accessed'] is not False:raise ValueError('TRAIN only')
    # Bind the terminal record's artifact digest. Its protocol receipt field
    # does not match this file; prefer the verified artifact identity.
    receipt=checked_json(source_root/'source-receipt.json','fec2097dc63f9b3df1f867e3fed87732df96edfb0bad0218dd57657e6102f04e')
    reports=[];works=[]
    for wid,spec in SPECS.items():
        w=next(w for w in payload['works'] if w['id']==wid)
        old_json(wid+'/source_clip.json');piece=load_piece(original_root/wid/'source_clip.json')
        if piece.split!='train':raise ValueError('clip not TRAIN')
        row=next(r for r in receipt['files'] if r['record_id']==w['source_id'] and r['kind']=='sidecar')
        if row['split']!='train':raise ValueError('sidecar not TRAIN')
        sidecar=checked_json(Path(row['path']),row['sha256'],row['size_bytes'])
        if sidecar['record_id']!=w['source_id']:raise ValueError('sidecar record mismatch')
        context={t-Q(*w['global_offset_qn']):h for t,h in harmonic_context(sidecar).items()}
        candidates=make_candidates(piece,context,spec)
        interventions={c['id']:Intervention(c['id'],c['patches'],c['context']) for c in candidates}
        directory=output/wid;directory.mkdir();presets=[]
        all_versions=[('original','Исходник',[],piece,[])]
        for c in candidates:
            changed=apply_interventions(piece,(interventions[c['id']],)).piece
            check=verify_variant(piece,changed,c)
            all_versions.append((c['id'],c['title'],[c['id']],changed,[check]))
        for stem,name,ids,changed,checks in all_versions:
            dump_piece(changed,directory/(stem+'.json'))
            midi,midi_report=routed_midi(changed);(directory/(stem+'.mid')).write_bytes(midi)
            audio=render_preview(event_rows(changed),directory/(stem+'.wav'),duration_seconds=w['duration_seconds'],gain=w['preview_gain'])
            presets.append(dict(name=name,ids=ids,file_stem=wid+'/'+stem,compatibility='original' if not ids else 'compatible',observable_checks=checks,all_observable_conditions_hold=True))
            reports.append(dict(work=w['source_id'],variant=stem,midi=midi_report,audio=audio,checks=checks))
        pairwise=[]
        for left,right in combinations(interventions,2):
            r=check_compatibility(interventions[left],interventions[right]);pairwise.append(dict(left=left,right=right,status=r.status,reasons=list(r.reasons)))
        works.append({**w,'title':w['title']+' · гармонические замены','interventions':[candidate_json(c) for c in candidates],
                      'presets':presets,'pairwise':pairwise,'applicable_pool_counts':None,'retained_counts':{'harmony':len(candidates)}})
        print(json.dumps({'work':wid,'changed_notes':{c['id']:len(c['patches']) for c in candidates}},ensure_ascii=False),flush=True)
    final={**payload,'version':'HarmonicReplacementPilot@1.0.0','works':works,'independent_source_works':len(works),'alternatives_only':True}
    (output/'pilot.json').write_text(json.dumps(final,ensure_ascii=False,indent=2)+'\n')
    (output/'verification.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2)+'\n')
    build_player(final,output/'index.html')
    return final


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();build(args.source_root,args.output)
