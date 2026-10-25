"""Fixed development examples for HCM-1; no model or TEST loader is invoked."""
from __future__ import annotations

from dataclasses import asdict, replace
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path

from music_critic.data import loads_piece, dumps_piece
from music_critic.data.timing import RationalTime as Q
from music_critic.pll.codec import EventCodec
from music_critic.pll.contracts import MaskQuery, ContextProfile
from music_critic.pll.masking import split_masked_piece


def normalize_split(value):
    if value is None:
        return None
    return {'val':'validation','dev':'validation'}.get(str(value).lower(),str(value).lower())


def select_validation(assignments, indices, *, per_source=6, seed=20260928):
    """Reject a whole component before selecting or accessing a file."""
    by_id={(r['dataset_id'],r['piece_id']):r for r in assignments}
    forbidden={r['component_fingerprint'] for r in assignments if normalize_split(r['split'])!='validation'}
    for source,rows in indices.items():
        for row in rows:
            a=by_id.get((source,row['piece_id']))
            if a and normalize_split(row.get('suggested_split')) not in (None,'validation'):
                forbidden.add(a['component_fingerprint'])
    result=[]
    for source,rows in sorted(indices.items()):
        eligible=[]
        for row in rows:
            a=by_id.get((source,row['piece_id']))
            native=normalize_split(row.get('suggested_split'))
            if (a and normalize_split(a['split'])=='validation' and native in (None,'validation')
                    and a['component_fingerprint'] not in forbidden):
                eligible.append(dict(source=source,piece_id=row['piece_id'],source_group_id=row['source_group_id'],
                    component_id=a['component_fingerprint'],global_split='validation',native_split=native,
                    path=row['canonical_relative_path'],sha256=row['canonical_sha256']))
        eligible.sort(key=lambda r:sha256(f"{seed}:{source}:{r['component_id']}:{r['piece_id']}".encode()).hexdigest())
        seen=set()
        for row in eligible:
            if row['component_id'] in seen:continue
            seen.add(row['component_id']);result.append(row)
            if len(seen)==per_source:break
        if len(seen)!=per_source:raise ValueError('insufficient validation components')
    return result


def read_bound(root, binding):
    root=Path(root).resolve();relative=Path(binding['path'])
    if relative.is_absolute() or '..' in relative.parts:raise ValueError('unsafe path')
    path=(root/relative).resolve()
    if not path.is_relative_to(root):raise ValueError('unsafe symlink')
    raw=path.read_bytes()
    if sha256(raw).hexdigest()!=binding['sha256'] or ('size_bytes' in binding and len(raw)!=binding['size_bytes']):
        raise ValueError('file integrity')
    return raw


def read_validation(root, row, accesses):
    if row['global_split']!='validation' or row['native_split'] not in (None,'validation'):
        raise ValueError('preload heldout guard')
    access=dict(piece_id=row['piece_id'],declared_split='validation',observed_split=None,verified=False)
    accesses.append(access)
    raw=read_bound(root,row)
    piece=loads_piece(raw.decode())
    access['observed_split']=piece.split
    if normalize_split(piece.split) not in (None,'validation'):
        raise ValueError('embedded split conflict')
    if (piece.piece_id,piece.dataset_name,piece.source_group_id)!=(row['piece_id'],row['source'],row['source_group_id']):
        raise ValueError('embedded identity conflict')
    if row['source']=='pop909_cl' and (any(n.channel==1 for n in piece.notes) or any(t.channel==1 for t in piece.tracks)):
        raise ValueError('annotation channel')
    access['verified']=True
    return replace(piece,split='validation',targets=(),annotations=(),key_signature_events=(),quality_flags=())


def query_json(query):
    return dict(start=[query.start.num,query.start.den],end=[query.end.num,query.end.den],track_ids=list(query.track_ids))


def query_from_json(raw):
    return MaskQuery(Q(*raw['start']),Q(*raw['end']),tuple(raw['track_ids']))


def fixed_queries(piece):
    if len(piece.bars)<2:raise ValueError('fewer than two bars')
    tracks=tuple(t.track_id for t in piece.tracks if any(n.track_id==t.track_id for n in piece.notes))
    result=[]
    for length in (1,2):
        start=(len(piece.bars)-length)//2
        first,last=piece.bars[start],piece.bars[start+length-1]
        result.append(MaskQuery(first.start_qn,last.start_qn+last.duration_qn,tracks))
    return tuple(result)


def pair_query(original, variant):
    if original.bars!=variant.bars or original.tracks!=variant.tracks or original.tempo_events!=variant.tempo_events:
        raise ValueError('paired geometry changed')
    left={n.note_id:n for n in original.notes};right={n.note_id:n for n in variant.notes}
    def content(n):
        return None if n is None else (n.track_id,n.pitch,n.onset_qn,n.duration_qn,n.velocity,
                                       n.channel,n.program,n.is_percussion,n.is_grace)
    changed={key for key in left.keys()|right.keys() if content(left.get(key))!=content(right.get(key))}
    if not changed:raise ValueError('no actual change')
    notes=[n for key in changed for n in (left.get(key),right.get(key)) if n is not None]
    start=min(n.onset_qn for n in notes);last=max(n.onset_qn for n in notes)
    bars=[b for b in original.bars if b.start_qn<=last and b.start_qn+b.duration_qn>start]
    if not bars or len(bars)>4:raise ValueError('changed span exceeds four bars')
    query=MaskQuery(bars[0].start_qn,bars[-1].start_qn+bars[-1].duration_qn,
                   tuple(t.track_id for t in original.tracks))
    return query,tuple(sorted(changed))


def paired_examples(original,variant,query,profile=ContextProfile()):
    a=split_masked_piece(original,query,profile);b=split_masked_piece(variant,query,profile)
    # IDs and hidden labels are already removed at this boundary. Include exact
    # query/track mapping, all observable note attributes and raw schedules.
    def identity(example):
        return sha256(json.dumps(asdict(example.context),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    left,right=identity(a),identity(b)
    if left!=right:raise ValueError('paired observed contexts differ')
    return a,b,left


def qn_seconds(piece, point):
    point=point.to_fraction() if isinstance(point,Q) else Fraction(point)
    tempos=sorted(piece.tempo_events,key=lambda x:x.onset_qn)
    if not tempos or tempos[0].onset_qn>Q(0):raise ValueError('tempo schedule missing')
    total=0.0
    for i,event in enumerate(tempos):
        start=event.onset_qn.to_fraction()
        if start>=point:break
        end=min(point,tempos[i+1].onset_qn.to_fraction() if i+1<len(tempos) else point)
        total+=float(end-start)*event.microseconds_per_quarter/1_000_000
    return total


def audio_events(piece, start, end):
    origin=qn_seconds(piece,start);duration=qn_seconds(piece,end)-origin
    tracks={t.track_id:t for t in piece.tracks};events=[]
    for note in piece.notes:
        a=note.onset_qn;b=a+note.duration_qn
        if a>=end or b<=start:continue
        track=tracks[note.track_id]
        events.append(dict(id=note.note_id,pitch=note.pitch,velocity=note.velocity if note.velocity is not None else 80,
            program=note.program if note.program is not None else (track.program or 0),is_percussion=note.is_percussion,
            onset_seconds=max(0,qn_seconds(piece,a)-origin),
            duration_seconds=max(0,qn_seconds(piece,min(b,end))-qn_seconds(piece,max(a,start)))))
    return events,duration


def save_piece(root,piece):
    raw=dumps_piece(piece).encode();digest=sha256(raw).hexdigest();relative='pieces/'+digest+'.json'
    path=Path(root)/relative;path.parent.mkdir(exist_ok=True)
    if not path.exists():path.write_bytes(raw)
    return dict(path=relative,sha256=digest,size_bytes=len(raw))
