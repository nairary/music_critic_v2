"""Authenticated, bounded census of weak graph-analysis supervision (no training)."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

from music_critic.experiments.supervision_stream import iter_top_object

EXPECTED = {
 'harmony': ['sonority','extensions','inversion','key','roman','function','tension_role'],
 'notes': ['chord_membership','key_degree','chord_degree','voice','temporal_role','preparation','resolution','tension_role'],
 'harmonic_processes': ['process','function','tension_role'],
 'ensemble': ['parts_voices','texture','interactions','tension_role'],
 'phrases': ['function','question_answer','tension_role'],
 'motifs': ['identity','transformations','tension_role'],
 'form': ['section','repetition_contrast','tension_role'],
}
OBJECT_LAYERS={'harmonic_process':'harmonic_processes','ensemble':'ensemble','phrase':'phrases','motif':'motifs','form':'form','section':'form'}

def q(v):
 if not isinstance(v,dict) or set(v)!={'num','den'} or type(v.get('num')) is not int or type(v.get('den')) is not int or v['den']<=0:
  raise ValueError('Invalid exact rational')
 result=Fraction(v['num'],v['den'])
 if (result.numerator,result.denominator)!=(v['num'],v['den']):raise ValueError('Unreduced exact rational')
 return result

def interval(r):return q(r['start_qn']),q(r['end_qn'])
def view(r):return r.get('view_id',r.get('harmony_view','unspecified'))
def known(f):return isinstance(f,dict) and f.get('available') is True and f.get('value') is not None and f.get('annotation_state','known')=='known'
def decision(f):
 if f is None:return 'missing'
 if not isinstance(f,dict) or type(f.get('available')) is not bool:raise ValueError('Invalid field availability')
 state=f.get('annotation_state','known' if f['available'] else 'unreviewed')
 if state not in ('known','not_applicable','unreviewed','unresolved','conflicting') or f['available']!=(state=='known') or (f.get('value') is not None)!=f['available']:
  raise ValueError('Contradictory field state/value/availability')
 return state
def category(value):
 if type(value) in (str,bool,int,float):return (type(value).__name__,str(value).lower() if type(value) is bool else str(value))
 if isinstance(value,dict) and isinstance(value.get('label'),str):return ('str',value['label'])
 return None

def check_deadline(deadline):
 if deadline is not None and time.monotonic()>deadline:raise TimeoutError('Census cooperative wall cap')


def validate_binding(e,work_id=None,role=None):
 if not isinstance(e,dict):raise ValueError('Invalid binding')
 rel=Path(e.get('path',''))
 if not e.get('path') or rel.is_absolute() or '..' in rel.parts:raise ValueError('Flat relative admission required')
 digest=e.get('sha256','')
 if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest) or type(e.get('size_bytes')) is not int or e['size_bytes']<0:raise ValueError('Invalid SHA/size binding')
 if work_id is not None and e.get('work_id')!=work_id:raise ValueError('Binding work identity mismatch')
 if role is not None and e.get('role')!=role:raise ValueError('Binding role mismatch')


def validate_admission(admission):
 if admission.get('schema')!='HiergraphSupervisionAdmission@1' or len(admission.get('works',[]))!=67:raise ValueError('Expected exact current67 admission')
 ids=set();pieces=set();families=set();bindings=[];paths=set()
 for w in admission['works']:
  for key,seen in [('id',ids),('piece_id',pieces),('family',families)]:
   val=w.get(key)
   if not isinstance(val,str) or not val or val in seen:raise ValueError('Missing or duplicate work '+key)
   seen.add(val)
  if set(w.get('bindings',{}))!={'canonical_piece','bundle','reviewed_report'}:raise ValueError('Expected three flat binding roles')
  for role,e in w['bindings'].items():
   validate_binding(e,w['id'],role)
   if e['path'] in paths:raise ValueError('Duplicate admitted payload path')
   paths.add(e['path']);bindings.append(e)
 entries=admission.get('entries')
 if entries is None or len(entries)!=len(bindings) or sorted(json.dumps(e,sort_keys=True) for e in entries)!=sorted(json.dumps(e,sort_keys=True) for e in bindings):raise ValueError('Flat entries disagree with per-work bindings')
 return admission


def load_small_object(path,deadline=None):
 result={}
 for key,value in iter_top_object(path,streamed_arrays=frozenset()):
  check_deadline(deadline);result[key]=value
 return result


def checked_path(base,e,deadline=None):
 validate_binding(e)
 rel=Path(e['path'])
 if rel.is_absolute() or '..' in rel.parts:raise ValueError('Flat relative admission required')
 p=base/rel
 if any(x.is_symlink() for x in (p,*(base/r for r in rel.parents if r!=Path('.')))) or not p.resolve().is_relative_to(base.resolve()) or not p.is_file():raise ValueError('Unsafe admitted path')
 h=sha256();size=0
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):
   check_deadline(deadline);h.update(b);size+=len(b)
 if size!=e['size_bytes'] or h.hexdigest()!=e['sha256']:raise ValueError('Input integrity mismatch: '+str(rel))
 return p

class Inventory:
 def __init__(self):
  self.fields={};self.classes=defaultdict(lambda:defaultdict(Counter));self.schemas=Counter()
 def add(self,layer,row,work,local):
  fields=row.get('fields',{});names=set(fields)|{'analysis.'+f for f in EXPECTED.get(layer,[])}
  for name in sorted(names):
   key=layer+'/'+name;f=fields.get(name);state=decision(f)
   if key not in self.fields:
    self.fields[key]={'states':Counter(),'origins_known':Counter(),'shapes':Counter(),'derivations':Counter(),'upstream_origins':Counter(),'known_work_counts':Counter(),'raw_value_hashes':set(),'confidence_nonnull_known':0,'dict_keys':Counter(),'examples':[]}
   s=self.fields[key];s['states'][state]+=1;local.setdefault(key,Counter())[state]+=1
   if not known(f):continue
   s['known_work_counts'][work]+=1;s['origins_known'][str(f.get('origin'))]+=1
   v=f['value'];s['shapes'][type(v).__name__]+=1
   raw=json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'));s['raw_value_hashes'].add(sha256(raw.encode()).hexdigest())
   if f.get('confidence') is not None:s['confidence_nonnull_known']+=1
   deriv=f.get('derivation') or {};s['derivations'][str(deriv.get('rule','none'))]+=1;s['upstream_origins'][str(deriv.get('upstream_origin','none'))]+=1
   if isinstance(v,dict):s['dict_keys'].update(v.keys())
   lab=category(v)
   if lab is not None:self.classes[key][lab][work]+=1
   if isinstance(v,dict):
    for k in ('root','quality','status'):
     leaf=category(v.get(k))
     if leaf is not None:self.classes[key+'.'+k][leaf][work]+=1
   if len(s['examples'])<3 and raw not in s['examples']:s['examples'].append(raw[:400])
 def result(self):
  out={}
  for key,s in self.fields.items():
   v={k:(dict(x) if isinstance(x,Counter) else x) for k,x in s.items() if k not in ('raw_value_hashes','known_work_counts')}
   wc=s['known_work_counts'];total=sum(wc.values());v.update(known=sum(wc.values()),works_with_known=len(wc),raw_distinct_values=len(s['raw_value_hashes']),known_by_work=dict(wc),top_work_share=max(wc.values(),default=0)/total if total else None,top3_work_share=sum(sorted(wc.values(),reverse=True)[:3])/total if total else None)
   out[key]=v
  return out
 def class_result(self):
  return {k:[{'label':label[1],'json_type':label[0],'count':sum(w.values()),'works':len(w),'by_work':dict(w)} for label,w in sorted(vals.items(),key=lambda p:(-sum(p[1].values()),p[0]))] for k,vals in self.classes.items()}

def union_length(spans):
 end=None;total=Fraction(0)
 for a,b in sorted(spans):
  if end is None or a>end:total+=b-a;end=b
  elif b>end:total+=b-end;end=b
 return total

def skim_fields(row):
 return {k:{'available':known(f),'value':f.get('value') if known(f) else None} for k,f in row.get('fields',{}).items() if k in ('analysis.function','analysis.sonority','analysis.key','analysis.chord_membership','analysis.temporal_role','analysis.resolution','analysis.preparation','analysis.process')}

def has_role_container(row):
 paths=[]
 def walk(x,path,depth):
  if depth>4:return
  if isinstance(x,dict):
   for k,v in x.items():
    if k in ('rationale','evidence','unknown_reason','decision_reason','literal_active_states'):continue
    p=path+'.'+k
    if any(t in k.lower() for t in ('member_role','participant_role','source_ids','target_ids','preparation_ids','resolution_ids')) and v:paths.append(p)
    if isinstance(v,(dict,list)):walk(v,p,depth+1)
  elif isinstance(x,list):
   for v in x[:200]:
    if isinstance(v,(dict,list)):walk(v,path+'[]',depth+1)
 walk(row,'P',0);return sorted(set(paths))

def analyse_work(w,base,inv,readlog,progress,deadline):
 def check_time():
  check_deadline(deadline)
 paths={}
 for role,e in w['bindings'].items():
  check_time();paths[role]=checked_path(base,e,deadline);readlog.append({**e,'verified':True})
  progress('verified',w['id'],role)
 # Canonical payload is at most 8MB in this admission; retain raw IDs/geometry only.
 c=load_small_object(paths['canonical_piece'],deadline)
 if c.get('schema_version')!='2.0.0' or c.get('split') not in (None,'train') or c.get('piece_id')!=w['piece_id'] or not isinstance(c.get('dataset_name'),str) or not c['dataset_name']:raise ValueError('Non-TRAIN or canonical schema/identity mismatch')
 if w.get('dataset_id',c['dataset_name'])!=c['dataset_name']:raise ValueError('Admission dataset identity mismatch')
 notes={n['note_id']:n for n in c['notes']}
 if len(notes)!=len(c['notes']):raise ValueError('Duplicate canonical note identity')
 tracks=Counter(n['track_id'] for n in notes.values());raw_anchors={Fraction(0),q(c['duration_qn'])};attacks=set()
 for n in notes.values():
  check_time();a=q(n['onset_qn']);b=a+q(n['duration_qn']);raw_anchors.update((a,b));attacks.add(a)
 for r in c.get('beats',[])+c.get('bars',[]):
  check_time();start=q(r['start_qn']);raw_anchors.add(start)
  # Canonical beats/bars store exact start+duration, not an end field.
  end=start+q(r['duration_qn']) if 'duration_qn' in r else q(r['end_qn'])
  if 'end_qn' in r and q(r['end_qn'])!=end:raise ValueError('Canonical metric end disagrees with duration')
  raw_anchors.add(end)
 note_geo={k:(q(n['onset_qn']),q(n['onset_qn'])+q(n['duration_qn'])) for k,n in notes.items()}
 metrics=Counter();layer_rows=Counter();local={};H={};contexts={};objects={};relations={};header={};scopekeys=Counter();scopevalues=Counter();pattrs=Counter();psig=Counter();rsig=Counter();rolepaths=Counter();H_members=defaultdict(Counter);hviews=defaultdict(list);ngroups=defaultdict(set);contextspans=defaultdict(list);vlocal=defaultdict(lambda:defaultdict(Counter));vrows=defaultdict(Counter)
 counts_expected=w['counts_from_metadata']
 for key,row in iter_top_object(paths['bundle'],streamed_arrays=frozenset({'harmonies','note_contexts','objects','relations'})):
  check_time()
  if key not in ('harmonies','note_contexts','objects','relations'):header[key]=row;continue
  if key=='harmonies':
   layer='harmony';inv.add(layer,row,w['id'],local);a,b=interval(row);hid=row['id']
   if hid in H:raise ValueError('Duplicate H identity')
   fields=skim_fields(row);H[hid]={'interval':(a,b),'view':view(row),'fields':fields};hviews[view(row)].append((a,b))
   metrics['H_total']+=1;metrics['H_positive_interval']+=int(b>a);metrics['H_start_on_raw_anchor']+=int(a in raw_anchors);metrics['H_end_on_raw_anchor']+=int(b in raw_anchors);metrics['H_both_on_raw_anchors']+=int(a in raw_anchors and b in raw_anchors);metrics['H_start_on_note_attack']+=int(a in attacks)
   for owner,d in [('top',row),('attributes',row.get('attributes',{}))]:
    if isinstance(d,dict):
     for k in d:
      if any(x in k.lower() for x in ('scope','track','part_id','voice_id','member')):scopekeys[owner+'.'+k]+=1
   for owner,d in [('top',row),('attributes',row.get('attributes',{}))]:
    if isinstance(d,dict):
     for k in ('interpretation_scope','participant_scope','scope'):
      if isinstance(d.get(k),str):scopevalues[owner+'.'+k+'='+json.dumps(d[k],ensure_ascii=False)]+=1
   explicit=[k for k in ('participant_scope','track_ids','part_ids','voice_ids','member_ids') if row.get(k) or row.get('attributes',{}).get(k)]
   metrics['H_explicit_participant_scope_or_ids']+=int(bool(explicit))
   template=row.get('note_template') or {};metrics['H_note_template']+=int(bool(template));metrics['H_template_members_complete']+=int(template.get('members_complete') is True)
   for names,label in [(('analysis.function','analysis.key'),'H_function_and_key'),(('analysis.function','analysis.sonority'),'H_function_and_sonority'),(('analysis.function','analysis.sonority','analysis.key'),'H_function_sonority_key')]:metrics[label]+=int(all(known(row['fields'].get(k)) for k in names))
  elif key=='note_contexts':
   layer='notes';inv.add(layer,row,w['id'],local);cid=row['id']
   if cid in contexts:raise ValueError('Duplicate context identity')
   fs=skim_fields(row);contexts[cid]={'note':row['note_id'],'H':row.get('harmony_id'),'interval':interval(row),'view':view(row),'fields':fs}
   metrics['h_total']+=1
  elif key=='objects':
   typ=row.get('object_type',row.get('type',row.get('kind')));layer=OBJECT_LAYERS.get(typ,'object:'+str(typ));inv.add(layer,row,w['id'],local)
   oid=row['id']
   if oid in objects:raise ValueError('Duplicate object identity')
   attrs=row.get('attributes') or {};objects[oid]={'type':typ,'interval':interval(row),'view':view(row),'members':row.get('member_ids',[]),'fields':skim_fields(row),'harmony_ids':attrs.get('harmony_ids',row.get('harmony_ids',[])),'role_paths':has_role_container(row)}
   if typ=='harmonic_process':
    metrics['P_total']+=1;pattrs.update(attrs.keys());rolepaths.update(objects[oid]['role_paths'])
  else:
   layer='relation:'+str(row.get('type'));rid=row['id']
   if rid in relations:raise ValueError('Duplicate relation identity')
   decision(row)
   relations[rid]={'value':row.get('value'),'type':row['type'],'source':row.get('source_ids',[]),'target':row.get('target_ids',[]),'known':known(row),'interval':interval(row),'view':view(row),'origin':row.get('origin')}
   metrics['relations_total']+=1
  layer_rows[layer]+=1;vrows[view(row)][layer]+=1
  for name in set(row.get('fields',{}))|{'analysis.'+f for f in EXPECTED.get(layer,[])}:
   vlocal[view(row)][layer+'/'+name][decision(row.get('fields',{}).get(name))]+=1
 if header.get('schema')!='EventTheoryEnrichment@1' or header.get('dataset_id')!=c['dataset_name'] or header.get('current_split')!='train' or header.get('piece_id')!=w['piece_id']:raise ValueError('Non-TRAIN bundle or identity mismatch')
 if header.get('canonical_piece',{}).get('sha256')!=w['bindings']['canonical_piece']['sha256']:raise ValueError('Canonical binding mismatch')
 for k,actual in [('notes',len(notes)),('harmonies',len(H)),('note_contexts',len(contexts)),('objects',len(objects)),('relations',len(relations))]:
  if counts_expected[k]!=actual:raise ValueError('Release count mismatch '+w['id']+' '+k)
 all_ids={**{k:'n' for k in notes},**{k:'H' for k in H},**{k:'h' for k in contexts},**{k:('P' if o['type']=='harmonic_process' else 'object:'+str(o['type'])) for k,o in objects.items()}}
 if len(all_ids)!=len(notes)+len(H)+len(contexts)+len(objects):raise ValueError('Duplicate cross-kind IDs')
 def kinds(ids):return ','.join(sorted(set(all_ids.get(k,'missing') for k in ids))) or 'empty'
 duration=q(c['duration_qn'])
 geometry={**note_geo,**{k:v['interval'] for group in (H,contexts,objects) for k,v in group.items()}}
 entities={**H,**contexts,**objects}
 def valid_scope(x):
  a,b=x['interval'];return 0<=a<=b<=duration and isinstance(x['view'],str) and bool(x['view'])
 def contains(owner,identity):
  if identity not in geometry:return False
  a,b=owner['interval'];l,r=geometry[identity]
  return a<=l<=r<=b and (identity in notes or entities[identity]['view']==owner['view'])
 def valid_context(x):
  return valid_scope(x) and x['note'] in note_geo and note_geo[x['note']][0]<=x['interval'][0]<=x['interval'][1]<=note_geo[x['note']][1]
 def valid_h_context(x):
  h=H.get(x['H'])
  return valid_context(x) and h is not None and valid_scope(h) and h['interval'][0]<h['interval'][1] and h['view']==x['view'] and h['interval'][0]<=x['interval'][0]<=x['interval'][1]<=h['interval'][1] and x['interval'][0]<h['interval'][1]
 outgoing=defaultdict(set);incoming=defaultdict(set)
 for rid,r in relations.items():
  check_time();typ=r['type'];metrics['p_type:'+typ]+=1;metrics['p_known:'+typ]+=int(r['known']);metrics['p_binary:'+typ]+=int(len(r['source'])==len(r['target'])==1)
  rsig[typ+'|'+kinds(r['source'])+'->'+kinds(r['target'])+'|'+str(len(r['source']))+'x'+str(len(r['target']))]+=1
  if any(k not in all_ids for k in r['source']+r['target']):metrics['p_missing_endpoint']+=1
  r['positive']=r['known'] and r['value'] is True
  ids=r['source']+r['target']
  r['valid']=valid_scope(r) and all(r[side] and len(r[side])==len(set(r[side])) for side in ('source','target')) and all(contains(r,x) for x in ids)
  if r['type'] in ('prepares','resolves_to'):
   r['valid']=r['valid'] and not set(r['source'])&set(r['target']) and all(all_ids.get(x) in ('n','h') for x in ids) and max((geometry[x][0] for x in r['source'] if x in geometry),default=duration+1)<=min((geometry[x][0] for x in r['target'] if x in geometry),default=-1)
  metrics['p_positive:'+typ]+=int(r['positive']);metrics['p_valid_positive:'+typ]+=int(r['positive'] and r['valid'])
  if r['positive']:
   for x in r['source']:outgoing[(typ,x,r['view'])].add(rid)
   for x in r['target']:incoming[(typ,x,r['view'])].add(rid)
 for cid,x in contexts.items():
  check_time();a,b=x['interval'];nid=x['note'];hid=x['H'];h=H.get(hid);fs=x['fields'];ngroups[nid].add(hid);contextspans[nid].append((a,b))
  if nid not in note_geo:metrics['h_unknown_note']+=1;continue
  na,nb=note_geo[nid];usable_h=valid_h_context(x);metrics['h_valid_bound_context']+=int(usable_h);metrics['h_valid_note_context']+=int(valid_context(x));metrics['h_within_note']+=int(na<=a<=b<=nb)
  if h is not None:
   metrics['h_bound_H']+=1;ha,hb=h['interval'];metrics['h_within_H']+=int(ha<=a<=b<=hb and a<hb);metrics['h_same_view_H']+=int(h['view']==x['view'])
   metrics['h_matches_full_note_H_intersection']+=int((a,b)==(max(na,ha),min(nb,hb)))
   metrics['h_bound_H_with_known_F']+=int(h['fields'].get('analysis.function',{}).get('available',False))
  else:metrics['h_unbound_H']+=1
  f=fs.get('analysis.chord_membership',{});mv=f.get('value')
  if f.get('available') and type(mv) is bool:
   metrics['membership_'+str(mv).lower()]+=1;H_members[hid]['positive' if mv else 'negative']+=1
   if h and h['fields'].get('analysis.function',{}).get('available'):
    metrics['joint_h_membership_H_function']+=1;metrics['usable_joint_h_membership_H_function']+=int(usable_h)
   if fs.get('analysis.temporal_role',{}).get('available'):
    metrics['joint_h_membership_temporal_role']+=1;metrics['usable_joint_h_membership_temporal_role']+=int(usable_h)
  else:metrics['membership_unknown']+=1;H_members[hid]['unknown']+=1
  for field,typ,direction in [('analysis.resolution','resolves_to','source'),('analysis.preparation','prepares','target')]:
   short=field.split('.')[-1];f=fs.get(field,{});v=f.get('value');state=v.get('status') if f.get('available') and isinstance(v,dict) else 'masked';metrics[short+'_'+str(state)]+=1
   if state=='linked':
    refs=v.get('relation_ids',[]);metrics[short+'_linked_no_refs']+=int(not refs)
    for ref in refs:
     rr=relations.get(ref)
     valid=rr is not None and rr['positive'] and rr['valid'] and valid_context(x) and rr['type']==typ and (cid in rr[direction] or nid in rr[direction]) and rr['view']==x['view']
     metrics[short+'_reference_valid' if valid else short+'_reference_invalid']+=1
   if state=='none':
    lookup=outgoing if direction=='source' else incoming
    conflict=lookup.get((typ,cid,x['view']),set())|lookup.get((typ,nid,x['view']),set());metrics[short+'_none_with_positive_edge']+=int(bool(conflict))
    metrics[short+'_none_has_reason']+=int(bool(v.get('reason')))
    metrics[short+'_usable_reviewed_none']+=int(valid_context(x) and isinstance(v.get('reason'),str) and bool(v['reason'].strip()) and v.get('relation_ids')==[] and not conflict)
  if fs.get('analysis.resolution',{}).get('available') and isinstance(fs['analysis.resolution']['value'],dict) and fs['analysis.resolution']['value'].get('status')=='linked':
   metrics['joint_resolution_temporal_role']+=int(fs.get('analysis.temporal_role',{}).get('available',False))
 for hid,m in H_members.items():
  check_time()
  if hid in H:metrics['H_with_known_positive_and_negative']+=int(m['positive']>0 and m['negative']>0)
 for nid,gs in ngroups.items():metrics['notes_with_multiple_H_contexts']+=int(len(gs-{None})>1)
 metrics['notes_with_context']=len(ngroups)
 metrics['notes_fully_duration_covered_by_context_union']=0
 for nid,spans in contextspans.items():
  check_time()
  if nid not in note_geo:continue
  a,b=note_geo[nid]
  if all(a<=l<=r<=b for l,r in spans) and union_length(spans)==b-a:metrics['notes_fully_duration_covered_by_context_union']+=1
 for oid,o in objects.items():
  check_time()
  if o['type']!='harmonic_process':continue
  mids=o['members'];signature=kinds(mids);psig[signature]+=1;metrics['P_with_members']+=int(bool(mids));metrics['P_member_references']+=len(mids);metrics['P_missing_member_references']+=sum(x not in all_ids for x in mids)
  for k in ('H','h','n','P'):metrics['P_with_direct_'+k+'_members']+=int(any(all_ids.get(m)==k for m in mids))
  hs={m for m in mids if m in H};attrs=o['harmony_ids'];attrs=attrs if isinstance(attrs,list) else [];validhs={m for m in attrs if m in H};hs|=validhs
  metrics['P_with_attribute_harmony_ids']+=int(bool(attrs));metrics['P_attribute_harmony_ids_missing']+=sum(m not in H for m in attrs)
  derivedhs={contexts[m]['H'] for m in mids if m in contexts and contexts[m]['H'] in H};metrics['P_with_H_via_h']+=int(bool(derivedhs));metrics['P_with_explicit_H_reference']+=int(bool(hs))
  pknown=o['fields'].get('analysis.process',{}).get('available',False);fknown=o['fields'].get('analysis.function',{}).get('available',False)
  metrics['P_process_and_function_known']+=int(pknown and fknown);metrics['P_process_known_with_h_members']+=int(pknown and any(m in contexts for m in mids));metrics['P_process_known_with_explicit_H']+=int(pknown and bool(hs));metrics['P_process_known_with_H_via_h']+=int(pknown and bool(derivedhs));metrics['P_with_role_like_container']+=int(bool(o['role_paths']));metrics['P_process_known_with_explicit_H_and_known_H_F']+=int(pknown and bool(hs) and any(H[h]['fields'].get('analysis.function',{}).get('available') for h in hs))
  valid_members=valid_scope(o) and bool(mids) and len(mids)==len(set(mids)) and all(contains(o,m) and (m not in contexts or valid_context(contexts[m])) for m in mids)
  valid_hs={h for h in hs if valid_scope(H[h]) and H[h]['interval'][0]<H[h]['interval'][1] and contains(o,h)} if valid_scope(o) else set()
  valid_h_members={m for m in mids if m in contexts and valid_h_context(contexts[m]) and contains(o,m)} if valid_scope(o) else set()
  metrics['P_valid_members']+=int(valid_members);metrics['usable_P_process_and_function_known']+=int(pknown and fknown and valid_members)
  metrics['usable_P_process_known_with_h_members']+=int(pknown and valid_members and bool(valid_h_members))
  metrics['usable_P_process_known_with_explicit_H']+=int(pknown and valid_members and bool(valid_hs))
  metrics['usable_P_process_known_with_explicit_H_and_known_H_F']+=int(pknown and valid_members and any(H[h]['fields'].get('analysis.function',{}).get('available') for h in valid_hs))
 report=load_small_object(paths['reviewed_report'],deadline);report_layers=defaultdict(list)
 if report.get('schema')!='WholeWorkAnnotationReport@2' or report.get('piece_id')!=w['piece_id'] or report.get('dataset_id')!=c['dataset_name'] or report.get('work_family_id')!=w['family'] or report.get('canonical_piece_sha256')!=w['bindings']['canonical_piece']['sha256']:raise ValueError('Report schema/identity mismatch')
 if not isinstance(report.get('views'),dict) or not report['views'] or set(report['views'])!=set(vrows):raise ValueError('Report views mismatch')
 for v,vr in report['views'].items():
  check_time()
  if set(vr.get('layers',{}))!=set(EXPECTED):raise ValueError('Expected all seven report layers')
  for layer,lr in vr.get('layers',{}).items():
   if type(lr.get('rows')) is not int or lr['rows']!=vrows[v][layer]:metrics['report_row_mismatches']+=1
   required={'analysis.'+f for f in EXPECTED[layer] if f!='tension_role'}
   if lr.get('rows',0)>0 and not required<=set(lr.get('fields',{})):raise ValueError('Missing report field accounting')
   report_layers[layer].append({'view_id':v,**{k:lr.get(k) for k in ('rows','review_status','reviewed_full_interval','structural_coverage_complete','content_accounted','all_required_decisions_reviewed','coverage')}})
   for name,fr in lr.get('fields',{}).items():
    counts=fr.get('counts',{});got=vlocal[v].get(layer+'/'+name,Counter())
    if not isinstance(counts,dict) or any(type(n) is not int or n<0 for n in counts.values()):raise ValueError('Invalid report counts')
    for state in set(counts)|set(got):
     if got[state]!=counts.get(state,0):metrics['report_count_mismatches']+=1
 metrics['canonical_native_split_null']=int(c.get('split') is None);metrics['canonical_native_split_train']=int(c.get('split')=='train');metrics['raw_notes']=len(notes);metrics['raw_tracks_with_notes']=len(tracks);metrics['raw_notes_with_source_voice']=sum(n.get('voice') is not None for n in notes.values());metrics['raw_notes_with_source_staff']=sum(n.get('staff') is not None for n in notes.values())
 hcoverage={str(v):{'H':len(sp),'union_qn':str(union_length(sp)),'summed_qn':str(sum((b-a for a,b in sp),Fraction(0))),'piece_duration_qn':str(q(c['duration_qn']))} for v,sp in hviews.items()}
 return {'id':w['id'],'family':w['family'],'genre':w['genre'],'title':w['title'],'piece_id':w['piece_id'],'dataset':c['dataset_name'],'requested_annotator':w.get('requested_annotator'),'full_work_review_complete':w.get('full_work_review_complete'),'complete_for_declared_scope':w.get('complete_for_declared_scope'),'counts':dict(metrics),'layers':dict(layer_rows),'fields':{k:dict(v) for k,v in local.items()},'track_note_counts':dict(tracks),'H_scope_keys':dict(scopekeys),'H_scope_values':dict(scopevalues),'H_coverage_by_view':hcoverage,'P_attribute_keys':dict(pattrs),'P_member_signatures':dict(psig),'P_role_like_paths':dict(rolepaths),'relation_signatures':dict(rsig),'report_layers':dict(report_layers)}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--admission',type=Path,required=True);ap.add_argument('--source-root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--max-seconds',type=int,default=600);args=ap.parse_args()
 started=time.monotonic();deadline=started+args.max_seconds;args.output.mkdir(parents=True,exist_ok=False)
 if args.max_seconds<=0:raise ValueError('Positive cooperative wall cap required')
 admission=validate_admission(load_small_object(args.admission,deadline))
 inv=Inventory();works=[];readlog=[];counts=Counter();supports=defaultdict(list)
 def progress(stage,work,role=''):
  (args.output/'progress.json').write_text(json.dumps({'stage':stage,'work':work,'role':role,'completed_works':len(works),'wall_seconds':time.monotonic()-started})+'\n')
 try:
  for w in admission['works']:
   check_deadline(deadline);row=analyse_work(w,args.source_root,inv,readlog,progress,deadline);works.append(row);counts.update(row['counts'])
   for key,value in row['counts'].items():
    if value>0:supports[key].append(row['id'])
   with (args.output/'per-work.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')
   progress('completed_work',w['id']);print(json.dumps({'work':w['id'],'done':len(works),'H':row['counts'].get('H_total',0),'P':row['counts'].get('P_total',0),'seconds':round(time.monotonic()-started,2)}),flush=True)
  summary={'schema':'HiergraphSupervisionCensus@1','works':len(works),'families':len({w['family'] for w in works}),'genres':dict(Counter(w['genre'] for w in works)),'datasets':dict(Counter(w['dataset'] for w in works)),'counts':dict(counts),'work_support':{k:len(v) for k,v in supports.items()},'work_support_ids':dict(supports),'raw_track_count_distribution':dict(Counter(str(w['counts']['raw_tracks_with_notes']) for w in works)),'full_work_review_complete_works':sum(w['full_work_review_complete'] is True for w in works),'complete_for_declared_scope_works':sum(w['complete_for_declared_scope'] is True for w in works),'input_admission_sha256':sha256(args.admission.read_bytes()).hexdigest(),'input_files_verified':len(readlog),'input_bound_bytes':sum(e['size_bytes'] for e in readlog),'wall_seconds':time.monotonic()-started,'training_updates':0,'model_inference':False,'test_accessed':False,'cap_semantics':'cooperative_wall_time_checked_between_rows_hash_chunks_and_postprocessing; a single JSON value or write may overrun','validity_gated_metrics_prefixes':['usable_','p_valid_positive:'],'role_like_container_is_shape_inventory_only':True}
  for name,obj in [('summary.json',summary),('field-inventory.json',inv.result()),('class-inventory.json',inv.class_result()),('read-log.json',readlog)]:
   check_deadline(deadline);(args.output/name).write_text(json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
  progress('completed',works[-1]['id']);print('COMPLETED',json.dumps({'works':len(works),'wall_seconds':summary['wall_seconds']}))
 except BaseException as error:
  (args.output/'failure.json').write_text(json.dumps({'type':type(error).__name__,'message':str(error),'completed_works':len(works),'wall_seconds':time.monotonic()-started},ensure_ascii=False)+'\n');(args.output/'read-log.json').write_text(json.dumps(readlog,ensure_ascii=False,indent=2)+'\n');raise

if __name__=='__main__':main()
