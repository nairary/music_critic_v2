"""Synthetic tests for census masks, joins, provenance and authentication."""
import importlib.util
import json
from hashlib import sha256
from pathlib import Path
import tempfile
import time
import unittest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('hiergraph_audit',ROOT/'scripts/audit_hiergraph_supervision.py')
m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m)

def t(n):return {'num':n,'den':1}
def f(v,state='known',origin='astra_individual_context'):
 return {'value':v if state=='known' else None,'available':state=='known','annotation_state':state,'origin':origin,'confidence':None}
def row(i,a,b,fields,**extra):return dict(id=i,start_qn=t(a),end_qn=t(b),view_id='v',fields=fields,**extra)

class CensusTests(unittest.TestCase):
 def fixture(self,root,two_views=False,split='train',metric_field=None,explicit_metric_end=False,conflicting_metric_end=False):
  c={'schema_version':'2.0.0','piece_id':'piece','dataset_name':'fixture','split':split,'duration_qn':t(2),'bars':[],'beats':[], 'notes':[
   {'note_id':'n1','onset_qn':t(0),'duration_qn':t(2),'track_id':'a','pitch':60},
   {'note_id':'n2','onset_qn':t(1),'duration_qn':t(1),'track_id':'b','pitch':62}]}
  hs=[row('H1',0,1,{'analysis.function':f('tonic'),'analysis.key':f('C major')}),row('H2',1,2,{'analysis.function':f('dominant')})]
  if two_views:hs.append(dict(row('H3',0,2,{'analysis.function':f(None,'unresolved')}),view_id='v2'))
  cs=[row('c1',0,1,{'analysis.chord_membership':f(True,'known','source_derived'),'analysis.temporal_role':f('structural'),'analysis.resolution':f({'status':'linked','relation_ids':['r1']}),'analysis.preparation':f({'status':'none','relation_ids':[],'reason':'reviewed'})},note_id='n1',harmony_id='H1'),
      row('c2',1,2,{'analysis.chord_membership':f(False),'analysis.temporal_role':f('suspension'),'analysis.resolution':f({'status':'none','relation_ids':[],'reason':'reviewed'})},note_id='n1',harmony_id='H2'),
      row('c3',1,2,{'analysis.chord_membership':f(None,'unresolved')},note_id='n2',harmony_id='H2')]
  if metric_field is not None:
   half={'num':1,'den':2}
   metric={'start_qn':{'num':1,'den':4},'duration_qn':{'num':1,'den':4}}
   if explicit_metric_end:
    metric['end_qn']=half
   if conflicting_metric_end:
    metric.update(end_qn=t(1))
   c[metric_field]=[metric]
   hs[0]['end_qn']=half;hs[1]['start_qn']=half
   cs[0]['end_qn']=half;cs[1]['start_qn']=half
  obj=row('P1',0,2,{'analysis.process':f({'label':'prolongation','comment':'not a new class'}),'analysis.function':f('tonic')},object_type='harmonic_process',member_ids=['c1','c2'],attributes={'harmony_ids':['H1','H2']})
  rel=dict(f(True),id='r1',type='resolves_to',source_ids=['c1'],target_ids=['c2'],start_qn=t(0),end_qn=t(2),view_id='v')
  canonical=root/'canonical.json';canonical.write_text(json.dumps(c))
  cb={'path':'canonical.json','sha256':sha256(canonical.read_bytes()).hexdigest(),'size_bytes':canonical.stat().st_size}
  bundle={'schema':'EventTheoryEnrichment@1','dataset_id':'fixture','current_split':'train','piece_id':'piece','canonical_piece':cb,'harmonies':hs,'note_contexts':cs,'objects':[obj],'relations':[rel]}
  report={'schema':'WholeWorkAnnotationReport@2','dataset_id':'fixture','piece_id':'piece','work_family_id':'family0','canonical_piece_sha256':cb['sha256'],'views':{}}
  for vid in ('v','v2') if two_views else ('v',):
   report['views'][vid]={'layers':{}}
   for layer in m.EXPECTED:
    scope=hs if layer=='harmony' else cs if layer=='notes' else [obj] if layer=='harmonic_processes' else []
    layerrows=[r for r in scope if m.view(r)==vid]
    fields={}
    for name in ('analysis.'+x for x in m.EXPECTED[layer] if x!='tension_role'):
     counts={state:0 for state in ('known','not_applicable','unreviewed','unresolved','conflicting','missing')}
     for r in layerrows:counts[m.decision(r['fields'].get(name))]+=1
     fields[name]={'counts':counts}
    report['views'][vid]['layers'][layer]={'rows':len(layerrows),'fields':fields}
  bindings={'canonical_piece':dict(cb,role='canonical_piece',work_id='work0')}
  for name,data in [('bundle',bundle),('reviewed_report',report)]:
   p=root/(name+'.json');p.write_text(json.dumps(data));bindings[name]={'path':p.name,'sha256':sha256(p.read_bytes()).hexdigest(),'size_bytes':p.stat().st_size,'role':name,'work_id':'work0'}
  return {'id':'work0','family':'family0','genre':'fixture','title':'fixture','piece_id':'piece','counts_from_metadata':{'notes':2,'harmonies':len(hs),'note_contexts':3,'objects':1,'relations':1},'bindings':bindings}
 def run_fixture(self,**kw):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);w=self.fixture(root,**kw);inv=m.Inventory();reads=[]
   result=m.analyse_work(w,root,inv,reads,lambda *a:None,time.monotonic()+10)
  return result,inv,reads
 def rewrite(self,root,w,role,mutate):
  path=root/w['bindings'][role]['path'];data=json.loads(path.read_text());mutate(data);path.write_text(json.dumps(data))
  w['bindings'][role].update(sha256=sha256(path.read_bytes()).hexdigest(),size_bytes=path.stat().st_size)
 def analyse(self,root,w):
  return m.analyse_work(w,root,m.Inventory(),[],lambda *args:None,time.monotonic()+10)
 def test_known_false_is_not_unknown_and_missing_not_negative(self):
  r,inv,_=self.run_fixture();c=r['counts']
  self.assertEqual((c['membership_true'],c['membership_false'],c['membership_unknown']),(1,1,1))
  self.assertEqual((c['resolution_linked'],c['resolution_none'],c['resolution_masked']),(1,1,1))
  self.assertEqual(c.get('resolution_reference_invalid',0),0)
  self.assertEqual(c.get('resolution_none_with_positive_edge',0),0)
  self.assertEqual(inv.result()['notes/analysis.chord_membership']['known'],2)
 def test_process_members_and_roles_are_different_claims(self):
  r,_,_=self.run_fixture();c=r['counts']
  self.assertEqual(c['P_with_direct_h_members'],1)
  self.assertEqual(c['P_with_direct_H_members'],0)
  self.assertEqual(c['P_with_explicit_H_reference'],1)
  self.assertEqual(c['P_process_known_with_explicit_H'],1)
  self.assertEqual(c['P_with_role_like_container'],0)
 def test_held_note_and_exact_context_join(self):
  r,_,_=self.run_fixture();c=r['counts']
  self.assertEqual(c['notes_with_multiple_H_contexts'],1)
  self.assertEqual(c['h_matches_full_note_H_intersection'],3)
  self.assertEqual(c['H_both_on_raw_anchors'],2)
  self.assertEqual(c['raw_tracks_with_notes'],2)
 def test_canonical_metric_duration_end_anchors_H_without_note_boundary(self):
  # Canonical beats/bars have start+duration. Their 1/2 end is absent
  # from raw note starts/ends (0,1,2) and the piece endpoints (0,2).
  for metric_field in ('beats','bars'):
   with self.subTest(metric_field=metric_field):
    r,_,_=self.run_fixture(metric_field=metric_field)
    self.assertEqual(r['counts']['H_end_on_raw_anchor'],2)
    self.assertEqual(r['counts']['H_both_on_raw_anchors'],2)
 def test_consistent_explicit_metric_end_is_preserved(self):
  for metric_field in ('beats','bars'):
   with self.subTest(metric_field=metric_field):
    r,_,_=self.run_fixture(metric_field=metric_field,explicit_metric_end=True)
    self.assertEqual(r['counts']['H_end_on_raw_anchor'],2)
    self.assertEqual(r['counts']['H_both_on_raw_anchors'],2)
 def test_conflicting_metric_end_is_rejected(self):
  for metric_field in ('beats','bars'):
   with self.subTest(metric_field=metric_field):
    with self.assertRaisesRegex(ValueError,'metric end disagrees with duration'):
     self.run_fixture(metric_field=metric_field,conflicting_metric_end=True)
 def test_report_comparison_is_view_scoped(self):
  r,inv,_=self.run_fixture(two_views=True)
  self.assertEqual(r['counts'].get('report_count_mismatches',0),0)
  self.assertEqual(inv.result()['harmony/analysis.function']['states'],{'known':2,'unresolved':1})
 def test_categories_project_label_not_explanatory_dict(self):
  _,inv,_=self.run_fixture();rows=inv.class_result()['harmonic_processes/analysis.process']
  self.assertEqual(rows[0]['label'],'prolongation');self.assertEqual(rows[0]['works'],1)
 def test_train_only_and_hash_failure(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);w=self.fixture(root,split='test')
   with self.assertRaisesRegex(ValueError,'Non-TRAIN'):m.analyse_work(w,root,m.Inventory(),[],lambda *a:None,time.monotonic()+10)
   (root/'bundle.json').write_text('{}')
   with self.assertRaisesRegex(ValueError,'integrity'):m.checked_path(root,w['bindings']['bundle'])
 def test_nullable_canonical_split_requires_current_train_bundle(self):
  r,_,_=self.run_fixture(split=None)
  self.assertEqual(r['counts']['canonical_native_split_null'],1)
  self.assertEqual(r['counts']['canonical_native_split_train'],0)
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);w=self.fixture(root,split=None)
   self.rewrite(root,w,'bundle',lambda b:b.update(current_split='test'))
   with self.assertRaisesRegex(ValueError,'Non-TRAIN'):self.analyse(root,w)
  r,_,_=self.run_fixture(split='train')
  self.assertEqual(r['counts']['canonical_native_split_train'],1)
  self.assertEqual(r['counts']['canonical_native_split_null'],0)
 def test_exact_rational_and_union(self):
  self.assertEqual(m.q({'num':1,'den':2}),m.Fraction(1,2))
  self.assertEqual(m.union_length([(m.q(t(0)),m.q(t(2))),(m.q(t(1)),m.q(t(3)))]),3)
  for value in ({'num':2,'den':4},{'num':1,'den':2,'extra':1},{'num':1,'den':0},{'num':True,'den':1},{'num':1.0,'den':1}):
   with self.assertRaises(ValueError):m.q(value)

 def test_known_false_relation_is_neither_positive_nor_valid_link(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);w=self.fixture(root)
   self.rewrite(root,w,'bundle',lambda b:b['relations'][0].update(value=False))
   c=self.analyse(root,w)['counts']
   self.assertEqual(c['p_known:resolves_to'],1)
   self.assertEqual(c['p_positive:resolves_to'],0)
   self.assertEqual(c['resolution_reference_invalid'],1)
   self.assertEqual(c.get('resolution_none_with_positive_edge',0),0)
 def test_reviewed_none_ignores_explicit_false_edge(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);w=self.fixture(root)
   def change(b):
    b['relations'][0]['value']=False
    b['note_contexts'][0]['fields']['analysis.resolution']=f({'status':'none','relation_ids':[],'reason':'reviewed'})
   self.rewrite(root,w,'bundle',change)
   c=self.analyse(root,w)['counts']
   self.assertEqual(c.get('resolution_none_with_positive_edge',0),0)
   self.assertEqual(c['resolution_usable_reviewed_none'],2)
 def test_scalar_H_scope_values_remain_shape_inventory(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);w=self.fixture(root)
   self.rewrite(root,w,'bundle',lambda b:b['harmonies'][0].update(attributes={'interpretation_scope':'contextual_structural_reduction'}))
   r=self.analyse(root,w)
   self.assertEqual(r['H_scope_values'],{'attributes.interpretation_scope="contextual_structural_reduction"':1})
 def test_typed_class_identity(self):
  inv=m.Inventory()
  for value in (False,'false',1,'1',1.0):inv.add('notes',{'fields':{'analysis.example':f(value)}},'w',{})
  rows=inv.class_result()['notes/analysis.example']
  self.assertEqual({(r['json_type'],r['label']) for r in rows},{('bool','false'),('str','false'),('int','1'),('str','1'),('float','1.0')})
 def test_identity_and_report_structure_fail_closed(self):
  changes=[('canonical_piece',lambda c:c.update(schema_version='bad')),
           ('bundle',lambda b:b.update(schema='bad')),
           ('bundle',lambda b:b.update(dataset_id='other')),
           ('reviewed_report',lambda r:r.update(schema='bad')),
           ('reviewed_report',lambda r:r.update(piece_id='other')),
           ('reviewed_report',lambda r:r.update(dataset_id='other')),
           ('reviewed_report',lambda r:r.update(work_family_id='other')),
           ('reviewed_report',lambda r:r.update(canonical_piece_sha256='0'*64)),
           ('reviewed_report',lambda r:r.update(views={})),
           ('reviewed_report',lambda r:r['views']['v']['layers'].pop('form'))]
  for role,change in changes:
   with self.subTest(role=role),tempfile.TemporaryDirectory() as td:
    root=Path(td);w=self.fixture(root);self.rewrite(root,w,role,change)
    with self.assertRaises(ValueError):self.analyse(root,w)
 def test_report_row_and_field_mismatch_are_visible(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);w=self.fixture(root)
   def change(r):
    layer=r['views']['v']['layers']['harmony'];layer['rows']=9;layer['fields']['analysis.function']['counts']['known']=9
   self.rewrite(root,w,'reviewed_report',change)
   c=self.analyse(root,w)['counts']
   self.assertEqual(c['report_row_mismatches'],1);self.assertEqual(c['report_count_mismatches'],1)
 def test_context_union_cannot_cover_note_from_outside(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);w=self.fixture(root)
   self.rewrite(root,w,'bundle',lambda b:b['note_contexts'][2].update(start_qn=t(2),end_qn=t(3)))
   c=self.analyse(root,w)['counts']
   self.assertEqual(c['notes_fully_duration_covered_by_context_union'],1)
   self.assertEqual(c['h_valid_note_context'],2)
 def test_cross_view_joint_presence_is_not_usable(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);w=self.fixture(root,two_views=True)
   def change(b):
    b['note_contexts'][0]['harmony_id']='H3';b['harmonies'][2]['fields']['analysis.function']=f('tonic')
   self.rewrite(root,w,'bundle',change)
   c=self.analyse(root,w)['counts']
   self.assertEqual(c['joint_h_membership_H_function'],2)
   self.assertEqual(c['usable_joint_h_membership_H_function'],1)
   self.assertEqual(c['h_valid_bound_context'],2)
 def test_cross_view_P_H_reference_not_usable(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);w=self.fixture(root,two_views=True)
   self.rewrite(root,w,'bundle',lambda b:b['objects'][0].update(attributes={'harmony_ids':['H3']}))
   c=self.analyse(root,w)['counts']
   self.assertEqual(c['P_process_known_with_explicit_H'],1)
   self.assertEqual(c['usable_P_process_known_with_explicit_H'],0)
 def test_cooperative_deadline(self):
  from unittest.mock import patch
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);w=self.fixture(root)
   with self.assertRaises(TimeoutError):m.checked_path(root,w['bindings']['bundle'],deadline=time.monotonic()-1)
   with patch.object(m.time,'monotonic',side_effect=[0,0,11]):
    with self.assertRaises(TimeoutError):m.analyse_work(w,root,m.Inventory(),[],lambda *args:None,deadline=10)
 def test_exact67_admission_unique_identity_and_flat_bindings(self):
  from copy import deepcopy
  works=[]
  for i in range(67):
   work={'id':str(i),'family':'family'+str(i),'piece_id':'piece'+str(i),'bindings':{}}
   for role in ('canonical_piece','bundle','reviewed_report'):
    work['bindings'][role]={'path':str(i)+'/'+role+'.json','sha256':'0'*64,'size_bytes':0,'role':role,'work_id':str(i)}
   works.append(work)
  admission={'schema':'HiergraphSupervisionAdmission@1','works':works,'entries':[e for w in works for e in w['bindings'].values()]}
  self.assertIs(m.validate_admission(admission),admission)
  changes=[lambda a:a['works'].pop(),lambda a:a['works'][1].update(id='0'),lambda a:a['works'][1].update(piece_id='piece0'),lambda a:a['works'][1].update(family='family0'),lambda a:a['works'][0]['bindings']['bundle'].update(path='../outside.json'),lambda a:a['works'][0]['bindings']['bundle'].update(role='wrong'),lambda a:a['entries'].pop()]
  for change in changes:
   bad=deepcopy(admission);change(bad)
   with self.assertRaises(ValueError):m.validate_admission(bad)
 def test_contradictory_annotation_mask_is_rejected(self):
  with self.assertRaises(ValueError):m.decision(dict(f(True),available=False))

if __name__=='__main__':unittest.main()
