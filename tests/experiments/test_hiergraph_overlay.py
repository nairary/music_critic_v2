from copy import deepcopy
import unittest
from music_critic.experiments.hierarchical_graph_analyzer.overlay import apply_accepted_overlay

class OverlayTests(unittest.TestCase):
    def fixture(self):
        binding=lambda path,digest:dict(path=path,sha256=digest*64,size_bytes=10)
        work=dict(id='work0',piece_id='piece0',family='family0',bindings=dict(canonical_piece=binding('piece.json','a'),bundle=binding('old.json','b')))
        work['bindings']['canonical_piece'].update(role='canonical_piece',work_id='work0');work['bindings']['bundle'].update(role='bundle',work_id='work0')
        admission=dict(works=[work],entries=list(work['bindings'].values()))
        split=dict(assignments=[dict(work_id='work0',piece_id='piece0',family_id='family0',split='fit')])
        row=dict(work_id='work0',piece_id='piece0',family='family0',assignment=deepcopy(split['assignments'][0]),independently_reviewed=True,review_id='REVIEW1',review_record_fingerprint='d'*64,basis3=dict(bundle_fingerprint='e'*64,profile_fingerprint='f'*64,assessment_fingerprint='1'*64),assets=dict(canonical_piece=binding('piece-copy.json','a'),baseline_bundle=binding('baseline-copy.json','b'),bundle=binding('new.json','c')),delta=dict(structure={'harmonies':dict(removed=0,shared_geometry_changes=0),'note_contexts':dict(removed=0,shared_geometry_changes=0)}))
        reviews={'REVIEW1':dict(status='completed',record_fingerprint='d'*64)}
        return admission,split,[row],reviews
    def test_exact_identity_replacement_preserves_inputs_and_bindings(self):
        args=self.fixture();before=deepcopy(args);out=apply_accepted_overlay(*args)
        self.assertEqual(out['works'][0]['bindings']['bundle']['path'],'new.json')
        self.assertEqual(out['entries'][1],out['works'][0]['bindings']['bundle'])
        self.assertEqual(out['works'][0]['bindings']['canonical_piece'],args[0]['works'][0]['bindings']['canonical_piece'])
        self.assertEqual(args,before)
    def test_rejects_unreviewed_wrong_identity_geometry_or_holdout(self):
        for mutation in ['baseline','canonical','split','review','basis','geometry','duplicate']:
            args=list(self.fixture());r=args[2][0]
            if mutation=='baseline':r['assets']['baseline_bundle']['sha256']='0'*64
            if mutation=='canonical':r['assets']['canonical_piece']['sha256']='0'*64
            if mutation=='split':args[1]['assignments'][0]['split']='dev'
            if mutation=='review':args[3]['REVIEW1']['status']='failed'
            if mutation=='basis':r['basis3'].pop('assessment_fingerprint')
            if mutation=='geometry':r['delta']['structure']['harmonies']['shared_geometry_changes']=1
            if mutation=='duplicate':args[2].append(deepcopy(r))
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):apply_accepted_overlay(*args)

    def test_preparation_rebases_explicit_readonly_fallback_and_freezes_metadata(self):
        from pathlib import Path
        import tempfile,json
        from hashlib import sha256
        from music_critic.experiments.hiergraph_export import canonical,file_binding
        from music_critic.experiments.hierarchical_graph_analyzer.overlay import prepare_overlay_export
        from music_critic.experiments.registry import seal_record
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);repo=base/'isolated';main=base/'main';repo.mkdir();main.mkdir()
            template=self.fixture();works=[];assignments=[];rows=[]
            for i in range(15):
                a,sp,rs,_=deepcopy(template);w=a['works'][0];r=rs[0]
                identity=f'work{i}';w.update(id=identity,piece_id=f'p{i}',family=f'f{i}')
                assignment=dict(work_id=identity,piece_id=f'p{i}',family_id=f'f{i}',split='fit');assignments.append(assignment)
                r.update(work_id=identity,piece_id=f'p{i}',family=f'f{i}',assignment=assignment)
                for role in ('canonical_piece','bundle'):
                    name=f'{identity}-{role}.json';(main/name).write_bytes(b'{}')
                    w['bindings'][role]=dict(file_binding(main/name),role=role,work_id=identity)
                r['assets'].update(canonical_piece=dict(w['bindings']['canonical_piece']),baseline_bundle=dict(w['bindings']['bundle']))
                name=f'{identity}-new.json';(repo/name).write_bytes(b'{}');r['assets']['bundle']=file_binding(repo/name)
                works.append(w);rows.append(r)
            record=seal_record(dict(schema_version='1.0.0',experiment_id='REVIEW1',kind='preparation',phase='test',title='synthetic',status='completed',question='test',hypothesis='test',decision_use='test',repository=dict(git_commit=None,branch=None,dirty=False),seeds=[],budget=dict(planned={},actual={}),data_access=dict(access_evidence_status='verified_primary_evidence',declared_policy=dict(source='synthetic',validation_allowed=False,test_inputs_allowed=False,test_targets_allowed=False,test_metrics_allowed=False,test_selection_allowed=False),evidence=['synthetic'],validation_accessed=False,test_inputs_accessed=False,test_targets_accessed=False,test_metrics_computed=False,test_used_for_selection=False),results=dict(summary='synthetic'),validity=dict(valid=True,reason=None),claims=dict(supported=['synthetic'],not_supported=[]),artifacts=[],import_provenance=dict(mode='synthetic')))
            (repo/'docs/experiments/records').mkdir(parents=True);(repo/'docs/experiments/records/REVIEW1.json').write_bytes(canonical(record))
            for row in rows:row['review_record_fingerprint']=record['record_fingerprint']
            admission=dict(works=works,entries=[b for w in works for b in w['bindings'].values()])
            split=dict(assignments=assignments,parent_manifests=[],fingerprint='s')
            config=dict(caps={})
            artifacts=[]
            for name,value in [('admission.json',admission),('split.json',split),('config.json',config)]:
                (repo/name).write_bytes(canonical(value));b=file_binding(repo/name);artifacts.append(dict(role=name,locator=name,sha256=b['sha256'],size_bytes=b['size_bytes']))
            (repo/'accepted.jsonl').write_bytes(b'\n'.join(canonical(r) for r in rows));b=file_binding(repo/'accepted.jsonl')
            parent=dict(status='completed',experiment_id='PARENT',artifacts=artifacts)
            release=dict(status='completed',experiment_id='RELEASE',results=dict(accepted_works=15),artifacts=[dict(role='manifest.jsonl',locator='accepted.jsonl',sha256=b['sha256'],size_bytes=b['size_bytes'])])
            result=prepare_overlay_export(repo,parent,release,repo/'metadata',source_fallback=main)
            self.assertEqual(result['source_root'],base.resolve())
            merged=json.loads((repo/'metadata/admission.json').read_bytes())
            self.assertEqual(merged['works'][0]['bindings']['canonical_piece']['path'],'main/work0-canonical_piece.json')
            self.assertEqual(merged['works'][0]['bindings']['bundle']['path'],'isolated/work0-new.json')
            self.assertEqual((repo/'metadata/split.json').read_bytes(),(repo/'split.json').read_bytes())
            self.assertTrue((repo/'metadata/provenance.json').is_file())
