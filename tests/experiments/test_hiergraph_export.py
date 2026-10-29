"""Synthetic safety and export mechanics; never reads the retained corpus."""
from hashlib import sha256
from pathlib import Path
import tempfile
import json
import importlib.util
import dataclasses
from copy import deepcopy
import unittest

from music_critic.experiments.hiergraph_export import (
    safe_relative, checked_read, build_family_split, write_gzip_json, read_gzip_json,
    export_targets, canonical, fingerprint, file_binding,
)


class SafetyTests(unittest.TestCase):
    def test_relative_admission_rejects_traversal_absolute_and_symlinks(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / 'a').write_text('safe')
            (root / 'link').symlink_to(root / 'a')
            for path in ('../x', '/tmp/x', 'a/../a', 'link', 'C:\\x'):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    safe_relative(root, path)
            self.assertEqual(safe_relative(root, 'a'), root.resolve() / 'a')

    def test_checked_read_detects_mutation_before_json(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); (root/'a').write_bytes(b'{}')
            binding = dict(path='a', sha256=sha256(b'{}').hexdigest(), size_bytes=2)
            self.assertEqual(checked_read(root, binding), b'{}')
            (root/'a').write_bytes(b'[]')
            with self.assertRaises(ValueError): checked_read(root, binding)

    def test_existing_family_assignments_are_preserved(self):
        works = [dict(id=f'w{i}', piece_id=f'p{i}', family=f'f{i}') for i in range(3)]
        result = build_family_split(works, [dict(works=[dict(family='f0', piece_id='p0', split='train'),
            dict(family='f1', piece_id='p1', split='selection')]),
            dict(works=[dict(family='f0', piece_id='p0', split='fit'), dict(family='f2', piece_id='p2', split='dev')])])
        self.assertEqual([r['split'] for r in result['assignments']], ['fit','selection','dev'])
        with self.assertRaises(ValueError):
            build_family_split(works, [dict(works=[dict(family='f0',piece_id='p0',split='dev')]),
                dict(works=[dict(family='f0',piece_id='p0',split='fit')])])

    def test_gzip_bytes_are_deterministic_and_integrity_checked(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); value={'known_false':False,'unknown':None}
            a=write_gzip_json(root/'a.gz',value,10000); b=write_gzip_json(root/'b.gz',value,10000)
            self.assertEqual(a['sha256'],b['sha256'])
            self.assertEqual(read_gzip_json(root,a,10000),value)
            (root/'a.gz').write_bytes(b'bad')
            with self.assertRaises(ValueError):read_gzip_json(root,a,10000)

def synthetic_inputs(root, role='fit', canonical_split='train', interval_fields=False):
    # Reuse the real leaf-consumer fixture after a regular package import.
    import music_critic.tasks.hiergraph_projection
    path=Path(__file__).resolve().parents[1]/'tasks'/'test_hiergraph_projection.py'
    spec=importlib.util.spec_from_file_location('hiergraph_synthetic_fixture',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    piece,bundle=module.fixture()
    piece=dataclasses.replace(piece,split=canonical_split)
    if interval_fields:
        note=piece.notes[0]
        piece=dataclasses.replace(piece,notes=(note,dataclasses.replace(note,note_id='note:E',pitch=64),
            dataclasses.replace(note,note_id='note:G',pitch=67)))
        for harmony,label,quality,members in ((bundle['harmonies'][0],'C_major','major',['0','4','7']),
                (bundle['harmonies'][1],'C_minor','minor',['0','3','7'])):
            harmony['fields']['analysis.sonority']=module.known(dict(label=label,root_pitch_class=0,
                quality=quality,tones=['C','E' if quality=='major' else 'Eb','G']))
            harmony['note_template']=dict(schema='AuthoredNoteTemplate@1',id='template:'+harmony['id'],
                basis='midi_pitch_class',enharmonic_policy='author_declared_pc_mapping',members=members,
                members_complete=True,evidence=['synthetic:complete-authored-template'])
        observed=dict(id='Hobs',start_qn=module.q(0),end_qn=module.q(2),view_id='observed',
            source_event_ids=['note:n','note:E','note:G'],
            attributes=dict(key_active_note_ids=['note:n','note:E','note:G'],key_pitches=[60,64,67],
                possible_pedal_held_note_ids=[]),
            fields={'analysis.sonority':module.known(dict(label='literal_key_collection',reading='complete_collection',
                root_pitch_class=0,tones=['C4','E4','G4']))})
        bundle['harmonies'].append(observed)
        for identity in ('note:n','note:E','note:G'):
            bundle['note_contexts'].append(dict(id='obs:'+identity,note_id=identity,harmony_id='Hobs',
                start_qn=module.q(0),end_qn=module.q(2),view_id='observed',fields={}))
    from music_critic.data import dumps_piece
    (root/'piece.json').write_text(dumps_piece(piece))
    cb=dict(file_binding(root/'piece.json'),role='canonical_piece',work_id='work0')
    bundle['canonical_piece']=cb
    bundle['harmonies'][1]['fields']['analysis.function']=module.known(None,'unresolved')
    (root/'bundle.json').write_bytes(canonical(bundle))
    bb=dict(file_binding(root/'bundle.json'),role='bundle',work_id='work0')
    work=dict(id='work0',piece_id=piece.piece_id,family='family0',bindings={'canonical_piece':cb,'bundle':bb})
    admission=dict(works=[work],entries=[cb,bb])
    split=build_family_split([work],[])
    split['assignments'][0]['split']=role;split['fingerprint']=fingerprint(split)
    (root/'admission.json').write_bytes(canonical(admission));(root/'split.json').write_bytes(canonical(split))
    from music_critic.tasks.hiergraph_projection import ONTOLOGY, VERSION, projection_contract
    config=dict(schema='HiergraphExportConfig@1',max_works=1,degree_frames={},exported_fields=sorted(ONTOLOGY),
        projection_version=VERSION,contract_fingerprint=projection_contract()['fingerprint'],
        allow_coarsened_fields=sorted(ONTOLOGY),admission_sha256=file_binding(root/'admission.json')['sha256'],
        split_sha256=file_binding(root/'split.json')['sha256'],caps=dict(worker_wall_seconds=30,total_wall_seconds=90))
    (root/'config.json').write_bytes(canonical(config))
    args=dict(admission_path=root/'admission.json',admission_sha256=config['admission_sha256'],
        split_path=root/'split.json',split_sha256=config['split_sha256'],config_path=root/'config.json',
        config_sha256=file_binding(root/'config.json')['sha256'],source_root=root,output=root/'result',test_mode=True)
    return args,work,config


class ExportTests(unittest.TestCase):
    def test_synthetic_end_to_end_false_unknown_support_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);args,work,_=synthetic_inputs(root)
            result=export_targets(**args)
            self.assertEqual(len(result['works']),1)
            from music_critic.tasks.hiergraph_projection import VERSION, projection_contract
            self.assertEqual(result['projection_version'],VERSION)
            self.assertEqual(result['contract_fingerprint'],projection_contract()['fingerprint'])
            receipt=json.loads((root/'result'/'read-receipt.json').read_bytes())
            self.assertEqual(receipt['distinct_files'],2)
            self.assertEqual(receipt['read_operations'],3)
            numeric=read_gzip_json(root/'result',result['works'][0]['numeric'],1024**2)
            values=numeric['targets']['note_contexts']['values']['h.chord_membership']
            masks=numeric['targets']['note_contexts']['masks']['h.chord_membership']
            self.assertEqual(masks,[True,True]);self.assertNotEqual(values[0],values[1])
            self.assertEqual(numeric['targets']['harmonies']['masks']['H.function'],[True,False])
            self.assertNotIn('harmonies',numeric['source'])
            heads=json.loads((root/'result'/'heads.json').read_bytes())
            self.assertEqual(heads['projection_version'],VERSION)
            self.assertEqual(heads['contract_fingerprint'],result['contract_fingerprint'])
            self.assertFalse(heads['fields']['h.chord_membership']['active'])
            self.assertIn('fewer_than_three_fit_families',heads['fields']['h.chord_membership']['deferred_reasons'])
            before=(root/'result'/'manifest.json').read_bytes()
            with self.assertRaises(ValueError):export_targets(**args)
            self.assertEqual((root/'result'/'manifest.json').read_bytes(),before)

    def test_interval_and_observed_scopes_survive_real_subprocess_encoding(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);args,_,_=synthetic_inputs(root,interval_fields=True)
            result=export_targets(**args)
            numeric=read_gzip_json(root/'result',result['works'][0]['numeric'],2*1024**2)
            projection=read_gzip_json(root/'result',result['works'][0]['projection'],2*1024**2)
            vocab=json.loads((root/'result'/'vocabulary.json').read_bytes())
            targets=numeric['targets']['harmonies']
            def decoded(field):
                classes=vocab['event']['fields']['harmonies'][field]
                return [classes[value] if mask else None for value,mask in zip(targets['values'][field],targets['masks'][field])]
            self.assertEqual(decoded('H.chord.interval_pc.4'),[True,False,None])
            self.assertEqual(decoded('H.chord.interval_pc.3'),[False,True,None])
            self.assertEqual(decoded('H.observed.pitch_class.4'),[None,None,True])
            self.assertEqual(decoded('H.observed.pitch_class.1'),[None,None,False])
            self.assertEqual(decoded('H.observed.interval_pc.4'),[None,None,True])
            self.assertEqual(decoded('H.observed.interval_pc.1'),[None,None,False])
            self.assertEqual(decoded('H.observed.root_pc'),[None,None,0])
            self.assertFalse(projection['mappings']['Hobs']['H.chord.interval_pc.4']['available'])
            heads=json.loads((root/'result'/'heads.json').read_bytes())
            self.assertIn('positive_unlabeled_negative_loss_protocol_not_admitted',
                heads['fields']['H.sonority.pitch_class.0']['deferred_reasons'])
            self.assertNotIn('positive_unlabeled_negative_loss_protocol_not_admitted',
                heads['fields']['H.chord.interval_pc.4']['deferred_reasons'])
            self.assertEqual(heads['fields']['H.chord.interval_pc.4']['fit_classes'],2)
            self.assertEqual(heads['fields']['H.chord.interval_pc.4']['fit_families'],1)
            self.assertFalse(heads['fields']['H.chord.interval_pc.4']['active'])
            # Octave/compound detail is explicitly coarsened into a modulo-12 set.
            self.assertEqual(result['coverage_by_field']['H.chord.interval_pc.4:coarsened'],2)
            self.assertEqual(result['coverage_by_field']['H.chord.interval_pc.4:masked'],1)
            self.assertEqual(numeric['schema'],'HiergraphNumericTargets@3')
            self.assertEqual(vocab['schema'],'HiergraphVocabulary@3')
            self.assertEqual(projection['schema'],'HiergraphProjection@3.0.0')

    def test_payload_mutation_retains_failed_attempt(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);args,_,_=synthetic_inputs(root)
            (root/'bundle.json').write_bytes(b'[]')
            with self.assertRaises(ValueError):export_targets(**args)
            self.assertEqual(json.loads((root/'result'/'failure.json').read_text())['status'],'failed')
            self.assertFalse((root/'result'/'manifest.json').exists())

    def test_mixed_family_split_rejected_before_payload(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);args,_,_=synthetic_inputs(root)
            split=json.loads((root/'split.json').read_bytes())
            split['assignments'][0]['split']='test';split['fingerprint']=fingerprint(split)
            (root/'split.json').write_bytes(canonical(split));args['split_sha256']=file_binding(root/'split.json')['sha256']
            config=json.loads((root/'config.json').read_bytes());config['split_sha256']=args['split_sha256']
            (root/'config.json').write_bytes(canonical(config));args['config_sha256']=file_binding(root/'config.json')['sha256']
            (root/'piece.json').unlink();(root/'bundle.json').unlink()
            with self.assertRaises(ValueError):export_targets(**args)
            self.assertFalse((root/'result').exists())

    def test_null_canonical_requires_named_authority(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);args,_,_=synthetic_inputs(root,canonical_split=None)
            with self.assertRaises(ValueError):export_targets(**args)
            failure=list((root/'result'/'project'/'work0').glob('failure.json'))
            self.assertEqual(len(failure),1)
            self.assertIn('null canonical split',failure[0].read_text())

    def test_explicit_null_authority_is_preserved_without_mutating_canonical(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);args,work,config=synthetic_inputs(root,canonical_split=None)
            before=(root/'piece.json').read_bytes()
            config['null_split_authority']=dict(work_id='work0',piece_id=work['piece_id'],
                canonical_sha256=work['bindings']['canonical_piece']['sha256'])
            (root/'config.json').write_bytes(canonical(config));args['config_sha256']=file_binding(root/'config.json')['sha256']
            result=export_targets(**args)
            self.assertEqual(result['status'],'completed')
            self.assertEqual((root/'piece.json').read_bytes(),before)

    def test_worker_memory_cap_leaves_failure_receipt(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);args,_,config=synthetic_inputs(root)
            config['caps']['worker_rss_bytes']=1
            (root/'config.json').write_bytes(canonical(config));args['config_sha256']=file_binding(root/'config.json')['sha256']
            with self.assertRaises(ValueError):export_targets(**args)
            self.assertTrue((root/'result'/'failure.json').exists())
            self.assertFalse((root/'result'/'manifest.json').exists())

    def test_real_heldout_canonical_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);args,_,_=synthetic_inputs(root,canonical_split='test')
            with self.assertRaises(ValueError):export_targets(**args)
            self.assertFalse((root/'result'/'manifest.json').exists())

    def test_prior_metadata_assignment_mutation_rejected_before_payload(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);args,work,config=synthetic_inputs(root)
            prior=dict(works=[dict(family='family0',piece_id=work['piece_id'],split='dev')])
            (root/'prior.json').write_bytes(canonical(prior))
            split=json.loads((root/'split.json').read_bytes());split['parent_manifests']=[file_binding(root/'prior.json')]
            split['fingerprint']=fingerprint(split);(root/'split.json').write_bytes(canonical(split))
            args['split_sha256']=file_binding(root/'split.json')['sha256'];config['split_sha256']=args['split_sha256']
            (root/'config.json').write_bytes(canonical(config));args['config_sha256']=file_binding(root/'config.json')['sha256']
            (root/'piece.json').unlink();(root/'bundle.json').unlink()
            with self.assertRaises(ValueError):export_targets(**args)
            self.assertFalse((root/'result').exists())

    def test_identity_aliases_rejected_before_payload_access(self):
        for alias_kind in ('piece_id','canonical_sha'):
            with self.subTest(alias_kind=alias_kind), tempfile.TemporaryDirectory() as td:
                root=Path(td);args,work,config=synthetic_inputs(root)
                alias=deepcopy(work);alias['id']='work1';alias['family']='family1'
                if alias_kind=='canonical_sha':alias['piece_id']='piece:other'
                for role,binding in alias['bindings'].items():
                    binding.update(work_id='work1',path='alias-'+binding['path'])
                    if role=='canonical_piece' and alias_kind=='piece_id':binding['sha256']='b'*64
                admission=dict(works=[work,alias],entries=list(work['bindings'].values())+list(alias['bindings'].values()))
                split=build_family_split([work,alias]);split['assignments'][1]['split']='dev';split['fingerprint']=fingerprint(split)
                (root/'admission.json').write_bytes(canonical(admission));(root/'split.json').write_bytes(canonical(split))
                args['admission_sha256']=file_binding(root/'admission.json')['sha256'];args['split_sha256']=file_binding(root/'split.json')['sha256']
                config.update(max_works=2,admission_sha256=args['admission_sha256'],split_sha256=args['split_sha256'])
                (root/'config.json').write_bytes(canonical(config));args['config_sha256']=file_binding(root/'config.json')['sha256']
                (root/'piece.json').unlink();(root/'bundle.json').unlink()
                with self.assertRaisesRegex(ValueError,'duplicate canonical'):export_targets(**args)
                self.assertFalse((root/'result').exists())

    def test_disallowed_coarsening_with_empty_fitted_vocab_never_activates_head(self):
        from music_critic.experiments.hiergraph_export import _summary, _heads
        from music_critic.tasks.hiergraph_projection import project_hiergraph_targets, build_projection_vocabulary
        import music_critic.tasks.hiergraph_projection
        path=Path(__file__).resolve().parents[1]/'tasks'/'test_hiergraph_projection.py'
        spec=importlib.util.spec_from_file_location('hiergraph_coarse_fixture',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        projections=[];summaries={};assignments={}
        for index in range(3):
            piece,bundle=module.fixture();piece=dataclasses.replace(piece,piece_id='piece:'+str(index));bundle['piece_id']=piece.piece_id
            family='family'+str(index);name='work'+str(index)
            projection=project_hiergraph_targets(piece,bundle,family_id=family)
            projections.append(projection);summaries[name]=_summary(projection)
            assignments[name]=dict(family_id=family,split='fit')
        vocab=build_projection_vocabulary(projections,fit_families=['family0','family1','family2'],
            split_fingerprint='a'*64,active_fields=['H.key.tonic_pc'],allow_coarsened_fields=[])
        self.assertEqual(vocab['event']['fields'].get('harmonies',{}).get('H.key.tonic_pc',[]),[])
        heads=_heads(vocab,summaries,assignments,['H.key.tonic_pc'])
        head=heads['fields']['H.key.tonic_pc']
        self.assertFalse(head['active']);self.assertEqual(head['fit_families'],0);self.assertEqual(head['fit_classes'],0)

    def test_stale_projection_version_and_contract_fail_before_payload_reads(self):
        for field in ('projection_version','contract_fingerprint'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as td:
                root=Path(td);args,_,config=synthetic_inputs(root)
                config[field]='stale'
                (root/'config.json').write_bytes(canonical(config));args['config_sha256']=file_binding(root/'config.json')['sha256']
                (root/'piece.json').unlink();(root/'bundle.json').unlink()
                with self.assertRaisesRegex(ValueError,'stale projection'):export_targets(**args)
                self.assertFalse((root/'result').exists())

    def test_empty_work_set_and_metadata_mutation_fail_before_output(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);args,_,_=synthetic_inputs(root)
            (root/'config.json').write_bytes(b'{}')
            with self.assertRaises(ValueError):export_targets(**args)
            self.assertFalse((root/'result').exists())
            args,_,_=synthetic_inputs(root)
            admission=json.loads((root/'admission.json').read_bytes());admission['works']=[]
            (root/'admission.json').write_bytes(canonical(admission));args['admission_sha256']=file_binding(root/'admission.json')['sha256']
            config=json.loads((root/'config.json').read_bytes());config['admission_sha256']=args['admission_sha256']
            (root/'config.json').write_bytes(canonical(config));args['config_sha256']=file_binding(root/'config.json')['sha256']
            with self.assertRaises(ValueError):export_targets(**args)
            self.assertFalse((root/'result').exists())


if __name__=='__main__':unittest.main()
