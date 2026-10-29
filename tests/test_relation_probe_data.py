"""Synthetic contracts, not evidence of musical correctness."""
import unittest
from music_critic.relation_probe.data import extract_targets, family_partition, checked_read
from pathlib import Path
import tempfile
import hashlib


def ctx(identity, note, status, relation_ids=(), start=0):
    return dict(id=identity,note_id=note,view_id='v',start_qn={'num':start,'den':1},
                end_qn={'num':start+1,'den':1},fields={'analysis.resolution':dict(
                    available=status!='unknown',value={'status':status,'relation_ids':list(relation_ids)},
                    origin='model_individual_context',confidence=None)})


class DataTests(unittest.TestCase):
    def test_exact_context_geometry_is_validated_against_raw_note(self):
        from music_critic.data import load_piece
        from music_critic.relation_probe.data import validate_target_geometry
        p=load_piece(Path(__file__).parent/'fixtures/data/canonical_piece_v2.json')
        c=ctx('bad',p.notes[0].note_id,'none',start=10000)
        with self.assertRaisesRegex(ValueError,'geometry'):
            validate_target_geometry(p,dict(current_split='train',piece_id=p.piece_id,
                dataset_id=p.dataset_name,note_contexts=[c],relations=[]))

    def test_fallback_views_never_merge(self):
        a=ctx('a','n1','linked',['r']);b=ctx('b','n2','none',start=1)
        a['harmony_view']='A';b['harmony_view']='B';a.pop('view_id');b.pop('view_id')
        relation=dict(id='r',type='resolves_to',available=True,value=True,harmony_view='A',source_ids=['a'],target_ids=['b'])
        rows,counts=extract_targets(dict(current_split='train',note_contexts=[a,b],relations=[relation]))
        self.assertFalse(any(x['label'] for x in rows));self.assertEqual(counts['cross_view'],1)

    def test_point_none_with_direct_positive_edge_is_unknown(self):
        c=ctx('a','n1','none');c['end_qn']=c['start_qn']
        r=dict(id='r',type='resolves_to',available=True,value=True,view_id='v',source_ids=['n1'],target_ids=['n2'],
               start_qn={'num':0,'den':1},end_qn={'num':1,'den':1})
        rows,counts=extract_targets(dict(current_split='train',note_contexts=[c],relations=[r]))
        self.assertEqual(rows,[]);self.assertEqual(counts['contradictory_none'],1)

    def test_missing_or_backward_link_does_not_supervise_existence(self):
        a=ctx('a','n1','linked',['missing'],start=3)
        self.assertEqual(extract_targets(dict(current_split='train',note_contexts=[a],relations=[]))[0],[])
        a['fields']['analysis.resolution']['value']['relation_ids']=['r']
        b=ctx('b','n2','none',start=1)
        r=dict(id='r',type='resolves_to',available=True,value=True,view_id='v',source_ids=['a'],target_ids=['b'])
        rows,counts=extract_targets(dict(current_split='train',note_contexts=[a,b],relations=[r]))
        self.assertFalse(any(x['label'] for x in rows));self.assertEqual(counts['invalid_direction'],1)

    def test_raw_export_preserves_time_and_serializes_without_theory(self):
        from music_critic.data import load_piece,dumps_piece,loads_piece
        from music_critic.relation_probe.data import sanitize
        p=load_piece(Path(__file__).parent/'fixtures/data/canonical_piece_v2.json')
        clean=loads_piece(dumps_piece(sanitize(p)))
        self.assertEqual([(n.onset_qn,n.duration_qn) for n in clean.notes],
                         [(n.onset_qn,n.duration_qn) for n in p.notes])
        self.assertEqual(clean.targets,())
        self.assertEqual(clean.annotations,())
        self.assertTrue(all(n.voice is None and n.staff is None for n in clean.notes))

    def test_unknown_is_not_negative_and_none_is_scoped(self):
        b=dict(current_split='train',note_contexts=[ctx('a','n1','linked',['r']),
             ctx('b','n2','none',start=1),ctx('c','n3','unknown',start=2)],relations=[dict(
             id='r',type='resolves_to',available=True,value=True,view_id='v',
             source_ids=['a'],target_ids=['b'])])
        rows,counts=extract_targets(b)
        self.assertEqual([(x['context_id'],x['label']) for x in rows],[('a',1),('b',0)])
        self.assertEqual(rows[0]['targets'],[{'note_id':'n2','time':[1,1]}])
        self.assertEqual(counts['unknown'],1)

    def test_hyperedge_is_not_cartesian_pairs(self):
        b=dict(current_split='train',note_contexts=[ctx('a','n1','linked',['r']),ctx('b','n2','none')],
               relations=[dict(id='r',type='resolves_to',available=True,value=True,view_id='v',
                   source_ids=['a'],target_ids=['b','a'])])
        rows,counts=extract_targets(b)
        self.assertEqual(rows[0]['targets'],[])
        self.assertEqual(counts['nonbinary_relation'],1)

    def test_wrong_split_rejected(self):
        with self.assertRaisesRegex(ValueError,'TRAIN'):
            extract_targets(dict(current_split='test',note_contexts=[],relations=[]))

    def test_hash_and_symlink_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp).resolve()/'x';p.write_bytes(b'ok')
            binding=dict(path=str(p),size_bytes=2,sha256=hashlib.sha256(b'ok').hexdigest())
            self.assertEqual(checked_read(binding),b'ok')
            p.write_bytes(b'no')
            with self.assertRaises(ValueError):checked_read(binding)
            q=p.parent/'link';q.symlink_to(p);binding['path']=str(q)
            with self.assertRaises(ValueError):checked_read(binding)

    def test_family_split_order_independent(self):
        names=[str(i) for i in range(20)]
        self.assertEqual(family_partition(names),family_partition(list(reversed(names))))
        split=family_partition(names)
        self.assertEqual(sum(x=='dev' for x in split.values()),4)


if __name__=='__main__':unittest.main()
