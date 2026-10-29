"""Synthetic exact-scope/partial-label contracts; no corpus evidence."""
import unittest

from music_critic.experiments.harmonic_event_reasoner.data import extract_memberships, candidate_labels


def time(n, d=1):
    return {'num': n, 'den': d}


class MembershipDataTests(unittest.TestCase):
    def bundle(self):
        return dict(current_split='train', harmonies=[dict(id='h', start_qn=time(0),
            end_qn=time(2), harmony_view='v')], note_contexts=[dict(id='c', note_id='n',
            harmony_id='h', harmony_view='v', start_qn=time(0), end_qn=time(1),
            fields={'analysis.chord_membership':dict(available=True, value=False,
                annotation_state='known', origin='fixture', confidence=None)})])

    def test_known_false_not_missing(self):
        rows, _ = extract_memberships(self.bundle())
        self.assertEqual(candidate_labels(rows[0], [('n', time(0), time(1))]), ([0], [True]))

    def test_unknown_and_partial_interval_do_not_become_negative(self):
        rows, _ = extract_memberships(self.bundle())
        self.assertEqual(candidate_labels(rows[0], [('n', time(0), time(2)), ('x', time(0), time(1))]), ([0, 0], [False, False]))

    def test_conflicting_overlapping_labels_mask(self):
        b=self.bundle();c=dict(b['note_contexts'][0]);c['id']='c2'
        c['fields']={'analysis.chord_membership':dict(available=True,value=True)}
        b['note_contexts'].append(c)
        rows,_=extract_memberships(b)
        self.assertEqual(candidate_labels(rows[0],[('n',time(0),time(1))])[1],[False])

    def test_other_view_not_joined(self):
        b=self.bundle();b['note_contexts'][0]['harmony_view']='other'
        rows,counts=extract_memberships(b)
        self.assertEqual(rows[0]['labels'],[])
        self.assertEqual(counts['cross_view'],1)

    def test_union_of_contiguous_same_labels(self):
        rows,_=extract_memberships(self.bundle())
        first=rows[0]['labels'][0];second={**first,'start':[1,1],'end':[2,1]}
        rows[0]['labels'].append(second)
        self.assertEqual(candidate_labels(rows[0],[('n',time(0),time(2))]),([0],[True]))

    def test_test_bundle_rejected(self):
        b=self.bundle();b['current_split']='test'
        with self.assertRaises(ValueError):extract_memberships(b)

    def test_not_applicable_cannot_be_negative(self):
        b=self.bundle()
        b['note_contexts'][0]['fields']['analysis.chord_membership']['annotation_state']='not_applicable'
        rows,_=extract_memberships(b)
        self.assertEqual(candidate_labels(rows[0],[('n',time(0),time(1))])[1],[False])

if __name__ == '__main__':unittest.main()
