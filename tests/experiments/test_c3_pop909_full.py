"""Synthetic contracts for explicitly authorized full POP909-CL evaluation."""
from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace

import pytest
import torch

from music_critic.adapters.pop909_cl import Pop909ClCorpusRecord, convert_pop909_cl_file
from music_critic.adapters.pop909_cl_synthetic import build_pop909_cl_smoke_midi
from music_critic.experiments.pop909_cl_frozen import c3_full
from music_critic.experiments.pop909_cl_frozen.transfer import BoundFile


def test_full_access_requires_explicit_test_authorization_before_io(tmp_path):
    missing = BoundFile(tmp_path / 'missing.json', 'a' * 64, 1)
    request = c3_full.FullRequest((missing,), missing, missing, tmp_path, False)
    with pytest.raises(ValueError, match='explicit_test_authorization_required'):
        c3_full.preflight(request)


def test_full_protocol_states_real_scoped_execution_and_keeps_metric_rules():
    protocol = c3_full.full_protocol()
    assert protocol['execution_policy'] == 'explicit_registered_full_pinned_POP909_CL_CPU_test_authorized'
    assert 'execution_policy' not in protocol['metric_policy']
    assert protocol['metric_policy']['macro_f1'] == 'unweighted_mean_over_observed_true_classes_only'
    assert protocol['metric_policy']['model_selection'] is False


def test_c3_predictions_select_beat_candidates_and_keep_full_quality_vocabulary():
    keys = ((0, 'slice', 's0'), (0, 'beat', 'b0'))
    quality = torch.zeros(2, 17)
    quality[0, 0] = 20
    quality[1, 16] = 20  # Outside subset9 must remain the prediction.
    output = SimpleNamespace(candidate_keys={'root': keys, 'quality': keys},
                             logits={'root': torch.zeros(2, 37), 'quality': quality})
    predictions, _ = c3_full.beat_predictions(output, ('b0',))
    assert predictions['quality'] == {'b0': 16}
    with pytest.raises(ValueError, match='beat_candidate_inventory'):
        c3_full.beat_predictions(output, ('b0', 'b1'))


def test_score_reader_keeps_test_identity_and_excludes_chord_channel(tmp_path):
    path = tmp_path / '001.mid'
    midi = build_pop909_cl_smoke_midi()
    midi.save(path)
    record = Pop909ClCorpusRecord('001', path, 'POP909_processed/001.mid', '001.mid',
                                sha256(path.read_bytes()).hexdigest(), 'pending:001',
                                'pop909-lineage:001')
    converted = convert_pop909_cl_file(record)
    record = replace(record, source_group_id=converted.piece.source_group_id)
    row = {'split': 'test', 'score_projection_sha256': converted.score_projection_sha256}
    with pytest.raises(ValueError, match='explicit_test_authorization_required'):
        c3_full.read_score(record, row, test_authorized=False)
    piece = c3_full.read_score(record, row, test_authorized=True)
    assert piece.split == 'test'
    assert not piece.targets and not piece.annotations
    assert {note.channel for note in piece.notes} == {0}
    assert len(piece.notes) == len(converted.piece.notes)


def test_invalid_record_policy_denies_test_before_payload_reads(monkeypatch, tmp_path):
    record = {'status': 'planned', 'protocol': {'c3_pop_full_binding_fingerprint': 'bound'},
              'data_access': {'declared_policy': {'validation_allowed': True,
                  'test_inputs_allowed': True, 'test_targets_allowed': False,
                  'test_metrics_allowed': True, 'test_selection_allowed': False}}}
    monkeypatch.setattr(c3_full, 'load_record', lambda _: record)
    with pytest.raises(ValueError, match='registered_access_policy_mismatch'):
        c3_full.registered_record(tmp_path / 'plan.json', {'binding_fingerprint': 'bound'})


def test_split_metrics_are_computed_from_observations_without_relabeling():
    pieces = [{'piece_id': 'p1', 'group_id': 'g', 'split': 'train', 'source_status': 'available',
               'event_count': 2, 'no_chord_span_count': 0, 'trailing_unannotated_span_count': 0}]
    rows = [dict(piece_id='p1', group_id='g', event_id=str(i), beat_id='b',
                 alignment_reason=None, root={'gold': 0, 'prediction': p, 'reason': None,
                 'label_reason': None, 'source_label': 0}, quality={'gold': 0, 'prediction': p, 'reason': None,
                 'label_reason': None, 'source_label': 'M'}) for i, p in enumerate((0, 1))]
    report = c3_full.summarize_populations(rows, pieces)
    assert report['all']['tasks']['root']['accuracy']['value'] == .5
    assert report['all']['tasks']['root']['observed_true_class_macro_f1']['value'] == pytest.approx(2 / 3)
    assert report['splits']['train']['tasks']['root']['coverage']['eligible_events'] == 2
    assert report['splits']['test']['tasks']['root']['accuracy']['value'] is None


def test_evaluate_reads_test_targets_only_after_both_real_c3_views(monkeypatch, tmp_path):
    from music_critic.critic.c3_backbone import C3ThreePassRelationGateModel
    path = tmp_path / 'POP909_processed' / '001.mid'
    path.parent.mkdir()
    build_pop909_cl_smoke_midi().save(path)
    record = Pop909ClCorpusRecord('001', path, 'POP909_processed/001.mid', '001.mid',
        sha256(path.read_bytes()).hexdigest(), 'pending:001', 'pop909-lineage:001')
    converted = convert_pop909_cl_file(record)
    row = dict(piece_id=converted.piece.piece_id, song_id='001', split='test',
        source_relative_path=record.relative_path, source_sha256=record.sha256,
        score_projection_sha256=converted.score_projection_sha256,
        source_group_id=converted.piece.source_group_id,
        lineage_group_id=record.lineage_group_id, component_fingerprint='synthetic-group')

    class CountingC3(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = C3ThreePassRelationGateModel()
            self.calls = 0

        def forward(self, graph):
            self.calls += 1
            return self.model(graph)

    model = CountingC3().eval().requires_grad_(False)
    metadata = dict(records=[row], checkpoint={'sha256': 'a' * 64, 'size_bytes': 1},
                    metric_protocol=c3_full.full_protocol(), candidate_policy='raw_beat',
                    binding_fingerprint='synthetic')
    monkeypatch.setattr(c3_full, 'preflight', lambda _: metadata)
    monkeypatch.setattr(c3_full, 'registered_record', lambda *_: {'experiment_id': 'synthetic'})
    monkeypatch.setattr(c3_full, 'load_c3_relationgate_adapter',
                        lambda *_args, **_kwargs: SimpleNamespace(model=model))
    def delayed_targets(*args, **kwargs):
        assert model.calls == 2
        return convert_pop909_cl_file(*args, **kwargs)
    monkeypatch.setattr(c3_full, 'convert_pop909_cl_file', delayed_targets)
    missing = BoundFile(tmp_path / 'unread-checkpoint', 'a' * 64, 1)
    report = c3_full.evaluate(c3_full.FullRequest((missing,), missing, missing, tmp_path, True),
        experiment_record=tmp_path / 'unread-record', output_root=tmp_path / 'result')
    assert report['access']['model_forwards'] == 2
    assert report['access']['target_reads']['test'] == 1
    assert report['pieces'][0]['split'] == 'test'
    assert report['views']['native']['all']['coverage']['source_chord_events'] > 0
    assert report['views']['native']['splits']['train']['coverage']['source_chord_events'] == 0
