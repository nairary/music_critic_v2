"""Whole-piece contracts on synthetic canonical inputs; no corpus is opened."""
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

from music_critic.data import dumps_piece, load_piece


def source(tmp_path, *, source_name='synthetic', coverage='full', suffix='one'):
    from music_critic.pll.whole_data import WholeSourceCandidate
    piece = replace(load_piece(Path(__file__).parents[1] / 'fixtures/data/canonical_piece_v2.json'),
                    dataset_name=source_name, split='train')
    raw = dumps_piece(piece).encode()
    name = suffix + '.json'
    (tmp_path / name).write_bytes(raw)
    candidate = WholeSourceCandidate(record_id=source_name + ':' + suffix, source=source_name,
        piece_id=piece.piece_id, source_group_id=piece.source_group_id, work_id='fixture:' + suffix,
        split='train', native_split='train', path=name, sha256=sha256(raw).hexdigest(),
        size_bytes=len(raw), coverage=coverage,
        coverage_evidence='Synthetic fixture with explicitly declared extent')
    return candidate, piece


def test_whole_preparation_keeps_complete_canonical_piece_and_explicit_extent(tmp_path):
    from music_critic.pll.whole_data import WholeDataManifest, prepare_whole_sources
    candidate, original = source(tmp_path)
    report = prepare_whole_sources([candidate], tmp_path / 'out', source_root=tmp_path,
        source_weights={'synthetic': 1.}, split_provenance='Synthetic closed split assignment')
    manifest = WholeDataManifest.load(tmp_path / 'out/manifest.json')
    prepared = manifest.load_piece(candidate.record_id)
    assert prepared.notes == original.notes
    assert prepared.bars == original.bars
    assert prepared.duration_qn == original.duration_qn
    assert not prepared.targets and not prepared.annotations
    assert manifest.coverage[candidate.record_id]['coverage'] == 'full'
    assert manifest.coverage[candidate.record_id]['source_sha256'] == candidate.sha256
    assert report['admitted'][0]['bars'] == 2
    assert report['verified_train_reads'] == 1
    assert report['test_payloads_read'] == 0


@pytest.mark.parametrize('coverage', ['unknown', 'excerpt'])
def test_nonfull_sources_are_excluded_before_payload_read_by_default(tmp_path, coverage):
    from music_critic.pll.whole_data import prepare_whole_sources
    c, _ = source(tmp_path, coverage=coverage)
    (tmp_path / c.path).unlink()
    with pytest.raises(ValueError, match='all_requested_sources_required'):
        prepare_whole_sources([c], tmp_path / 'out', source_root=tmp_path,
            source_weights={'synthetic': 1.}, split_provenance='Synthetic closed split assignment')
    receipt = json.loads((tmp_path / 'out/prepare-receipt.json').read_text())
    assert receipt['accesses'] == []
    assert receipt['exclusions'][0]['reason'] == 'coverage_policy'


def test_multiple_complete_records_for_one_source_work_are_rejected(tmp_path):
    from music_critic.pll.whole_data import prepare_whole_sources
    c, _ = source(tmp_path)
    other = replace(c, record_id='other', path='does-not-exist.json')
    with pytest.raises(ValueError, match='one_record_per_source_work'):
        prepare_whole_sources([c, other], tmp_path / 'out', source_root=tmp_path,
            source_weights={'synthetic': 1.}, split_provenance='Synthetic closed split assignment')
    assert not (tmp_path / 'out').exists()


def test_whole_sampler_retains_distant_bars_and_exact_resume(tmp_path):
    from music_critic.data import RationalTime as Q
    from music_critic.pll.whole_data import WholeDataManifest, WholeMaskSampler, prepare_whole_sources
    from music_critic.pll.whole_contracts import WholePieceConfig
    c, p = source(tmp_path)
    p = replace(p, bars=(*p.bars, *(replace(p.bars[-1], bar_id=f'bar:{i:03d}', index=i,
        start_qn=Q(1 + 4 * (i - 1))) for i in range(2, 40))), duration_qn=Q(157),
        beats=(*p.beats, *(replace(b, beat_id=f'beat:{i}-{b.index_in_bar}', bar_id=f'bar:{i:03d}',
            start_qn=b.start_qn + Q(4 * (i - 1))) for i in range(2, 40)
            for b in p.beats if b.bar_id == p.bars[-1].bar_id)))
    raw = dumps_piece(p).encode()
    (tmp_path / c.path).write_bytes(raw)
    c = replace(c, sha256=sha256(raw).hexdigest(), size_bytes=len(raw))
    prepare_whole_sources([c], tmp_path / 'out', source_root=tmp_path,
        source_weights={'synthetic': 1.}, split_provenance='Synthetic closed split assignment')
    manifest = WholeDataManifest.load(tmp_path / 'out/manifest.json')
    sampler = WholeMaskSampler(manifest, seed=9)
    assert len(sampler.draw().example.context.piece.bars) == 40
    state = json.loads(json.dumps(sampler.state_dict()))
    expected = [sampler.draw() for _ in range(3)]
    restored = WholeMaskSampler(manifest, seed=9)
    restored.load_state_dict(state)
    assert [restored.draw() for _ in range(3)] == expected
    with pytest.raises(ValueError, match='state_mismatch'):
        WholeMaskSampler(manifest, seed=9, whole_config=WholePieceConfig(max_piece_bars=128)).load_state_dict(state)


def cli_module():
    import importlib.util
    path = Path(__file__).parents[2] / 'scripts/prepare_pll_whole_sources.py'
    spec = importlib.util.spec_from_file_location('whole_prepare_cli', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def planned(tmp_path, *, damage=False):
    from dataclasses import asdict
    from music_critic.experiments.registry import DATA_ACCESS_FIELDS, seal_record, write_record_once
    module = cli_module()
    c, _ = source(tmp_path)
    if damage:
        c = replace(c, sha256='0' * 64)
    candidates = tmp_path / 'candidates.json'
    candidates.write_text(json.dumps(dict(schema='PLLWholeSourceCandidates@1', records=[asdict(c)],
        source_weights={'synthetic': 1.}, split_provenance='Synthetic fixture split')))
    records = tmp_path / 'records'
    records.mkdir()
    record = records / 'PLL-WHOLE-SYNTHETIC-001.json'
    write_record_once(record, seal_record(dict(schema_version='1.0.0', experiment_id=record.stem,
        kind='data_preparation', phase='HCM-1', title='Synthetic whole preparation', status='planned',
        question='Whole-piece mechanics?', hypothesis='Synthetic only', decision_use='No scientific inference',
        repository={'git_commit': 'a' * 40}, seeds=[], budget={}, results={}, validity={}, claims={},
        artifacts=[], import_provenance={},
        pll_whole_prepare=module.preparation_bindings(candidates.read_bytes(), tmp_path),
        data_access={**{key: None for key in DATA_ACCESS_FIELDS},
            'access_evidence_status': 'not_applicable_planning_record',
            'declared_policy': dict(validation_allowed=False, test_inputs_allowed=False,
                test_targets_allowed=False, test_metrics_allowed=False, test_selection_allowed=False,
                source='Synthetic fixture only')})))
    return module, candidates, record


def test_registered_cli_finalizes_success_and_forbids_reuse(tmp_path):
    from music_critic.experiments.registry import load_record
    module, candidates, record = planned(tmp_path)
    report = module.prepare_registered(candidates, record, tmp_path / 'out', source_root=tmp_path)
    assert report['status'] == 'completed'
    assert load_record(record)['status'] == 'completed'
    assert (tmp_path / 'EXPERIMENT_LEDGER.md').exists()
    before = record.read_bytes()
    with pytest.raises(ValueError, match='planned_record'):
        module.prepare_registered(candidates, record, tmp_path / 'out', source_root=tmp_path)
    assert record.read_bytes() == before


def test_registered_cli_failed_hash_preserves_unknown_access(tmp_path):
    from music_critic.experiments.registry import DATA_ACCESS_FIELDS, load_record
    module, candidates, record = planned(tmp_path, damage=True)
    with pytest.raises(ValueError, match='hash_mismatch'):
        module.prepare_registered(candidates, record, tmp_path / 'out', source_root=tmp_path)
    result = load_record(record)
    assert result['status'] == 'failed'
    assert result['data_access']['access_evidence_status'] == 'pending_primary_evidence'
    assert all(result['data_access'][k] is None for k in DATA_ACCESS_FIELDS)


@pytest.mark.parametrize('source_name', ['dilemmadata', 'hooktheory', 'pop909_cl', 'pdmx'])
def test_four_source_names_retain_canonical_boundaries(tmp_path, source_name):
    from music_critic.pll.whole_data import WholeDataManifest, prepare_whole_sources
    c, p = source(tmp_path, source_name=source_name)
    prepare_whole_sources([c], tmp_path / 'out', source_root=tmp_path,
        source_weights={source_name: 1.}, split_provenance='Synthetic source assignment')
    manifest = WholeDataManifest.load(tmp_path / 'out/manifest.json')
    restored = manifest.load_piece(c.record_id)
    assert restored.bars == p.bars and restored.notes == p.notes
    assert manifest.records[0].source == source_name


@pytest.mark.parametrize('damage', ['path', 'backslash', 'symlink', 'split_component', 'native_alias'])
def test_metadata_preflight_closes_unsafe_and_heldout_payloads(tmp_path, damage):
    from music_critic.pll.whole_data import prepare_whole_sources
    c, _ = source(tmp_path)
    (tmp_path / c.path).unlink()
    if damage == 'path':
        candidates = [replace(c, path='../absent.json')]
    elif damage == 'backslash':
        candidates = [replace(c, path='..\\absent.json')]
    elif damage == 'symlink':
        (tmp_path / 'link.json').symlink_to(tmp_path.parent / 'absent.json')
        candidates = [replace(c, path='link.json')]
    elif damage == 'split_component':
        candidates = [c, replace(c, record_id='heldout', split='test')]
    else:
        candidates = [c, replace(c, source='pdmx', record_id='alias', native_split='test')]
    with pytest.raises(ValueError):
        prepare_whole_sources(candidates, tmp_path / 'out', source_root=tmp_path,
            source_weights={'synthetic': 1.}, split_provenance='Synthetic split assignment')
    receipt = tmp_path / 'out/prepare-receipt.json'
    if receipt.exists():
        assert json.loads(receipt.read_text())['accesses'] == []


def test_unknown_extent_requires_explicit_opt_in_and_stays_unknown(tmp_path):
    from music_critic.pll.whole_data import WholeDataManifest, prepare_whole_sources
    c, _ = source(tmp_path, coverage='unknown')
    prepare_whole_sources([c], tmp_path / 'out', source_root=tmp_path,
        source_weights={'synthetic': 1.}, split_provenance='Synthetic split assignment',
        allowed_coverage=('unknown',))
    manifest = WholeDataManifest.load(tmp_path / 'out/manifest.json')
    assert manifest.coverage[c.record_id]['coverage'] == 'unknown'


def test_preparation_refuses_over_budget_piece_without_truncation(tmp_path):
    from music_critic.pll.whole_data import prepare_whole_sources
    from music_critic.pll.whole_contracts import WholePieceConfig
    c, _ = source(tmp_path)
    with pytest.raises(ValueError, match='all_requested_sources_required'):
        prepare_whole_sources([c], tmp_path / 'out', source_root=tmp_path,
            source_weights={'synthetic': 1.}, split_provenance='Synthetic split assignment',
            whole_config=WholePieceConfig(max_piece_bars=1))
    receipt = json.loads((tmp_path / 'out/prepare-receipt.json').read_text())
    assert receipt['exclusions'][0]['reason'] == 'whole_piece_bar_limit'
    assert not list((tmp_path / 'out/pieces').iterdir())


def test_manifest_rejects_forged_full_coverage_and_changed_source_boundary(tmp_path):
    from music_critic.pll.data import seal_manifest
    from music_critic.pll.whole_data import WholeDataManifest, prepare_whole_sources
    c, _ = source(tmp_path)
    prepare_whole_sources([c], tmp_path / 'out', source_root=tmp_path,
        source_weights={'synthetic': 1.}, split_provenance='Synthetic split assignment')
    payload = json.loads((tmp_path / 'out/manifest.json').read_text())
    payload['whole_piece']['records'][c.record_id]['coverage'] = 'excerpt'
    with pytest.raises(ValueError, match='coverage_invalid'):
        WholeDataManifest(tmp_path / 'out', seal_manifest(payload))
    payload['whole_piece']['records'][c.record_id]['coverage'] = 'full'
    payload['whole_piece']['records'][c.record_id]['end_qn'] = [4, 1]
    manifest = WholeDataManifest(tmp_path / 'out', seal_manifest(payload))
    with pytest.raises(ValueError, match='source_boundary_mismatch'):
        manifest.load_piece(c.record_id)


def test_cli_inspection_rejects_duplicate_json_and_does_not_read_music(tmp_path, capsys):
    from dataclasses import asdict
    module = cli_module()
    c, _ = source(tmp_path)
    (tmp_path / c.path).unlink()
    path = tmp_path / 'candidates.json'
    path.write_text(json.dumps(dict(schema='PLLWholeSourceCandidates@1', records=[asdict(c)],
        source_weights={'synthetic': 1.}, split_provenance='Synthetic split assignment')))
    assert module.main(['inspect', '--candidates', str(path), '--source-root', str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)['data_accessed'] is False
    path.write_text('{"schema":"PLLWholeSourceCandidates@1","schema":"other"}')
    assert module.main(['inspect', '--candidates', str(path), '--source-root', str(tmp_path)]) == 2
    assert 'duplicate_json_key' in capsys.readouterr().err


@pytest.mark.parametrize('observed_split,field', [('test', 'test_inputs_accessed'),
                                                ('validation', 'validation_accessed')])
def test_registered_embedded_heldout_incident_keeps_observed_access(tmp_path, observed_split, field):
    from music_critic.experiments.registry import canonical_json_bytes, load_record, seal_record
    module, candidates, record = planned(tmp_path)
    document = json.loads(candidates.read_text())
    p = load_piece(tmp_path / 'one.json')
    raw = dumps_piece(replace(p, split=observed_split, quality_flags=())).encode()
    (tmp_path / 'one.json').write_bytes(raw)
    document['records'][0].update(sha256=sha256(raw).hexdigest(), size_bytes=len(raw))
    candidates.write_text(json.dumps(document))
    bound = load_record(record)
    bound['pll_whole_prepare'] = module.preparation_bindings(candidates.read_bytes(), tmp_path)
    record.write_bytes(canonical_json_bytes(seal_record(bound)) + b'\n')
    with pytest.raises(ValueError, match='embedded_nontrain'):
        module.prepare_registered(candidates, record, tmp_path / 'out', source_root=tmp_path)
    final = load_record(record)
    assert final['status'] == 'invalid'
    assert final['data_access'][field] is True
    expected_violations = [field]
    if observed_split == 'test' and p.targets:
        expected_violations.append('test_targets_accessed')
        assert final['data_access']['test_targets_accessed'] is True
    assert final['data_access']['policy_violations'] == expected_violations
    assert final['data_access']['access_evidence_status'] == 'partial_primary_evidence'
    assert final['data_access']['test_metrics_computed'] is None
    assert json.loads((tmp_path / 'out/receipt.json').read_text())['status'] == 'invalid'


def test_registered_keyboard_interrupt_finalizes_aborted_with_reason(tmp_path, monkeypatch):
    from music_critic.experiments.registry import load_record
    module, candidates, record = planned(tmp_path)
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt()
    monkeypatch.setattr(module, 'prepare_whole_sources', interrupted)
    with pytest.raises(KeyboardInterrupt):
        module.prepare_registered(candidates, record, tmp_path / 'out', source_root=tmp_path)
    final = load_record(record)
    assert final['status'] == 'aborted'
    assert final['validity']['reason'] == 'KeyboardInterrupt'
