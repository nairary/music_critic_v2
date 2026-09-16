import json
from pathlib import Path

import pytest

from scripts.audit_dlc_changes_alignment import audit, digest, patterns, select_pieces
from music_critic.tasks.dlc_changes import source_events


def event(record_id, identity, changes):
    e = source_events([{'changes': changes, 'unfolded_harmony_index': identity,
                       'chord': 'V', 'valid_chord_label': 'True'}], record_id)[0]
    return e


def test_selection_covers_patterns_and_tokens_deterministically(tmp_path):
    rows = [event('dlc:a:a', '0', '64'), event('dlc:a:b', '0', '+9'), event('dlc:a:c', '0', 'v#9')]
    p = tmp_path/'events.jsonl'; p.write_text(''.join(json.dumps(e)+'\n' for e in rows))
    records = {e['record_id']: {'size_bytes': 100} for e in rows}
    selection = select_pieces(p, records, maximum=3)
    assert selection['records'] == sorted(records)
    assert selection['covered_tokens'] == ['+9', '4', '6', 'v#9']
    assert selection == select_pieces(p, records, maximum=3)


def test_normalized_shorthand_not_mistaken_for_literal_parentheses():
    e = event('dlc:a:a', '0', 'b3'); e['source_context']['chord'] = ['Ger6']
    assert patterns(e) == ['Ger6']


def test_audit_rejects_nontrain_before_target_payload_read(tmp_path, monkeypatch):
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'records': [{'record_id': 'dlc:test:a', 'split': 'test'}]}))
    parent = tmp_path/'parent.json'
    parent.write_text(json.dumps({'status': 'completed', 'artifacts': [{'locator': str(manifest),
        'sha256': digest(manifest), 'size_bytes': manifest.stat().st_size}]}))
    original = Path.open
    def guard(path, *args, **kwargs):
        assert path in (manifest, parent), 'must not open source/target payload'
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guard)
    with pytest.raises(AssertionError): audit(tmp_path, parent, tmp_path/'output')
