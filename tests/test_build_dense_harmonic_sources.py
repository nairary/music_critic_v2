"""Synthetic source geometry tests; no corpus payload is used."""
from fractions import Fraction
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path

import pytest

from music_critic.adapters.dilemmadata import (
    DilemmadataRawTargetAlignmentEvidence, DilemmadataSourceRowBinding, _alignment_evidence_fingerprint,
)
from music_critic.data import RationalTime, dumps_piece, load_piece
from scripts.build_dense_harmonic_sources import extract_source_packet, parse_score_harmonies


def score(measures):
    return f'<museScore><Score><Staff id="1">{measures}</Staff></Score></museScore>'.encode()


def test_score_locations_define_harmony_boundary_before_next_note_attack():
    xml = score('''<Measure><voice><TimeSig><sigN>4</sigN><sigD>4</sigD></TimeSig>
      <Harmony><name>I</name></Harmony><Chord><durationType>whole</durationType></Chord>
      <location><fractions>-1/2</fractions></location><Harmony><name>V7</name></Harmony>
      </voice></Measure>''')
    parsed = parse_score_harmonies(xml)
    assert [(h['label'], h['start_qn'], h['end_qn']) for h in parsed['harmonies']] == [
        ('I', {'num': 0, 'den': 1}, {'num': 2, 'den': 1}),
        ('V7', {'num': 2, 'den': 1}, {'num': 4, 'den': 1}),
    ]
    assert parsed['harmonies'][-1]['end_origin'] == 'explicit_score_end'


def test_explicit_repeat_preserves_silent_measures_in_unfolded_clock():
    xml = score('''<Measure len="1/4"><voice><TimeSig><sigN>3</sigN><sigD>4</sigD></TimeSig>
      <Harmony><name>I</name></Harmony><Rest><durationType>quarter</durationType></Rest></voice></Measure>
      <Measure><endRepeat>2</endRepeat><voice><Rest><durationType>measure</durationType></Rest></voice></Measure>
      <Measure><startRepeat/><voice><Harmony><name>V</name></Harmony><Rest><durationType>measure</durationType></Rest></voice></Measure>
      <Measure><endRepeat>2</endRepeat><voice><Rest><durationType>measure</durationType></Rest></voice></Measure>''')
    parsed = parse_score_harmonies(xml)
    assert [m['mc'] for m in parsed['playthrough_measures']] == [1, 2, 1, 2, 3, 4, 3, 4]
    assert [h['start_qn'] for h in parsed['harmonies']] == [
        {'num': 0, 'den': 1}, {'num': 4, 'den': 1}, {'num': 8, 'den': 1}, {'num': 14, 'den': 1}]
    assert parsed['score_end_qn'] == {'num': 20, 'den': 1}


def test_unsupported_timing_and_ambiguous_harmony_fail_closed():
    for payload in [
        '<Tuplet><normalNotes>2</normalNotes><actualNotes>3</actualNotes></Tuplet>',
        '<location><measures>1</measures></location>',
        '<Harmony><name>V</name></Harmony>',
    ]:
        xml = score(f'''<Measure><voice><TimeSig><sigN>4</sigN><sigD>4</sigD></TimeSig>
          <Harmony><name>I</name></Harmony>{payload}</voice></Measure>''')
        with pytest.raises(ValueError):
            parse_score_harmonies(xml)


def extraction_fixture():
    base = load_piece(Path(__file__).parent / 'fixtures/data/canonical_piece_v2.json')
    note = replace(next(n for n in base.notes if not n.is_percussion), note_id='note:dense-0',
                   onset_qn=RationalTime(0), duration_qn=RationalTime(2), pitch=60, staff=1, voice=1)
    piece = replace(base, notes=(note,), annotations=(), targets=())
    rid, source_sha = 'dlc:fixture:dense', 'a' * 64
    evidence = DilemmadataRawTargetAlignmentEvidence(
        version='1.1.0', record_id=rid, piece_id=piece.piece_id,
        record_binding_sha256='b'*64, raw_projection_sha256='c'*64,
        canonical_piece_sha256=sha256(dumps_piece(piece).encode()).hexdigest(),
        raw_source_fields=('pitch',), rows=(DilemmadataSourceRowBinding(0,2,note.onset_qn,note.note_id,False),),
        fingerprint='')
    evidence = replace(evidence, fingerprint=_alignment_evidence_fingerprint(evidence))
    row = {'quarterbeats_playthrough':'0','duration':'1/2','mc_playthrough':'1','mc':'1','mn':'1',
           'unfolded_harmony_index':'0','label':'I','pitch':'60','staff':'1','voice':'1',
           'is_note_onset':'True','tpc':'0','step':'C','alter':'0','octave':'4','note_degree':'5',
           'tpc_is_in_label':'False','tpc_is_root':'False','tpc_is_bass':'False','valid_chord_label':'True',
           'root_tpc':'1','localkey_tpc':'1','chord_tones':'(0, 4, 1)','added_tones':'()', 'changes':''}
    metadata = {'case_id':'fixture','record_id':rid,'source_sha256':source_sha,
                'source_version_id':f'{rid}@sha256:{source_sha}','current_split':'train',
                'source_to_canonical_offset_qn':'0/1','parts':[{'part_id':'part:one','staff_ids':['1']}]}
    xml = score('''<Measure><voice><TimeSig><sigN>4</sigN><sigD>4</sigD></TimeSig>
      <Harmony><name>I</name></Harmony><location><fractions>1/4</fractions></location>
      <Harmony><name>V</name></Harmony></voice></Measure>''')
    return piece, asdict(evidence), [row], xml, metadata


def test_native_false_stays_known_and_scale_degree_is_not_chord_degree():
    packet = extract_source_packet(*extraction_fixture())
    segment, = packet['note_segments']
    assert segment['values']['tpc_is_in_label'] is False
    assert segment['availability']['tpc_is_in_label'] is True
    assert segment['values']['note_degree'] == '5'
    assert packet['coordinate_contract']['note_degree'] == 'source_scale_degree_not_chord_degree'
    assert packet['harmonies'][0]['values']['added_tones'] == '()'
    assert packet['harmonies'][0]['availability']['added_tones'] is True
    assert packet['harmonies'][1]['values']['root_tpc'] is None
    assert packet['harmonies'][1]['availability']['root_tpc'] is False
    assert segment['harmony_source_index'] == '0'
    assert segment['end_qn'] == {'num': 2, 'den': 1}
    assert packet['harmonies'][0]['end_qn'] == {'num': 1, 'den': 1}


def test_invalid_source_chord_masks_native_semantics_but_keeps_note_geometry():
    piece, evidence, rows, xml, meta = extraction_fixture()
    rows[0]['valid_chord_label'] = 'False'
    packet = extract_source_packet(piece, evidence, rows, xml, meta)
    assert packet['note_segments'][0]['binding_available'] is True
    assert packet['note_segments'][0]['availability']['tpc_is_root'] is False
    assert packet['harmonies'][0]['availability']['root_tpc'] is False
    assert packet['harmonies'][0]['availability']['label'] is True


def test_mismatching_source_score_label_fails_closed():
    piece, evidence, rows, xml, meta = extraction_fixture()
    rows[0]['label'] = 'invented'
    with pytest.raises(ValueError, match='label mismatch'):
        extract_source_packet(piece, evidence, rows, xml, meta)
