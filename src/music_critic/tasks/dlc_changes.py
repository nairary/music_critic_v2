"""Lossless DLC changes targets, separate from raw music and model inputs.

This parses the source notation, not preparation/resolution relations. The
lexical convention follows ms3 changes2list; unlike findall we reject any
unconsumed characters. Vocabulary fitting is explicitly TRAIN-only.
"""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Iterable, Mapping
from fractions import Fraction
from hashlib import sha256
import io
import json
from pathlib import Path
import re

SCHEMA = "DLCChangesTargets@1.0.0"
TOKEN = re.compile(r"(?P<marker>[+\-^v]?)(?P<alteration>#+|b+)?(?P<degree>1[0-9]|[0-9])")
NULLS = {"<NA>", "nan", "NaN", "None", "null"}
CONTEXT_FIELDS = ("label", "chord", "chord_type", "numeral", "figbass",
                  "localkey", "globalkey", "relativeroot")


def parse_changes(value: str) -> list[dict]:
    """Parse all characters, preserving order/multiplicity and interval numbers.

    Empty string is lexical empty, not evidence of a negative label. The source
    event validity gate decides whether absence can supervise a model.
    Numbers 0..19 are source syntax, not a claim about musical interpretation.
    """
    if not isinstance(value, str) or value in NULLS:
        raise ValueError("missing changes")
    result = []
    position = 0
    while position < len(value):
        match = TOKEN.match(value, position)
        if match is None:
            raise ValueError(f"invalid changes at offset {position}: {value!r}")
        alteration = match['alteration'] or ""
        result.append({"token": match[0], "marker": match['marker'],
                       "alteration": alteration,
                       "alteration_semitones": alteration.count('#') - alteration.count('b'),
                       "degree": int(match['degree'])})
        position = match.end()
    return result


def encode_presence(event: Mapping, vocabulary: list[str]) -> tuple[list[int], list[bool]]:
    """Dense multilabel view. Unknown atoms mask the entire event, not negatives."""
    if len(set(vocabulary)) != len(vocabulary):
        raise ValueError("duplicate vocabulary class")
    positives = set(event['positive_tokens'])
    usable = bool(event['target_mask']) and positives.issubset(vocabulary)
    return [int(token in positives) if usable else 0 for token in vocabulary], [usable] * len(vocabulary)


def source_events(rows: Iterable[Mapping[str, str]], record_id: str) -> list[dict]:
    """Deduplicate repeated TSV note rows by source annotation identity.

    Row numbers are TSV line numbers, including the header at line 1. Observed
    note onsets are exact rationals as serialized; they are not inferred label
    boundaries. No canonical note IDs, intervals, or durations are fabricated.
    """
    groups: dict[str, list[tuple[int, Mapping]]] = {}
    for line, row in enumerate(rows, 2):
        identity = row.get('unfolded_harmony_index', '')
        valid_identity = isinstance(identity, str) and re.fullmatch(r"[0-9]+", identity) is not None
        key = str(int(identity)) if valid_identity else f"missing-row-{line}"
        groups.setdefault(key, []).append((line, row))
    result = []
    for identity, members in groups.items():
        reasons = set()
        if identity.startswith('missing-'):
            reasons.add('missing_annotation_id')
        changes = {row.get('changes') for _, row in members}
        if len(changes) != 1:
            reasons.add('conflicting_changes')
        if any(row.get('valid_chord_label') != 'True' for _, row in members):
            reasons.add('invalid_or_missing_chord_validity')
        if any(not row.get('chord') or row.get('chord') in NULLS for _, row in members):
            reasons.add('missing_chord')
        contexts = {field: sorted({row.get(field, '') for _, row in members}) for field in CONTEXT_FIELDS}
        if any(len(values) > 1 for values in contexts.values()):
            reasons.add('conflicting_chord_context')
        atoms = []
        if len(changes) == 1:
            try:
                atoms = parse_changes(next(iter(changes)))
            except ValueError:
                reasons.add('missing_or_unparseable_changes')
        onsets = set()
        onset_evidence = []
        for line, row in members:
            value = row.get('quarterbeats_playthrough', '')
            try:
                onset = Fraction(value)
                onsets.add(onset)
                if row.get('a_isOnset') == 'True':
                    onset_evidence.append({"row": line, "onset_qn": str(onset)})
            except (ValueError, ZeroDivisionError, TypeError):
                pass
        result.append({
            'schema': SCHEMA, 'record_id': record_id, 'source_annotation_id': identity,
            'source_row_numbers': [line for line, _ in members],
            'source_changes_values': sorted(changes, key=lambda x: '' if x is None else x),
            'source_context': contexts, 'atoms': atoms,
            'positive_tokens': sorted({atom['token'] for atom in atoms}),
            'target_mask': not reasons, 'mask_reasons': sorted(reasons),
            'annotation_confidence': 'source_annotation_not_independently_expert_verified',
            'first_observed_note_onset_qn': str(min(onsets)) if onsets else None,
            'source_chord_onset_rows': onset_evidence,
            'canonical_alignment_status': 'not_performed',
        })
    return result


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n'


def export_train_targets(manifest_path: Path, output: Path) -> dict:
    """Read only declared TRAIN TSVs, hash-bound to the manifest; never overwrite.

    A failed output directory is retained as incomplete evidence. A retry must
    use a new path. A completed receipt binds target, vocabulary, and sources.
    """
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    records = manifest['records']
    if not records or any(r.get('split') != 'train' for r in records):
        raise ValueError('all records must be TRAIN before any source access')
    if len({r['record_id'] for r in records}) != len(records):
        raise ValueError('duplicate record_id')
    root = Path(manifest['root']).resolve()
    paths = []
    for record in records:
        relative = Path(record['relative_path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('unsafe source path')
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError('source path escapes root')
        if not record['record_id'].startswith('dlc:') or relative.parts[:2] != ('pitch_arrays', 'DLC'):
            raise ValueError('DLC source required')
        if not re.fullmatch('[0-9a-f]{64}', record['sha256']):
            raise ValueError('source checksum required')
        paths.append(path)
    if len(set(paths)) != len(paths):
        raise ValueError('duplicate source path')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'manifest.json').write_bytes(manifest_bytes)
    counts = Counter()
    support = Counter()
    reasons = Counter()
    piece_support = Counter()
    target_path = output / 'events.jsonl'
    with target_path.open('w', encoding='utf-8') as stream:
        for record, path in sorted(zip(records, paths), key=lambda pair: pair[0]['record_id']):
            payload = path.read_bytes()
            if sha256(payload).hexdigest() != record['sha256'] or len(payload) != record['size_bytes']:
                raise ValueError(f"source binding mismatch: {record['record_id']}")
            rows = csv.DictReader(io.StringIO(payload.decode('utf-8-sig')), delimiter='\t')
            events = source_events(rows, record['record_id'])
            seen_tokens = set()
            for event in events:
                event['source'] = {k: record[k] for k in ('relative_path', 'sha256', 'component_id', 'split')}
                stream.write(_json(event))
                counts['events'] += 1
                counts['note_rows'] += len(event['source_row_numbers'])
                reasons.update(event['mask_reasons'])
                if event['target_mask']:
                    counts['unmasked_events'] += 1
                    counts['nonempty_unmasked_events' if event['positive_tokens'] else 'empty_unmasked_events'] += 1
                    support.update(event['positive_tokens'])
                    seen_tokens.update(event['positive_tokens'])
                else:
                    counts['masked_events'] += 1
            piece_support.update(seen_tokens)
            counts['files'] += 1
    vocabulary = {'schema': SCHEMA, 'fit_split': 'train', 'classes': sorted(support),
                  'event_support': dict(sorted(support.items())),
                  'piece_support': dict(sorted(piece_support.items()))}
    vocabulary_path = output / 'vocabulary.json'
    vocabulary_path.write_text(_json(vocabulary), encoding='utf-8')
    receipt = {'schema': SCHEMA, 'status': 'completed', 'counts': dict(counts),
               'vocabulary_size': len(support), 'mask_reasons': dict(reasons),
               'manifest_sha256': sha256(manifest_bytes).hexdigest(),
               'source_payload_hash_checks': len(records),
               'validation_accessed': False, 'test_accessed': False,
               'canonical_alignment_performed': False, 'model_execution': False,
               'artifacts': {p.name: {'sha256': sha256(p.read_bytes()).hexdigest(),
                                      'size_bytes': p.stat().st_size}
                             for p in (target_path, vocabulary_path)}}
    (output / 'receipt.json').write_text(_json(receipt), encoding='utf-8')
    return receipt
