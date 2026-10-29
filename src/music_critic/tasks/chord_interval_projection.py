"""Pure source-bound chord components and separately scoped observed sonorities.

No roots are estimated from notes. Canonical notes authenticate observed scope;
physical realizations never silently become interpreted chord components.
"""
from bisect import bisect_left, bisect_right
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import json
import re
from types import MappingProxyType

from music_critic.tasks.chord_structure_projection import factor_sonority

VERSION = 'ChordIntervalProjection@1'
INTERVAL_FIELDS = MappingProxyType({
    **{f'H.chord.interval_pc.{pc}': (False, True) for pc in range(12)},
    **{f'H.observed.pitch_class.{pc}': (False, True) for pc in range(12)},
    **{f'H.observed.interval_pc.{pc}': (False, True) for pc in range(12)},
    'H.observed.root_pc': tuple(range(12)),
})
_PITCH = re.compile(r'([A-Ga-g])((?:#{1,3}|b{1,3})?)(-?\d+)?')
_NATURAL = dict(C=0, D=2, E=4, F=5, G=7, A=9, B=11)
_LITERAL_QUALITIES = {'major': {0, 4, 7}, 'minor': {0, 3, 7}, 'diminished': {0, 3, 6},
                      'major7': {0, 4, 7, 11}, 'dominant7': {0, 4, 7, 10}}
_SCALE_SCOPES = {'scale', 'scale_collection', 'key_scale', 'tonality_collection', 'diatonic_scale', 'key_scale_collection', 'scale_pitch_classes'}


def _result(value=None, state='masked', rule='scope_not_admitted', *, loss=(), scope=None, frame=None, diagnostic=None):
    return dict(value=value, available=state in ('exact', 'coarsened'), state=state, rule=rule,
                version=VERSION, information_loss=sorted(set(loss)), reference_frame=frame,
                diagnostic=diagnostic, source_scope=scope, source_paths=['analysis.sonority'])


def _q(value):
    if hasattr(value, 'to_fraction'):
        value = value.to_fraction()
    if isinstance(value, Fraction):
        return value
    if not isinstance(value, Mapping) or set(value) != {'num', 'den'}:
        raise ValueError('exact rational required')
    n, d = value['num'], value['den']
    if type(n) is not int or type(d) is not int or d <= 0:
        raise ValueError('invalid rational')
    result = Fraction(n, d)
    if (result.numerator, result.denominator) != (n, d):
        raise ValueError('reduced rational required')
    return result


def parse_tones(value):
    """Parse explicit MIDI/PC or spelled tones; never guess numeric formats.

    midi_pitches is None if octave information is incomplete. PCs do not assert
    octave doubling. spelling_classes retains the existing note-template basis.
    """
    if not isinstance(value, Mapping):
        return dict(_result(state='invalid', rule='structured_tones'), pcs=[], midi_pitches=None)
    tones = value.get('tones')
    if not isinstance(tones, list) or not tones:
        return dict(_result(state='missing', rule='no_explicit_tones'), pcs=[], midi_pitches=None)
    fmt = value.get('tones_format', 'spelled_pitch')
    pcs, pitches, spellings, losses = set(), [], set(), set()
    try:
        for tone in tones:
            if fmt in ('midi', 'pitch_class'):
                if type(tone) is not int or not 0 <= tone < (128 if fmt == 'midi' else 12):
                    raise ValueError('invalid explicit integer tone')
                pcs.add(tone % 12)
                if fmt == 'midi':
                    pitches.append(tone); losses.add('octave_removed')
            elif fmt == 'spelled_pitch':
                text = tone.replace('♯', '#').replace('♭', 'b') if type(tone) is str else None
                if text is not None:
                    alias = re.fullmatch(r'([A-Ga-g])-(flat|sharp)', text)
                    if alias:
                        text = alias[1] + ('b' if alias[2] == 'flat' else '#')
                    bare_flat = re.fullmatch(r'([A-Ga-g])(-{1,3})', text)
                    if bare_flat:
                        text = bare_flat[1] + 'b' * len(bare_flat[2])
                match = _PITCH.fullmatch(text) if text is not None else None
                if match is None:
                    raise ValueError('unsupported spelled tone')
                step, accidental, octave = match.groups(); step = step.upper()
                alteration = accidental.count('#') - accidental.count('b')
                pc = _NATURAL[step] + alteration
                pcs.add(pc % 12); spellings.add(f'{step}:{alteration}')
                losses.add('enharmonic_spelling_removed')
                if octave is not None:
                    pitch = 12 * (int(octave) + 1) + pc
                    if not 0 <= pitch <= 127:
                        raise ValueError('spelled MIDI tone outside range')
                    pitches.append(pitch); losses.add('octave_removed')
            else:
                return dict(_result(state='unsupported', rule='tone_format_not_admitted'), pcs=[], midi_pitches=None)
    except ValueError as error:
        return dict(_result(state='invalid', rule='explicit_tone_grammar', diagnostic=str(error)), pcs=[], midi_pitches=None)
    return dict(_result(sorted(pcs), 'coarsened' if losses else 'exact', 'explicit_tone_grammar', loss=losses),
                pcs=sorted(pcs), midi_pitches=sorted(set(pitches)) if len(pitches) == len(tones) else None,
                spelling_classes=sorted(spellings) if fmt == 'spelled_pitch' else None)


@dataclass(frozen=True)
class NoteContext:
    """Reusable exact interval index; built once per canonical piece."""
    notes: Mapping
    ordered: tuple
    onsets: tuple
    boundaries: tuple
    ends: tuple
    tree_size: int

    def active(self, start, end):
        limit = bisect_left(self.onsets, end)
        result = []
        stack = [(1, 0, self.tree_size)]
        while stack:
            node, lo, hi = stack.pop()
            if lo >= limit or self.ends[node] is None or self.ends[node] <= start:
                continue
            if hi - lo == 1:
                result.append(self.ordered[lo])
            else:
                mid = (lo + hi) // 2
                stack.extend(((node * 2, lo, mid), (node * 2 + 1, mid, hi)))
        return result


def build_note_context(notes):
    """Accept note_id -> CanonicalNote or dict with canonical timing/pitch keys."""
    if isinstance(notes, NoteContext):
        return notes
    if not isinstance(notes, Mapping):
        raise ValueError('canonical note mapping required')
    normalized, ordered, boundaries = {}, [], set()
    for identity, note in notes.items():
        get = note.get if isinstance(note, Mapping) else lambda name, default=None: getattr(note, name, default)
        if not isinstance(identity, str) or not identity or get('note_id', identity) != identity:
            raise ValueError('canonical note identity mismatch')
        pitch, drum = get('pitch'), get('is_percussion', False)
        if type(pitch) is not int or not 0 <= pitch < 128 or type(drum) is not bool:
            raise ValueError('invalid canonical note pitch/percussion')
        start, duration = _q(get('onset_qn')), _q(get('duration_qn'))
        if start < 0 or duration < 0:
            raise ValueError('invalid canonical note geometry')
        end = start + duration
        normalized[identity] = (pitch, start, end, drum)
        if not drum and duration > 0:
            ordered.append((start, end, identity, pitch)); boundaries.update((start, end))
    ordered.sort(); size = 1
    while size < len(ordered):
        size *= 2
    ends = [None] * (2 * size)
    for i, item in enumerate(ordered):
        ends[size + i] = item[1]
    for i in range(size - 1, 0, -1):
        known = [v for v in (ends[2 * i], ends[2 * i + 1]) if v is not None]
        ends[i] = max(known) if known else None
    return NoteContext(MappingProxyType(normalized), tuple(ordered), tuple(n[0] for n in ordered),
                       tuple(sorted(boundaries)), tuple(ends), size)


def _ids(value):
    if not isinstance(value, list) or any(type(i) is not str or not i for i in value) or len(value) != len(set(value)):
        raise ValueError('explicit distinct note IDs required')
    return set(value)


def _observed_evidence(row, source, parsed, context):
    start, end = _q(row['start_qn']), _q(row['end_qn'])
    if end <= start:
        raise ValueError('positive observed interval required')
    active = context.active(start, end)
    active_ids = {n[2] for n in active}
    ids = _ids(row.get('source_event_ids'))
    if ids != active_ids:
        raise ValueError('source IDs differ from all active pitched notes')
    actual = {n[3] for n in active}
    if parsed['midi_pitches'] is not None and set(parsed['midi_pitches']) != actual:
        raise ValueError('authored MIDI tones differ from canonical notes')
    if set(parsed['pcs']) != {p % 12 for p in actual}:
        raise ValueError('authored PCs differ from canonical notes')
    attrs = source.get('attributes', {})
    if 'key_active_note_ids' in attrs or 'key_pitches' in attrs:
        if _ids(attrs.get('key_active_note_ids')) != active_ids:
            raise ValueError('key-active IDs differ from canonical notes')
        pitches = attrs.get('key_pitches')
        if not isinstance(pitches, list) or any(type(p) is not int for p in pitches) or set(pitches) != actual:
            raise ValueError('key-active pitches differ from canonical notes')
    if attrs.get('possible_pedal_held_note_ids', []):
        raise ValueError('additional pedal scope not admitted')
    if bisect_right(context.boundaries, start) != bisect_left(context.boundaries, end):
        raise ValueError('observed interval crosses an internal note boundary')
    return active_ids, actual


def _root(mapping):
    if not isinstance(mapping, Mapping) or not mapping.get('available'):
        return None
    pc = mapping.get('value')
    return pc if type(pc) is int and 0 <= pc < 12 and mapping.get('state') in ('exact', 'coarsened') else None


def _literal_root(row, source, parsed, ids, actual):
    literal = source.get('attributes', {}).get('completion012_literal_reading')
    if literal is None:
        return None
    if not isinstance(literal, Mapping):
        raise ValueError('literal reading object required')
    if literal.get('harmony_id') != row.get('id') or literal.get('start_qn') != row['start_qn'] or literal.get('end_qn') != row['end_qn']:
        raise ValueError('literal H identity/interval mismatch')
    if literal.get('structural_root_asserted') is not False or literal.get('function_asserted') is not False:
        raise ValueError('literal root scope must not assert structural root/function')
    if _ids(literal.get('note_ids')) != ids:
        raise ValueError('literal participant IDs mismatch')
    pitches, pcs = literal.get('pitches'), literal.get('pitch_classes')
    if not isinstance(pitches, list) or any(type(p) is not int for p in pitches) or set(pitches) != actual:
        raise ValueError('literal pitches mismatch')
    if not isinstance(pcs, list) or any(type(pc) is not int for pc in pcs) or set(pcs) != set(parsed['pcs']):
        raise ValueError('literal PCs mismatch')
    root = literal.get('literal_root_pitch_class'); quality = literal.get('literal_quality')
    if type(root) is not int or not 0 <= root < 12 or quality not in _LITERAL_QUALITIES:
        raise ValueError('literal root/quality not admitted')
    if {(pc - root) % 12 for pc in parsed['pcs']} != _LITERAL_QUALITIES[quality]:
        raise ValueError('literal root/quality contradicts observed pitches')
    for name in ('analysis.extensions', 'analysis.inversion'):
        field = row.get('fields', {}).get(name)
        if not isinstance(field, Mapping) or not field.get('available') or field.get('annotation_state', 'known') != 'known':
            continue
        attrs = field.get('attributes', {})
        if not isinstance(attrs, Mapping):
            raise ValueError('invalid literal field metadata')
        if not any(key in attrs for key in ('literal_root_pitch_class', 'literal_quality')) and attrs.get('target_scope') != 'instantaneous_complete_pitch_class_sonority':
            # An independent structural inversion is not evidence about this
            # literal root, even when it is known.
            continue
        if attrs.get('literal_root_pitch_class') != root or attrs.get('literal_quality') != quality:
            raise ValueError('literal field root/quality mismatch')
        if attrs.get('target_scope') != 'instantaneous_complete_pitch_class_sonority' or attrs.get('structural_root_asserted') is not False:
            raise ValueError('literal field scope mismatch')
        if attrs.get('reference_interval') != dict(start_qn=row['start_qn'], end_qn=row['end_qn']) or _ids(attrs.get('reference_ids')) != ids:
            raise ValueError('literal field interval/participants mismatch')
    return root


def _complete_chord(row, value, parsed):
    template = row.get('note_template')
    if not isinstance(template, Mapping) or template.get('members_complete') is not True:
        return False
    # Completeness in an observed list never licenses chord-component negatives.
    factors = factor_sonority(value)
    attributes = row.get('fields', {}).get('analysis.sonority', {}).get('attributes', {})
    explicit_chord_scope = any(container.get(key) in ('chord_components', 'interpreted_chord_components')
                               for container in (value, attributes) for key in ('scope', 'target_scope', 'reading'))
    if not explicit_chord_scope and not any(factors[name]['available'] for name in ('H.sonority.core', 'H.sonority.seventh')):
        return False
    identity, evidence = template.get('id'), template.get('evidence')
    if (template.get('schema') != 'AuthoredNoteTemplate@1' or not isinstance(identity, str) or not identity.strip()
            or not isinstance(evidence, list) or not evidence
            or any(not isinstance(item, str) or not item.strip() for item in evidence)):
        raise ValueError('authenticated authored template required')
    members = _ids(template.get('members')); basis = template.get('basis')
    if basis == 'midi_pitch_class' and template.get('enharmonic_policy') == 'author_declared_pc_mapping':
        expected = {str(pc) for pc in parsed['pcs']}
    elif basis == 'midi_pitch' and parsed['midi_pitches'] is not None:
        expected = {str(p) for p in parsed['midi_pitches']}
    elif basis == 'spelled_pitch_class' and parsed.get('spelling_classes') is not None:
        expected = set(parsed['spelling_classes'])
    else:
        raise ValueError('template pitch basis not admitted')
    if members != expected:
        raise ValueError('complete template contradicts authored tones')
    return True


def _emit(result, prefix, values, complete, parsed, scope, frame, *, root_loss=()):
    loss = set(parsed['information_loss']) | set(root_loss)
    for pc in range(12):
        present = pc in values
        if present or complete:
            result[prefix + str(pc)] = _result(present, 'coarsened' if loss else 'exact',
                'complete_source_set' if complete else 'authored_positive_only', loss=loss, scope=scope, frame=frame)


def project_intervals(row, *, resolved_root, notes):
    """Return 37 independent envelopes; no in-place mutation or file access."""
    result = {name: _result(scope='interpreted_chord_components' if name.startswith('H.chord.') else
                          'instantaneous_observed_sonority', frame='root_relative_pc' if '.interval_pc.' in name else 'absolute_pc')
              for name in INTERVAL_FIELDS}
    source = row.get('fields', {}).get('analysis.sonority')
    if not isinstance(source, Mapping) or not source.get('available') or source.get('annotation_state', 'known') != 'known':
        return result
    value = source.get('value'); parsed = parse_tones(value)
    if not parsed['available']:
        for name in result:
            result[name].update(state=parsed['state'], rule=parsed['rule'], diagnostic=parsed['diagnostic'])
        return result
    attrs = source.get('attributes', {})
    if not isinstance(attrs, Mapping):
        for mapping in result.values():
            mapping.update(state='invalid', rule='source_attributes_not_object')
        return result
    scopes = {container[k] for container in (value, attrs) for k in ('scope', 'target_scope', 'reading', 'label')
              if isinstance(container.get(k), str)}
    if scopes & _SCALE_SCOPES:
        for mapping in result.values():
            mapping.update(state='unsupported', rule='scale_scope_not_chord_or_observed')
        return result
    observed = (any(k in attrs for k in ('key_active_note_ids', 'key_pitches', 'completion012_literal_reading')) or
                value.get('reading') in ('complete_collection', 'exact_observed_aggregate_root_unresolved', 'observed_sonority'))
    declared = {container[k] for container in (value, attrs) for k in ('scope', 'target_scope')
                if isinstance(container.get(k), str)}
    observed_scopes = {'observed_sonority', 'instantaneous_complete_pitch_class_sonority', 'key_active_sonority'}
    observed = observed or bool(declared & observed_scopes)
    if declared - observed_scopes - {'chord_components', 'interpreted_chord_components'}:
        return result
    root = _root(resolved_root)
    if observed:
        try:
            ids, actual = _observed_evidence(row, source, parsed, build_note_context(notes))
        except (ValueError, KeyError, TypeError) as error:
            for name in result:
                if name.startswith('H.observed.'):
                    result[name].update(state='invalid', rule='observed_scope_binding', diagnostic=str(error))
            return result
        scope = 'instantaneous_observed_sonority'
        _emit(result, 'H.observed.pitch_class.', set(parsed['pcs']), True, parsed, scope, 'absolute_pc')
        if scopes & {'chord_components', 'interpreted_chord_components'}:
            # Canonical geometry establishes the absolute aggregate, but an
            # interpreted-root assertion cannot become an observed-root label.
            for name in result:
                if name == 'H.observed.root_pc' or name.startswith('H.observed.interval_pc.'):
                    result[name].update(state='ambiguous', rule='conflicting_source_scopes',
                                        diagnostic='observed and interpreted scope declarations conflict')
            return result
        # An outer interpreted root has a different scope. Only this observed
        # sonority's own root components can compete with its literal reading.
        own_root = factor_sonority(value)['H.sonority.root_pc']
        root = _root(own_root)
        root_loss = own_root.get('information_loss', ()) if root is not None else ()
        try:
            literal = _literal_root(row, source, parsed, ids, actual)
            if literal is not None:
                if own_root['state'] in ('ambiguous', 'invalid'):
                    raise ValueError('own observed root components conflict')
                if root is not None and root != literal:
                    raise ValueError('direct and literal observed roots disagree')
                root = literal
                root_loss = ()
        except (ValueError, KeyError, TypeError) as error:
            for name in result:
                if name.startswith('H.observed.interval_pc.') or name == 'H.observed.root_pc':
                    result[name].update(state='ambiguous', rule='literal_root_binding', diagnostic=str(error))
            return result
        if root is not None:
            result['H.observed.root_pc'] = _result(root, 'coarsened' if root_loss else 'exact', 'scoped_observed_root', loss=root_loss, scope=scope, frame='absolute_pc')
            _emit(result, 'H.observed.interval_pc.', {(pc - root) % 12 for pc in parsed['pcs']}, True,
                  parsed, scope, 'observed_root_relative_pc', root_loss=root_loss)
    elif root is not None and 'root_unresolved' not in str(value.get('reading', '')) and value.get('label') not in ('literal_key_collection', 'observed_trumpet_sonority'):
        try:
            complete = _complete_chord(row, value, parsed)
        except (ValueError, KeyError, TypeError) as error:
            for name in result:
                if name.startswith('H.chord.'):
                    result[name].update(state='ambiguous', rule='chord_completeness_binding', diagnostic=str(error))
            return result
        _emit(result, 'H.chord.interval_pc.', {(pc - root) % 12 for pc in parsed['pcs']}, complete, parsed,
              'interpreted_chord_components', 'chord_root_relative_pc', root_loss=resolved_root.get('information_loss', ()))
        paths = list(dict.fromkeys(['analysis.sonority', *resolved_root.get('source_paths', [])]))
        for name, mapping in result.items():
            if name.startswith('H.chord.'):
                mapping['source_paths'] = paths.copy()
    for mapping in result.values():
        mapping['evidence'] = list(source.get('evidence', []))
    return result


def interval_contract():
    contract = dict(schema=VERSION, fields=dict(INTERVAL_FIELDS),
        tone_formats=['spelled_pitch', 'midi', 'pitch_class'], pitch_grammar=_PITCH.pattern,
        spelling_aliases=['Unicode flats/sharps', 'E-flat/A-sharp', 'bare trailing flat hyphens'],
        intervals='(tone_pc-root_pc)%12', scopes=['interpreted_chord_components', 'instantaneous_observed_sonority'],
        chord_closure='explicit_interpreted_scope_or_supported_authored_chord_and_matching_AuthoredNoteTemplate@1_members_complete',
        template_provenance='nonempty_string_id_and_nonempty_list_of_nonempty_string_evidence',
        observed_closure='exact_all_active_pitched_note_IDs_and_tones_no_internal_boundaries',
        literal_qualities={name: sorted(values) for name, values in _LITERAL_QUALITIES.items()},
        interpreted_root='caller_final_reconciled_root', observed_root='own_sonority_or_scoped_literal_only',
        explicit_scales=sorted(_SCALE_SCOPES), conflicting_scopes='authenticated_observed_absolute_only',
        unknown_is_negative=False, root_in_set_required=False, omission_inferred=False,
        note_geometry='exact_half_open_nonpercussion_positive_duration')
    contract = json.loads(json.dumps(contract))
    contract['fingerprint'] = sha256(json.dumps(contract, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
    return contract
