"""Opt-in HIERGRAPH v0 projections over admitted T07/T08, never raw inputs.

No files, statistical fitting, musical inference or constructor-specific rules
live here. Callers authenticate source bytes, authorize family membership and
freeze a downstream split/active-head manifest before fitting numerical IDs.
"""
from collections import Counter, defaultdict
from pathlib import PurePosixPath, PureWindowsPath
import json
from copy import deepcopy
from hashlib import sha256
import re
from types import MappingProxyType

from music_critic.tasks.event_theory_enrichment import (
    EnrichmentError, canonical, require, validate_enrichment,
    build_vocabulary, encode_enrichment,
)
from music_critic.tasks.whole_work_annotation import (
    _decision, _object_type, _semantic_indexes, _directed_note_decision, _validate_objects,
    build_object_vocabulary, encode_object_fields,
)
from music_critic.tasks.harmonic_projection_primitives import DEGREES, spelling_pitch_class, COMMON_QUALITY_INTERVALS

from music_critic.tasks.chord_structure_projection import FACTOR_FIELDS, factor_sonority, chord_structure_contract
from music_critic.tasks.chord_interval_projection import (
    INTERVAL_FIELDS, interval_contract, parse_tones, build_note_context, project_intervals,
)

VERSION = 'HiergraphProjection@3.0.0'
F = ('tonic', 'dominant', 'predominant', 'subdominant', 'linear_connection', 'coloristic', 'modal_center')
TEMPORAL_ROLES = ('structural', 'arpeggiation', 'passing', 'neighbor', 'ornamental_repetition',
                  'pedal', 'appoggiatura', 'anticipation', 'suspension', 'escape',
                  'changing_tone', 'retardation', 'mixed')
VOICE_ROLES = ('melody', 'harmonic_support', 'bass', 'rhythmic_support', 'countermelody',
               'inner_voice', 'pedal_support', 'mixed', 'ornamental')
PROCESSES = ('progression', 'prolongation', 'cadence', 'loop', 'sequence', 'tonicization',
             'pedal', 'chromatic_connection', 'vamp', 'harmonic_stasis', 'modulation', 'mixed')
P_FUNCTIONS = ('mixed', 'dominant', 'tonic', 'linear_connection', 'predominant', 'coloristic',
               'modal_center', 'nonfunctional', 'subdominant')
# Authored analysis.sonority.quality only. No substring matching or label parsing.
QUALITY_ALIASES = MappingProxyType({
    'major': 'major triad', 'minor': 'minor triad', 'diminished': 'diminished triad',
    'augmented': 'augmented triad', 'dominant_seventh': 'dominant seventh chord',
    'dominant seventh': 'dominant seventh chord', 'minor_seventh': 'minor seventh chord',
    'minor seventh': 'minor seventh chord', 'major_seventh': 'major seventh chord',
    'diminished_seventh': 'diminished seventh chord', 'diminished seventh': 'diminished seventh chord',
    'half_diminished_seventh': 'half-diminished seventh chord',
    'half-diminished seventh': 'half-diminished seventh chord',
})
QUALITIES = tuple(sorted(set(QUALITY_ALIASES.values())))
INVERSION_ALIASES = MappingProxyType({
    'root_position': 'root', 'first_inversion': 'first', 'second_inversion': 'second',
    'third_inversion': 'third', 'root': 'root', 'first': 'first', 'second': 'second', 'third': 'third',
})
# Only these explicit known modes coarsen to other. Unknown never maps to other.
OTHER_MODES = ('dorian', 'phrygian', 'lydian', 'mixolydian', 'locrian', 'blues reference')
DEGREE_FRAMES = MappingProxyType({'analysis.chord_degree': 'H.root.major_reference',
                                 'analysis.key_degree': 'H.tonic.major_reference'})
H_FIELDS = MappingProxyType({
    'H.function': F, 'H.key.tonic_pc': tuple(range(12)), 'H.key.mode': ('major', 'minor', 'other'),
    'H.sonority.root_pc': tuple(range(12)), 'H.sonority.quality': QUALITIES,
    'H.bass_pc': tuple(range(12)), 'H.inversion': ('root', 'first', 'second', 'third'),
    **{f'H.sonority.pitch_class.{pc}': (True,) for pc in range(12)},
})
h_FIELDS = MappingProxyType({
    'h.chord_membership': (False, True), 'h.chord_degree': DEGREES, 'h.key_degree': DEGREES,
    'h.chord_degree.compound': ('9', '11', '13'), 'h.key_degree.compound': ('9', '11', '13'),
    'h.temporal_role': TEMPORAL_ROLES, 'h.voice_role': VOICE_ROLES,
    'h.preparation.linked': (False, True), 'h.resolution.linked': (False, True),
})
P_FIELDS = MappingProxyType({'P.process': PROCESSES, 'P.function': P_FUNCTIONS})
ONTOLOGY = MappingProxyType({**H_FIELDS, **FACTOR_FIELDS, **INTERVAL_FIELDS, **h_FIELDS, **P_FIELDS})


def _hash(value):
    return sha256(canonical(value).encode('utf-8')).hexdigest()


def projection_field_manifest():
    """Machine-readable source/type/frame/encoding contract for every field."""
    result = {}
    for name, values in ONTOLOGY.items():
        scope = 'harmonies' if name.startswith('H.') else 'note_contexts' if name.startswith('h.') else 'objects/harmonic_process'
        frame = None
        if name in INTERVAL_FIELDS:
            paths = ['analysis.sonority']
            if name.startswith('H.chord.interval_pc.'):
                paths = ['analysis.root', 'analysis.root_pitch_class', *paths]
                frame = 'declared_chord_root_relative_pc'
            elif name.startswith('H.observed.interval_pc.'):
                frame = 'instantaneous_observed_root_relative_pc'
            else:
                frame = 'instantaneous_observed_absolute_pc'
        elif name.startswith('H.key.'):
            component = name.rsplit('.', 1)[-1]
            paths = (['analysis.tonic_pc', 'analysis.tonic', 'analysis.key'] if component == 'tonic_pc'
                     else ['analysis.mode', 'analysis.key'])
            frame = 'absolute_pc' if component == 'tonic_pc' else 'local_key_mode'
        elif name in FACTOR_FIELDS and name not in ('H.sonority.root_pc', 'H.bass_pc', 'H.inversion'):
            paths = ['analysis.sonority']
            frame = 'H.root.major_reference' if name == 'H.bass_degree' else 'authored_chord_structure'
        elif name.startswith('H.sonority.') or name == 'H.bass_pc':
            paths = ['analysis.sonority']
            if name.endswith('root_pc'):
                paths = ['analysis.root_pitch_class', 'analysis.root', *paths]
            elif name == 'H.bass_pc':
                paths = ['analysis.bass_pitch_class', 'analysis.bass', *paths]
            frame = None if name.endswith('quality') else 'absolute_pc'
        elif name == 'H.inversion':
            paths = ['analysis.inversion', 'analysis.sonority']
            frame = 'common_basic_inversion_with_known_quality_cardinality'
        elif name.startswith(('h.chord_degree', 'h.key_degree')):
            path = 'analysis.' + name.split('.')[1]
            paths, frame = [path], DEGREE_FRAMES[path]
        elif name.startswith(('h.preparation.', 'h.resolution.')):
            paths = ['analysis.' + name.split('.')[1], 'relations']
            frame = 'incoming_prepares' if name.startswith('h.preparation.') else 'outgoing_resolves_to'
        else:
            paths = ['analysis.' + name.split('.')[1]]
        result[name] = dict(task_id='hiergraph.v1.' + name, scope=scope, source_paths=paths,
            value_type=type(values[0]).__name__, reference_frame=frame, semantic_ids=list(values),
            projection_version=VERSION, encoding_kind='closed_categorical_index',
            supervision_regime='positive_unlabeled' if name.startswith('H.sonority.pitch_class.') else 'fully_supervised',
            unavailable_sentinel=-1, availability='independent_component_mask', default_active=False,
            confidence_policy='single_source_confidence_else_null')
    return result


def projection_contract():
    """One constructor-independent ontology/rule identity (not fitted IDs)."""
    contract = dict(schema=VERSION, ontology=dict(ONTOLOGY), fields=projection_field_manifest(), quality_aliases=dict(QUALITY_ALIASES),
                    inversion_aliases=dict(INVERSION_ALIASES), other_modes=OTHER_MODES,
                    degree_frames=dict(DEGREE_FRAMES), pitch_grammar='common.spelling@1',
                    key_grammar='literal_pitch_optional_mode@1',
                    tones_grammar='spelled_pitch_with_optional_octave_or_explicit_midi_or_pitch_class@2',
                    numerical_encoders=['EventTheoryVocabulary@1', 'EventTheoryObjectVocabulary@1'],
                    source_paths={'H': 'harmonies/analysis.*', 'h': 'note_contexts/analysis.*',
                                  'P': 'objects/harmonic_process/analysis.*'},
                    deferred=['analysis.roman', 'analysis.voice', 'arbitrary_fields'],
                    pitch_set_regime='positive_unlabeled', no_object_from_missing=False,
                    quality_intervals=dict(COMMON_QUALITY_INTERVALS), inversion_check='known_cardinality@1', chord_structure=chord_structure_contract(),
                    source_storage=['inline','authenticated_binding'],
                    chord_intervals=interval_contract())
    contract['fingerprint'] = _hash(contract)
    return contract


def _result(value=None, state='unsupported', rule='no_supported_rule', loss=(), frame=None, diagnostic=None):
    return dict(value=value, available=state in ('exact', 'coarsened'), state=state,
                rule=rule, version=VERSION, information_loss=list(loss), reference_frame=frame,
                diagnostic=diagnostic)


def _literal(value, vocabulary, rule):
    if any(type(value) is type(v) and value == v for v in vocabulary):
        return _result(value, 'exact', rule)
    return _result(state='invalid' if type(value) not in {type(v) for v in vocabulary} else 'unsupported', rule=rule)


def _pitch(value, *, pc=False):
    if pc:
        return (_result(value, 'exact', 'explicit_pc') if type(value) is int and 0 <= value < 12
                else _result(state='invalid', rule='explicit_pc'))
    if type(value) is not str:
        return _result(state='invalid', rule='spelled_pitch')
    # Scoped spelling aliases. This does not change the AN/DLC source grammar.
    match = re.fullmatch(r'([A-Ga-g])-(flat|sharp)', value)
    spelling = (match[1] + ('b' if match[2] == 'flat' else '#')) if match else value
    spelling = spelling.replace('♭', 'b').replace('♯', '#')
    pitch = spelling_pitch_class(spelling)
    return (_result(pitch, 'coarsened', 'spelling_to_pc', ('enharmonic_spelling_removed',))
            if pitch is not None else _result(rule='spelled_pitch'))


def _mode(value):
    if value in ('major', 'minor') and type(value) is str:
        return _result(value, 'exact', 'literal_mode')
    if type(value) is str and value in OTHER_MODES:
        return _result('other', 'coarsened', 'known_mode_to_other', ('mode_detail_removed',))
    if value in (None, 'unknown'):
        return _result(state='masked', rule='unknown_mode', diagnostic='mode_unavailable')
    return _result(state='unsupported' if type(value) is str else 'invalid', rule='literal_mode')


def _combine(candidates):
    """Explicit components precede parsed labels, but disagreements mask locally."""
    if not candidates:
        return _result(state='missing', rule='no_component')
    primary = next((c for c in candidates if c['state'] != 'missing'), candidates[0])
    if not primary['available']:
        return deepcopy(primary)
    conflicting = [c for c in candidates if c['state'] == 'ambiguous' or c.get('diagnostic') == 'conflicting']
    if conflicting:
        return deepcopy(conflicting[0])
    good = [c for c in candidates if c['available']]
    if len({canonical(c['value']) for c in good}) > 1:
        return _result(state='ambiguous', rule='component_agreement', diagnostic='conflicting_components')
    # Lower-priority undecodable descriptions do not erase a supported explicit
    # component. Nor do their spelling losses change an exact explicit-PC route.
    # Preserve the secondary diagnostics rather than asserting their agreement.
    result = deepcopy(primary)
    result['component_routes'] = [
        {k: deepcopy(c[k]) for k in ('value', 'state', 'rule', 'information_loss', 'diagnostic')}
        for c in candidates]
    return result



def _key(value):
    if type(value) is str:
        parts = value.split(' ', 1)
        tonic = _pitch(parts[0])
        mode = _mode(parts[1]) if len(parts) == 2 else _result(state='missing', rule='mode_not_declared')
        return {'tonic_pc': tonic, 'mode': mode}
    if isinstance(value, dict):
        return {'tonic_pc': _combine([_pitch(value[k], pc=k == 'tonic_pc')
                                     for k in ('tonic_pc', 'tonic') if k in value]),
                'mode': _mode(value['mode']) if 'mode' in value else _result(state='missing')}
    return {k: _result(state='invalid', rule='literal_key') for k in ('tonic_pc', 'mode')}


def _degree(value, path, frame):
    expected = DEGREE_FRAMES[path]
    unavailable = _result(rule='spelled_degree', frame=expected, diagnostic='reference_frame_not_admitted')
    if frame != expected:
        return {'generic': unavailable, 'compound': deepcopy(unavailable)}
    text = str(value) if type(value) is int else value
    match = re.fullmatch(r'(bbb|bb|b|###|##|#)?(1[13]|9|[1-7])', text) if type(text) is str else None
    if match is None:
        bad = _result(state='unsupported' if type(value) in (str, int) else 'invalid', rule='spelled_degree', frame=frame)
        return {'generic': bad, 'compound': deepcopy(bad)}
    accidental, degree = match[1] or '', int(match[2])
    generic = accidental + str((degree - 1) % 7 + 1)
    return {'generic': _result(generic, 'coarsened' if degree > 7 else 'exact',
                               'spelled_degree', ('compound_in_separate_component',) if degree > 7 else (), frame),
            'compound': (_result(str(degree), 'exact', 'explicit_compound_degree', frame=frame)
                         if degree > 7 else _result(state='missing', rule='no_compound_evidence', frame=frame))}


def _tones(value):
    # The compatibility PC fields remain positive-only. Richer closed-set
    # observed targets have separate names and exact canonical evidence.
    parsed = parse_tones(value)
    if not parsed['available']:
        return {str(i): _result(state=parsed['state'], rule=parsed['rule'],
                               diagnostic=parsed.get('diagnostic')) for i in range(12)}
    return {str(i): (_result(True, parsed['state'], 'authored_tones_positive_only',
                             parsed['information_loss'], 'absolute_pc')
                     if i in parsed['pcs'] else _result(state='missing', rule='absence_not_negative', frame='absolute_pc'))
            for i in range(12)}


def _sonority(value):
    keys = ['root_pc', 'quality', 'bass_pc'] + ['pitch_class.' + str(i) for i in range(12)]
    if not isinstance(value, dict):
        return {k: _result(state='invalid', rule='structured_sonority') for k in keys}
    quality = value.get('quality')
    if quality is None:
        q = _result(state='missing', rule='explicit_quality')
    elif type(quality) is str and quality in QUALITY_ALIASES:
        q = _result(QUALITY_ALIASES[quality], 'exact', 'scoped_quality_alias')
    else:
        q = _literal(quality, QUALITIES, 'explicit_quality')
    result = {'root_pc': _combine([_pitch(value[k], pc=k == 'root_pitch_class')
                                   for k in ('root_pitch_class', 'root') if k in value]),
              'bass_pc': _combine([_pitch(value[k], pc=k == 'bass_pitch_class')
                                   for k in ('bass_pitch_class', 'bass') if k in value]), 'quality': q}
    result.update({'pitch_class.' + k: v for k, v in _tones(value).items()})
    return result


def project_field(source_path, field, *, degree_frame=None):
    """Project one validated source field; raw input is never mutated.

    Returned components have independent masks and states. Contextual relations
    and multi-source agreement are handled by project_hiergraph_targets.
    """
    components = ({'tonic_pc', 'mode'} if source_path == 'analysis.key' else
                  {'generic', 'compound'} if source_path in DEGREE_FRAMES else
                  {'root_pc', 'quality', 'bass_pc', *('pitch_class.' + str(i) for i in range(12))}
                  if source_path == 'analysis.sonority' else {'value'})
    state = 'missing' if field is None else _decision(field)
    if state != 'known':
        return {k: _result(state='missing' if state == 'missing' else 'masked',
                           rule='source_' + state, diagnostic=state) for k in sorted(components)}
    value = field['value']
    if source_path == 'analysis.key':
        return _key(value)
    if source_path == 'analysis.sonority':
        return _sonority(value)
    if source_path in DEGREE_FRAMES:
        return _degree(value, source_path, degree_frame)
    finite = {'analysis.function': F, 'analysis.temporal_role': TEMPORAL_ROLES,
              'analysis.voice_role': VOICE_ROLES, 'analysis.chord_membership': (False, True),
              'P/analysis.process': PROCESSES, 'P/analysis.function': P_FUNCTIONS}
    if source_path in finite:
        result = _literal(value, finite[source_path], 'scoped_literal')
        if source_path == 'analysis.chord_membership' and type(value) is not bool:
            result = _result(state='invalid', rule='boolean_membership')
        return {'value': result}
    if source_path == 'analysis.inversion':
        result = (_result(INVERSION_ALIASES[value], 'exact', 'basic_inversion_alias')
                  if type(value) is str and value in INVERSION_ALIASES else _result(rule='basic_inversion_alias'))
        return {'value': result}
    return {'value': _result(rule='deferred_open_value', diagnostic='preserved_not_categorical')}


def _target(mapping, sources):
    """Reuse the T07 typed field envelope, with explicit derivation evidence."""
    known_sources = [f for f in sources.values() if f is not None and f['available']]
    confidence = known_sources[0]['confidence'] if len(known_sources) == 1 else None
    return dict(value=deepcopy(mapping['value']), available=mapping['available'],
                origin='source_derived' if mapping['available'] else 'unknown', confidence=confidence,
                evidence=[{'projection': VERSION, 'rule': mapping['rule'], 'source_paths': sorted(sources)}],
                unknown_reason=None if mapping['available'] else mapping['state'] + ':' + mapping['rule'])


def _project_row(row, scope, degree_frames, indexes, note_context):
    fields = row.get('fields', {})
    mapped = {}
    def add(target, mapping, paths):
        mapping = deepcopy(mapping)
        mapping['source_paths'] = list(paths)
        mapping['source_states'] = {p: 'missing' if fields.get(p) is None else _decision(fields[p]) for p in paths}
        mapping['source_origins'] = {p: fields[p].get('origin') for p in paths if fields.get(p) is not None}
        mapping['source_confidences'] = {p: fields[p].get('confidence') for p in paths if fields.get(p) is not None}
        mapped[target] = mapping
    def simple(path, target, parser_path=None):
        add(target, project_field(parser_path or path, fields.get(path))['value'], [path])
    if scope == 'harmonies':
        simple('analysis.function', 'H.function')
        key = project_field('analysis.key', fields.get('analysis.key'))
        for component, explicit in [('tonic_pc', ('analysis.tonic_pc', 'analysis.tonic')), ('mode', ('analysis.mode',))]:
            candidates = []
            for path in explicit:
                if path in fields:
                    state = _decision(fields[path])
                    candidates.append((_mode(fields[path]['value']) if component == 'mode' else
                                       _pitch(fields[path]['value'], pc=path.endswith('tonic_pc')))
                                      if state == 'known' else _result(state='masked', rule='source_' + state, diagnostic=state))
            candidates.append(key[component])
            add('H.key.' + component, _combine(candidates), [*explicit, 'analysis.key'])
        son = project_field('analysis.sonority', fields.get('analysis.sonority'))
        for component, m in son.items():
            add('H.bass_pc' if component == 'bass_pc' else 'H.sonority.' + component, m, ['analysis.sonority'])
        source_sonority = fields.get('analysis.sonority')
        state = 'missing' if source_sonority is None else _decision(source_sonority)
        factors = (factor_sonority(source_sonority['value']) if state == 'known' else
                   {name: _result(state='missing' if state == 'missing' else 'masked',
                                  rule='source_' + state, diagnostic=state) for name in FACTOR_FIELDS})
        for name, factor in factors.items():
            if name in ('H.sonority.root_pc', 'H.bass_pc'):
                factor = _combine([mapped[name], factor])
            if name != 'H.inversion':
                add(name, factor, ['analysis.sonority'])
        # Explicit inversion and the literal slash-bass interpretation must agree.
        direct_inversion = project_field('analysis.inversion', fields.get('analysis.inversion'))['value']
        inferred_inversion = factors['H.inversion']
        if (direct_inversion['available'] and factors['H.bass_degree']['available'] and
                inferred_inversion.get('rule') == 'bass_outside_basic_inversion'):
            inferred_inversion = _result(state='ambiguous', rule='inversion_bass_conflict',
                diagnostic='explicit inversion disagrees with a bass outside the basic inversion system')
        add('H.inversion', _combine([direct_inversion, inferred_inversion]),
            ['analysis.inversion', 'analysis.sonority'])
        inversion, quality = mapped['H.inversion'], mapped['H.sonority.quality']
        if inversion['available'] and quality['available']:
            intervals = COMMON_QUALITY_INTERVALS.get(quality['value'])
            ordinal = ('root', 'first', 'second', 'third').index(inversion['value'])
            if intervals is not None and ordinal >= len(intervals):
                add('H.inversion', _result(state='ambiguous', rule='known_quality_cardinality',
                    diagnostic='inversion_cardinality_inconsistent'), ['analysis.inversion', 'analysis.sonority'])
        for path, target, pc in [('analysis.root', 'H.sonority.root_pc', False),
                                 ('analysis.root_pitch_class', 'H.sonority.root_pc', True),
                                 ('analysis.bass', 'H.bass_pc', False),
                                 ('analysis.bass_pitch_class', 'H.bass_pc', True)]:
            if path in fields:
                f = fields[path]
                state = _decision(f)
                m = _pitch(f['value'], pc=pc) if state == 'known' else _result(state='masked', rule='source_' + state, diagnostic=state)
                add(target, _combine([m, mapped[target]]), [path, *mapped[target]['source_paths']])
        # Final outer root/bass evidence can invalidate symbol-derived properties.
        # Directly authored inversion is independently supervised and stays separate.
        dependencies = [mapped['H.sonority.root_pc'], mapped['H.bass_pc']]
        absolute_conflict = any(item['state'] == 'ambiguous' for item in dependencies)
        independent_degree = mapped['H.bass_degree']['rule'] == 'explicit_bass_degree'
        if absolute_conflict or (not independent_degree and any(not item['available'] for item in dependencies)):
            state = 'ambiguous' if any(item['state'] == 'ambiguous' for item in dependencies) else 'masked'
            paths = list(dict.fromkeys(mapped['H.sonority.root_pc']['source_paths'] + mapped['H.bass_pc']['source_paths']))
            if mapped['H.bass_degree']['available']:
                add('H.bass_degree', _result(state=state, rule='unavailable_root_or_bass_dependency',
                    diagnostic='final root/bass cannot support derived bass degree'), paths)
            if not direct_inversion['available'] and mapped['H.inversion']['available']:
                add('H.inversion', _result(state=state, rule='unavailable_root_or_bass_dependency',
                    diagnostic='final root/bass cannot support inferred inversion'), paths)
        intervals = project_intervals(row, resolved_root=mapped['H.sonority.root_pc'], notes=note_context)
        for name, mapping in intervals.items():
            paths = list(mapping.get('source_paths', ['analysis.sonority']))
            if name.startswith('H.chord.interval_pc.'):
                paths = list(dict.fromkeys(paths + mapped['H.sonority.root_pc']['source_paths']))
            add(name, mapping, paths)
    elif scope == 'note_contexts':
        for path, target in [('analysis.chord_membership', 'h.chord_membership'),
                             ('analysis.temporal_role', 'h.temporal_role'), ('analysis.voice_role', 'h.voice_role')]:
            simple(path, target)
        for path in DEGREE_FRAMES:
            result = project_field(path, fields.get(path), degree_frame=degree_frames.get(path))
            if row.get('harmony_id') is None:
                result = {k: (_result(state='unsupported', rule='missing_H_reference', frame=DEGREE_FRAMES[path])
                              if m['available'] else m) for k, m in result.items()}
            base = 'h.' + path.removeprefix('analysis.')
            for component, m in result.items():
                add(base if component == 'generic' else base + '.compound', m, [path])
        for name in ('preparation', 'resolution'):
            path = 'analysis.' + name
            f = fields.get(path)
            if f is not None and _decision(f) == 'known':
                try:
                    _directed_note_decision(row, path, f['value'], indexes)
                except (EnrichmentError, TypeError, KeyError) as error:
                    m = _result(state='invalid', rule='directed_decision', diagnostic=str(error))
                else:
                    m = _result(f['value']['status'] == 'linked', 'exact', 'directed_decision')
            else:
                m = _result(state='missing' if f is None else 'masked', rule='directed_decision')
            add('h.' + name + '.linked', m, [path])
    elif scope == 'harmonic_process':
        simple('analysis.process', 'P.process', 'P/analysis.process')
        simple('analysis.function', 'P.function', 'P/analysis.function')
    return mapped


def project_hiergraph_targets(piece, bundle, *, family_id, degree_frames=None, source_binding=None):
    """Validate admitted source geometry; return target-only shadow plus evidence."""
    validate_enrichment(piece, bundle)
    _validate_objects(bundle)
    require(isinstance(family_id, str) and bool(family_id), 'explicit family admission required')
    degree_frames = dict(degree_frames or {})
    require(set(degree_frames) <= set(DEGREE_FRAMES) and all(DEGREE_FRAMES[k] == v for k, v in degree_frames.items()),
            'unsupported degree reference frame')
    for scope in ('harmonies', 'note_contexts', 'objects'):
        for row in bundle.get(scope, []):
            for f in row.get('fields', {}).values():
                _decision(f)
    indexes = _semantic_indexes(piece, bundle)
    note_context = build_note_context({note.note_id: note for note in piece.notes})
    if source_binding is None:
        shadow = deepcopy(bundle)
    else:
        _validate_source_bindings(source_binding)
        shadow = {k: deepcopy(v) for k,v in bundle.items()
                  if k not in ('harmonies','note_contexts','objects','relations')}
        for scope in ('harmonies','note_contexts','objects'):
            shadow[scope] = [{k: deepcopy(v) for k,v in row.items() if k != 'fields'}
                             for row in bundle.get(scope, [])]
        shadow['relations'] = deepcopy(bundle.get('relations', []))
    evidence, deferred = {}, []
    for scope in ('harmonies', 'note_contexts', 'objects'):
        for source, target in zip(bundle.get(scope, []), shadow.get(scope, [])):
            kind = ('harmonic_process' if _object_type(source) == 'harmonic_process' else 'deferred_object') if scope == 'objects' else scope
            mappings = _project_row(source, kind, degree_frames, indexes, note_context)
            target['fields'] = {name: _target(m, {p: source.get('fields', {}).get(p) for p in m['source_paths']})
                                for name, m in mappings.items()}
            evidence[source['id']] = mappings
            consumed = {p for m in mappings.values() for p in m['source_paths']}
            for path in sorted(set(source.get('fields', {})) - consumed):
                deferred.append(dict(scope=scope, id=source['id'], source_path=path,
                                     source_state=_decision(source['fields'][path]),
                                     reason='preserved_not_categorical'))
    # Addressed relations reuse exact list endpoints. Never classify their dicts
    # or turn a many-to-many relation into independent binary positives.
    for relation in shadow.get('relations', []):
        source = deepcopy(relation)
        state = _decision(source)
        m = (_literal(source['value'], (False, True), 'addressed_boolean') if state == 'known'
             else _result(state='masked', rule='source_' + state, diagnostic=state))
        if type(source['value']) is not bool and state == 'known':
            m = _result(state='unsupported', rule='addressed_boolean')
        if source['type'] not in ('prepares', 'resolves_to'):
            m = _result(rule='deferred_relation_type')
        if m['available']:
            participants = indexes[0]
            ids = source['source_ids'] + source['target_ids']
            view = source.get('view_id', source.get('harmony_view', 'unspecified'))
            same_view = all(participants[i][0] in ('note', 'note_contexts') and
                            (participants[i][0] == 'note' or participants[i][1] == view) for i in ids)
            forward = max(participants[i][2] for i in source['source_ids']) <= min(
                participants[i][2] for i in source['target_ids'])
            if not same_view or not forward or set(source['source_ids']) & set(source['target_ids']):
                m = _result(state='invalid', rule='addressed_scope', diagnostic='endpoint_view_kind_order_or_self_conflict')
        relation.update(_target(m, {'relation': source}))
        relation.pop('annotation_state', None)
        evidence[source['id']] = {'p.addressed': m}
    result = dict(schema=VERSION, contract_fingerprint=projection_contract()['fingerprint'],
                  family_id=family_id, degree_frames=degree_frames,
                  source_fingerprint=_hash(bundle), source=None, bundle=shadow, mappings=evidence, deferred=deferred)
    result['source'] = (deepcopy(bundle) if source_binding is None else
                        _source_reference(result, source_binding))
    result['fingerprint'] = _hash(result)
    return result


def _validate_source_bindings(bindings):
    require(isinstance(bindings, dict) and set(bindings) == {'bundle','canonical_piece'},
            'canonical and bundle source bindings required')
    for role, binding in bindings.items():
        require(isinstance(binding, dict), 'source binding must be an object')
        path = binding.get('path')
        require(isinstance(path, str) and bool(path) and '\\' not in path and
                not PurePosixPath(path).is_absolute() and not PureWindowsPath(path).drive and
                '..' not in PurePosixPath(path).parts, 'flat relative source binding required')
        require(type(binding.get('size_bytes')) is int and binding['size_bytes'] >= 0 and
                isinstance(binding.get('sha256'),str) and re.fullmatch('[0-9a-f]{64}', binding['sha256']),
                'source SHA/size binding required')
        require(binding.get('role',role) == role, 'source binding role mismatch')


def _source_reference(projection, bindings):
    _validate_source_bindings(bindings)
    return dict(schema='HiergraphSourceBinding@1', bindings=deepcopy(bindings),
                semantic_fingerprint=projection['source_fingerprint'],
                dataset_id=projection['bundle']['dataset_id'], piece_id=projection['bundle']['piece_id'],
                note='Raw evidence is retained in the SHA/size-bound original files, not copied here.')


def compact_projection(projection, source_binding):
    """Retain raw evidence by authenticated reference, without duplicating it.

    Caller must verify the admitted files before supplying these bindings.
    The reference is provenance, not evidence that this function read the files.
    """
    _validate_projection(projection)
    result = {k: deepcopy(v) for k,v in projection.items() if k not in ('source','fingerprint')}
    result['source'] = _source_reference(projection,source_binding)
    result['upstream_projection_fingerprint'] = projection['fingerprint']
    result['fingerprint'] = _hash(result)
    return result


def _validate_projection(projection):
    require(projection.get('schema') == VERSION and projection.get('contract_fingerprint') == projection_contract()['fingerprint'],
            'projection contract mismatch')
    require(_hash({k: v for k, v in projection.items() if k != 'fingerprint'}) == projection.get('fingerprint'),
            'projection fingerprint mismatch')
    source = projection['source']
    if source.get('schema') == 'HiergraphSourceBinding@1':
        _validate_source_bindings(source['bindings'])
        require(source['semantic_fingerprint'] == projection['source_fingerprint'] and
                (source['dataset_id'],source['piece_id']) ==
                (projection['bundle']['dataset_id'],projection['bundle']['piece_id']),
                'projection source reference mismatch')
    else:
        require(_hash(source) == projection['source_fingerprint'], 'projection source mismatch')


def _active_bundle(projection, active_fields, allow_coarsened_fields):
    source = projection['bundle']
    # Avoid retaining unrelated native/source annotation groups while fitting.
    # Their full immutable values already live in projection['source'].
    shadow = {k: source[k] for k in ('schema', 'dataset_id', 'piece_id', 'current_split')}
    for scope in ('harmonies', 'note_contexts', 'objects'):
        shadow[scope] = deepcopy(source.get(scope, []))
    for scope in ('harmonies', 'note_contexts', 'objects'):
        for row in shadow.get(scope, []):
            row['fields'] = {k: v for k, v in row['fields'].items() if k in active_fields}
            for name, field in row['fields'].items():
                if projection['mappings'][row['id']][name]['state'] == 'coarsened' and name not in allow_coarsened_fields:
                    field.update(value=None, available=False, origin='unknown',
                                 unknown_reason='coarsening_not_admitted')
    # P/p losses are opt-in later; this slice exports addresses in evidence only.
    shadow['relations'] = []
    return shadow


def build_projection_vocabulary(projections, *, fit_families, split_fingerprint, active_fields,
                                allow_coarsened_fields=()):
    """Explicit downstream fit admission; a global TRAIN flag is insufficient."""
    require(not isinstance(fit_families, str), 'fit families must be explicit string IDs')
    fit_families = tuple(fit_families)
    require(all(isinstance(f, str) and f for f in fit_families), 'fit families must be explicit string IDs')
    fit_families = tuple(sorted(set(fit_families)))
    active_fields = tuple(sorted(set(active_fields)))
    require(bool(fit_families) and all(isinstance(f, str) and f for f in fit_families), 'fit families required')
    require(type(split_fingerprint) is str and re.fullmatch('[0-9a-f]{64}', split_fingerprint), 'split fingerprint required')
    require(bool(active_fields) and set(active_fields) <= set(ONTOLOGY), 'explicit supported active fields required')
    allow_coarsened_fields = tuple(sorted(set(allow_coarsened_fields)))
    require(set(allow_coarsened_fields) <= set(active_fields), 'coarsening must name active fields')
    identities, seen_identities = [], set()
    counts = Counter()
    value_sets = {scope: defaultdict(set) for scope in ('harmonies','note_contexts')}
    object_sets = defaultdict(lambda: defaultdict(set))
    for projection in projections:
        _validate_projection(projection)
        require(projection['family_id'] in fit_families, 'non-fit family cannot fit vocabulary')
        b = projection['bundle']
        identity = (b['dataset_id'], b['piece_id'])
        require(identity not in seen_identities, 'duplicate fit source')
        seen_identities.add(identity)
        identities.append((*identity, projection['family_id'], projection['fingerprint']))
        bundle = _active_bundle(projection, active_fields, allow_coarsened_fields)
        event_part, object_part = build_vocabulary([bundle]), build_object_vocabulary([bundle])
        for scope, fields in event_part['fields'].items():
            for name, values in fields.items():
                value_sets[scope][name].update(canonical(v) for v in values)
        for kind, fields in object_part['fields'].items():
            object_sets[kind]
            for name, values in fields.items():
                object_sets[kind][name].update(canonical(v) for v in values)
        for scope in ('harmonies','note_contexts','objects'):
            for row in bundle.get(scope,[]):
                for name,field in row['fields'].items():
                    if field['available']:
                        counts[(name, canonical(field['value']))] += 1
    require(bool(identities), 'nonempty admitted fit projections required')
    event = build_vocabulary([])
    event['fields'] = {scope: {name: [json.loads(v) for v in sorted(vs)]
                             for name,vs in sorted(fields.items())} for scope,fields in value_sets.items()}
    event['fingerprint'] = _hash({k:v for k,v in event.items() if k != 'fingerprint'})
    objects = build_object_vocabulary([])
    objects['fields'] = {kind: {name: [json.loads(v) for v in sorted(vs)]
                               for name,vs in sorted(fields.items())} for kind,fields in sorted(object_sets.items())}
    objects['fingerprint'] = _hash({k:v for k,v in objects.items() if k != 'fingerprint'})
    result = dict(schema='HiergraphVocabulary@3', contract_fingerprint=projection_contract()['fingerprint'],
                  fit_families=fit_families, split_fingerprint=split_fingerprint,
                  active_fields=active_fields, allow_coarsened_fields=allow_coarsened_fields, fit_sources=sorted(identities),
                  event=event, objects=objects)
    result['fit_support'] = {}
    for name in active_fields:
        scope = 'harmonies' if name.startswith('H.') else 'note_contexts' if name.startswith('h.') else None
        fields = (result['event']['fields'].get(scope, {}) if scope else
                  result['objects']['fields'].get('harmonic_process', {}))
        lookup = {canonical(value): i for i, value in enumerate(fields.get(name, []))}
        result['fit_support'][name] = [
            dict(value=value, count=counts[(name, canonical(value))],
                 active_id=lookup.get(canonical(value)), unseen_in_fit=counts[(name, canonical(value))] == 0)
            for value in ONTOLOGY[name]]
    result['fingerprint'] = _hash(result)
    return result


def encode_projection(piece, projection, vocabulary, *, split_fingerprint):
    """Frozen event/object encoders plus complete projection/OOV accounting."""
    _validate_projection(projection)
    require(vocabulary.get('schema') == 'HiergraphVocabulary@3', 'projection vocabulary required')
    require(_hash({k: v for k, v in vocabulary.items() if k != 'fingerprint'}) == vocabulary.get('fingerprint'),
            'vocabulary fingerprint mismatch')
    require(vocabulary['contract_fingerprint'] == projection['contract_fingerprint'], 'vocabulary contract mismatch')
    require(split_fingerprint == vocabulary['split_fingerprint'], 'downstream split mismatch')
    bundle = _active_bundle(projection, vocabulary['active_fields'], vocabulary['allow_coarsened_fields'])
    numeric = encode_enrichment(piece, bundle, vocabulary['event'])
    objects = encode_object_fields(piece, bundle, vocabulary['objects'])
    # Event encoder omits fields never present in its fit input; account and
    # preserve those explicitly. Objects already provide this behavior.
    for scope in ('harmonies', 'note_contexts'):
        names = {n for row in bundle.get(scope, []) for n in row['fields']}
        for name in sorted(names - set(numeric[scope]['values'])):
            rows = bundle.get(scope, [])
            numeric[scope]['values'][name] = [-1] * len(rows)
            numeric[scope]['masks'][name] = [False] * len(rows)
            count = sum(row['fields'].get(name, {}).get('available', False) for row in rows)
            if count:
                numeric['oov_counts'][scope + ':' + name] = count
    coverage = Counter()
    for row in projection['mappings'].values():
        for name, m in row.items():
            coverage[name + ':' + m['state']] += 1
    return dict(schema='HiergraphNumericTargets@3', projection_fingerprint=projection['fingerprint'],
                vocabulary_fingerprint=vocabulary['fingerprint'], targets=numeric, objects=objects,
                coverage=dict(sorted(coverage.items())), evidence=deepcopy(projection['mappings']),
                deferred=deepcopy(projection['deferred']),
                coarsening_excluded_counts=dict(Counter(
                    name for row in projection['mappings'].values() for name, m in row.items()
                    if m['state'] == 'coarsened' and name in vocabulary['active_fields']
                    and name not in vocabulary['allow_coarsened_fields'])),
                source_fingerprint=projection['source_fingerprint'], source=deepcopy(projection['source']),
                addressed_relations=deepcopy(projection['bundle'].get('relations', [])))


def projection_encoding_specs(vocabulary):
    """Bind *fitted* IDs to existing TargetEncodingSpec, never ontology order.

    Requires the project's regular ontology dependencies. Fields with no fit
    support have no trainable spec; fit_support still reports their full
    ontology. Raw Roman/voice identity retain deferred CPU encoding policies.
    """
    from music_critic.tasks.encoding import TargetEncodingSpec, TARGET_ENCODING_REGISTRY_VERSION
    require(vocabulary.get('schema') == 'HiergraphVocabulary@3' and
            _hash({k: v for k, v in vocabulary.items() if k != 'fingerprint'}) == vocabulary.get('fingerprint'),
            'vocabulary fingerprint mismatch')
    require(vocabulary['contract_fingerprint'] == projection_contract()['fingerprint'], 'vocabulary contract mismatch')
    specs = []
    for name in vocabulary['active_fields']:
        scope = 'harmonies' if name.startswith('H.') else 'note_contexts' if name.startswith('h.') else None
        fields = (vocabulary['event']['fields'].get(scope, {}) if scope else
                  vocabulary['objects']['fields'].get('harmonic_process', {}))
        values = fields.get(name, [])
        if not values:
            continue
        specs.append(TargetEncodingSpec(
            task_id='hiergraph.v1.' + name, registry_version=TARGET_ENCODING_REGISTRY_VERSION,
            encoding_kind='closed_categorical_index', dtype='torch.long', shape='[N]',
            vocabulary=tuple(canonical(v) for v in values), unavailable_sentinel=-1,
            model_ready=True, deferred_reason=None,
            supervision_regime='positive_unlabeled' if name.startswith('H.sonority.pitch_class.') else 'fully_supervised'))
    for name in ('H.roman', 'h.voice_identity'):
        specs.append(TargetEncodingSpec(
            task_id='hiergraph.v1.' + name, registry_version=TARGET_ENCODING_REGISTRY_VERSION,
            encoding_kind='open_string_cpu', dtype='cpu.str', shape='[N] CPU', vocabulary=None,
            unavailable_sentinel=None, model_ready=False, deferred_reason='preserved_not_categorical',
            supervision_regime='deferred_open_vocabulary'))
    return tuple(specs)
