"""Pure, bounded chord-symbol factorization of authored target descriptions.

Unmentioned intervals have no negative meaning. These are symbolic factors,
not inferred notes, pitch-set completeness, harmonic functions or model inputs.
"""
from hashlib import sha256
import json
import re

from music_critic.tasks.harmonic_projection_primitives import DEGREES, spelling_pitch_class

VERSION = 'ChordStructureProjection@1'
CORES = ('major', 'minor', 'diminished', 'augmented', 'sus2', 'sus4', 'power')
EXTENSIONS = ('9', 'b9', '#9', '11', '#11', '13', 'b13')
ADDITIONS = ('2', '4', '6', '9', '11', '13', '#11')
ALTERATIONS = ('b5', '#5', 'b9', '#9', '#11', 'b13')
OMISSIONS = ('1', '3', '5', '7', '9', '11', '13')
BASS_DEGREES = DEGREES + tuple(a + d for d in ('9', '11', '13') for a in ('bb', 'b', '', '#', '##'))
FACTOR_FIELDS = {
    'H.sonority.core': CORES, 'H.sonority.seventh': ('b7', '7', 'bb7'),
    **{f'H.sonority.{group}.{token}': (False, True)
       for group, tokens in (('extension', EXTENSIONS), ('addition', ADDITIONS),
                             ('alteration', ALTERATIONS), ('omission', OMISSIONS))
       for token in tokens},
    'H.sonority.root_pc': tuple(range(12)), 'H.bass_pc': tuple(range(12)),
    'H.bass_degree': BASS_DEGREES, 'H.inversion': ('root', 'first', 'second', 'third'),
}


def _result(value=None, state='missing', rule='unmentioned_component', loss=(), frame=None, diagnostic=None):
    return dict(value=value, available=state in ('exact', 'coarsened'), state=state,
                rule=rule, version=VERSION, information_loss=list(loss),
                reference_frame=frame, diagnostic=diagnostic)


# Exact aliases are scoped to authored analysis.sonority.quality; no prose search.
# Each entry is (abstract core, seventh, explicitly named additional factors).
QUALITY_ALIASES = {
    'major': ('major', None, ()), 'major triad': ('major', None, ()),
    'major_core': ('major', None, ()),
    'minor': ('minor', None, ()), 'minor triad': ('minor', None, ()),
    'diminished': ('diminished', None, ()), 'diminished triad': ('diminished', None, ()),
    'augmented': ('augmented', None, ()), 'augmented triad': ('augmented', None, ()),
    'dominant_seventh': ('major', 'b7', ()), 'dominant seventh': ('major', 'b7', ()),
    'dominant seventh chord': ('major', 'b7', ()),
    'minor_seventh': ('minor', 'b7', ()), 'minor seventh': ('minor', 'b7', ()),
    'minor seventh chord': ('minor', 'b7', ()),
    'major_seventh': ('major', '7', ()), 'major seventh chord': ('major', '7', ()),
    'diminished_seventh': ('diminished', 'bb7', ()),
    'diminished seventh': ('diminished', 'bb7', ()),
    'diminished seventh chord': ('diminished', 'bb7', ()),
    'fully diminished seventh': ('diminished', 'bb7', ()),
    'half_diminished_seventh': ('diminished', 'b7', ()),
    'half-diminished seventh': ('diminished', 'b7', ()),
    'half-diminished seventh chord': ('diminished', 'b7', ()),
    'augmented_seventh': ('augmented', 'b7', ()),
    'suspended_fourth': ('sus4', None, ()), 'suspended fourth': ('sus4', None, ()),
    'suspended_second': ('sus2', None, ()),
    'major_add9': ('major', None, ('addition.9',)),
    'dominant ninth': ('major', 'b7', ('extension.9',)),
    'dominant seventh with thirteenth': ('major', 'b7', ('extension.13',)),
    'diminished seventh with eleventh': ('diminished', 'bb7', ('extension.11',)),
    'major seventh with added thirteenth': ('major', '7', ('addition.13',)),
    'rootless major added ninth': ('major', None, ('addition.9', 'omission.1')),
    'rootless minor added ninth': ('minor', None, ('addition.9', 'omission.1')),
    'incomplete dominant seventh, fifth absent': ('major', 'b7', ('omission.5',)),
    'incomplete minor seventh, fifth absent': ('minor', 'b7', ('omission.5',)),
    'incomplete diminished seventh, fifth absent': ('diminished', 'bb7', ('omission.5',)),
    'incomplete major triad, fifth absent': ('major', None, ('omission.5',)),
    'no_third': (None, None, ('omission.3',)),
}
PITCH_PATTERN = r'[A-Ga-g](?:#{1,3}|b{1,3}|♭{1,3}|♯{1,3})?'
SYMBOL_PATTERN = rf'(?P<root>{PITCH_PATTERN})(?P<suffix>[^/]*)(?:/(?P<bass>{PITCH_PATTERN}))?'
LEGACY_PATTERN = r'([A-G])(?:(sharp|flat))?_(major|minor|diminished|augmented)'
TOKEN_PATTERN = r'(?:add(?:#11|13|11|9|6|4|2)|(?:no|omit)(?:13|11|9|7|5|3|1)|maj7|M7|Δ7|bb7|b7|[#b](?:13|11|9|5)|s11|69|13|11|9|7|6|2|4|alt)'
CORE_PREFIXES = (('sus2', 'sus2'), ('sus4', 'sus4'), ('sus', 'sus4'),
                 ('dim', 'diminished'), ('aug', 'augmented'),
                 ('min', 'minor'), ('maj', 'major'), ('M', 'major'),
                 ('Δ', 'major'), ('m', 'minor'), ('-', 'minor'),
                 ('ø', 'diminished'), ('°', 'diminished'), ('o', 'diminished'),
                 ('+', 'augmented'), ('5', 'power'))


def _blank(state='missing', diagnostic=None):
    return {field: _result(state=state, diagnostic=diagnostic) for field in FACTOR_FIELDS}


def _write(result, field, value, rule, state='exact', loss=(), frame=None):
    old = result[field]
    if old['state'] == 'ambiguous':
        return
    if old['available'] and old['value'] != value:
        result[field] = _result(state='ambiguous', rule='contradictory_explicit_components',
                                diagnostic='incompatible explicit values for ' + field, frame=frame)
    else:
        result[field] = _result(value, state, rule, loss, frame)


def _spelling(value):
    if isinstance(value, dict) and ('available' in value or 'annotation_state' in value):
        if value.get('annotation_state', 'known') != 'known' or value.get('available') is False:
            return None
        value = value.get('value')
    if type(value) is not str:
        return None
    match = re.fullmatch(r'([A-Ga-g])-(flat|sharp)', value)
    if match:
        value = match[1] + ('b' if match[2] == 'flat' else '#')
    value = value.replace('♭', 'b').replace('♯', '#')
    return value if spelling_pitch_class(value) is not None else None


def _pitch(value, explicit_pc=False):
    if isinstance(value, dict) and ('available' in value or 'annotation_state' in value):
        if value.get('annotation_state', 'known') != 'known' or value.get('available') is False:
            return _result(state='masked', rule='explicit_component_unavailable')
        value = value.get('value')
    if value is None:
        return _result(state='masked', rule='explicit_component_unavailable')
    if explicit_pc:
        return (_result(value, 'exact', 'explicit_pitch_class', frame='absolute_pc')
                if type(value) is int and 0 <= value < 12 else
                _result(state='invalid', rule='explicit_pitch_class'))
    spelling = _spelling(value)
    if spelling is None:
        return _result(state='unsupported' if type(value) is str else 'invalid', rule='literal_spelling')
    return _result(spelling_pitch_class(spelling), 'coarsened', 'spelling_to_pc',
                   ('spelling_removed',), 'absolute_pc')


def _omissions(result, rule):
    # no9 alone explicitly excludes the natural ninth, never altered ninths.
    for token in ('9', '11', '13'):
        if result['H.sonority.omission.' + token]['available']:
            _write(result, 'H.sonority.extension.' + token, False, rule)
            addition = 'H.sonority.addition.' + token
            if result[addition]['available'] and result[addition]['value'] is True:
                _write(result, addition, False, rule)
    if result['H.sonority.omission.7']['available'] and result['H.sonority.seventh']['available']:
        result['H.sonority.seventh'] = _result(state='ambiguous', rule='seventh_explicitly_omitted',
                                              diagnostic='seventh and no7 both explicit')


def _symbol(label):
    if type(label) is not str:
        return _blank('invalid', 'chord label must be a string'), None, None
    legacy = re.fullmatch(LEGACY_PATTERN, label)
    if legacy:
        label = legacy[1] + ({'sharp': '#', 'flat': 'b'}.get(legacy[2], '')) + {
            'major': '', 'minor': 'm', 'diminished': 'dim', 'augmented': 'aug'}[legacy[3]]
    match = re.fullmatch(SYMBOL_PATTERN, label)
    if match is None:
        return _blank('unsupported', 'outside anchored literal chord grammar'), None, None
    root, suffix, bass = match['root'], match['suffix'], match['bass']
    # Spaces and plus separators only occur between explicit extension tokens.
    # A leading + remains the augmented-core sign.
    suffix = suffix.strip()
    if re.fullmatch(r'[A-Za-z0-9#b♭♯Δø°+(),\s-]*', suffix) is None:
        return _blank('unsupported', 'outside chord suffix alphabet'), None, None
    suffix = suffix.replace('♭', 'b').replace('♯', '#')
    if suffix != '+' and re.search(r',\s*,|\+\s*\+|[, +]$', suffix):
        return _blank('unsupported', 'malformed interval separator'), None, None
    if suffix.count('(') != suffix.count(')') or suffix.count('(') > 1:
        return _blank('unsupported', 'unbalanced or nested chord parentheses'), None, None
    if '(' in suffix and not re.fullmatch(r'[^()]*\([^()]*\)', suffix):
        return _blank('unsupported', 'parentheses must enclose the final interval list'), None, None
    result = _blank()
    core, prefix = 'major', ''
    for candidate, candidate_core in CORE_PREFIXES:
        if suffix.startswith(candidate):
            prefix, core = candidate, candidate_core
            suffix = suffix[len(candidate):]
            break
    if not prefix:
        suspended = re.fullmatch(r'(.*?)(sus2|sus4|sus)(.*)', suffix)
        if suspended:
            core = 'sus2' if suspended[2] == 'sus2' else 'sus4'
            suffix = suspended[1] + suspended[3]
    # Major marker applies to the following unaltered seventh/extension.
    major_seventh = prefix in ('maj', 'M', 'Δ')
    diminished_seventh = prefix in ('dim', '°', 'o')
    half_diminished = prefix == 'ø'
    addition_list = bool(re.fullmatch(r'\(add[^()]*\)', suffix))
    compact = re.sub(r'[\s(),]', '', suffix)
    compact = compact.replace('+', '')  # added-interval delimiter after core
    tokens = []
    while compact:
        token = re.match(TOKEN_PATTERN, compact)
        if token is None:
            return _blank('unsupported', 'unsupported chord suffix'), None, None
        tokens.append(token[0])
        compact = compact[len(token[0]):]
    rule = 'anchored_literal_chord_symbol'
    _write(result, 'H.sonority.core', core, rule)
    result['H.sonority.root_pc'] = _pitch(root)
    if bass:
        result['H.bass_pc'] = _pitch(bass)
    for token in tokens:
        if token.startswith(('no', 'omit')):
            degree = token[2:] if token.startswith('no') else token[4:]
            _write(result, 'H.sonority.omission.' + degree, True, rule)
        elif token.startswith('add'):
            _write(result, 'H.sonority.addition.' + token[3:], True, rule)
        elif token == 'alt':
            for alteration in ALTERATIONS:
                field = 'H.sonority.alteration.' + alteration
                if not result[field]['available']:
                    result[field] = _result(state='unsupported', rule='unspecified_alt',
                                            diagnostic='alt does not specify individual alterations')
        elif token in ('maj7', 'M7', 'Δ7', 'b7', 'bb7'):
            _write(result, 'H.sonority.seventh', {'maj7': '7', 'M7': '7', 'Δ7': '7'}.get(token, token), rule)
        elif addition_list and token in ('9', '11', '13'):
            _write(result, 'H.sonority.addition.' + token, True, rule)
        elif token in ('7', '9', '11', '13'):
            seventh = '7' if major_seventh else 'bb7' if diminished_seventh else 'b7'
            _write(result, 'H.sonority.seventh', seventh, rule)
            if token != '7':
                _write(result, 'H.sonority.extension.' + token, True, rule)
        elif token == '69':
            for degree in ('6', '9'):
                _write(result, 'H.sonority.addition.' + degree, True, rule)
        elif token in ('2', '4', '6'):
            _write(result, 'H.sonority.addition.' + token, True, rule)
        else:
            token = '#11' if token == 's11' else token
            _write(result, 'H.sonority.alteration.' + token, True, rule)
            if token in EXTENSIONS:
                _write(result, 'H.sonority.extension.' + token, True, rule)
    if half_diminished and not result['H.sonority.seventh']['available']:
        _write(result, 'H.sonority.seventh', 'b7', rule)
    _omissions(result, rule)
    return result, _spelling(root), _spelling(bass) if bass else None


def _quality(value):
    if isinstance(value, dict) and ('available' in value or 'annotation_state' in value):
        if value.get('annotation_state', 'known') != 'known' or value.get('available') is False:
            return _blank('masked', 'explicit quality unavailable')
        value = value.get('value')
    if value is None:
        return _blank('masked', 'explicit quality unavailable')
    if type(value) is not str or value not in QUALITY_ALIASES:
        return _blank('unsupported' if type(value) is str else 'invalid', 'quality outside finite alias table')
    core, seventh, positives = QUALITY_ALIASES[value]
    result = _blank()
    if core:
        _write(result, 'H.sonority.core', core, 'scoped_quality_alias')
    if seventh:
        _write(result, 'H.sonority.seventh', seventh, 'scoped_quality_alias')
    for field in positives:
        _write(result, 'H.sonority.' + field, True, 'scoped_quality_alias')
    _omissions(result, 'scoped_quality_alias')
    return result


def _combine(routes):
    """Explicit fields precede symbols; contradiction masks just this field."""
    routes = [route for route in routes if route['state'] != 'missing']
    if not routes:
        return _result()
    if routes[0]['state'] in ('masked', 'invalid', 'ambiguous'):
        return routes[0].copy()
    known = [route for route in routes if route['available']]
    if any(route['state'] == 'ambiguous' for route in routes) or len({json.dumps(r['value']) for r in known}) > 1:
        return _result(state='ambiguous', rule='conflicting_authored_components',
                       diagnostic='explicit field and literal symbol disagree')
    if known:
        selected = known[0].copy()
        if any(route['state'] in ('unsupported', 'invalid') for route in routes):
            selected['diagnostic'] = 'unsupported alternative authored component retained by caller'
        return selected
    return routes[0].copy()


def _bass_degree(root, bass, result):
    if not root or not bass:
        return _result(rule='spelled_bass_degree', diagnostic='both root and bass spelling required')
    names = 'CDEFGAB'
    generic = (names.index(bass[0].upper()) - names.index(root[0].upper())) % 7 + 1
    expected = (0, 2, 4, 5, 7, 9, 11)[generic - 1]
    alteration = (spelling_pitch_class(bass) - spelling_pitch_class(root) - expected) % 12
    alteration = alteration - 12 if alteration > 6 else alteration
    if not -3 <= alteration <= 3:
        return _result(rule='spelled_bass_degree', diagnostic='bass alteration outside bounded grammar')
    accidental = ('#' * alteration) if alteration > 0 else ('b' * -alteration)
    degree = accidental + str(generic)
    compound = {2: 9, 4: 11, 6: 13}.get(generic)
    if compound is not None:
        token = accidental + str(compound)
        for group in ('extension', 'addition'):
            mapped = result.get('H.sonority.' + group + '.' + token)
            if mapped and mapped['available'] and mapped['value'] is True:
                degree = token
                break
    if degree not in BASS_DEGREES:
        return _result(rule='spelled_bass_degree', diagnostic='bass degree outside finite grammar')
    return _result(degree, 'exact', 'spelled_bass_degree', frame='H.root.major_reference')


def _inversion(result):
    degree = result['H.bass_degree']
    if not degree['available']:
        return _result(state=degree['state'], rule='spelled_basic_inversion', diagnostic=degree['diagnostic'])
    degree = degree['value']
    core = result['H.sonority.core']
    seventh = result['H.sonority.seventh']
    if degree == '1':
        if result['H.sonority.omission.1']['available']:
            return _result(state='ambiguous', rule='omitted_bass_member', diagnostic='bass degree explicitly omitted')
        return _result('root', 'exact', 'spelled_basic_inversion')
    thirds = {'major': '3', 'minor': 'b3', 'diminished': 'b3', 'augmented': '3'}
    fifths = {'major': '5', 'minor': '5', 'diminished': 'b5', 'augmented': '#5', 'power': '5', 'sus2': '5', 'sus4': '5'}
    if core['available']:
        for ordinal, mapping in (('first', thirds), ('second', fifths)):
            member_matches = mapping.get(core['value']) == degree
            if ordinal == 'second':
                fifth_routes = {token: result['H.sonority.alteration.' + token]
                                for token in ('b5', '#5')}
                explicit_fifths = {token for token, route in fifth_routes.items()
                                   if route['available'] and route['value'] is True}
                if explicit_fifths:
                    # An authored alteration replaces the core's default fifth.
                    member_matches = degree in explicit_fifths
                elif any(route['state'] != 'missing' for route in fifth_routes.values()):
                    # alt leaves the fifth unspecified; the core cannot fill it.
                    member_matches = False
            if member_matches:
                generic = '3' if ordinal == 'first' else '5'
                if result['H.sonority.omission.' + generic]['available']:
                    return _result(state='ambiguous', rule='omitted_bass_member', diagnostic='bass degree explicitly omitted')
                return _result(ordinal, 'exact', 'spelled_basic_inversion')
    if seventh['available'] and seventh['value'] == degree:
        return _result('third', 'exact', 'spelled_basic_inversion')
    return _result(state='masked', rule='bass_outside_basic_inversion', diagnostic='no basic inversion licensed by this bass degree')


def factor_sonority(value):
    """Factor one authored sonority dict; no pitches-to-quality inference or I/O.

    Explicit scalar fields are independently reconciled with a fully anchored
    literal label. Unknown prose stays unavailable. Raw text/provenance belongs
    to the caller and is never changed. Negative interval presence is emitted
    only for an explicitly omitted natural extension.
    """
    if not isinstance(value, dict):
        return _blank('invalid', 'sonority must be an object')
    symbol, root, bass = _symbol(value['label']) if 'label' in value else (_blank(), None, None)
    quality = _quality(value['quality']) if 'quality' in value else _blank()
    result = {field: _combine([quality[field], symbol[field]]) if field.startswith('H.sonority.')
              and field != 'H.sonority.root_pc' else symbol[field].copy() for field in FACTOR_FIELDS}
    for target, fields in (('H.sonority.root_pc', ('root_pc', 'root_pitch_class', 'root')),
                           ('H.bass_pc', ('bass_pc', 'bass_pitch_class', 'bass'))):
        routes = [_pitch(value[field], explicit_pc=field != fields[-1]) for field in fields if field in value]
        result[target] = _combine(routes + [symbol[target]])
        spelling_key = fields[-1]
        if spelling_key in value:
            spelling = _spelling(value[spelling_key])
            if target == 'H.sonority.root_pc':
                root = spelling
            else:
                bass = spelling
    if result['H.sonority.root_pc']['available'] and result['H.bass_pc']['available']:
        degree = _bass_degree(root, bass, result)
    else:
        degree = _result(state='ambiguous' if any(result[f]['state'] == 'ambiguous' for f in
                         ('H.sonority.root_pc', 'H.bass_pc')) else 'missing',
                         rule='bass_degree_requires_consistent_root_and_bass')
    if 'bass_degree' in value:
        declared = value['bass_degree']
        route = (_result(declared, 'exact', 'explicit_bass_degree', frame='H.root.major_reference')
                 if type(declared) is str and declared in BASS_DEGREES else
                 _result(state='invalid', rule='explicit_bass_degree'))
        generic = lambda text: re.sub(r'(9|11|13)$', lambda m: {'9': '2', '11': '4', '13': '6'}[m[0]], text)
        if route['available'] and degree['available'] and generic(declared) == generic(degree['value']):
            degree = route if re.search(r'(9|11|13)$', declared) else degree
        else:
            degree = _combine([route, degree])
    _omissions(result, 'reconciled_explicit_omission')
    result['H.bass_degree'] = degree
    result['H.inversion'] = _inversion(result)
    return result


def chord_structure_contract():
    """Self-hashed finite grammar/alias contract shared by all H constructors."""
    contract = dict(schema=VERSION, fields=FACTOR_FIELDS, quality_aliases=QUALITY_ALIASES,
                    pitch_grammar=PITCH_PATTERN, explicit_pitch_grammar='common.spelling@1 plus scoped Unicode/E-flat aliases',
                    symbol_grammar=SYMBOL_PATTERN,
                    legacy_grammar=LEGACY_PATTERN, interval_grammar=TOKEN_PATTERN,
                    core_prefixes=CORE_PREFIXES, extension_presence='explicit_positive_only',
                    omission='natural_extension_false_only_when_explicitly_omitted',
                    compound_bass='spelling_degree_with_explicit_extension_or_addition',
                    inversion_fifth='explicit_alteration_replaces_core_fifth_unspecified_alt_masks_default@1',
                    no_inference_from_tones=True, absent_is_negative=False,
                    merge='explicit_priority_independent_conflicts_mask_component@1')
    contract = json.loads(json.dumps(contract))
    contract['fingerprint'] = sha256(json.dumps(contract, sort_keys=True, separators=(',', ':'),
                                               ensure_ascii=False).encode()).hexdigest()
    return contract
