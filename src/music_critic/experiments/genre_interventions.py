"""Bounded multi-note listening interventions, conditioned on declared roles.

Genres choose a rhythm role; they do not make an intervention a stylistic error.
Roles are source/auditor inputs, never inferred from pitch or track order here.
Every result remains an unreviewed listening hypothesis, not a quality label.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import fields

from music_critic.data.schema import CanonicalPiece
from music_critic.data.timing import RationalTime as Q
from music_critic.data.validation import validate_or_raise
from music_critic.experiments.music_interventions import NotePatch

ZERO = Q(0)
ROLES = frozenset({"melody", "accompaniment", "bass", "drums"})
GENRES = frozenset({"classical", "rock", "electronic", "hiphop"})


def _rat(q):
    return [q.num, q.den]


def _roles(piece, roles):
    if not isinstance(roles, dict) or set(roles) - ROLES:
        raise ValueError("unknown role names")
    notes = {n.note_id: n for n in piece.notes}
    result = {}
    for key in ROLES:
        value = roles.get(key, set())
        if not isinstance(value, (set, frozenset)) or any(not isinstance(x, str) for x in value):
            raise ValueError("roles must contain note-ID sets")
        if value - notes.keys():
            raise ValueError("unknown role note IDs")
        result[key] = set(value)
    pitched = result["accompaniment"] | result["bass"]
    if result["melody"] & pitched or result["drums"] & (pitched | result["melody"]):
        raise ValueError("conflicting melody/pitched/drum roles")
    if any(notes[k].is_percussion for k in pitched | result["melody"]):
        raise ValueError("pitched role contains percussion")
    if any(not notes[k].is_percussion for k in result["drums"]):
        raise ValueError("drum role must reference observed percussion")
    return result


def _inside(notes, a, b):
    return [n for n in notes if n.duration_qn > ZERO and a <= n.onset_qn
            and n.onset_qn + n.duration_qn <= b]


def _windows(piece, count=2):
    """Complete consecutive metric bars; no invented fixed 4/4 grid."""
    bars = piece.bars
    for i in range(len(bars) - count + 1):
        group = bars[i:i + count]
        if any(b.is_incomplete or b.is_pickup for b in group):
            continue
        if any(x.start_qn + x.duration_qn != y.start_qn for x, y in zip(group, group[1:])):
            continue
        yield group[0].start_qn, group[-1].start_qn + group[-1].duration_qn, group


def _signature(patches):
    return tuple((p.note_id, p.pitch, p.onset_qn, p.duration_qn, p.delete)
                 for p in sorted(patches, key=lambda p: p.note_id))


def build_genre_interventions(piece: CanonicalPiece, roles: dict[str, set[str]], genre: str):
    """Return (candidate dictionaries, explicit omitted-operator reasons).

    Bass may also belong to accompaniment. Other role overlaps are errors.
    Operators with insufficient observed support are omitted, never fabricated.
    Candidate ``check`` data contains JSON-compatible exact rational pairs.
    """
    validate_or_raise(piece)
    if genre not in GENRES:
        raise ValueError("unknown genre")
    roles = _roles(piece, roles)
    by_id = {n.note_id: n for n in piece.notes}
    role_notes = {k: sorted((by_id[x] for x in v), key=lambda n: (n.onset_qn, n.pitch, n.note_id))
                  for k, v in roles.items()}
    accompaniment = sorted((by_id[x] for x in roles["accompaniment"] | roles["bass"]),
                           key=lambda n: (n.onset_qn, n.pitch, n.note_id))
    candidates, omitted, signatures = [], [], set()

    def omit(ident, aspect, reason):
        omitted.append(dict(id=ident, aspect=aspect, reason=reason))

    def add(ident, aspect, title, description, patches, support, *, kind, minimum=4, **checks):
        patches = tuple(sorted(patches, key=lambda p: p.note_id))
        if len(patches) < minimum or sum(by_id[p.note_id].duration_qn > ZERO for p in patches) < minimum:
            omit(ident, aspect, "Недостаточно реально изменяемых нот для этого оператора.")
            return False
        sig = _signature(patches)
        if sig in signatures:
            omit(ident, aspect, "Точная последовательность правок уже представлена другим вариантом.")
            return False
        signatures.add(sig)
        ids = tuple(p.note_id for p in patches)
        starts = [by_id[k].onset_qn for k in ids]
        ends = [by_id[k].onset_qn + by_id[k].duration_qn for k in ids]
        lo, hi = min(starts), max(ends)
        context = tuple(sorted(n.note_id for n in piece.notes
                               if n.onset_qn < hi and n.onset_qn + n.duration_qn > max(ZERO, lo-Q(4))))
        candidates.append(dict(id=ident, aspect=aspect, title=title, description=description,
                               patches=patches, context=context, onset=lo,
                               check=dict(kind=kind, minimum_count=minimum, support_ids=list(support),
                                          genre=genre, source_piece_id=piece.piece_id,
                                          quality_preference="unreviewed", **checks)))
        return True

    # Keep the whole accompaniment internally coherent while separating it
    # from the explicitly retained melodic line.
    if len(accompaniment) >= 4 and roles["melody"] and all(n.pitch < 127 for n in accompaniment):
        add("H1", "harmony", "Всё сопровождение на полутон выше мелодии",
            "На всём фрагменте сопровождение, включая назначенный бас, сдвинуто на полутон. "
            "Исходная мелодическая партия и ударные сохранены; это конфликт партий, а не доказанная неверная функция.",
            [NotePatch(n.note_id, pitch=n.pitch+1) for n in accompaniment],
            [n.note_id for n in accompaniment], kind="accompaniment_transposition", semitones=1)
    else:
        omit("H1", "harmony", "Нужны отдельная мелодическая партия и минимум четыре ноты сопровождения без выхода за MIDI127.")

    chosen = None
    for a, b, _ in _windows(piece):
        ns = _inside(role_notes["bass"], a, b)
        witnesses = _inside([n for n in piece.notes if n.note_id not in roles["bass"] and not n.is_percussion], a, b)
        delta = 6 if ns and max(n.pitch for n in ns) <= 121 else -6
        if len(ns) >= 4 and witnesses and all(0 <= n.pitch+delta <= 127 for n in ns):
            chosen = (a, b, ns, delta); break
    if chosen:
        a, b, ns, delta = chosen
        add("H2", "harmony", "Бас в конфликте с остальными партиями",
            "В двух тактах назначенный бас перенесён на тритон относительно сохранённых верхних партий. "
            "Проверяется рассогласование баса и созвучий; функциональная ошибка здесь не размечена.",
            [NotePatch(n.note_id, pitch=n.pitch+delta) for n in ns], [n.note_id for n in ns],
            kind="bass_tritone", semitones=delta, window=[_rat(a), _rat(b)])
    else:
        omit("H2", "harmony", "Нет двух полных тактов с четырьмя назначенными басовыми нотами и сохранённой верхней партией.")

    rhythm_role = {"classical": "accompaniment", "rock": "drums", "electronic": "drums", "hiphop": "bass"}[genre]
    if genre == "electronic" and not roles["drums"] and roles["bass"]:
        # Corpus electronic MIDI may omit drums. Use its explicitly supplied
        # bass against the preserved lead rather than synthesizing percussion.
        rhythm_role = "bass"
    rhythm_notes = accompaniment if rhythm_role == "accompaniment" else role_notes[rhythm_role]
    witness_ids = roles["drums"] if genre == "hiphop" else roles["melody"] | roles["bass"]
    witness_ids -= {n.note_id for n in rhythm_notes}
    chosen = None
    delay = Q(1, 2)
    for a, b, _ in _windows(piece):
        # Whole attack groups are delayed; no last note is silently dropped to
        # fit a boundary. We may use following context inside the piece.
        ns = [n for n in rhythm_notes if a <= n.onset_qn < b and n.duration_qn > ZERO]
        simultaneous = sorted({n.onset_qn for n in ns} & {by_id[k].onset_qn for k in witness_ids})
        if len(ns) >= 4 and len(simultaneous) >= 2 and all(n.onset_qn+n.duration_qn+delay <= piece.duration_qn for n in ns):
            chosen = (a, b, ns, simultaneous); break
    if chosen:
        a, b, ns, simultaneous = chosen
        role_label = {"accompaniment": "сопровождение", "drums": "ударные", "bass": "бас"}[rhythm_role]
        add("R1", "rhythm", f"{role_label.capitalize()} отстаёт на половину четверти",
            f"Во фрагменте из двух тактов {role_label} сдвинуто на половину четвертной ноты. "
            "Ранее совместные атаки расходятся; исходные высоты, длительности и остальные партии сохранены. "
            "Это не утверждение, что синкопа сама по себе ошибочна.",
            [NotePatch(n.note_id, onset_qn=n.onset_qn+delay) for n in ns], [n.note_id for n in ns],
            kind="role_delay", role=rhythm_role, delay_qn=_rat(delay), window=[_rat(a), _rat(b)],
            witness_ids=sorted(witness_ids), coincident_onsets=[_rat(t) for t in simultaneous])
    else:
        omit("R1", "rhythm", "Нет двух полных тактов с четырьмя нотами нужной партии и двумя совместными атаками с сохранённой партией.")

    chosen = None
    if roles["drums"]:
        # GM bass/snare hits are named only on explicitly observed drum notes.
        drums = [n for n in role_notes["drums"] if n.pitch in {35, 36, 38, 40} and n.duration_qn > ZERO]
        for a, b, bars in _windows(piece):
            left, right = bars
            if left.duration_qn != right.duration_qn or left.meter_event_id != right.meter_event_id:
                continue
            left_notes = [n for n in drums if left.start_qn <= n.onset_qn < right.start_qn]
            right_pattern = {(n.pitch, n.onset_qn-right.start_qn) for n in drums if right.start_qn <= n.onset_qn < b}
            established = [n for n in left_notes if (n.pitch, n.onset_qn-left.start_qn) in right_pattern]
            if len(established) >= 2:
                chosen = (established, "drum_hit_removal", a, right.start_qn, dict(reference_bar=[_rat(right.start_qn), _rat(b)])); break
    elif genre == "classical":
        for a, b, _ in _windows(piece):
            ns = _inside(accompaniment, a, b)
            times = sorted({n.onset_qn for n in ns})
            selected = [n for n in ns if n.onset_qn in set(times[::2])]
            if len(times) >= 4 and len(selected) >= 2 and roles["melody"]:
                chosen = (selected, "alternating_attack_removal", a, b, {}); break
    if chosen:
        ns, kind, a, b, extra = chosen
        description = ("В одном такте убраны удары бас-барабана/малого барабана, повторяющиеся на тех же позициях следующего такта. Остальные удары сохранены."
                       if kind == "drum_hit_removal" else
                       "В двух тактах удалена каждая вторая группа атак сопровождения; верхняя партия продолжает исходный ритм. Это наблюдаемое прореживание, без выдуманной разметки сильных долей.")
        add("R2", "rhythm", "Пропуски в установленном рисунке", description,
            [NotePatch(n.note_id, delete=True) for n in ns], [n.note_id for n in ns],
            kind=kind, minimum=2, window=[_rat(a), _rat(b)], **extra)
    else:
        omit("R2", "rhythm", "Нет повторённых ударов kick/snare или подходящих групп атак сопровождения.")

    groups = defaultdict(list)
    for n in role_notes["melody"]:
        if n.duration_qn > ZERO:
            groups[n.onset_qn].append(n)
    attack_groups = sorted(groups.items())
    chosen = None
    for i in range(len(attack_groups)-3):
        group = attack_groups[i:i+4]
        if any(len(ns) != 1 for _, ns in group):
            continue
        ns = [v[0] for _, v in group]
        if any(y.onset_qn-x.onset_qn > Q(2) or abs(y.pitch-x.pitch) > 7 for x, y in zip(ns, ns[1:])):
            continue
        for polarity in (1, -1):
            pitches = [n.pitch + 12*polarity*(1 if j % 2 == 0 else -1) for j, n in enumerate(ns)]
            if all(0 <= p <= 127 for p in pitches) and all(abs(y-x) >= 12 for x, y in zip(pitches, pitches[1:])):
                chosen = (ns, pitches); break
        if chosen: break
    if chosen:
        ns, pitches = chosen
        add("M1", "melody", "Четыре атаки с резкими октавными разрывами",
            "Четыре последовательные одиночные атаки назначенной мелодической партии попеременно перенесены на октаву вверх и вниз. "
            "Исходные шаги не превышали квинту, новые составляют не меньше октавы. Классы высот и ритм сохранены.",
            [NotePatch(n.note_id, pitch=p) for n, p in zip(ns, pitches)], [n.note_id for n in ns],
            kind="octave_contour", ordered_ids=[n.note_id for n in ns])
    else:
        omit("M1", "melody", "Нет четырёх последовательных одиночных атак с достаточно плавным исходным контуром и допустимым регистром.")

    chosen = None
    for a, b, _ in _windows(piece):
        ns = _inside(role_notes["melody"], a, b)
        if len({n.pitch for n in ns}) < 3:
            continue
        pitch = sorted(n.pitch for n in ns)[len(ns)//2]
        changed = [n for n in ns if n.pitch != pitch]
        if len(changed) >= 4:
            chosen = (a, b, ns, changed, pitch); break
    if chosen:
        a, b, ns, changed, pitch = chosen
        add("M2", "melody", "Мелодический рисунок заменён одной высотой",
            "В двух тактах все назначенные мелодические ноты сведены к одной высоте. "
            "Ритм и сопровождение сохранены; исчезновение контура проверяется по исходным нотам.",
            [NotePatch(n.note_id, pitch=pitch) for n in changed], [n.note_id for n in ns],
            kind="melody_flattening", target_pitch=pitch, window=[_rat(a), _rat(b)])
    else:
        omit("M2", "melody", "Недостаточно различающихся высот и четырёх изменяемых мелодических нот в двух полных тактах.")

    chosen = None
    for role in ("bass", "accompaniment"):
        ns_all = role_notes[role] if role == "bass" else accompaniment
        for a, b, _ in _windows(piece):
            ns = _inside(ns_all, a, b)
            touching = [n for n in ns_all if n.duration_qn > ZERO and n.onset_qn < b and n.onset_qn+n.duration_qn > a]
            if len(ns) >= 4 and len(ns) < len(piece.notes) and len(touching) == len(ns):
                chosen = (role, a, b, ns); break
        if chosen: break
    if chosen:
        role, a, b, ns = chosen
        label = "бас" if role == "bass" else "сопровождение"
        add("A1", "arrangement", f"На два такта исчезает {label}",
            f"Удалены полностью лежащие в этих двух тактах ноты назначенной партии «{label}». "
            "Остальная фактура сохранена. Это потеря слоя аранжировки, а не обязательная ошибка: допустимость оценивается на слух.",
            [NotePatch(n.note_id, delete=True) for n in ns], [n.note_id for n in ns],
            kind="role_dropout", role=role, window=[_rat(a), _rat(b)])
    else:
        omit("A1", "arrangement", "Нет выделенного баса/сопровождения с четырьмя нотами внутри двух тактов.")

    chosen = None
    for a, b, bars in _windows(piece):
        left, right = bars
        if left.duration_qn != right.duration_qn or left.meter_event_id != right.meter_event_id or left.metric_offset_qn != right.metric_offset_qn:
            continue
        boundaries = (a, right.start_qn, b)
        if any(n.onset_qn < t < n.onset_qn+n.duration_qn for n in piece.notes for t in boundaries):
            continue
        ns_left = [n for n in piece.notes if a <= n.onset_qn < right.start_qn]
        ns_right = [n for n in piece.notes if right.start_qn <= n.onset_qn < b]
        if len(ns_left) < 2 or len(ns_right) < 2:
            continue
        def pattern(ns, offset):
            return sorted((n.track_id, n.pitch, n.onset_qn-offset, n.duration_qn, n.velocity,
                           str(n.staff), str(n.voice)) for n in ns)
        if pattern(ns_left, a) == pattern(ns_right, right.start_qn):
            continue
        chosen = (a, b, right.start_qn, left.duration_qn, ns_left, ns_right); break
    if chosen:
        a, b, mid, length, left_notes, right_notes = chosen
        add("S1", "structure", "Два соседних такта поменяны местами",
            "Два различающихся полных такта одинаковой длительности переставлены во всех партиях. "
            "Размер, внутренний ритм каждого такта и общая длина сохранены; это изменение локального порядка, не разметка куплета или припева.",
            [NotePatch(n.note_id, onset_qn=n.onset_qn+length) for n in left_notes]
            + [NotePatch(n.note_id, onset_qn=n.onset_qn-length) for n in right_notes],
            [n.note_id for n in left_notes+right_notes], kind="bar_order_swap",
            window=[_rat(a), _rat(b)], midpoint_qn=_rat(mid), bar_length_qn=_rat(length))
    else:
        omit("S1", "structure", "Нет двух различных одинаковых по метру тактов без пересекающих границы нот.")
    return candidates, omitted


def verify_genre_variant(original: CanonicalPiece, changed: CanonicalPiece, candidate: dict,
                         roles: dict[str, set[str]]) -> dict:
    """Recompute field-level and operator-level effects on materialized notes.

    This intentionally does not call the candidate generator or composer and
    never returns a positive/negative human quality judgement.
    """
    validate_or_raise(original)
    validate_or_raise(changed)
    roles = _roles(original, roles)
    before = {n.note_id: n for n in original.notes}
    after = {n.note_id: n for n in changed.notes}
    check = candidate["check"]
    if check["source_piece_id"] != original.piece_id:
        raise ValueError("candidate belongs to another source")
    patches = {p.note_id: p for p in candidate["patches"]}
    if len(patches) != len(candidate["patches"]) or set(patches)-before.keys():
        raise ValueError("duplicate or unknown patch IDs")
    if len(patches) < check["minimum_count"] or sum(before[k].duration_qn > ZERO for k in patches) < check["minimum_count"]:
        raise ValueError("insufficient changed-note support")
    removed = {k for k, p in patches.items() if p.delete}
    if after.keys() != before.keys()-removed:
        raise ValueError("unexpected added or deleted notes")
    for k, old in before.items():
        patch = patches.get(k)
        if k in removed:
            continue
        new = after[k]
        if patch is None:
            if new != old:
                raise ValueError("unplanned change to an untouched note")
            continue
        written = patch.written_fields
        if all(getattr(old, field) == getattr(patch, field) for field in written):
            raise ValueError("no-op patch")
        allowed = set(written) | {"provenance_id"}
        if "pitch" in written:
            allowed |= {"spelling_step", "spelling_alter"}
            if new.spelling_step is not None or new.spelling_alter is not None:
                raise ValueError("stale pitch spelling")
        if {"onset_qn", "duration_qn"} & written:
            timing = {"source_onset_ticks", "source_duration_ticks", "source_onset_seconds", "source_duration_seconds"}
            allowed |= timing
            if any(getattr(new, field) is not None for field in timing):
                raise ValueError("stale absolute source timing")
        for field in fields(old):
            name = field.name
            if name in written and getattr(new, name) != getattr(patch, name):
                raise ValueError("planned field not materialized")
            if name not in allowed and getattr(new, name) != getattr(old, name):
                raise ValueError("unplanned field change")
    for attr in ("duration_qn", "tracks", "bars", "beats", "meter_events", "tempo_events", "key_signature_events", "split", "source_group_id"):
        if getattr(original, attr) != getattr(changed, attr):
            raise ValueError("raw piece context changed")
    if changed.targets or any(a.layer == "target_alignment" for a in changed.annotations):
        raise ValueError("stale supervised targets")
    support = check["support_ids"]
    if len(support) != len(set(support)) or set(support)-before.keys() or set(patches)-set(support):
        raise ValueError("invalid support IDs")
    kind = check["kind"]
    if "window" in check:
        start, end = (Q(*r) for r in check["window"])
        count = 1 if kind == "drum_hit_removal" else 2
        if (start, end) not in {(a, b) for a, b, _ in _windows(original, count)}:
            raise ValueError("operator window is not the stated full metric bars")
    if candidate["aspect"] in {"harmony", "rhythm", "arrangement"} and set(patches) & roles["melody"]:
        raise ValueError("designated melody not preserved")
    if kind in {"accompaniment_transposition", "bass_tritone", "octave_contour", "melody_flattening"}:
        if any(p.written_fields != {"pitch"} for p in patches.values()):
            raise ValueError("pitch operator changed other fields")
    if kind in {"accompaniment_transposition", "bass_tritone"}:
        eligible = roles["accompaniment"] | roles["bass"] if kind == "accompaniment_transposition" else roles["bass"]
        if set(patches)-eligible or any(after[k].pitch-before[k].pitch != check["semitones"] for k in patches):
            raise ValueError("role transposition mismatch")
        if kind == "accompaniment_transposition" and set(patches) != eligible:
            raise ValueError("whole accompaniment was not shifted")
        if kind == "bass_tritone" and abs(check["semitones"]) != 6:
            raise ValueError("not a tritone")
        if kind == "bass_tritone":
            expected = {n.note_id for n in _inside([before[k] for k in roles["bass"]], start, end)}
            witnesses = _inside([n for n in original.notes if n.note_id not in roles["bass"] and not n.is_percussion], start, end)
            if set(patches) != expected or not witnesses:
                raise ValueError("bass support or retained upper context mismatch")
    elif kind == "role_delay":
        delay = Q(*check["delay_qn"])
        eligible = roles[check["role"]] | (roles["bass"] if check["role"] == "accompaniment" else set())
        if delay != Q(1, 2) or set(patches)-eligible:
            raise ValueError("invalid delayed role or offset")
        if any(p.written_fields != {"onset_qn"} or after[k].onset_qn-before[k].onset_qn != delay for k, p in patches.items()):
            raise ValueError("incorrect attack delay")
        expected = {k for k in eligible if start <= before[k].onset_qn < end and before[k].duration_qn > ZERO}
        if set(patches) != expected:
            raise ValueError("delay omitted part of an observed role attack group")
        witnesses = set(check["witness_ids"])
        if witnesses-before.keys() or witnesses & set(patches):
            raise ValueError("invalid preserved attack witnesses")
        coincident = {before[k].onset_qn for k in patches} & {before[k].onset_qn for k in witnesses}
        if len(coincident) < 2 or sorted(_rat(t) for t in coincident) != sorted(check["coincident_onsets"]):
            raise ValueError("source simultaneous attack evidence missing")
    elif kind in {"drum_hit_removal", "alternating_attack_removal", "role_dropout"}:
        if set(patches) != removed:
            raise ValueError("removal operator contains non-deletion")
        if kind == "drum_hit_removal":
            a, b = (Q(*r) for r in check["window"])
            c, d = (Q(*r) for r in check["reference_bar"])
            if b-a != d-c:
                raise ValueError("drum reference duration mismatch")
            observed = {(n.pitch, n.onset_qn-c) for n in original.notes if n.note_id in roles["drums"] and c <= n.onset_qn < d}
            if removed-roles["drums"] or any(before[k].pitch not in {35,36,38,40} or (before[k].pitch,before[k].onset_qn-a) not in observed for k in removed):
                raise ValueError("deleted drum hits lack repeated source evidence")
        elif kind == "alternating_attack_removal":
            a, b = (Q(*r) for r in check["window"])
            ns = _inside([before[k] for k in roles["accompaniment"] | roles["bass"]], a, b)
            selected_times = set(sorted({n.onset_qn for n in ns})[::2])
            if removed != {n.note_id for n in ns if n.onset_qn in selected_times}:
                raise ValueError("wrong alternating attack groups")
        else:
            eligible = roles[check["role"]] | (roles["bass"] if check["role"] == "accompaniment" else set())
            a, b = (Q(*r) for r in check["window"])
            if removed != {n.note_id for n in _inside([before[k] for k in eligible], a, b)}:
                raise ValueError("role dropout scope mismatch")
            if any(n.note_id in eligible and n.duration_qn > ZERO and n.onset_qn < b and n.onset_qn+n.duration_qn > a for n in changed.notes):
                raise ValueError("role dropout left a sustained note inside its window")
    elif kind == "octave_contour":
        ids = check["ordered_ids"]
        if set(ids) != set(patches) or len(ids) != 4 or set(ids)-roles["melody"]:
            raise ValueError("invalid melody contour support")
        groups = defaultdict(list)
        for k in roles["melody"]:
            if before[k].duration_qn > ZERO:
                groups[before[k].onset_qn].append(k)
        source_groups = sorted(groups.items())
        consecutive = [[xs[0] for _, xs in source_groups[i:i+4]]
                       for i in range(len(source_groups)-3)
                       if all(len(xs) == 1 for _, xs in source_groups[i:i+4])]
        if ids not in consecutive or any(before[b].onset_qn-before[a].onset_qn > Q(2) for a,b in zip(ids,ids[1:])):
            raise ValueError("contour notes are not consecutive isolated source attacks")
        if any(abs(after[k].pitch-before[k].pitch) != 12 for k in ids):
            raise ValueError("melody pitch classes not preserved")
        if any(abs(before[b].pitch-before[a].pitch) > 7 or abs(after[b].pitch-after[a].pitch) < 12 for a, b in zip(ids,ids[1:])):
            raise ValueError("contour break not established")
    elif kind == "melody_flattening":
        if set(support)-roles["melody"] or len({before[k].pitch for k in support}) < 3 or {after[k].pitch for k in support} != {check["target_pitch"]}:
            raise ValueError("melody flattening not established")
        expected = {n.note_id for n in _inside([before[k] for k in roles["melody"]], start, end)}
        if set(support) != expected or set(patches) != {k for k in expected if before[k].pitch != check["target_pitch"]}:
            raise ValueError("incomplete melodic flattening window")
    elif kind == "bar_order_swap":
        a, b = (Q(*r) for r in check["window"])
        mid, length = Q(*check["midpoint_qn"]), Q(*check["bar_length_qn"])
        if mid-a != length or b-mid != length:
            raise ValueError("unequal swapped bars")
        bars = [bar for bar in original.bars if bar.start_qn in (a, mid)]
        if len(bars) != 2 or any(bar.duration_qn != length for bar in bars) or bars[0].meter_event_id != bars[1].meter_event_id or bars[0].metric_offset_qn != bars[1].metric_offset_qn:
            raise ValueError("bar swap meter mismatch")
        observed = {n.note_id for n in original.notes if a <= n.onset_qn < b}
        if observed != set(patches) or any(n.onset_qn < t < n.onset_qn+n.duration_qn for n in original.notes for t in (a,mid,b)):
            raise ValueError("bar swap crosses a sustained note")
        for k, p in patches.items():
            expected = before[k].onset_qn + (length if before[k].onset_qn < mid else -length)
            if p.written_fields != {"onset_qn"} or after[k].onset_qn != expected:
                raise ValueError("incorrect bar permutation")
        def observed_pattern(lo, hi):
            return sorted((n.track_id, n.pitch, n.onset_qn-lo, n.duration_qn, n.velocity,
                           str(n.staff), str(n.voice)) for n in original.notes if lo <= n.onset_qn < hi)
        if observed_pattern(a,mid) == observed_pattern(mid,b):
            raise ValueError("identical bars have no structural effect")
    else:
        raise ValueError("unknown intervention kind")
    return dict(id=candidate["id"], passed=True, kind=kind, changed_notes=len(patches),
                removed_notes=len(removed), quality_preference="unreviewed",
                primary_aspect_is_hypothesis=True)
