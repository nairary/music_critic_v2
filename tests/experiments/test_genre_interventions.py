from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from music_critic.data import RationalTime as Q, dumps_piece, load_piece, validate_or_raise
from music_critic.experiments.genre_interventions import build_genre_interventions, verify_genre_variant
from music_critic.experiments.music_interventions import (
    Intervention, InterventionError, NotePatch, apply_interventions, check_compatibility,
)


@pytest.fixture
def ensemble():
    p = load_piece(Path(__file__).parents[1] / "fixtures/data/canonical_piece_v2.json")
    bars = tuple(replace(p.bars[1], bar_id=f"bar:{i}", index=i, start_qn=Q(4*i), display_number=str(i+1)) for i in range(4))
    beats = tuple(replace(p.beats[1], beat_id=f"beat:{i}", bar_id=bars[i//4].bar_id,
                          start_qn=Q(i), index_in_bar=i%4, position_in_bar_qn=Q(i%4),
                          is_downbeat=i%4==0, strength=1.0 if i%4==0 else .5) for i in range(16))
    pitched, drum = p.notes[0], p.notes[2]
    notes, roles = [], {k:set() for k in ["melody", "accompaniment", "bass", "drums"]}
    for i in range(16):
        for role, pitch, template in [("melody", [72,74,76,77,79,77,76,74][i%8], pitched),
                                       ("accompaniment", 60+i//4, pitched),
                                       ("bass", 48+i//4, pitched),
                                       ("drums", 36 if i%2==0 else 38, drum)]:
            k = f"note:{role}-{i}"
            notes.append(replace(template, note_id=k, pitch=pitch, onset_qn=Q(i), duration_qn=Q(1,4),
                                 source_onset_ticks=None, source_duration_ticks=None,
                                 spelling_step=None, spelling_alter=None))
            roles[role].add(k)
    order={t.track_id:i for i,t in enumerate(p.tracks)}
    notes.sort(key=lambda n:(n.onset_qn,order[n.track_id],n.pitch,n.duration_qn,n.note_id))
    p = replace(p, duration_qn=Q(16), notes=tuple(notes), bars=bars, beats=beats, annotations=(), targets=(), quality_flags=(), split="train")
    validate_or_raise(p)
    return p, roles


def materialize(p, c):
    return apply_interventions(p, (Intervention(c["id"], c["patches"], c["context"]),)).piece


@pytest.mark.parametrize("genre", ["classical", "rock", "electronic", "hiphop"])
def test_all_multi_note_operators_have_exact_verified_effects(ensemble, genre):
    p, roles = ensemble
    original = dumps_piece(p)
    roles_before = deepcopy(roles)
    candidates, omitted = build_genre_interventions(p, roles, genre)
    assert {c["id"] for c in candidates} == {"H1","H2","R1","R2","M1","M2","A1","S1"}
    assert not omitted
    for c in candidates:
        result = materialize(p, c)
        report = verify_genre_variant(p, result, c, roles)
        assert report["passed"] and report["quality_preference"] == "unreviewed"
        assert report["changed_notes"] >= (2 if c["id"] == "R2" else 4)
    assert dumps_piece(p) == original
    assert roles == roles_before


@pytest.mark.parametrize("genre,expected", [("classical","accompaniment"),("rock","drums"),("electronic","drums"),("hiphop","bass")])
def test_genre_changes_observed_rhythm_role_not_pitch_or_duration(ensemble, genre, expected):
    p, roles = ensemble
    c = next(c for c in build_genre_interventions(p, roles, genre)[0] if c["id"] == "R1")
    assert c["check"]["role"] == expected
    changed = materialize(p,c)
    before = {n.note_id:n for n in p.notes}
    after = {n.note_id:n for n in changed.notes}
    for patch in c["patches"]:
        assert patch.note_id in roles[expected] or (expected == "accompaniment" and patch.note_id in roles["bass"])
        a,b=before[patch.note_id],after[patch.note_id]
        assert b.onset_qn-a.onset_qn == Q(1,2)
        assert (a.pitch,a.duration_qn) == (b.pitch,b.duration_qn)


def test_bass_may_also_belong_to_accompaniment_without_duplicate_edits(ensemble):
    p,roles=ensemble
    roles["accompaniment"] |= roles["bass"]
    cs,_=build_genre_interventions(p,roles,"rock")
    c=next(c for c in cs if c["id"]=="H1")
    assert len(c["patches"])==32
    assert len({patch.note_id for patch in c["patches"]})==32
    assert verify_genre_variant(p,materialize(p,c),c,roles)["passed"]


@pytest.mark.parametrize("issue", ["unknown_key","unknown_id","list_instead_set","melody_overlap","drums_overlap","pitched_drum","drum_pitched"])
def test_ambiguous_or_unobserved_roles_fail_closed(ensemble,issue):
    p,r=ensemble
    if issue=="unknown_key":r["lead"] = set()
    if issue=="unknown_id":r["bass"].add("note:absent")
    if issue=="list_instead_set":r["bass"] = list(r["bass"])
    if issue=="melody_overlap":r["accompaniment"].add(next(iter(r["melody"])))
    if issue=="drums_overlap":r["bass"].add(next(iter(r["drums"])))
    if issue=="pitched_drum":r["drums"] = set();r["melody"].add("note:drums-0")
    if issue=="drum_pitched":r["melody"] = set();r["drums"].add("note:melody-0")
    with pytest.raises(ValueError):build_genre_interventions(p,r,"rock")


def test_unknown_genre_is_not_silently_mapped(ensemble):
    with pytest.raises(ValueError,match="genre"):build_genre_interventions(*ensemble,"jazz")


def test_missing_roles_omit_instead_of_inventing_melody_or_drums(ensemble):
    p,_=ensemble
    cs,om=build_genre_interventions(p,{},"rock")
    assert {c["id"] for c in cs} == {"S1"}
    assert {c["id"] for c in om} == {"H1","H2","R1","R2","M1","M2","A1"}
    assert all(c["reason"] for c in om)


def test_pitch_ceiling_omits_h1_whole_role_instead_of_partial_shift(ensemble):
    p,r=ensemble
    ns=tuple(replace(n,pitch=127) if n.note_id=="note:accompaniment-0" else n for n in p.notes)
    order={t.track_id:i for i,t in enumerate(p.tracks)}
    p=replace(p,notes=tuple(sorted(ns,key=lambda n:(n.onset_qn,order[n.track_id],n.pitch,n.duration_qn,n.note_id))))
    cs,om=build_genre_interventions(p,r,"rock")
    assert "H1" not in {c["id"] for c in cs}
    assert any(c["id"]=="H1" for c in om)


def test_drum_removal_requires_repeated_pitch_and_position(ensemble):
    p,r=ensemble
    # Each bar receives a distinct kick/snare-like pitch; only first bar's36/38
    # remain named hits, so no adjacent source repetition establishes R2.
    p=replace(p,notes=tuple(replace(n,pitch=n.pitch+8*(int(n.onset_qn.to_fraction())//4)) if n.is_percussion else n for n in p.notes))
    cs,om=build_genre_interventions(p,r,"rock")
    assert "R2" not in {c["id"] for c in cs}
    assert any(c["id"]=="R2" for c in om)


def test_no_simultaneous_attacks_means_no_rhythm_delay_claim(ensemble):
    p,r=ensemble
    p=replace(p,notes=tuple(sorted((replace(n,onset_qn=n.onset_qn+Q(1,8)) if n.is_percussion else n for n in p.notes),key=lambda n:(n.onset_qn,1 if n.is_percussion else 0,n.pitch,n.duration_qn,n.note_id))))
    cs,om=build_genre_interventions(p,r,"rock")
    assert "R1" not in {c["id"] for c in cs}
    assert any(c["id"]=="R1" for c in om)


def test_classical_r2_removes_whole_alternating_attack_groups(ensemble):
    p,r=ensemble
    r["drums"] = set()
    cs,_=build_genre_interventions(p,r,"classical")
    c=next(c for c in cs if c["id"]=="R2")
    assert c["check"]["kind"]=="alternating_attack_removal"
    changed=materialize(p,c)
    assert verify_genre_variant(p,changed,c,r)["passed"]
    assert {patch.note_id for patch in c["patches"]}=={f"note:{role}-{i}" for role in ["accompaniment","bass"] for i in [0,2,4,6]}


def test_identical_bar_patterns_not_claimed_as_structural_change(ensemble):
    p,r=ensemble
    # Repeat first bar exactly in every bar; IDs alone do not count as change.
    first={n.note_id.rsplit("-",1)[0]+"-"+str(int(n.onset_qn.to_fraction())%4):n for n in p.notes if n.onset_qn<Q(4)}
    ns=[]
    for n in p.notes:
        key=n.note_id.rsplit("-",1)[0]+"-"+str(int(n.onset_qn.to_fraction())%4)
        ns.append(replace(n,pitch=first[key].pitch))
    p=replace(p,notes=tuple(ns))
    cs,om=build_genre_interventions(p,r,"rock")
    assert "S1" not in {c["id"] for c in cs}
    assert any(c["id"]=="S1" for c in om)


def test_sustained_notes_block_structure_and_full_dropout(ensemble):
    p,r=ensemble
    ns=tuple(replace(n,duration_qn=Q(2)) if n.note_id.startswith("note:bass-") and int(n.onset_qn.to_fraction()) in [3,7,11] else n for n in p.notes)
    p=replace(p,notes=ns)
    cs,om=build_genre_interventions(p,r,"hiphop")
    assert "S1" not in {c["id"] for c in cs}
    # It may explicitly fall back to accompaniment; never claims all bass gone.
    a=next((c for c in cs if c["id"]=="A1"),None)
    assert (a is not None and a["check"]["role"]=="accompaniment") or any(c["id"]=="A1" for c in om)


def test_non_monophonic_designated_line_does_not_get_fake_contour_proof(ensemble):
    p,r=ensemble
    # Assign existing accompaniment notes to melody: every attack becomes a
    # dyad, and M1's singleton/contiguous support must fail.
    r["melody"] |= r["accompaniment"]
    r["accompaniment"] = set()
    cs,om=build_genre_interventions(p,r,"rock")
    assert "M1" not in {c["id"] for c in cs}
    assert any(c["id"]=="M1" for c in om)


@pytest.mark.parametrize("ident",["H1","H2","R1","R2","M1","M2","A1","S1"])
def test_verifier_rejects_unrelated_extra_note_edit(ensemble,ident):
    p,r=ensemble
    c=next(c for c in build_genre_interventions(p,r,"rock")[0] if c["id"]==ident)
    result=materialize(p,c)
    edited={patch.note_id for patch in c["patches"]}
    untouched=next(n for n in result.notes if n.note_id not in edited)
    tampered=replace(result,notes=tuple(replace(n,velocity=n.velocity-1) if n.note_id==untouched.note_id else n for n in result.notes))
    with pytest.raises(ValueError,match="unplanned"):verify_genre_variant(p,tampered,c,r)


def test_verifier_rejects_mutated_check_even_when_pitch_patch_matches(ensemble):
    p,r=ensemble
    c=next(c for c in build_genre_interventions(p,r,"rock")[0] if c["id"]=="H2")
    result=materialize(p,c)
    c=deepcopy(c);c["check"]["semitones"]=5
    with pytest.raises(ValueError):verify_genre_variant(p,result,c,r)


def test_verifier_rejects_noop_and_duplicate_patch_support(ensemble):
    p,r=ensemble
    c=next(c for c in build_genre_interventions(p,r,"rock")[0] if c["id"]=="H1")
    result=materialize(p,c)
    bad=deepcopy(c);bad["patches"]+=(bad["patches"][0],)
    with pytest.raises(ValueError,match="duplicate"):verify_genre_variant(p,result,bad,r)
    before={n.note_id:n for n in p.notes}
    bad=deepcopy(c);bad["patches"]=tuple(NotePatch(x.note_id,pitch=before[x.note_id].pitch) for x in bad["patches"])
    with pytest.raises(ValueError,match="no-op"):verify_genre_variant(p,p,bad,r)


def test_actual_cross_operator_conflicts_remain_explicit(ensemble):
    p,r=ensemble
    cs={c["id"]:c for c in build_genre_interventions(p,r,"rock")[0]}
    a,b=(Intervention(cs[k]["id"],cs[k]["patches"],cs[k]["context"]) for k in ["H2","A1"])
    assert check_compatibility(a,b).status=="conflict"
    with pytest.raises(InterventionError,match="delete/edit"):apply_interventions(p,(a,b))


def test_deterministic_selection(ensemble):
    p,r=ensemble
    assert build_genre_interventions(p,r,"hiphop")==build_genre_interventions(p,r,"hiphop")


def test_electronic_without_drum_role_uses_declared_bass_against_lead(ensemble):
    p,r=ensemble
    r["drums"]=set()
    cs,om=build_genre_interventions(p,r,"electronic")
    c=next(c for c in cs if c["id"]=="R1")
    assert c["check"]["role"]=="bass"
    assert verify_genre_variant(p,materialize(p,c),c,r)["passed"]
    assert any(c["id"]=="R2" for c in om)


def test_zero_duration_grace_is_retained_without_false_sounding_dropout_failure(ensemble):
    p,r=ensemble
    grace=replace(next(n for n in p.notes if n.note_id=="note:bass-0"),note_id="note:bass-grace",onset_qn=Q(1,8),duration_qn=Q(0),is_grace=True)
    order={t.track_id:i for i,t in enumerate(p.tracks)}
    p=replace(p,notes=tuple(sorted((*p.notes,grace),key=lambda n:(n.onset_qn,order[n.track_id],n.pitch,n.duration_qn,n.note_id))))
    r["bass"].add(grace.note_id)
    c=next(c for c in build_genre_interventions(p,r,"rock")[0] if c["id"]=="A1")
    result=materialize(p,c)
    assert grace in result.notes
    assert verify_genre_variant(p,result,c,r)["passed"]


def test_verifier_rejects_support_relabelled_as_non_bar_window(ensemble):
    p,r=ensemble
    c=next(c for c in build_genre_interventions(p,r,"rock")[0] if c["id"]=="H2")
    result=materialize(p,c)
    bad=deepcopy(c);bad["check"]["window"]=[[0,1],[7,1]]
    with pytest.raises(ValueError,match="metric bars"):verify_genre_variant(p,result,bad,r)


def test_verifier_rejects_nonconsecutive_contour_order(ensemble):
    p,r=ensemble
    c=next(c for c in build_genre_interventions(p,r,"rock")[0] if c["id"]=="M1")
    result=materialize(p,c)
    bad=deepcopy(c);bad["check"]["ordered_ids"].reverse()
    with pytest.raises(ValueError,match="consecutive"):verify_genre_variant(p,result,bad,r)
