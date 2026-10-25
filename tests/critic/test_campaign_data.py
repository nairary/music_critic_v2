"""Synthetic preparation and explicit split/role contracts, never real files."""
from dataclasses import replace
from pathlib import Path
import json
from hashlib import sha256

import pytest

from music_critic.data import RationalTime as Q, dumps_piece, load_piece, validate_or_raise
from music_critic.critic.campaign_data import (PILOT_ROLE_POLICY, pilot_roles, CampaignSource,
    CampaignDataConfig, SOURCES_SCHEMA, build_campaign_dataset, load_campaign_dataset, inspect_manifest)


def ensemble(identifier="0"):
    p = load_piece(Path(__file__).parents[1] / "fixtures/data/canonical_piece_v2.json")
    bars = tuple(replace(p.bars[1], bar_id=f"bar:{i}", index=i, start_qn=Q(4 * i)) for i in range(4))
    beats = tuple(replace(p.beats[1], beat_id=f"beat:{i}", bar_id=f"bar:{i // 4}", start_qn=Q(i),
                          index_in_bar=i % 4, position_in_bar_qn=Q(i % 4), is_downbeat=i % 4 == 0)
                  for i in range(16))
    notes = []
    for i in range(16):
        for staff, pitch in ((1, (72, 74, 76, 77, 79, 77, 76, 74)[i % 8]),
                             (2, 60 + i // 4), (3, 48 + i // 4)):
            notes.append(replace(p.notes[0], note_id=f"note:{identifier}-{staff}-{i}", staff=staff, voice=1,
                                 pitch=pitch + int(identifier), onset_qn=Q(i), duration_qn=Q(1, 4),
                                 source_onset_ticks=None, source_duration_ticks=None))
    notes.sort(key=lambda n: (n.onset_qn, n.pitch, n.duration_qn, n.note_id))
    p = replace(p, piece_id=f"piece:synthetic-{identifier}", source_group_id=f"work-{identifier}",
                notes=tuple(notes), bars=bars, beats=beats, duration_qn=Q(16), split="train",
                annotations=(), targets=(), quality_flags=())
    validate_or_raise(p)
    return p


def test_pilot_roles_require_actual_staff_voice_and_keep_inference_provenance():
    p = ensemble()
    roles, provenance = pilot_roles(p)
    assert len(roles["melody"]) == 16 and len(roles["accompaniment"]) == 32 and len(roles["bass"]) == 16
    assert roles["bass"] <= roles["accompaniment"]
    assert provenance["policy"] == PILOT_ROLE_POLICY
    assert provenance["gold_roles"] is False
    assert not roles["drums"]
    with pytest.raises(ValueError, match="staff"):
        pilot_roles(replace(p, notes=tuple(replace(n, staff=None) for n in p.notes)))


def source(identifier):
    p = ensemble(str(identifier))
    roles, provenance = pilot_roles(p)
    return CampaignSource(str(identifier), p.source_group_id, f"component-{identifier}", p, roles, provenance)


def test_dataset_is_deterministic_disjoint_verified_and_excludes_arrangement():
    sources = tuple(source(i) for i in range(4))
    config = CampaignDataConfig(seed=17, head_dev_fraction=.25)
    result = build_campaign_dataset(sources, config)
    repeated = build_campaign_dataset(sources[::-1], config)
    assert result.fingerprint == repeated.fingerprint
    assert [p.pair_id for p in result.pairs] == [p.pair_id for p in repeated.pairs]
    assert len(result.pairs) == 28
    assert {p.split for p in result.pairs} == {"train", "head_dev"}
    works = {split: {p.work_id for p in result.pairs if p.split == split} for split in ("train", "head_dev")}
    assert works["train"].isdisjoint(works["head_dev"])
    for pair in result.pairs:
        assert pair.operator != "A1"
        assert pair.targets.provenance["quality_preference"] == "unreviewed"
        assert sum(pair.targets.pair_mask) == 1
        assert pair.targets.sample_ids == (pair.original.piece_id, pair.variant.piece_id)
        assert pair.targets.provenance["generation_roles"]["gold_roles"] is False


def test_known_work_component_and_raw_aliases_cannot_cross_split():
    a, b, c, d = (source(i) for i in range(4))
    alias = replace(a, source_id="alias", work_id="alias-work", component_id="alias-component",
                    piece=replace(a.piece, piece_id="piece:alias", source_group_id="alias-work"))
    b = replace(b, component_id=a.component_id)
    result = build_campaign_dataset((a, b, c, d, alias))
    assert result.split_assignments[a.source_id] == result.split_assignments[b.source_id] == result.split_assignments[alias.source_id]


@pytest.mark.parametrize("ambiguity", ("overlap", "grace"))
def test_melody_pairs_require_monophony_in_the_actual_operator_region(ambiguity):
    a, b = source(0), source(1)
    first_melody = min((n for n in a.piece.notes if n.note_id in a.roles["melody"]), key=lambda n: n.onset_qn)
    if ambiguity == "overlap":
        notes = tuple(replace(n, duration_qn=Q(3, 2)) if n.note_id == first_melody.note_id else n for n in a.piece.notes)
    else:
        grace = replace(first_melody, note_id="note:grace", duration_qn=Q(0), is_grace=True)
        notes = tuple(sorted((*a.piece.notes, grace), key=lambda n: (n.onset_qn, n.track_id, n.pitch, n.duration_qn, n.note_id)))
        a = replace(a, roles={**a.roles, "melody": a.roles["melody"] | {grace.note_id}})
    piece = replace(a.piece, notes=notes)
    result = build_campaign_dataset((replace(a, piece=piece), b))
    assert not any(p.source_id == a.source_id and p.operator.startswith("M") for p in result.pairs)
    assert any(p.source_id == a.source_id and p.operator.startswith("H") for p in result.pairs)
    assert sum(row.get("reason") == "designated_melody_not_monophonic_in_operator_support" for row in result.omissions) == 2


def test_raw_alias_fingerprint_ignores_note_id_tie_order():
    from music_critic.critic.campaign_data import observation_fingerprint
    p = ensemble()
    first = replace(p.notes[0], note_id="note:alias-a", velocity=40)
    second = replace(first, note_id="note:alias-b", velocity=100)
    key = lambda n: (n.onset_qn, n.track_id, n.pitch, n.duration_qn, n.note_id)
    a = replace(p, notes=tuple(sorted((first, second, *p.notes[1:]), key=key)))
    b = replace(p, notes=tuple(sorted((replace(first, note_id="note:alias-z"), second, *p.notes[1:]), key=key)))
    validate_or_raise(a); validate_or_raise(b)
    assert observation_fingerprint(a) == observation_fingerprint(b)


def test_unknown_role_provenance_and_nontrain_sources_fail_closed():
    a, b = source(0), source(1)
    with pytest.raises(ValueError, match="provenance"):
        build_campaign_dataset((replace(a, role_provenance={}), b))
    with pytest.raises(ValueError, match="split"):
        build_campaign_dataset((replace(a, source_split="test"), b))
    with pytest.raises(ValueError, match="bar budget"):
        build_campaign_dataset((a, b), CampaignDataConfig(max_bars=2))


def test_manifest_loader_is_bound_and_checks_all_splits_before_payloads(tmp_path):
    rows = []
    for i in range(2):
        src = source(i)
        raw = dumps_piece(src.piece).encode()
        (tmp_path / f"piece{i}.json").write_bytes(raw)
        rows.append({"source_id": src.source_id, "work_id": src.work_id, "component_id": src.component_id,
            "genre": src.genre, "source_split": "train", "roles": {k: sorted(v) for k, v in src.roles.items()},
            "role_provenance": src.role_provenance,
            "piece": {"path": f"piece{i}.json", "sha256": sha256(raw).hexdigest(), "size_bytes": len(raw)}})
    path = tmp_path / "sources.json"
    def config():
        raw = json.dumps({"schema": SOURCES_SCHEMA, "sources": rows}).encode()
        path.write_bytes(raw)
        return {"sources_manifest": str(path), "sources_manifest_sha256": sha256(raw).hexdigest()}
    settings = config()
    assert inspect_manifest(settings)["piece_payloads_accessed"] is False
    assert len(load_campaign_dataset(settings).pairs) == 14
    with pytest.raises(ValueError, match="manifest integrity"):
        load_campaign_dataset({**settings, "sources_manifest_sha256": "0" * 64})
    rows[0]["piece"]["path"] = "must-not-open.json"
    rows[1]["source_split"] = "test"
    with pytest.raises(ValueError, match="split"):
        load_campaign_dataset(config())


def test_source_clip_uses_exact_bars_and_rejects_unavailable_roles_without_guessing():
    from music_critic.critic.campaign_data import make_pilot_clip
    p = ensemble()
    clip, roles, provenance, receipt = make_pilot_clip(p, max_bars=2)
    assert len(clip.bars) == 2 and clip.duration_qn == Q(8)
    assert len(clip.notes) == 24 and set().union(*roles.values()) == {n.note_id for n in clip.notes}
    assert receipt["source_piece_id"] == p.piece_id and receipt["source_bar_start"] == 0
    assert provenance["gold_roles"] is False
    with pytest.raises(ValueError, match="eligible"):
        make_pilot_clip(replace(p, notes=tuple(replace(n, staff=None) for n in p.notes)))


def test_authenticated_train_admission_rejects_conflicting_splits_and_components():
    from music_critic.critic.campaign_data import _admit_bound_train_source
    p = ensemble()
    row = {"record_id": "dlc:fixture:0", "split": "train", "source_component_id": p.source_group_id,
           "piece": {"sha256": "0" * 64}}
    for split in (None, "train"):
        admitted, provenance = _admit_bound_train_source(replace(p, split=split), row, "1" * 64)
        assert admitted.split == "train"
        assert dict(provenance.details)["observed_embedded_split"] == split
    for split in ("validation", "test"):
        with pytest.raises(ValueError, match="split contradicts"):
            _admit_bound_train_source(replace(p, split=split), row, "1" * 64)
    with pytest.raises(ValueError, match="component"):
        _admit_bound_train_source(replace(p, source_group_id="other"), row, "1" * 64)


def test_archive_preparation_reads_only_authenticated_train_piece_members(tmp_path, monkeypatch):
    import io
    import tarfile
    from music_critic.critic.campaign_data import prepare_frozen_train_sources
    archive = tmp_path / "payload.tar.gz"
    records = []
    with tarfile.open(archive, "w:gz") as tar:
        for i in range(3):
            raw = dumps_piece(replace(ensemble(str(i)), split=None, source_group_id=f"component{i}")).encode()
            name = f"record{i}/piece.json"
            info = tarfile.TarInfo(name); info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))
            records.append({"record_id": f"dlc:fixture:{i}", "source_component_id": f"component{i}", "split": "train",
                "piece": {"path": f"/frozen/{name}", "sha256": sha256(raw).hexdigest(), "size_bytes": len(raw)}})
        raw = b"this is not a canonical payload and must never be opened"
        info = tarfile.TarInfo("validation/piece.json"); info.size = len(raw)
        tar.addfile(info, io.BytesIO(raw))
    records.append({"record_id": "dlc:validation", "source_component_id": "component-val", "split": "validation",
                    "piece": {"path": "/frozen/validation/piece.json", "sha256": "0" * 64, "size_bytes": 0}})
    def write_binding(name, value):
        raw = json.dumps(value).encode(); path = tmp_path / name; path.write_bytes(raw)
        return {"path": str(path), "sha256": sha256(raw).hexdigest(), "size_bytes": len(raw)}
    manifest = write_binding("input.json", {"records": records})
    receipt = write_binding("receipt.json", {"destination": "/frozen", "test_payload_packaged": False,
        "parts": [{"path": archive.name, "sha256": sha256(archive.read_bytes()).hexdigest(), "size_bytes": archive.stat().st_size}]})
    config = prepare_frozen_train_sources(manifest, receipt, tmp_path / "output", max_sources=2, max_inputs=3)
    summary = json.loads((tmp_path / "output" / "prepare-receipt.json").read_text())
    assert summary["train_piece_payloads_read"] == 3
    assert summary["validation_payloads_read"] == summary["test_payloads_read"] == 0
    assert summary["selected_sources"] == 2
    dataset = load_campaign_dataset(config)
    assert len(dataset.pairs) == 14
    assert all(any(p.source == "music_critic.critic.frozen_train_admission"
                   and ("observed_embedded_split", None) in p.details
                   and ("assigned_split", "train") in p.details
                   for p in pair.original.provenance) for pair in dataset.pairs)
    with pytest.raises(FileExistsError):
        prepare_frozen_train_sources(manifest, receipt, tmp_path / "output", max_sources=2)
    import music_critic.critic.campaign_data as module
    clock = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    def refused(*args, **kwargs):
        clock[0] += 4000
        raise ValueError("fixture has unavailable roles")
    monkeypatch.setattr(module, "make_pilot_clip", refused)
    with pytest.raises(RuntimeError, match="two-hour"):
        prepare_frozen_train_sources(manifest, receipt, tmp_path / "refused", max_sources=2, max_inputs=3)
