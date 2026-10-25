"""Synthetic metadata only; none of these tests reads musical payloads."""
import json
from pathlib import Path

import pytest

from scripts.build_corpus400_catalog import aggregate_hook_families, build_candidates, file_facts, write_catalog


def clip(identity, native="TRAIN", artist="An Artist", song="A Song"):
    return {"clip_id": identity, "metadata": {"split": native,
        "hooktheory": {"artist": artist, "song": song}}}


def assignment(identity, split="train", component=None):
    return {"piece_id": "piece:hooktheory-" + identity, "split": split,
            "component_fingerprint": component or identity}


def test_all_family_clips_contribute_heldout_even_outside_join():
    families, by_clip = aggregate_hook_families(
        [clip("joined"), clip("not_in_pair", "TEST")],
        [assignment("joined"), assignment("not_in_pair", "test")])
    family = families[by_clip["joined"]]
    assert family["clip_count"] == 2
    assert family["native_splits"] == ["test", "train"]
    assert family["known_heldout"] is True
    assert family["all_native_current_train"] is False


def pdmx(path="./data/one.json", title="A Song", artist="An Artist"):
    return {"dataset": "pdmx", "id": path,
        "titles": [{"field": "song_name", "value": title}],
        "creators": [{"field": "artist_name", "value": artist}],
        "source_metadata": {"path": path, "genres": "rock", "license": "publicdomain"}}


def pair(creator="exact_normalized_creator", right=None):
    return {"creator_evidence": creator,
        "left": {"dataset": "hooktheory", "id": '["An Artist","A Song"]',
            "clip_ids": ["joined"], "titles": [{"value": "A Song"}],
            "creators": [{"value": "An Artist"}]}, "right": right or pdmx()}


def test_pdmx_gate_aggregates_family_and_keeps_title_only_untrusted():
    candidates, families = build_candidates(hook_clips=[clip("joined"), clip("elsewhere", "TEST")],
        assignments=[assignment("joined"), assignment("elsewhere", "test")],
        pairs=[pair(), pair("no_exact_creator_agreement", pdmx("./data/other.json", artist="Different Artist"))])
    by_source = {r["source_id"]: r for r in candidates}
    assert by_source["./data/one.json"]["admission"]["state"] == "quarantined"
    other = by_source["./data/other.json"]
    assert other["analytical_sources"][0]["identity_match"] == "title_only_unverified"
    assert other["admission"]["state"] == "supplemental_admission_required"
    assert other["admission"]["payload_read_allowed"] is False
    assert families


def test_versions_deduplicate_family_without_completeness_or_genre_claim():
    candidates, _ = build_candidates(extra_records=[pdmx(), pdmx("./data/two.json")])
    assert len(candidates) == 2
    assert len({r["work_family_id"] for r in candidates}) == 1
    assert {r["family_rank"] for r in candidates} == {1, 2}
    assert all(r["complete_for_declared_scope"] is False for r in candidates)
    assert candidates[0]["genre_candidates"][0]["status"] == "candidate_only"


def test_missing_identity_is_not_deduplicated_or_promoted_to_train():
    a, b = pdmx(), pdmx("./data/two.json")
    for row in (a, b):
        row["titles"] = []; row["creators"] = []
    candidates, _ = build_candidates(extra_records=[a, b])
    assert len({r["work_family_id"] for r in candidates}) == 2
    assert all(r["identity_confidence"] == "missing_title_or_creator" for r in candidates)


def test_inherited_train_never_overrides_known_current_conflict():
    source = {"dataset": "pop909", "id": "001", "titles": [{"value": "Song"}],
        "creators": [{"value": "Artist"}], "membership": {"native_split": "train", "current_split": "train"}}
    release = {"dataset_id": "pop909_cl", "piece_id": "piece:pop909-cl-001",
        "current_split": "train", "source_metadata": {"source_identity": "001"},
        "canonical_piece": {"path": "outputs/one.json", "sha256": "a" * 64, "size_bytes": 3},
        "bundle": {"path": "outputs/one.targets.json.gz", "sha256": "b" * 64, "size_bytes": 4}}
    candidates, _ = build_candidates(release_rows=[release], catalogs=[source],
        assignments=[{"piece_id": release["piece_id"], "split": "validation"}])
    row = candidates[0]
    assert row["admission"]["state"] == "quarantined"
    assert row["admission"]["inherited_receipts"]
    assert row["source_bindings"]["canonical_piece"] == release["canonical_piece"]


def test_stat_does_not_open_payload_and_rejects_escape(tmp_path, monkeypatch):
    payload = tmp_path / "piece.json"
    payload.write_text("This is deliberately not JSON")
    monkeypatch.setattr(Path, "read_bytes", lambda *a, **k: pytest.fail("payload read"))
    monkeypatch.setattr(Path, "read_text", lambda *a, **k: pytest.fail("payload read"))
    assert file_facts(tmp_path, {"path": "piece.json"})["exists"] is True
    assert file_facts(tmp_path, {"path": "../escape"})["status"] == "unsafe_path"


def test_output_is_immutable_and_summary_does_not_claim_quota(tmp_path):
    candidates, families = build_candidates(extra_records=[pdmx()])
    out = tmp_path / "catalog"
    summary = write_catalog(out, candidates, families, [])
    assert summary["complete_works"] == 0
    assert summary["quota_complete"] is False
    assert json.loads((out / "candidates.jsonl").read_text())["candidate_id"] == candidates[0]["candidate_id"]
    with pytest.raises(FileExistsError): write_catalog(out, candidates, families, [])


def test_component_heldout_is_rejected_without_same_title_clip():
    families, ids = aggregate_hook_families([clip("one")],
        [assignment("one", component="shared"), assignment("unrelated", "test", "shared")])
    assert families[ids["one"]]["known_heldout"] is True


def test_catalog_heldout_version_blocks_train_version_of_same_family():
    hidden = pdmx("./data/heldout.json")
    hidden["membership"] = {"current_split": "validation"}
    candidates, _ = build_candidates(catalogs=[hidden], extra_records=[pdmx()])
    assert candidates[0]["admission"]["state"] == "quarantined"
    assert "catalog_family_has_heldout" in candidates[0]["admission"]["reasons"]


def test_symlink_escape_is_not_statted_as_available(tmp_path):
    repo = tmp_path / "repo"; repo.mkdir()
    external = tmp_path / "external"; external.write_text("private")
    (repo / "payload").symlink_to(external)
    assert file_facts(repo, {"path": "payload"})["status"] == "unsafe_path"


def test_duplicate_clip_identity_is_rejected():
    with pytest.raises(ValueError, match="duplicate HookTheory"):
        aggregate_hook_families([clip("one"), clip("one")], [])


@pytest.mark.parametrize("lane", ["extra", "pair"])
def test_heldout_from_every_candidate_lane_blocks_other_family_versions(lane):
    heldout = pdmx("./data/heldout.json")
    heldout["membership"] = {"current_split": "test"}
    args = {"extra_records": [pdmx()]}
    if lane == "extra": args["extra_records"].append(heldout)
    else: args["pairs"] = [pair(right=heldout)]
    candidates, _ = build_candidates(**args)
    assert len(candidates) == 2
    assert all(r["admission"]["state"] == "quarantined" for r in candidates)


def test_duplicate_catalog_identity_cannot_erase_prior_test_evidence():
    heldout, train = pdmx(), pdmx()
    heldout["membership"] = {"current_split": "test"}
    train["membership"] = {"current_split": "train"}
    with pytest.raises(ValueError, match="duplicate catalog identity"):
        build_candidates(catalogs=[heldout, train])
