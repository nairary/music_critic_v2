"""Synthetic authored-package admission; never campaign data or TEST."""
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

from music_critic.data import RationalTime as Q, dumps_piece, load_piece
from music_critic.experiments.music_interventions import Intervention, NotePatch, apply_interventions
from music_critic.critic.authored_data import load_authored_dataset


def packet(tmp_path, *, aspect="structure", patches=None, reviewed=True):
    original = load_piece(Path(__file__).parents[1] / "fixtures/data/canonical_piece_v2.json")
    original = replace(original, piece_id="piece:authored-fixture", source_group_id="work:fixture", split="train", targets=(), annotations=(), quality_flags=())
    patch = NotePatch(original.notes[0].note_id, pitch=original.notes[0].pitch + 1)
    interventions = patches or (Intervention("S1", (patch,)),)
    variant = apply_interventions(original, tuple(interventions)).piece

    def write(name, payload, raw=False):
        data = payload.encode() if raw else (json.dumps(payload, sort_keys=True) + "\n").encode()
        (tmp_path / name).write_bytes(data)
        return {"path": name, "sha256": sha256(data).hexdigest(), "size_bytes": len(data)}

    source = write("original.json", dumps_piece(original), raw=True)
    changed = write("variant.json", dumps_piece(variant), raw=True)
    variants = []
    for intervention in interventions:
        rows = []
        for item in intervention.patches:
            row = {"note_id": item.note_id}
            for name in item.written_fields:
                value = getattr(item, name)
                row[name] = [value.num, value.den] if isinstance(value, Q) else value
            rows.append(row)
        variants.append({"id": intervention.id, "aspect": aspect, "patches": rows, "context_note_ids": list(intervention.context_note_ids), "rationale": "Synthetic declared intervention", "uncertainty": "No human preference", "quality_preference": "unknown"})
    parents = [v["id"] for v in variants]
    selected = parents[0] if len(parents) == 1 else "C1"
    plan = write("plan.json", {"variants": variants, "combinations": [] if len(parents) == 1 else [{"id": "C1", "parents": parents, "rationale": "Synthetic combination", "uncertainty": "Unknown preference"}]})
    row = {"pair_id": "pair:fixture", "work_id": "work:fixture", "component_id": "component:fixture", "split": "train", "genre": "classical", "edit_count": len(parents), "source_scope": "train", "original": source, "variant": changed, "plan": plan, "variant_id": selected}
    if reviewed:
        row["direction_review"] = write("review.json", {"schema": "AuthoredDirectionReview@1.0.0", "evidence_kind": "weak_author_intent", "status": "accepted", "scope": "single" if len(parents) == 1 else "joint_combination", "work_id": row["work_id"], "aspect": aspect, "parents": parents, "original_sha256": source["sha256"], "variant_sha256": changed["sha256"], "plan_sha256": plan["sha256"], "direction": 1, "reviewer": "synthetic-test-author", "rationale": "Explicit synthetic direction, not a measured listener preference", "uncertainty": "Human preference unknown", "human_preference": "unknown"})
    manifest = {"schema": "AuthoredPairManifest@1.0.0", "source_id": "synthetic-authored-source", "split_id": "synthetic-disjoint-split", "target_version": "AuthoredGlobalPairTargets@1.0.0", "pairs": [row]}
    binding = write("manifest.json", manifest)
    return manifest, tmp_path / "manifest.json", binding["sha256"], write


def test_global_structure_pair_replays_without_old_local_bar_swap(tmp_path):
    _, path, checksum, _ = packet(tmp_path)
    data = load_authored_dataset(path, checksum)
    pair, = data.pairs
    assert pair.edit_count == 1 and pair.genre == "classical"
    assert pair.targets.pair_mask == (False, False, False, True)
    assert pair.targets.preference == (0, 0, 0, 1)
    assert pair.targets.local_targets == ({}, {})
    assert pair.targets.provenance["human_preference"] == "unknown"
    assert pair.original.targets == pair.variant.targets == ()
    assert data.split_assignments == {"work:fixture": "train"}
    assert data.fingerprint == load_authored_dataset(path, checksum).fingerprint


def test_combination_merges_different_fields_on_one_note_and_requires_joint_review(tmp_path):
    source = load_piece(Path(__file__).parents[1] / "fixtures/data/canonical_piece_v2.json")
    note = source.notes[0]
    edits = (Intervention("H1", (NotePatch(note.note_id, pitch=note.pitch + 1),)),
             Intervention("H2", (NotePatch(note.note_id, duration_qn=Q(1, 2)),)))
    manifest, path, checksum, write = packet(tmp_path, aspect="harmony", patches=edits)
    pair, = load_authored_dataset(path, checksum).pairs
    assert pair.edit_count == 2
    assert pair.targets.pair_mask == (True, False, False, False)
    assert pair.targets.provenance["changed_note_ids"] == (note.note_id,)
    changed = next(n for n in pair.variant.notes if n.note_id == note.note_id)
    assert changed.pitch == note.pitch + 1 and changed.duration_qn == Q(1, 2)
    review = json.loads((tmp_path / "review.json").read_text())
    review["scope"] = "single"
    manifest["pairs"][0]["direction_review"] = write("review.json", review)
    check = write("manifest.json", manifest)
    with pytest.raises(ValueError, match="scope mismatch"):
        load_authored_dataset(path, check["sha256"])


@pytest.mark.parametrize("reviewed", (False, True))
def test_unavailable_direction_is_retained_masked_never_an_implicit_negative(tmp_path, reviewed):
    manifest, path, checksum, write = packet(tmp_path, reviewed=reviewed)
    if reviewed:
        review = json.loads((tmp_path / "review.json").read_text())
        review.update(status="unknown", direction=0, uncertainty="unknown")
        manifest["pairs"][0]["direction_review"] = write("review.json", review)
        checksum = write("manifest.json", manifest)["sha256"]
    data = load_authored_dataset(path, checksum)
    pair, = data.pairs
    assert pair.targets.pair_mask == (False,) * 4 and pair.targets.preference == (0,) * 4
    assert data.omissions[0]["pair_retained"] is True
    assert data.source_summary["available_aspect_targets"]["structure"] == 0


def test_combination_without_direction_receipt_never_inherits_single_intents(tmp_path):
    source = load_piece(Path(__file__).parents[1] / "fixtures/data/canonical_piece_v2.json")
    edits = tuple(Intervention(f"H{i}", (NotePatch(n.note_id, pitch=n.pitch + 1),))
                  for i, n in enumerate(source.notes[:3], 1))
    _, path, checksum, _ = packet(tmp_path, aspect="harmony", patches=edits, reviewed=False)
    pair, = load_authored_dataset(path, checksum).pairs
    assert pair.edit_count == 3 and not any(pair.targets.pair_mask)


def test_deletion_remains_an_exact_global_pair_without_local_labels(tmp_path):
    source = load_piece(Path(__file__).parents[1] / "fixtures/data/canonical_piece_v2.json")
    edit = Intervention("S1", (NotePatch(source.notes[0].note_id, delete=True),))
    _, path, checksum, _ = packet(tmp_path, patches=(edit,))
    pair, = load_authored_dataset(path, checksum).pairs
    assert len(pair.variant.notes) == len(pair.original.notes) - 1
    assert pair.targets.provenance["deleted_note_ids"] == (source.notes[0].note_id,)
    assert pair.targets.local_targets == ({}, {})


def test_mixed_aspect_combination_is_not_admitted_as_single_aspect(tmp_path):
    source = load_piece(Path(__file__).parents[1] / "fixtures/data/canonical_piece_v2.json")
    edits = tuple(Intervention(f"H{i}", (NotePatch(n.note_id, pitch=n.pitch + 1),))
                  for i, n in enumerate(source.notes[:2], 1))
    manifest, path, _, write = packet(tmp_path, aspect="harmony", patches=edits)
    plan = json.loads((tmp_path / "plan.json").read_text())
    plan["variants"][1]["aspect"] = "melody"
    manifest["pairs"][0]["plan"] = write("plan.json", plan)
    checksum = write("manifest.json", manifest)["sha256"]
    with pytest.raises(ValueError, match="same known aspect"):
        load_authored_dataset(path, checksum)


@pytest.mark.parametrize("field,value", (("split", "final_holdout"), ("source_scope", "test")))
def test_forbidden_row_rejected_before_any_pair_payload_is_opened(tmp_path, field, value):
    manifest, path, _, write = packet(tmp_path)
    bad = {**manifest["pairs"][0], "pair_id": "pair:forbidden", field: value}
    bad["original"] = {**bad["original"], "path": "must-not-exist.json"}
    manifest["pairs"].append(bad)
    checksum = write("manifest.json", manifest)["sha256"]
    (tmp_path / "original.json").unlink()
    with pytest.raises(ValueError, match="forbidden"):
        load_authored_dataset(path, checksum)


def test_declared_work_or_component_cannot_cross_train_dev(tmp_path):
    manifest, path, _, write = packet(tmp_path)
    second = {**manifest["pairs"][0], "pair_id": "pair:alias", "work_id": "other-work", "split": "head_dev"}
    manifest["pairs"].append(second)
    checksum = write("manifest.json", manifest)["sha256"]
    with pytest.raises(ValueError, match="cross-split"):
        load_authored_dataset(path, checksum)


def test_rebound_but_unplanned_note_change_fails_exact_replay(tmp_path):
    manifest, path, _, write = packet(tmp_path)
    raw = json.loads((tmp_path / "variant.json").read_text())
    raw["notes"][-1]["velocity"] = 11
    manifest["pairs"][0]["variant"] = write("variant.json", raw)
    checksum = write("manifest.json", manifest)["sha256"]
    with pytest.raises(ValueError, match="exact replay"):
        load_authored_dataset(path, checksum)


def test_false_review_byte_binding_is_rejected(tmp_path):
    manifest, path, _, write = packet(tmp_path)
    review = json.loads((tmp_path / "review.json").read_text())
    review["original_sha256"] = "f" * 64
    manifest["pairs"][0]["direction_review"] = write("review.json", review)
    checksum = write("manifest.json", manifest)["sha256"]
    with pytest.raises(ValueError, match="binding/scope"):
        load_authored_dataset(path, checksum)


@pytest.mark.parametrize("field", ("manifest", "original", "plan", "direction_review"))
def test_file_bytes_are_bound_before_parsing(tmp_path, field):
    manifest, path, checksum, _ = packet(tmp_path)
    target = path if field == "manifest" else tmp_path / manifest["pairs"][0][field]["path"]
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError, match="integrity"):
        load_authored_dataset(path, checksum)


def test_exact_rational_fields_reject_floats(tmp_path):
    manifest, path, _, write = packet(tmp_path)
    plan = json.loads((tmp_path / "plan.json").read_text())
    plan["variants"][0]["patches"][0]["onset_qn"] = [0.5, 1]
    manifest["pairs"][0]["plan"] = write("plan.json", plan)
    checksum = write("manifest.json", manifest)["sha256"]
    with pytest.raises(ValueError, match="integer rational"):
        load_authored_dataset(path, checksum)


def test_path_traversal_and_symlink_escapes_are_forbidden(tmp_path):
    manifest, path, _, write = packet(tmp_path)
    manifest["pairs"][0]["original"]["path"] = "../outside.json"
    checksum = write("manifest.json", manifest)["sha256"]
    with pytest.raises(ValueError, match="unsafe relative path"):
        load_authored_dataset(path, checksum)


def test_pair_swap_preserves_masks_and_reverses_direction(tmp_path):
    _, path, checksum, _ = packet(tmp_path)
    target = load_authored_dataset(path, checksum).pairs[0].targets
    swapped = target.swapped()
    assert swapped.sample_ids == target.sample_ids[::-1]
    assert swapped.pair_mask == target.pair_mask and swapped.preference == (0, 0, 0, -1)
    assert swapped.swapped() == target
