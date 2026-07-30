from __future__ import annotations

from collections import defaultdict

import pytest

from music_critic.data import RationalTime, validate_piece
from music_critic.graph import build_raw_graph, graph_fingerprint
from music_critic.ssl.bounded_fixture import (
    build_phase7a_bounded_fixture,
)
from music_critic.ssl.hierarchy_fixture import (
    BAR_CONTAINS_NOTE_EDGE,
    BAR_CONTAINS_ONSET_EDGE,
    BEAT_CONTAINS_ONSET_EDGE,
    NOTE_ACTIVE_AT_BEAT_EDGE,
    ONSET_STARTS_NOTE_EDGE,
    PHASE8A_BOUND_PHASE7A_FIXTURE_FINGERPRINT,
    PHASE8A_HIERARCHY_DATASET_ID,
    PHASE8A_HIERARCHY_FIXTURE_CONTRACT_VERSION,
    PHASE8A_HIERARCHY_FIXTURE_POLICY,
    PHASE8A_HIERARCHY_ORACLE_PIECE_ID,
    PHASE8A_HIERARCHY_ORACLE_SOURCE_GROUP_ID,
    PHASE8A_HIERARCHY_POLICY_ORACLES,
    PHASE8A_ORACLE_BAR_CONTAINS_NOTE,
    PHASE8A_ORACLE_BAR_CONTAINS_ONSET,
    PHASE8A_ORACLE_BEAT_CONTAINS_ONSET,
    PHASE8A_ORACLE_NOTE_ACTIVE_AT_BEAT,
    PHASE8A_ORACLE_ONSET_STARTS_NOTE,
    PHASE8A_ORACLE_TRACK_CONTAINS_NOTE,
    TRACK_CONTAINS_NOTE_EDGE,
    Phase8AHierarchyFixture,
    build_phase8a_hierarchy_fixture,
    build_phase8a_hierarchy_oracle_piece,
)


@pytest.fixture(scope="module")
def hierarchy_fixture() -> Phase8AHierarchyFixture:
    return build_phase8a_hierarchy_fixture()


def _pairs(graph, edge_type) -> tuple[tuple[int, int], ...]:
    return tuple(
        (int(source), int(destination))
        for source, destination in graph[edge_type].edge_index.t().tolist()
    )


def _destinations(
    pairs: tuple[tuple[int, int], ...],
) -> dict[int, tuple[int, ...]]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for source, destination in pairs:
        grouped[source].append(destination)
    return {
        source: tuple(destinations)
        for source, destinations in grouped.items()
    }


def test_wrapper_is_versioned_deterministic_and_exact(
    hierarchy_fixture: Phase8AHierarchyFixture,
) -> None:
    repeated = build_phase8a_hierarchy_fixture()

    assert hierarchy_fixture.contract_version == (
        PHASE8A_HIERARCHY_FIXTURE_CONTRACT_VERSION
    ) == "1.0.0"
    assert hierarchy_fixture.policy == PHASE8A_HIERARCHY_FIXTURE_POLICY
    assert repeated.composition_payload() == (
        hierarchy_fixture.composition_payload()
    )
    assert repeated.fingerprint_bundle() == (
        hierarchy_fixture.fingerprint_bundle()
    )
    assert hierarchy_fixture.fingerprint_bundle() == {
        "kind": "phase8a_hierarchy_bounded",
        "hierarchy_fixture_fingerprint": (
            "ffd0d4c7db80323b8f1f8d72c1e4b7e530151c1b95dd68033e1a30273dd98a1b"
        ),
        "phase7a_base_fixture_fingerprint": (
            "9f959d91d6805101983711511abcf89450e24b1886417632ea37fd0dc96ba922"
        ),
        "supplemental_oracle_fingerprint": (
            "75885a66c8f131711650c20ba7180033f2c8074dbffff82fd6021c8aef1e9359"
        ),
        "supplemental_raw_graph_fingerprint": (
            "8061c20f6394b3689a179d4d3ba7f3e418071b14a08bf4183cc8f22f1975d18c"
        ),
    }
    assert hierarchy_fixture.oracle_composition.canonical_piece_fingerprint == (
        "c92c2a16b14b224c88227e7922f1300dcbfb14230d64a8623e08f65d85c1ea90"
    )


def test_base_plus_supplement_counts_and_splits_are_exact(
    hierarchy_fixture: Phase8AHierarchyFixture,
) -> None:
    assert hierarchy_fixture.count_summary() == {
        "split": "all",
        "piece_count": 6,
        "track_count": 14,
        "bar_count": 15,
        "beat_count": 60,
        "onset_count": 42,
        "note_count": 93,
        "track_bar_cell_count": 34,
        "nonempty_track_bar_cell_count": 34,
        "polyphonic_onset_count": 39,
        "multi_onset_beat_count": 1,
        "cross_bar_sustained_note_count": 1,
    }
    assert hierarchy_fixture.count_summary("train") == {
        "split": "train",
        "piece_count": 4,
        "track_count": 9,
        "bar_count": 10,
        "beat_count": 40,
        "onset_count": 27,
        "note_count": 57,
        "track_bar_cell_count": 22,
        "nonempty_track_bar_cell_count": 22,
        "polyphonic_onset_count": 24,
        "multi_onset_beat_count": 1,
        "cross_bar_sustained_note_count": 1,
    }
    assert hierarchy_fixture.count_summary("validation") == {
        "split": "validation",
        "piece_count": 2,
        "track_count": 5,
        "bar_count": 5,
        "beat_count": 20,
        "onset_count": 15,
        "note_count": 36,
        "track_bar_cell_count": 12,
        "nonempty_track_bar_cell_count": 12,
        "polyphonic_onset_count": 15,
        "multi_onset_beat_count": 0,
        "cross_bar_sustained_note_count": 0,
    }

    train_identities = set(hierarchy_fixture.identities("train"))
    validation_identities = set(
        hierarchy_fixture.identities("validation")
    )
    train_groups = {
        piece.source_group_id for piece in hierarchy_fixture.train_pieces
    }
    validation_groups = {
        piece.source_group_id
        for piece in hierarchy_fixture.validation_pieces
    }
    assert train_identities.isdisjoint(validation_identities)
    assert train_groups.isdisjoint(validation_groups)
    assert (
        PHASE8A_HIERARCHY_DATASET_ID,
        PHASE8A_HIERARCHY_ORACLE_PIECE_ID,
    ) in train_identities
    assert (
        PHASE8A_HIERARCHY_DATASET_ID,
        PHASE8A_HIERARCHY_ORACLE_PIECE_ID,
    ) not in validation_identities
    assert len(hierarchy_fixture.raw_samples("train")) == 4
    assert len(hierarchy_fixture.raw_samples("validation")) == 2

    for piece in (
        hierarchy_fixture.train_pieces
        + hierarchy_fixture.validation_pieces
    ):
        assert piece.targets == ()
        assert piece.annotations == ()
        assert not validate_piece(piece).errors


def test_phase7a_fixture_and_fingerprints_remain_bit_exact(
    hierarchy_fixture: Phase8AHierarchyFixture,
) -> None:
    direct = build_phase7a_bounded_fixture()
    bound = hierarchy_fixture.phase7a_fixture

    assert direct == bound
    assert direct.train_pieces == hierarchy_fixture.train_pieces[:-1]
    assert direct.validation_pieces == hierarchy_fixture.validation_pieces
    assert direct.fixture_fingerprint == (
        PHASE8A_BOUND_PHASE7A_FIXTURE_FINGERPRINT
    )
    assert direct.fingerprint_bundle() == {
        "kind": "bounded",
        "bounded_fixture_fingerprint": (
            "9f959d91d6805101983711511abcf89450e24b1886417632ea37fd0dc96ba922"
        ),
        "split_fingerprint": (
            "89715a23b35ead69a1a314845414d01c6b56bdfbcc913e931719f17020bbef8d"
        ),
        "train_composition_fingerprint": (
            "218b51f2a212b5158b244bb22f8b28952ec79d8ecf9fc2ff5861dc24b9e770bf"
        ),
        "validation_composition_fingerprint": (
            "5730dfa44b90912cfca10bdacf489800054da8331f6a030e8dd7ab7cb461d7cd"
        ),
    }
    assert direct.count_summary()["piece_count"] == 5
    assert direct.count_summary()["note_count"] == 84


def test_oracle_piece_relations_are_hand_computed_exact(
    hierarchy_fixture: Phase8AHierarchyFixture,
) -> None:
    piece = hierarchy_fixture.supplemental_piece
    repeated_piece = build_phase8a_hierarchy_oracle_piece()
    graph = build_raw_graph(piece, assume_valid=True)

    assert piece == repeated_piece
    assert piece.dataset_name == PHASE8A_HIERARCHY_DATASET_ID
    assert piece.piece_id == PHASE8A_HIERARCHY_ORACLE_PIECE_ID
    assert piece.source_group_id == (
        PHASE8A_HIERARCHY_ORACLE_SOURCE_GROUP_ID
    )
    assert len(piece.tracks) == 2
    assert len(piece.bars) == 3
    assert len(piece.beats) == 12
    assert int(graph["onset"].num_nodes) == 6
    assert len(piece.notes) == 9
    assert _pairs(graph, ONSET_STARTS_NOTE_EDGE) == (
        PHASE8A_ORACLE_ONSET_STARTS_NOTE
    )
    assert _pairs(graph, BEAT_CONTAINS_ONSET_EDGE) == (
        PHASE8A_ORACLE_BEAT_CONTAINS_ONSET
    )
    assert _pairs(graph, BAR_CONTAINS_ONSET_EDGE) == (
        PHASE8A_ORACLE_BAR_CONTAINS_ONSET
    )
    assert _pairs(graph, BAR_CONTAINS_NOTE_EDGE) == (
        PHASE8A_ORACLE_BAR_CONTAINS_NOTE
    )
    assert _pairs(graph, TRACK_CONTAINS_NOTE_EDGE) == (
        PHASE8A_ORACLE_TRACK_CONTAINS_NOTE
    )
    assert _pairs(graph, NOTE_ACTIVE_AT_BEAT_EDGE) == (
        PHASE8A_ORACLE_NOTE_ACTIVE_AT_BEAT
    )
    assert graph_fingerprint(graph) == (
        hierarchy_fixture.oracle_composition.raw_graph_fingerprint
    )


def test_oracle_exercises_polyphony_multi_onset_start_anchor_and_intersection(
    hierarchy_fixture: Phase8AHierarchyFixture,
) -> None:
    piece = hierarchy_fixture.supplemental_piece
    graph = build_raw_graph(piece, assume_valid=True)
    onset_notes = _destinations(
        _pairs(graph, ONSET_STARTS_NOTE_EDGE)
    )
    beat_onsets = _destinations(
        _pairs(graph, BEAT_CONTAINS_ONSET_EDGE)
    )
    bar_onsets = _destinations(
        _pairs(graph, BAR_CONTAINS_ONSET_EDGE)
    )
    bar_notes = _destinations(_pairs(graph, BAR_CONTAINS_NOTE_EDGE))
    track_notes = _destinations(
        _pairs(graph, TRACK_CONTAINS_NOTE_EDGE)
    )
    active_beats = _destinations(
        _pairs(graph, NOTE_ACTIVE_AT_BEAT_EDGE)
    )

    assert onset_notes[0] == (0, 1)
    assert beat_onsets[0] == (0, 1)
    assert tuple(
        note
        for onset in beat_onsets[0]
        for note in onset_notes[onset]
    ) == (0, 1, 2)
    assert bar_onsets[1] == (3, 4)
    assert bar_notes[1] == (4, 5, 6)
    assert set(track_notes[1]) & set(bar_notes[1]) == {5, 6}

    sustained = piece.notes[3]
    assert sustained.onset_qn == RationalTime(3)
    assert sustained.onset_qn + sustained.duration_qn == RationalTime(5)
    assert piece.bars[1].start_qn == RationalTime(4)
    assert 3 in bar_notes[0]
    assert 3 not in bar_notes[1]
    assert active_beats[3] == (3, 4)
    assert 4 in active_beats[3]


def test_policy_oracles_bind_exact_descendants_and_collateral(
    hierarchy_fixture: Phase8AHierarchyFixture,
) -> None:
    assert hierarchy_fixture.oracle_composition.policy_oracles == (
        PHASE8A_HIERARCHY_POLICY_ORACLES
    )
    by_policy = {
        oracle.policy: oracle
        for oracle in PHASE8A_HIERARCHY_POLICY_ORACLES
    }

    onset = by_policy["onset_pitch_descendants"]
    assert onset.selected_local_unit_indices == (0,)
    assert onset.selected_local_note_descendants == (0, 1)
    assert onset.visible_pitched_note_count == 7
    assert len(onset.collateral_peer_note_indices) == 7
    assert onset.collateral_owner_track_indices == (0, 1)
    assert onset.realized_mask_fraction == (2, 9)

    beat = by_policy["beat_pitch_descendants"]
    assert beat.selected_local_onset_indices == (0, 1)
    assert beat.selected_local_note_descendants == (0, 1, 2)
    assert beat.visible_pitched_note_count == 6
    assert len(beat.collateral_peer_note_indices) == 6
    assert beat.realized_mask_fraction == (1, 3)

    bar = by_policy["contiguous_bar_pitch_span"]
    assert (bar.span_start_bar_index, bar.span_end_bar_index) == (1, 1)
    assert bar.selected_local_onset_indices == (3, 4)
    assert bar.selected_local_note_descendants == (4, 5, 6)
    assert 3 not in bar.selected_local_note_descendants
    assert 3 in bar.collateral_peer_note_indices
    assert bar.visible_pitched_note_count == 6
    assert bar.realized_mask_fraction == (1, 3)

    track_bar = by_policy["track_bar_pitch_span"]
    assert track_bar.selected_local_track_index == 1
    assert (
        track_bar.span_start_bar_index,
        track_bar.span_end_bar_index,
    ) == (1, 1)
    assert track_bar.selected_local_note_descendants == (5, 6)
    assert track_bar.collateral_peer_note_indices == (1, 8)
    assert track_bar.collateral_owner_track_indices == (1,)
    assert track_bar.visible_pitched_note_count == 7
    assert track_bar.realized_mask_fraction == (2, 9)
