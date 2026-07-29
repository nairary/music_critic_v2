from __future__ import annotations

from copy import deepcopy
from fractions import Fraction

import pytest
import torch

from music_critic.data import validate_piece
from music_critic.graph import (
    MANDATORY_EDGE_TYPES,
    MANDATORY_NODE_TYPES,
    RAW_FEATURE_REGISTRY,
    graph_fingerprint,
)
from music_critic.ssl.bounded_fixture import (
    PHASE7A_BOUNDED_DATASET_ID,
    PHASE7A_BOUNDED_FIXTURE_CONTRACT_VERSION,
    PHASE7A_BOUNDED_FIXTURE_POLICY,
    PHASE7A_PITCH_MUTATION_CONTRACT_VERSION,
    PHASE7A_PITCH_MUTATION_POLICY,
    PHASE7A_PITCH_MUTATION_POLICY_FINGERPRINT,
    Phase7ABoundedFixture,
    Phase7ABoundedFixtureError,
    build_phase7a_bounded_fixture,
    mutate_piece_pitch_group,
)
from music_critic.ssl.data import collate_ssl_samples
from music_critic.ssl.engine import _runtime_mutation_source_binding
from music_critic.ssl.masking import (
    build_mask_plan,
    build_mask_plans_for_batch,
)


@pytest.fixture(scope="module")
def bounded_fixture() -> Phase7ABoundedFixture:
    return build_phase7a_bounded_fixture()


def _plans(
    fixture: Phase7ABoundedFixture,
    split: str,
):
    batch = collate_ssl_samples(fixture.raw_samples(split))
    plans = build_mask_plans_for_batch(
        batch,
        global_seed=42,
        epoch=0,
        encoder_view_index=0,
        requested_mask_rate=0.30,
        stage="train" if split == "train" else "validation",
    )
    return batch, plans


def _collateral_count(plans, node_type: str) -> int:
    return sum(
        len(mask.local_node_indices)
        for plan in plans
        for mask in plan.collateral_feature_masks
        if mask.node_type == node_type
    )


def test_fixture_is_deterministic_target_free_and_split_disjoint(
    bounded_fixture: Phase7ABoundedFixture,
) -> None:
    repeated = build_phase7a_bounded_fixture()
    train_identities = set(bounded_fixture.identities("train"))
    validation_identities = set(
        bounded_fixture.identities("validation")
    )
    train_groups = {
        piece.source_group_id for piece in bounded_fixture.train_pieces
    }
    validation_groups = {
        piece.source_group_id
        for piece in bounded_fixture.validation_pieces
    }

    assert bounded_fixture.contract_version == (
        PHASE7A_BOUNDED_FIXTURE_CONTRACT_VERSION
    ) == "1.0.0"
    assert bounded_fixture.policy == PHASE7A_BOUNDED_FIXTURE_POLICY
    assert len(train_identities) == len(bounded_fixture.train_pieces) == 3
    assert len(validation_identities) == (
        len(bounded_fixture.validation_pieces)
    ) == 2
    assert train_identities.isdisjoint(validation_identities)
    assert train_groups.isdisjoint(validation_groups)
    assert repeated.train_pieces == bounded_fixture.train_pieces
    assert repeated.validation_pieces == bounded_fixture.validation_pieces
    assert (
        repeated.composition_payload()
        == bounded_fixture.composition_payload()
    )
    assert (
        repeated.fingerprint_bundle()
        == bounded_fixture.fingerprint_bundle()
    )

    lookup = bounded_fixture.piece_lookup()
    assert set(lookup) == train_identities | validation_identities
    for identity, piece in lookup.items():
        assert bounded_fixture.piece_by_identity(*identity) is piece
        assert piece.dataset_name == PHASE7A_BOUNDED_DATASET_ID
        assert piece.split in {"train", "validation"}
        assert piece.annotations == ()
        assert piece.targets == ()
        assert not validate_piece(piece).errors


def test_fixture_fingerprints_and_composition_are_exact(
    bounded_fixture: Phase7ABoundedFixture,
) -> None:
    assert bounded_fixture.fixture_fingerprint == (
        "9f959d91d6805101983711511abcf89450e24b1886417632ea37fd0dc96ba922"
    )
    assert bounded_fixture.split_fingerprint == (
        "89715a23b35ead69a1a314845414d01c6b56bdfbcc913e931719f17020bbef8d"
    )
    assert bounded_fixture.train_composition_fingerprint == (
        "218b51f2a212b5158b244bb22f8b28952ec79d8ecf9fc2ff5861dc24b9e770bf"
    )
    assert bounded_fixture.validation_composition_fingerprint == (
        "5730dfa44b90912cfca10bdacf489800054da8331f6a030e8dd7ab7cb461d7cd"
    )
    assert bounded_fixture.fingerprint_bundle() == {
        "kind": "bounded",
        "bounded_fixture_fingerprint": (
            bounded_fixture.fixture_fingerprint
        ),
        "split_fingerprint": bounded_fixture.split_fingerprint,
        "train_composition_fingerprint": (
            bounded_fixture.train_composition_fingerprint
        ),
        "validation_composition_fingerprint": (
            bounded_fixture.validation_composition_fingerprint
        ),
    }
    assert bounded_fixture.count_summary() == {
        "split": "all",
        "piece_count": 5,
        "track_count": 12,
        "bar_count": 12,
        "beat_count": 48,
        "onset_count": 36,
        "note_count": 84,
        "pitch_min": 40,
        "pitch_max": 99,
        "distinct_pitch_class_count": 12,
        "distinct_octave_count": 6,
        "distinct_duration_count": 4,
        "distinct_position_in_bar_count": 3,
    }
    assert bounded_fixture.count_summary("train")["note_count"] == 48
    assert (
        bounded_fixture.count_summary("validation")["note_count"]
        == 36
    )

    samples = {
        (sample.dataset_id, sample.piece_id): sample
        for split in ("train", "validation")
        for sample in bounded_fixture.raw_samples(split)
    }
    for composition in bounded_fixture.composition:
        sample = samples[composition.identity]
        assert sample.raw_graph_fingerprint == (
            composition.raw_graph_fingerprint
        )
        assert graph_fingerprint(sample.raw_graph) == (
            composition.raw_graph_fingerprint
        )
        assert sample.raw_graph.raw_only is True


def test_fixture_has_multitrack_multibar_and_raw_feature_variety(
    bounded_fixture: Phase7ABoundedFixture,
) -> None:
    all_pieces = (
        bounded_fixture.train_pieces
        + bounded_fixture.validation_pieces
    )
    pitches = {
        note.pitch for piece in all_pieces for note in piece.notes
    }
    pitch_classes = {pitch % 12 for pitch in pitches}
    octaves = {pitch // 12 for pitch in pitches}
    durations = {
        note.duration_qn for piece in all_pieces for note in piece.notes
    }
    positions = {
        (note.onset_qn.num * 2 // note.onset_qn.den) % 8
        for piece in all_pieces
        for note in piece.notes
    }

    assert all(len(piece.tracks) >= 2 for piece in all_pieces)
    assert all(len(piece.bars) >= 2 for piece in all_pieces)
    assert all(len(piece.notes) >= 12 for piece in all_pieces)
    assert len(pitches) >= 30
    assert pitch_classes == set(range(12))
    assert len(octaves) == 6
    assert len(durations) == 4
    assert len(positions) == 3
    assert any(len(piece.tracks) == 3 for piece in all_pieces)
    assert any(len(piece.bars) == 3 for piece in all_pieces)


def test_mask_rate_and_primary_peer_owner_counts_are_exact(
    bounded_fixture: Phase7ABoundedFixture,
) -> None:
    train_batch, train_plans = _plans(bounded_fixture, "train")
    validation_batch, validation_plans = _plans(
        bounded_fixture,
        "validation",
    )

    assert train_batch.node_count == 114
    assert train_batch.edge_count == 740
    assert validation_batch.node_count == 83
    assert validation_batch.edge_count == 546
    assert tuple(
        plan.selected_local_node_indices for plan in train_plans
    ) == (
        (1, 11, 13, 14, 15),
        (0, 5, 10, 13, 14),
        (3, 10, 11),
    )
    assert tuple(
        plan.selected_local_node_indices for plan in validation_plans
    ) == (
        (2, 4, 6, 12, 13),
        (1, 4, 6, 11, 14),
    )
    assert tuple(plan.selected_count for plan in train_plans) == (5, 5, 3)
    assert tuple(plan.maskable_node_count for plan in train_plans) == (
        18,
        18,
        12,
    )
    assert tuple(plan.selected_count for plan in validation_plans) == (
        5,
        5,
    )
    assert all(
        plan.requested_mask_rate == 0.30
        for plan in train_plans + validation_plans
    )
    assert tuple(plan.realized_mask_rate for plan in train_plans) == (
        5 / 18,
        5 / 18,
        3 / 12,
    )
    assert tuple(
        plan.realized_mask_rate for plan in validation_plans
    ) == (5 / 18, 5 / 18)

    assert sum(plan.selected_count for plan in train_plans) == 13
    assert _collateral_count(train_plans, "note") == 35
    assert _collateral_count(train_plans, "track") == 7
    assert sum(plan.maskable_node_count for plan in train_plans) == 48
    assert Fraction(13, 48) == Fraction(
        sum(plan.selected_count for plan in train_plans),
        sum(plan.maskable_node_count for plan in train_plans),
    )

    assert sum(plan.selected_count for plan in validation_plans) == 10
    assert _collateral_count(validation_plans, "note") == 26
    assert _collateral_count(validation_plans, "track") == 5
    assert sum(plan.maskable_node_count for plan in validation_plans) == 36
    assert Fraction(10, 36) == Fraction(5, 18)
    assert all(
        0 < plan.realized_mask_rate < 1
        for plan in train_plans + validation_plans
    )

    assert tuple(plan.fingerprint for plan in train_plans) == (
        "f07c83364859e4f28b499d821985f9fb20c3be866c4d5e6f4bea237d3e16647c",
        "3b5c90bc0016a528cb840ee9c3a3214e52cbd2d0eafbad2aa6ded52e0729da5d",
        "42da3df81221b200303fd9184097e59bc7d4b85eca94a26ac7648f14bc120751",
    )
    assert tuple(plan.fingerprint for plan in validation_plans) == (
        "3f135a44278feff1d7af514895f924d796988521403b13573266cd2f7af823e8",
        "3d53144db3405b3d504d186ae6e6dfa4bf9f154afded82563f6a7575a41459db",
    )


def test_coherent_pitch_mutation_rebuilds_all_raw_dependencies(
    bounded_fixture: Phase7ABoundedFixture,
) -> None:
    piece = bounded_fixture.train_pieces[0]
    source_sample = bounded_fixture.raw_samples("train")[0]
    source_plan = build_mask_plan(
        source_sample.raw_graph,
        dataset_id=source_sample.dataset_id,
        piece_id=source_sample.piece_id,
        global_seed=42,
        epoch=0,
        requested_mask_rate=0.30,
        stage="train",
    )
    mutation = mutate_piece_pitch_group(
        piece,
        source_plan.selected_local_node_indices,
    )
    repeated = mutate_piece_pitch_group(
        piece,
        source_plan.selected_local_node_indices,
    )

    assert mutation.selected_local_node_indices == (
        1,
        11,
        13,
        14,
        15,
    )
    assert mutation.source_pitches == (74, 76, 90, 52, 67)
    assert mutation.mutated_pitches == (53, 51, 37, 75, 60)
    assert mutation.contract_version == (
        PHASE7A_PITCH_MUTATION_CONTRACT_VERSION
    )
    assert mutation.policy == PHASE7A_PITCH_MUTATION_POLICY
    assert mutation.policy_fingerprint == (
        PHASE7A_PITCH_MUTATION_POLICY_FINGERPRINT
    )
    assert mutation.policy_fingerprint == (
        "55c9c82b10153c21d158fb3287c3c01deea10b2a427b08d1266e1c89cdc32227"
    )
    assert mutation.mutation_instance_fingerprint == (
        "c2221c8c4b5bbcc25cb8575303a41963847393ea7b55f84a6b0d419fe7491f76"
    )
    assert mutation.source_raw_graph_fingerprint == (
        "34dc03a460d4204b429e938b2701c57b1a0da6356bd6b79038a36d921bf814d7"
    )
    assert mutation.mutated_raw_graph_fingerprint == (
        "cfcbf2e437246daeede793a4d146ee3dff68d6254796016239c80b64ad8b497c"
    )
    assert repeated.mutated_raw_graph_fingerprint == (
        mutation.mutated_raw_graph_fingerprint
    )
    assert repeated.mutation_instance_fingerprint == (
        mutation.mutation_instance_fingerprint
    )
    assert mutation.mutated_piece.annotations == ()
    assert mutation.mutated_piece.targets == ()
    assert not validate_piece(mutation.mutated_piece).errors
    assert mutation.changed_feature_slots == (
        ("track", "continuous", "mean_pitch"),
        ("track", "continuous", "pitch_std"),
        ("track", "continuous", "min_pitch"),
        ("track", "continuous", "max_pitch"),
        ("note", "categorical", "pitch"),
        ("note", "categorical", "pitch_class"),
        ("note", "categorical", "octave"),
        ("note", "continuous", "track_relative_pitch"),
    )
    for node_type in MANDATORY_NODE_TYPES:
        assert (
            mutation.source_raw_graph[node_type].entity_id
            == mutation.mutated_raw_graph[node_type].entity_id
        )
    for edge_type in MANDATORY_EDGE_TYPES:
        assert torch.equal(
            mutation.source_raw_graph[edge_type].edge_index,
            mutation.mutated_raw_graph[edge_type].edge_index,
        )

    note_continuous = RAW_FEATURE_REGISTRY.names(
        "note", "continuous"
    )
    onset_column = note_continuous.index("onset_qn")
    duration_column = note_continuous.index("duration_qn")
    relative_pitch_column = note_continuous.index(
        "track_relative_pitch"
    )
    assert torch.equal(
        mutation.source_raw_graph["note"].x_cont[
            :, (onset_column, duration_column)
        ],
        mutation.mutated_raw_graph["note"].x_cont[
            :, (onset_column, duration_column)
        ],
    )
    peer_rows = next(
        mask.local_node_indices
        for mask in source_plan.collateral_feature_masks
        if mask.node_type == "note"
    )
    assert peer_rows
    assert any(
        not torch.equal(
            mutation.source_raw_graph["note"].x_cont[
                row, relative_pitch_column
            ],
            mutation.mutated_raw_graph["note"].x_cont[
                row, relative_pitch_column
            ],
        )
        for row in peer_rows
    )

    mutated_sample = mutation.raw_sample(mutated=True)
    mutated_plan = build_mask_plan(
        mutated_sample.raw_graph,
        dataset_id=mutated_sample.dataset_id,
        piece_id=mutated_sample.piece_id,
        global_seed=42,
        epoch=0,
        requested_mask_rate=0.30,
        stage="train",
    )
    assert mutated_plan == source_plan
    assert collate_ssl_samples((mutated_sample,)).sample_count == 1


def test_runtime_source_binding_rejects_masked_feature_drift(
    bounded_fixture: Phase7ABoundedFixture,
) -> None:
    batch = collate_ssl_samples(
        bounded_fixture.raw_samples("train")
    )
    mutations = tuple(
        mutate_piece_pitch_group(piece, (0,))
        for piece in bounded_fixture.train_pieces
    )

    exact = _runtime_mutation_source_binding(batch, mutations)
    assert exact["passed"] is True
    assert all(row["passed"] is True for row in exact["per_sample"])

    drifted = deepcopy(batch)
    pitch_column = RAW_FEATURE_REGISTRY.names(
        "note",
        "categorical",
    ).index("pitch")
    original = int(
        drifted.raw_graph_batch["note"].x_cat[0, pitch_column]
    )
    drifted.raw_graph_batch["note"].x_cat[0, pitch_column] = (
        original + 1
    ) % 128

    rejected = _runtime_mutation_source_binding(
        drifted,
        mutations,
    )
    assert rejected["passed"] is False
    assert rejected["per_sample"][0]["identity_exact"] is True
    assert rejected["per_sample"][0]["fingerprint_exact"] is False


@pytest.mark.parametrize(
    "indices",
    ((), (0, 0), (-1,), (10_000,)),
)
def test_coherent_pitch_mutation_rejects_ambiguous_inputs(
    bounded_fixture: Phase7ABoundedFixture,
    indices: tuple[int, ...],
) -> None:
    with pytest.raises(
        Phase7ABoundedFixtureError,
        match="phase7a.fixture.mutation_",
    ):
        mutate_piece_pitch_group(
            bounded_fixture.train_pieces[0],
            indices,
        )


def test_fixture_lookup_and_summary_reject_unknown_values(
    bounded_fixture: Phase7ABoundedFixture,
) -> None:
    with pytest.raises(
        Phase7ABoundedFixtureError,
        match="identity_unknown",
    ):
        bounded_fixture.piece_by_identity("missing", "piece:missing")
    with pytest.raises(
        Phase7ABoundedFixtureError,
        match="split_invalid",
    ):
        bounded_fixture.count_summary("test")
