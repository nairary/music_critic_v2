from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Batch

from music_critic.data import CanonicalPiece, load_piece
from music_critic.graph import (
    RAW_FEATURE_REGISTRY,
    build_raw_graph,
    graph_fingerprint,
)
from music_critic.ssl.contracts import (
    MASK_PLAN_CONTRACT_VERSION,
    CollateralFeatureMask,
    MaskPlan,
    SSLContractError,
)
from music_critic.ssl.field_registry import (
    MASKABLE_FIELD_REGISTRY_FINGERPRINT,
    MASKABLE_FIELD_REGISTRY_VERSION,
    NOTE_PITCH_GROUP,
    SSL_MASKABLE_FIELD_REGISTRY,
)
from music_critic.ssl.masking import (
    build_batched_mask_plans,
    build_mask_plan,
)
from music_critic.ssl.views import build_feature_mask_overlay


FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "data"
    / "canonical_piece_v2.json"
)


@pytest.fixture(scope="module")
def canonical_piece() -> CanonicalPiece:
    return load_piece(FIXTURE_PATH)


@pytest.fixture(scope="module")
def raw_graph(canonical_piece: CanonicalPiece):
    return build_raw_graph(canonical_piece)


def _plan(raw_graph, canonical_piece: CanonicalPiece, **overrides) -> MaskPlan:
    arguments = {
        "dataset_id": canonical_piece.dataset_name,
        "piece_id": canonical_piece.piece_id,
        "global_seed": 42,
        "epoch": 3,
        "encoder_view_index": 0,
        "requested_mask_rate": 0.5,
        "stage": "train",
    }
    arguments.update(overrides)
    return build_mask_plan(raw_graph, **arguments)


def _recreate_plan(plan: MaskPlan, **overrides) -> MaskPlan:
    arguments = {
        "mask_policy": plan.mask_policy,
        "mask_policy_version": plan.mask_policy_version,
        "dataset_id": plan.dataset_id,
        "piece_id": plan.piece_id,
        "stage": plan.stage,
        "epoch": plan.epoch,
        "encoder_view_index": plan.encoder_view_index,
        "selected_node_type": plan.selected_node_type,
        "selected_local_node_indices": plan.selected_local_node_indices,
        "primary_feature_group": plan.primary_feature_group,
        "collateral_feature_masks": plan.collateral_feature_masks,
        "requested_mask_rate": plan.requested_mask_rate,
        "maskable_node_count": plan.maskable_node_count,
        "realized_mask_rate": plan.realized_mask_rate,
        "global_seed": plan.global_seed,
        "stable_seed": plan.stable_seed,
        "stable_seed_sha256": plan.stable_seed_sha256,
    }
    arguments.update(overrides)
    return MaskPlan.create(**arguments)


def _collateral(plan: MaskPlan, node_type: str) -> CollateralFeatureMask:
    matches = tuple(
        mask
        for mask in plan.collateral_feature_masks
        if mask.node_type == node_type
    )
    assert len(matches) == 1
    return matches[0]


def test_same_inputs_produce_bit_exact_mask_plan(
    raw_graph,
    canonical_piece: CanonicalPiece,
) -> None:
    first = _plan(raw_graph, canonical_piece)
    second = _plan(raw_graph, canonical_piece)

    assert first == second
    assert first.to_dict() == second.to_dict()
    assert first.contract_version == MASK_PLAN_CONTRACT_VERSION == "1.0.0"
    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 64
    assert first.stable_seed == second.stable_seed
    assert first.stable_seed_sha256 == second.stable_seed_sha256
    assert first.sample_identity == (
        canonical_piece.dataset_name,
        canonical_piece.piece_id,
    )


def test_batch_order_does_not_change_per_sample_plans(
    raw_graph,
    canonical_piece: CanonicalPiece,
) -> None:
    other_piece = replace(
        canonical_piece,
        dataset_name="synthetic-other",
        piece_id="piece:synthetic-other",
        source_group_id="group:synthetic-other",
        quality_flags=(),
    )
    other_graph = build_raw_graph(other_piece)
    forward = build_batched_mask_plans(
        Batch.from_data_list([raw_graph, other_graph]),
        dataset_ids=(
            canonical_piece.dataset_name,
            other_piece.dataset_name,
        ),
        piece_ids=(canonical_piece.piece_id, other_piece.piece_id),
        global_seed=19,
        epoch=4,
        requested_mask_rate=0.5,
    )
    reverse = build_batched_mask_plans(
        Batch.from_data_list([other_graph, raw_graph]),
        dataset_ids=(
            other_piece.dataset_name,
            canonical_piece.dataset_name,
        ),
        piece_ids=(other_piece.piece_id, canonical_piece.piece_id),
        global_seed=19,
        epoch=4,
        requested_mask_rate=0.5,
    )

    assert {plan.sample_identity: plan for plan in forward} == {
        plan.sample_identity: plan for plan in reverse
    }


def test_train_epoch_changes_plan_when_an_alternative_exists(
    raw_graph,
    canonical_piece: CanonicalPiece,
) -> None:
    first = _plan(raw_graph, canonical_piece, epoch=0)
    second = _plan(raw_graph, canonical_piece, epoch=1)

    assert 0 < first.selected_count < first.maskable_node_count
    assert first.selected_local_node_indices != second.selected_local_node_indices
    assert first.stable_seed != second.stable_seed
    assert first.fingerprint != second.fingerprint


def test_validation_plan_is_fixed_across_caller_epochs(
    raw_graph,
    canonical_piece: CanonicalPiece,
) -> None:
    first = _plan(
        raw_graph,
        canonical_piece,
        stage="validation",
        epoch=0,
    )
    later = _plan(
        raw_graph,
        canonical_piece,
        stage="validation",
        epoch=999,
    )

    assert first == later
    assert first.epoch == later.epoch == 0


def test_zero_node_and_singleton_plans_are_explicit(
    canonical_piece: CanonicalPiece,
) -> None:
    empty_piece = replace(
        canonical_piece,
        notes=(),
        annotations=(),
        targets=(),
        quality_flags=(),
    )
    empty_graph = build_raw_graph(empty_piece)
    empty = _plan(empty_graph, empty_piece, requested_mask_rate=0.3)
    assert empty.maskable_node_count == 0
    assert empty.selected_local_node_indices == ()
    assert empty.realized_mask_rate == 0.0
    assert all(
        not mask.local_node_indices
        for mask in empty.collateral_feature_masks
    )
    assert all(
        not slot.global_node_indices
        for slot in build_feature_mask_overlay(empty_graph, empty).slot_masks
    )

    singleton_piece = replace(
        canonical_piece,
        notes=(canonical_piece.notes[0],),
        annotations=(),
        targets=(),
        quality_flags=(),
    )
    singleton_graph = build_raw_graph(singleton_piece)
    singleton = _plan(
        singleton_graph,
        singleton_piece,
        epoch=11,
        requested_mask_rate=0.01,
    )
    assert singleton.maskable_node_count == 1
    assert singleton.selected_local_node_indices == (0,)
    assert singleton.realized_mask_rate == 1.0
    assert _collateral(
        singleton, "track"
    ).local_node_indices == (0,)
    assert _collateral(
        singleton, "note"
    ).local_node_indices == ()

    relative_pitch = RAW_FEATURE_REGISTRY.names(
        "note", "continuous"
    ).index("track_relative_pitch")
    assert not singleton_graph["note"].x_cont_available[0, relative_pitch]
    relative_slot = next(
        slot
        for slot in build_feature_mask_overlay(
            singleton_graph, singleton
        ).slot_masks
        if slot.feature_name == "track_relative_pitch"
    )
    assert relative_slot.global_node_indices == (0,)
    assert relative_slot.field.mask_availability


def test_duplicate_batch_identities_receive_the_same_local_plan(
    raw_graph,
    canonical_piece: CanonicalPiece,
) -> None:
    batch = Batch.from_data_list([raw_graph, raw_graph])
    identities = (
        canonical_piece.dataset_name,
        canonical_piece.dataset_name,
    )
    piece_ids = (canonical_piece.piece_id, canonical_piece.piece_id)
    plans = build_batched_mask_plans(
        batch,
        dataset_ids=identities,
        piece_ids=piece_ids,
        global_seed=23,
        epoch=5,
        requested_mask_rate=0.5,
    )

    assert plans[0] == plans[1]
    overlay = build_feature_mask_overlay(batch, plans)
    note_count = int(raw_graph["note"].num_nodes)
    expected = plans[0].selected_local_node_indices + tuple(
        note_count + index for index in plans[1].selected_local_node_indices
    )
    pitch_slot = next(
        slot
        for slot in overlay.slot_masks
        if slot.node_type == "note" and slot.feature_name == "pitch"
    )
    assert pitch_slot.global_node_indices == expected


def test_zero_and_unit_mask_rates_have_exact_boundary_behavior(
    raw_graph,
    canonical_piece: CanonicalPiece,
) -> None:
    zero = _plan(raw_graph, canonical_piece, requested_mask_rate=0.0)
    assert zero.selected_local_node_indices == ()
    assert zero.realized_mask_rate == 0.0
    assert all(
        not mask.local_node_indices
        for mask in zero.collateral_feature_masks
    )

    unit = _plan(raw_graph, canonical_piece, requested_mask_rate=1.0)
    assert unit.selected_local_node_indices == tuple(
        range(int(raw_graph["note"].num_nodes))
    )
    assert unit.realized_mask_rate == 1.0
    assert _collateral(unit, "track").local_node_indices == tuple(
        range(int(raw_graph["track"].num_nodes))
    )
    assert _collateral(unit, "note").local_node_indices == ()


@pytest.mark.parametrize(
    "invalid_rate",
    (-0.01, 1.01, float("nan"), float("inf"), True, "0.3"),
)
def test_invalid_mask_rates_are_rejected(
    raw_graph,
    canonical_piece: CanonicalPiece,
    invalid_rate: object,
) -> None:
    with pytest.raises(SSLContractError, match="mask_rate"):
        _plan(
            raw_graph,
            canonical_piece,
            requested_mask_rate=invalid_rate,
        )


@pytest.mark.parametrize(
    "invalid_seed",
    (-1, 1 << 63, True, 1.5, "42"),
)
def test_invalid_global_seeds_are_rejected(
    raw_graph,
    canonical_piece: CanonicalPiece,
    invalid_seed: object,
) -> None:
    with pytest.raises(SSLContractError, match="global_seed"):
        _plan(raw_graph, canonical_piece, global_seed=invalid_seed)


def test_invalid_primary_and_collateral_indices_are_rejected(
    raw_graph,
    canonical_piece: CanonicalPiece,
) -> None:
    plan = _plan(raw_graph, canonical_piece)
    with pytest.raises(SSLContractError, match="out-of-range"):
        replace(
            plan,
            selected_local_node_indices=(plan.maskable_node_count,),
        )
    with pytest.raises(SSLContractError, match="non-negative"):
        replace(
            _collateral(plan, "track"),
            local_node_indices=(-1,),
        )

    out_of_range_collateral = replace(
        _collateral(plan, "track"),
        local_node_indices=(int(raw_graph["track"].num_nodes),),
    )
    peer = _collateral(plan, "note")
    invalid_plan = _recreate_plan(
        plan,
        collateral_feature_masks=(peer, out_of_range_collateral),
    )
    with pytest.raises(SSLContractError, match="track index is out of range"):
        build_feature_mask_overlay(raw_graph, invalid_plan)


def test_overlay_rejects_in_range_but_wrong_collateral_ownership(
    raw_graph,
    canonical_piece: CanonicalPiece,
) -> None:
    plan = _plan(raw_graph, canonical_piece)
    omitted = replace(
        _collateral(plan, "track"),
        local_node_indices=(),
    )
    peer = _collateral(plan, "note")
    validly_fingerprinted_but_leaky = _recreate_plan(
        plan,
        collateral_feature_masks=(peer, omitted),
    )

    with pytest.raises(
        SSLContractError,
        match="differs from selected-note ownership",
    ):
        build_feature_mask_overlay(
            raw_graph,
            validly_fingerprinted_but_leaky,
        )


def test_note_pitch_registry_and_owner_track_collateral_are_exact(
    raw_graph,
    canonical_piece: CanonicalPiece,
) -> None:
    assert MASKABLE_FIELD_REGISTRY_VERSION == "1.0.0"
    assert len(MASKABLE_FIELD_REGISTRY_FINGERPRINT) == 64
    assert NOTE_PITCH_GROUP.name == "note_pitch_group"
    assert tuple(
        (field.kind, field.feature_name, field.mask_availability)
        for field in NOTE_PITCH_GROUP.primary_fields
    ) == (
        ("categorical", "pitch", True),
        ("categorical", "pitch_class", True),
        ("categorical", "octave", True),
        ("continuous", "track_relative_pitch", True),
    )
    assert tuple(
        (field.kind, field.feature_name, field.mask_availability)
        for field in NOTE_PITCH_GROUP.collateral_fields
    ) == (
        ("continuous", "mean_pitch", True),
        ("continuous", "pitch_std", True),
        ("continuous", "min_pitch", True),
        ("continuous", "max_pitch", True),
    )
    assert tuple(
        (field.kind, field.feature_name, field.mask_availability)
        for field in NOTE_PITCH_GROUP.peer_note_collateral_fields
    ) == (("continuous", "track_relative_pitch", True),)
    primary, collateral = SSL_MASKABLE_FIELD_REGISTRY.resolve_group(
        "note_pitch_group"
    )
    assert tuple(field.availability_tensor_name for field in primary) == (
        "x_cat_available",
        "x_cat_available",
        "x_cat_available",
        "x_cont_available",
    )
    assert tuple(field.availability_tensor_name for field in collateral) == (
        "x_cont_available",
        "x_cont_available",
        "x_cont_available",
        "x_cont_available",
    )

    plan = _plan(raw_graph, canonical_piece)
    note_owner = {}
    track_indices, note_indices = raw_graph[
        ("track", "contains_note", "note")
    ].edge_index.tolist()
    for track_index, note_index in zip(
        track_indices, note_indices, strict=True
    ):
        note_owner[note_index] = track_index
    expected_owners = tuple(
        sorted(
            {
                note_owner[note_index]
                for note_index in plan.selected_local_node_indices
            }
        )
    )
    collateral_mask = _collateral(plan, "track")
    assert isinstance(collateral_mask, CollateralFeatureMask)
    assert collateral_mask.reason == "owner_track_pitch_statistics"
    assert collateral_mask.local_node_indices == expected_owners
    assert collateral_mask.features == NOTE_PITCH_GROUP.collateral_fields
    peer_mask = _collateral(plan, "note")
    selected = set(plan.selected_local_node_indices)
    assert peer_mask.local_node_indices == tuple(
        note_index
        for note_index in range(int(raw_graph["note"].num_nodes))
        if note_index not in selected
        and note_owner[note_index] in set(expected_owners)
    )
    assert (
        peer_mask.features
        == NOTE_PITCH_GROUP.peer_note_collateral_fields
    )


def test_overlay_hides_value_and_availability_contributions_exactly(
    raw_graph,
    canonical_piece: CanonicalPiece,
) -> None:
    plan = _plan(raw_graph, canonical_piece)
    overlay = build_feature_mask_overlay(raw_graph, plan)
    token = torch.tensor([-3.0, -2.0, -1.0, 0.0])
    bound = overlay.bind(token)

    expected_slots = (
        ("primary", "note", "categorical", "pitch"),
        ("primary", "note", "categorical", "pitch_class"),
        ("primary", "note", "categorical", "octave"),
        (
            "primary_with_peer_collateral",
            "note",
            "continuous",
            "track_relative_pitch",
        ),
        ("collateral", "track", "continuous", "mean_pitch"),
        ("collateral", "track", "continuous", "pitch_std"),
        ("collateral", "track", "continuous", "min_pitch"),
        ("collateral", "track", "continuous", "max_pitch"),
    )
    assert tuple(
        (slot.role, slot.node_type, slot.kind, slot.feature_name)
        for slot in overlay.slot_masks
    ) == expected_slots

    for slot in overlay.slot_masks:
        row_count = int(raw_graph[slot.node_type].num_nodes)
        value = torch.arange(
            row_count * token.numel(), dtype=torch.float32
        ).reshape(row_count, token.numel())
        availability = value + 1000.0
        masked_value, masked_availability = (
            bound.replace_feature_contributions(
                node_type=slot.node_type,
                kind=slot.kind,
                feature_name=slot.feature_name,
                value_contribution=value,
                availability_contribution=availability,
            )
        )
        row_mask = overlay.feature_row_mask(
            node_type=slot.node_type,
            kind=slot.kind,
            feature_name=slot.feature_name,
        )
        assert torch.equal(
            masked_value[row_mask],
            token.expand(int(row_mask.count_nonzero()), -1),
        )
        assert torch.equal(
            masked_availability[row_mask],
            torch.zeros_like(masked_availability[row_mask]),
        )
        assert torch.equal(masked_value[~row_mask], value[~row_mask])
        assert torch.equal(
            masked_availability[~row_mask],
            availability[~row_mask],
        )

    unmasked_value = torch.ones(
        (int(raw_graph["note"].num_nodes), token.numel())
    )
    unmasked_availability = torch.full_like(unmasked_value, 2.0)
    same_value, same_availability = bound.replace_feature_contributions(
        node_type="note",
        kind="continuous",
        feature_name="duration_qn",
        value_contribution=unmasked_value,
        availability_contribution=unmasked_availability,
    )
    assert same_value is unmasked_value
    assert same_availability is unmasked_availability


def test_overlay_supports_autocast_mixed_contribution_dtypes(
    raw_graph,
    canonical_piece: CanonicalPiece,
) -> None:
    plan = _plan(raw_graph, canonical_piece)
    bound = build_feature_mask_overlay(raw_graph, plan).bind(torch.zeros(4))
    value = torch.ones(
        (int(raw_graph["note"].num_nodes), 4),
        dtype=torch.bfloat16,
    )
    availability = torch.full(
        value.shape,
        2.0,
        dtype=torch.float32,
    )

    masked_value, masked_availability = (
        bound.replace_feature_contributions(
            node_type="note",
            kind="categorical",
            feature_name="pitch",
            value_contribution=value,
            availability_contribution=availability,
        )
    )

    assert masked_value.dtype == torch.bfloat16
    assert masked_availability.dtype == torch.float32
    row_mask = bound.overlay.feature_row_mask(
        node_type="note",
        kind="categorical",
        feature_name="pitch",
    )
    assert torch.equal(
        masked_value[row_mask],
        torch.zeros_like(masked_value[row_mask]),
    )
    assert torch.equal(
        masked_availability[row_mask],
        torch.zeros_like(masked_availability[row_mask]),
    )


def test_plan_and_overlay_leave_raw_graph_fingerprint_unchanged(
    raw_graph,
    canonical_piece: CanonicalPiece,
) -> None:
    before = graph_fingerprint(raw_graph)
    plan = _plan(raw_graph, canonical_piece)
    overlay = build_feature_mask_overlay(raw_graph, plan)
    overlay.bind(torch.zeros(4))

    assert graph_fingerprint(raw_graph) == before


def test_selections_are_valid_distinct_rows_without_replacement(
    raw_graph,
    canonical_piece: CanonicalPiece,
) -> None:
    plans = tuple(
        _plan(
            raw_graph,
            canonical_piece,
            epoch=epoch,
            requested_mask_rate=0.75,
        )
        for epoch in range(8)
    )
    for plan in plans:
        selected = plan.selected_local_node_indices
        assert selected == tuple(sorted(set(selected)))
        assert len(selected) == 4
        assert all(0 <= index < plan.maskable_node_count for index in selected)
        assert plan.realized_mask_rate == len(selected) / plan.maskable_node_count
    assert all(
        left.selected_local_node_indices
        != right.selected_local_node_indices
        for left, right in zip(plans[:-1], plans[1:], strict=True)
    )
