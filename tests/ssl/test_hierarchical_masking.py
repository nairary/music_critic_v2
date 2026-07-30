from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import pickle
import subprocess
import sys

import pytest
import torch
from torch import Tensor
from torch_geometric.data import Batch

import music_critic.ssl.hierarchical_masking as hierarchy_masking_module
from music_critic.data import QualityFlag, TargetArray
from music_critic.graph import build_raw_graph, graph_fingerprint
from music_critic.ssl.bounded_fixture import (
    Phase7ABoundedFixture,
    build_phase7a_bounded_fixture,
    mutate_piece_pitch_group,
)
from music_critic.ssl.contracts import MaskPlan, StableSeed
from music_critic.ssl.data import collate_ssl_samples
from music_critic.ssl.field_registry import NOTE_PITCH_GROUP
from music_critic.ssl.hierarchical_masking import (
    BEAT_PITCH_DESCENDANTS,
    CONTIGUOUS_BAR_PITCH_SPAN,
    HIERARCHICAL_MASK_PLAN_CONTRACT_VERSION,
    HIERARCHY_MASK_POLICIES,
    HIERARCHY_MASK_POLICY_CONTRACT_FINGERPRINT,
    HIERARCHY_MASK_POLICY_VERSION,
    HIERARCHY_POLICY_CONFIG_CONTRACT_VERSION,
    HIERARCHY_PREPARED_BINDING_PROFILE_VERSION,
    HIERARCHY_SELECTION_EVIDENCE_CONTRACT_VERSION,
    INDEPENDENT_NOTE_PITCH,
    MAX_SPAN_BARS,
    MAX_SPAN_BUDGET_ERROR_SLACK,
    MAX_SPAN_SELECTION_POOL_SIZE,
    ONSET_PITCH_DESCENDANTS,
    SPAN_SELECTION_METHOD,
    TRACK_BAR_PITCH_SPAN,
    HierarchicalMaskPlan,
    HierarchyMaskContractError,
    HierarchyMaskPolicyConfig,
    HierarchyMaskResolution,
    HierarchyMaskUnavailableError,
    SelectedHierarchyUnits,
    build_batched_hierarchy_mask_plans,
    build_batched_hierarchy_mask_resolutions,
    build_hierarchy_mask_plan,
)
from music_critic.ssl.hierarchy_fixture import (
    PHASE8A_HIERARCHY_POLICY_ORACLES,
    PHASE8A_ORACLE_BAR_CONTAINS_NOTE,
    PHASE8A_ORACLE_BAR_CONTAINS_ONSET,
    PHASE8A_ORACLE_BEAT_CONTAINS_ONSET,
    PHASE8A_ORACLE_NOTE_ACTIVE_AT_BEAT,
    PHASE8A_ORACLE_ONSET_STARTS_NOTE,
    PHASE8A_ORACLE_TRACK_CONTAINS_NOTE,
    Phase8AHierarchyFixture,
    build_phase8a_hierarchy_fixture,
)
from music_critic.ssl.hierarchy_leakage import (
    AUDITED_RAW_FEATURE_REGISTRY_FINGERPRINT,
    PHASE8A_PITCH_LEAKAGE_AUDIT,
    PHASE8A_PITCH_LEAKAGE_AUDIT_CONTRACT_VERSION,
    build_phase8a_pitch_leakage_audit,
)
from music_critic.ssl.masking import (
    prepare_hierarchy_mask_binding,
    prepare_mask_binding,
)
from music_critic.ssl.views import build_feature_mask_overlay


_ONSET_STARTS_NOTE = ("onset", "starts_note", "note")
_BEAT_CONTAINS_ONSET = ("beat", "contains_onset", "onset")
_BAR_CONTAINS_BEAT = ("bar", "contains_beat", "beat")
_BAR_CONTAINS_ONSET = ("bar", "contains_onset", "onset")
_BAR_CONTAINS_NOTE = ("bar", "contains_note", "note")
_TRACK_CONTAINS_NOTE = ("track", "contains_note", "note")
_NOTE_ACTIVE_AT_BEAT = ("note", "active_at", "beat")

_HIERARCHICAL_POLICIES = (
    ONSET_PITCH_DESCENDANTS,
    BEAT_PITCH_DESCENDANTS,
    CONTIGUOUS_BAR_PITCH_SPAN,
    TRACK_BAR_PITCH_SPAN,
)

_ORACLE_PLANNER_PARAMETERS = {
    ONSET_PITCH_DESCENDANTS: (1, 0.23),
    BEAT_PITCH_DESCENDANTS: (4, 0.34),
    CONTIGUOUS_BAR_PITCH_SPAN: (0, 0.34),
    TRACK_BAR_PITCH_SPAN: (0, 0.23),
}


def test_fail_closed_pitch_leakage_audit_covers_the_raw_registry() -> None:
    audit = build_phase8a_pitch_leakage_audit()

    assert audit == PHASE8A_PITCH_LEAKAGE_AUDIT
    assert audit.fingerprint == (
        "27fc135b61649e5b892036dd0aacc92f679493ff671320c8235d33396a7c9949"
    )
    assert audit.contract_version == (
        PHASE8A_PITCH_LEAKAGE_AUDIT_CONTRACT_VERSION
    ) == "1.0.0"
    assert audit.classified_raw_feature_count == 68
    assert audit.raw_feature_registry_fingerprint == (
        AUDITED_RAW_FEATURE_REGISTRY_FINGERPRINT
    )
    assert audit.primary_note_pitch_fields == (
        ("note", "categorical", "pitch"),
        ("note", "categorical", "pitch_class"),
        ("note", "categorical", "octave"),
        ("note", "continuous", "track_relative_pitch"),
    )
    assert audit.peer_note_collateral_fields == (
        ("note", "continuous", "track_relative_pitch"),
    )
    assert audit.owner_track_collateral_fields == (
        ("track", "continuous", "mean_pitch"),
        ("track", "continuous", "pitch_std"),
        ("track", "continuous", "min_pitch"),
        ("track", "continuous", "max_pitch"),
    )
    assert len(audit.visible_raw_fields) == 60
    assert len(
        {
            *audit.primary_note_pitch_fields,
            *audit.owner_track_collateral_fields,
            *audit.visible_raw_fields,
        }
    ) == audit.classified_raw_feature_count


@pytest.fixture(scope="module")
def bounded_fixture() -> Phase7ABoundedFixture:
    return build_phase7a_bounded_fixture()


@pytest.fixture(scope="module")
def hierarchy_fixture() -> Phase8AHierarchyFixture:
    return build_phase8a_hierarchy_fixture()


@pytest.fixture(scope="module")
def oracle_piece(hierarchy_fixture: Phase8AHierarchyFixture):
    return hierarchy_fixture.supplemental_piece


@pytest.fixture(scope="module")
def oracle_graph(hierarchy_fixture: Phase8AHierarchyFixture):
    return hierarchy_fixture.raw_samples("train")[-1].raw_graph


@pytest.fixture(scope="module")
def source_piece(bounded_fixture: Phase7ABoundedFixture):
    return bounded_fixture.train_pieces[0]


@pytest.fixture(scope="module")
def source_graph(bounded_fixture: Phase7ABoundedFixture):
    return bounded_fixture.raw_samples("train")[0].raw_graph


def _single_policy_config(
    policy: str,
    *,
    min_span_bars: int = 1,
    max_span_bars: int = 2,
    span_selection_pool_size: int = 4,
    span_budget_error_slack: int = 1,
) -> HierarchyMaskPolicyConfig:
    return HierarchyMaskPolicyConfig.create(
        weights={policy: 1.0},
        min_span_bars=min_span_bars,
        max_span_bars=max_span_bars,
        span_selection_pool_size=span_selection_pool_size,
        span_budget_error_slack=span_budget_error_slack,
    )


def _plan(
    graph,
    piece,
    policy: str,
    *,
    global_seed: int = 42,
    epoch: int = 0,
    encoder_view_index: int = 0,
    requested_mask_rate: float = 0.30,
    stage: str = "train",
    config: HierarchyMaskPolicyConfig | None = None,
):
    return build_hierarchy_mask_plan(
        graph,
        dataset_id=piece.dataset_name,
        piece_id=piece.piece_id,
        policy=policy,
        global_seed=global_seed,
        epoch=epoch,
        encoder_view_index=encoder_view_index,
        requested_mask_rate=requested_mask_rate,
        stage=stage,
        policy_config=config or _single_policy_config(policy),
    )


def _recreate_hierarchical_plan(
    plan: HierarchicalMaskPlan,
    *,
    selection: SelectedHierarchyUnits | None = None,
    requested_hidden_note_count: int | None = None,
) -> HierarchicalMaskPlan:
    return HierarchicalMaskPlan.create(
        dataset_id=plan.dataset_id,
        piece_id=plan.piece_id,
        stage=plan.stage,
        epoch=plan.epoch,
        encoder_view_index=plan.encoder_view_index,
        global_seed=plan.global_seed,
        stable_seed=StableSeed(
            plan.stable_seed,
            plan.stable_seed_sha256,
        ),
        requested_mask_rate=plan.requested_mask_rate,
        requested_hidden_note_count=(
            plan.requested_hidden_note_count
            if requested_hidden_note_count is None
            else requested_hidden_note_count
        ),
        resolved_policy=plan.resolved_policy,
        policy_configuration=plan.policy_configuration,
        relevant_structure_fingerprint=(
            plan.relevant_structure_fingerprint
        ),
        selection=selection or plan.selection,
        collateral_feature_masks=plan.collateral_feature_masks,
        pitched_note_count=plan.pitched_note_count,
        available=plan.available,
        unavailable_reason=plan.unavailable_reason,
    )


def _pairs(graph, edge_type) -> tuple[tuple[int, int], ...]:
    source, target = graph[edge_type].edge_index.tolist()
    return tuple(zip(source, target, strict=True))


def _adjacency(graph, edge_type) -> dict[int, tuple[int, ...]]:
    result: dict[int, list[int]] = {}
    for source, target in _pairs(graph, edge_type):
        result.setdefault(source, []).append(target)
    return {
        source: tuple(sorted(targets))
        for source, targets in result.items()
    }


def _note_descendants_for_onsets(
    graph,
    onsets: tuple[int, ...],
) -> tuple[int, ...]:
    starts = _adjacency(graph, _ONSET_STARTS_NOTE)
    return tuple(
        sorted(
            {
                note
                for onset in onsets
                for note in starts.get(onset, ())
            }
        )
    )


def test_oracle_fixture_relations_are_exact(
    oracle_graph,
) -> None:
    assert _pairs(oracle_graph, _ONSET_STARTS_NOTE) == (
        PHASE8A_ORACLE_ONSET_STARTS_NOTE
    )
    assert _pairs(oracle_graph, _BEAT_CONTAINS_ONSET) == (
        PHASE8A_ORACLE_BEAT_CONTAINS_ONSET
    )
    assert _pairs(oracle_graph, _BAR_CONTAINS_ONSET) == (
        PHASE8A_ORACLE_BAR_CONTAINS_ONSET
    )
    assert _pairs(oracle_graph, _BAR_CONTAINS_NOTE) == (
        PHASE8A_ORACLE_BAR_CONTAINS_NOTE
    )
    assert _pairs(oracle_graph, _TRACK_CONTAINS_NOTE) == (
        PHASE8A_ORACLE_TRACK_CONTAINS_NOTE
    )
    assert _pairs(oracle_graph, _NOTE_ACTIVE_AT_BEAT) == (
        PHASE8A_ORACLE_NOTE_ACTIVE_AT_BEAT
    )


@pytest.mark.parametrize("oracle", PHASE8A_HIERARCHY_POLICY_ORACLES)
def test_hand_computed_policy_oracles(
    oracle_graph,
    oracle_piece,
    oracle,
) -> None:
    seed, rate = _ORACLE_PLANNER_PARAMETERS[oracle.policy]
    plan = _plan(
        oracle_graph,
        oracle_piece,
        oracle.policy,
        global_seed=seed,
        requested_mask_rate=rate,
        config=_single_policy_config(
            oracle.policy,
            min_span_bars=1,
            max_span_bars=1,
            span_selection_pool_size=1,
        ),
    )
    collateral = {
        mask.node_type: mask.local_node_indices
        for mask in plan.collateral_feature_masks
    }

    assert isinstance(plan, HierarchicalMaskPlan)
    assert plan.contract_version == (
        HIERARCHICAL_MASK_PLAN_CONTRACT_VERSION
    ) == "1.1.0"
    assert plan.policy_version == HIERARCHY_MASK_POLICY_VERSION == "1.1.0"
    assert plan.available
    assert plan.resolved_policy == oracle.policy
    assert plan.selected_unit_node_type == oracle.selected_unit_node_type
    assert (
        plan.selected_local_unit_indices
        == oracle.selected_local_unit_indices
    )
    assert (
        plan.selected_local_note_indices
        == oracle.selected_local_note_descendants
    )
    assert plan.primary_masked_count == oracle.primary_masked_count
    assert (
        plan.visible_pitched_note_count
        == oracle.visible_pitched_note_count
    )
    assert (
        plan.selected_local_track_index
        == oracle.selected_local_track_index
    )
    assert plan.span_start_bar_index == oracle.span_start_bar_index
    assert plan.span_end_bar_index == oracle.span_end_bar_index
    assert collateral["note"] == oracle.collateral_peer_note_indices
    assert collateral["track"] == oracle.collateral_owner_track_indices
    assert plan.realized_mask_rate == (
        oracle.realized_mask_fraction[0]
        / oracle.realized_mask_fraction[1]
    )
    assert plan.selected_local_note_indices == tuple(
        sorted(set(plan.selected_local_note_indices))
    )


def test_onset_polyphony_is_selected_as_an_indivisible_descendant_unit(
    oracle_graph,
    oracle_piece,
) -> None:
    plan = _plan(
        oracle_graph,
        oracle_piece,
        ONSET_PITCH_DESCENDANTS,
        global_seed=20,
        requested_mask_rate=0.23,
        config=_single_policy_config(
            ONSET_PITCH_DESCENDANTS,
            min_span_bars=1,
            max_span_bars=1,
        ),
    )
    starts = _adjacency(oracle_graph, _ONSET_STARTS_NOTE)
    selected = set(plan.selected_local_note_indices)

    assert all(len(starts[onset]) == 2 for onset in plan.selected_local_unit_indices)
    assert selected == {
        note
        for onset in plan.selected_local_unit_indices
        for note in starts[onset]
    }
    assert not any(
        bool(selected.intersection(notes)) and not set(notes) <= selected
        for notes in starts.values()
    )


def test_beat_descendants_follow_only_the_two_raw_forward_relations(
    oracle_graph,
    oracle_piece,
) -> None:
    plan = _plan(
        oracle_graph,
        oracle_piece,
        BEAT_PITCH_DESCENDANTS,
        global_seed=7,
        requested_mask_rate=0.34,
        config=_single_policy_config(
            BEAT_PITCH_DESCENDANTS,
            min_span_bars=1,
            max_span_bars=1,
        ),
    )
    beat_onsets = _adjacency(oracle_graph, _BEAT_CONTAINS_ONSET)
    selected_onsets = tuple(
        onset
        for beat in plan.selected_local_unit_indices
        for onset in beat_onsets.get(beat, ())
    )

    assert plan.selected_local_note_indices == _note_descendants_for_onsets(
        oracle_graph,
        selected_onsets,
    )


def test_bar_and_track_bar_descendants_are_exact_start_anchored_sets(
    oracle_graph,
    oracle_piece,
) -> None:
    bar_plan = _plan(
        oracle_graph,
        oracle_piece,
        CONTIGUOUS_BAR_PITCH_SPAN,
        global_seed=0,
        requested_mask_rate=0.34,
        config=_single_policy_config(
            CONTIGUOUS_BAR_PITCH_SPAN,
            min_span_bars=1,
            max_span_bars=1,
            span_selection_pool_size=1,
        ),
    )
    track_plan = _plan(
        oracle_graph,
        oracle_piece,
        TRACK_BAR_PITCH_SPAN,
        global_seed=0,
        requested_mask_rate=0.23,
        config=_single_policy_config(
            TRACK_BAR_PITCH_SPAN,
            min_span_bars=1,
            max_span_bars=1,
        ),
    )
    bar_onsets = _adjacency(oracle_graph, _BAR_CONTAINS_ONSET)
    selected_bar_onsets = tuple(
        onset
        for bar in bar_plan.selected_local_unit_indices
        for onset in bar_onsets.get(bar, ())
    )
    bar_descendants = _note_descendants_for_onsets(
        oracle_graph,
        selected_bar_onsets,
    )
    track_notes = set(
        _adjacency(oracle_graph, _TRACK_CONTAINS_NOTE)[
            track_plan.selected_local_track_index
        ]
    )
    selected_track_bar_onsets = tuple(
        onset
        for bar in track_plan.selected_local_unit_indices
        for onset in bar_onsets.get(bar, ())
    )
    start_descendants = set(
        _note_descendants_for_onsets(
            oracle_graph,
            selected_track_bar_onsets,
        )
    )

    assert bar_plan.selected_local_note_indices == bar_descendants
    assert track_plan.selected_local_note_indices == tuple(
        sorted(track_notes & start_descendants)
    )
    assert bar_plan.span_length_bars == len(
        bar_plan.selected_local_unit_indices
    )
    assert track_plan.span_length_bars == len(
        track_plan.selected_local_unit_indices
    )
    assert bar_plan.span_end_bar_index - bar_plan.span_start_bar_index + 1 == (
        bar_plan.span_length_bars
    )


def test_sustained_note_from_before_span_is_not_a_start_descendant(
    oracle_graph,
    oracle_piece,
) -> None:
    plan = _plan(
        oracle_graph,
        oracle_piece,
        CONTIGUOUS_BAR_PITCH_SPAN,
        global_seed=0,
        requested_mask_rate=0.34,
        config=_single_policy_config(
            CONTIGUOUS_BAR_PITCH_SPAN,
            min_span_bars=1,
            max_span_bars=1,
            span_selection_pool_size=1,
        ),
    )
    selected_beats = {
        beat
        for bar, beat in _pairs(oracle_graph, _BAR_CONTAINS_BEAT)
        if bar in set(plan.selected_local_unit_indices)
    }
    active_inside_span = {
        note
        for note, beat in _pairs(oracle_graph, _NOTE_ACTIVE_AT_BEAT)
        if beat in selected_beats
    }

    assert plan.selected_local_unit_indices == (1,)
    assert 3 in active_inside_span
    assert 3 not in plan.selected_local_note_indices
    assert 3 not in {
        note
        for bar, note in _pairs(oracle_graph, _BAR_CONTAINS_NOTE)
        if bar in set(plan.selected_local_unit_indices)
    }


@pytest.mark.parametrize("policy", _HIERARCHICAL_POLICIES)
def test_same_inputs_are_deterministic_and_seed_evidence_is_bound(
    source_graph,
    source_piece,
    policy: str,
) -> None:
    first = _plan(source_graph, source_piece, policy)
    repeated = _plan(source_graph, source_piece, policy)
    changed_seed = _plan(
        source_graph,
        source_piece,
        policy,
        global_seed=43,
    )

    assert first == repeated
    assert first.to_dict() == repeated.to_dict()
    assert first.fingerprint == repeated.fingerprint
    assert first.stable_seed_sha256 == repeated.stable_seed_sha256
    assert changed_seed.fingerprint != first.fingerprint
    assert changed_seed.stable_seed_sha256 != first.stable_seed_sha256


@pytest.mark.parametrize("policy", _HIERARCHICAL_POLICIES)
def test_validation_epoch_is_canonicalized_to_zero(
    source_graph,
    source_piece,
    policy: str,
) -> None:
    zero = _plan(
        source_graph,
        source_piece,
        policy,
        stage="validation",
        epoch=0,
    )
    later = _plan(
        source_graph,
        source_piece,
        policy,
        stage="validation",
        epoch=999,
    )

    assert zero == later
    assert zero.epoch == later.epoch == 0


def _actual_span(
    plan: HierarchicalMaskPlan,
) -> tuple[int | None, int, int, tuple[int, ...]]:
    assert plan.span_start_bar_index is not None
    assert plan.span_end_bar_index is not None
    return (
        plan.selected_local_track_index,
        plan.span_start_bar_index,
        plan.span_end_bar_index,
        plan.selected_local_note_indices,
    )


def _single_bar_track_candidates(
    graph,
) -> tuple[tuple[int, int, int, tuple[int, ...]], ...]:
    track_notes = _adjacency(graph, _TRACK_CONTAINS_NOTE)
    bar_onsets = _adjacency(graph, _BAR_CONTAINS_ONSET)
    note_count = int(graph["note"].num_nodes)
    candidates = []
    for track in range(int(graph["track"].num_nodes)):
        owned = set(track_notes[track])
        for bar in range(int(graph["bar"].num_nodes)):
            descendants = tuple(
                note
                for note in _note_descendants_for_onsets(
                    graph,
                    bar_onsets[bar],
                )
                if note in owned
            )
            if descendants and len(descendants) < note_count:
                candidates.append((bar, bar, track, descendants))
    return tuple(candidates)


def test_unique_closest_track_span_has_repeatable_near_optimal_diversity(
    oracle_graph,
    oracle_piece,
) -> None:
    config = _single_policy_config(
        TRACK_BAR_PITCH_SPAN,
        min_span_bars=1,
        max_span_bars=1,
        span_selection_pool_size=4,
        span_budget_error_slack=1,
    )
    target_count = 2
    candidates = _single_bar_track_candidates(oracle_graph)
    errors = tuple(
        abs(len(candidate[3]) - target_count)
        for candidate in candidates
    )
    canonical_pool = tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                abs(len(candidate[3]) - target_count),
                candidate[2],
                candidate[0],
                candidate[1],
                candidate[3],
            ),
        )[: config.span_selection_pool_size]
    )

    def sequence() -> tuple[HierarchicalMaskPlan, ...]:
        return tuple(
            _plan(
                oracle_graph,
                oracle_piece,
                TRACK_BAR_PITCH_SPAN,
                global_seed=42,
                epoch=epoch,
                requested_mask_rate=0.30,
                config=config,
            )
            for epoch in range(64)
        )

    first = sequence()
    repeated = sequence()
    actual = tuple(_actual_span(plan) for plan in first)

    assert len(candidates) == 6
    assert errors.count(0) == 1
    assert errors.count(1) == 5
    assert first == repeated
    assert actual == tuple(_actual_span(plan) for plan in repeated)
    assert len(set(actual)) == 4
    assert sum(
        plan.selection.span_selected_budget_error == 0
        for plan in first
    ) == 14
    assert sum(
        plan.selection.span_selected_budget_error == 1
        for plan in first
    ) == 50
    assert all(
        (
            plan.span_start_bar_index,
            plan.span_end_bar_index,
            plan.selected_local_track_index,
            plan.selected_local_note_indices,
        )
        in canonical_pool
        for plan in first
    )
    for plan in first:
        selection = plan.selection
        assert selection.total_valid_candidate_count == 6
        assert selection.span_best_budget_error == 0
        assert selection.span_tolerance_candidate_count == 6
        assert selection.span_admissible_pool_count == 4
        assert selection.span_configured_pool_size_limit == 4
        assert selection.span_configured_budget_error_slack == 1
        assert selection.span_selected_budget_error in {0, 1}
        assert selection.span_selected_descendant_count == len(
            plan.selected_local_note_indices
        )
        assert (
            selection.span_realized_mask_rate
            == plan.realized_mask_rate
        )
        assert selection.span_selection_method == SPAN_SELECTION_METHOD
        assert plan.visible_pitched_note_count >= 1


def test_span_pool_one_is_exact_closest_control_and_config_is_bound(
    oracle_graph,
    oracle_piece,
) -> None:
    diverse = _single_policy_config(
        TRACK_BAR_PITCH_SPAN,
        min_span_bars=1,
        max_span_bars=1,
        span_selection_pool_size=4,
        span_budget_error_slack=1,
    )
    closest = _single_policy_config(
        TRACK_BAR_PITCH_SPAN,
        min_span_bars=1,
        max_span_bars=1,
        span_selection_pool_size=1,
        span_budget_error_slack=1,
    )
    strict = _single_policy_config(
        TRACK_BAR_PITCH_SPAN,
        min_span_bars=1,
        max_span_bars=1,
        span_selection_pool_size=4,
        span_budget_error_slack=0,
    )
    closest_plans = tuple(
        _plan(
            oracle_graph,
            oracle_piece,
            TRACK_BAR_PITCH_SPAN,
            epoch=epoch,
            requested_mask_rate=0.30,
            config=closest,
        )
        for epoch in range(64)
    )
    strict_plans = tuple(
        _plan(
            oracle_graph,
            oracle_piece,
            TRACK_BAR_PITCH_SPAN,
            epoch=epoch,
            requested_mask_rate=0.30,
            config=strict,
        )
        for epoch in range(64)
    )
    diverse_plan = _plan(
        oracle_graph,
        oracle_piece,
        TRACK_BAR_PITCH_SPAN,
        epoch=0,
        requested_mask_rate=0.30,
        config=diverse,
    )

    assert len({_actual_span(plan) for plan in closest_plans}) == 1
    assert len({_actual_span(plan) for plan in strict_plans}) == 1
    assert all(
        plan.selection.span_selected_budget_error == 0
        and plan.selection.span_admissible_pool_count == 1
        for plan in (*closest_plans, *strict_plans)
    )
    assert closest.fingerprint != diverse.fingerprint
    assert strict.fingerprint != diverse.fingerprint
    assert closest_plans[0].fingerprint != diverse_plan.fingerprint


def test_span_sequence_binds_seed_view_and_canonical_validation_epoch(
    oracle_graph,
    oracle_piece,
) -> None:
    config = _single_policy_config(
        TRACK_BAR_PITCH_SPAN,
        min_span_bars=1,
        max_span_bars=1,
    )

    def sequence(
        *,
        global_seed: int,
        encoder_view_index: int,
    ) -> tuple[tuple[int | None, int, int, tuple[int, ...]], ...]:
        return tuple(
            _actual_span(
                _plan(
                    oracle_graph,
                    oracle_piece,
                    TRACK_BAR_PITCH_SPAN,
                    global_seed=global_seed,
                    epoch=epoch,
                    encoder_view_index=encoder_view_index,
                    requested_mask_rate=0.30,
                    config=config,
                )
            )
            for epoch in range(16)
        )

    baseline = sequence(global_seed=42, encoder_view_index=0)
    assert baseline == sequence(
        global_seed=42,
        encoder_view_index=0,
    )
    assert baseline != sequence(
        global_seed=43,
        encoder_view_index=0,
    )
    assert baseline != sequence(
        global_seed=42,
        encoder_view_index=1,
    )
    zero = _plan(
        oracle_graph,
        oracle_piece,
        TRACK_BAR_PITCH_SPAN,
        stage="validation",
        epoch=0,
        requested_mask_rate=0.30,
        config=config,
    )
    later = _plan(
        oracle_graph,
        oracle_piece,
        TRACK_BAR_PITCH_SPAN,
        stage="validation",
        epoch=999,
        requested_mask_rate=0.30,
        config=config,
    )
    assert zero == later
    assert _actual_span(zero) == _actual_span(later)
    assert zero.epoch == later.epoch == 0


def test_span_candidate_enumeration_order_does_not_change_selection(
    oracle_graph,
    oracle_piece,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _single_policy_config(
        TRACK_BAR_PITCH_SPAN,
        min_span_bars=1,
        max_span_bars=1,
    )
    expected = _plan(
        oracle_graph,
        oracle_piece,
        TRACK_BAR_PITCH_SPAN,
        epoch=17,
        requested_mask_rate=0.30,
        config=config,
    )
    original = hierarchy_masking_module._track_bar_span_candidates

    def reversed_candidates(*, index, config):
        return tuple(reversed(original(index=index, config=config)))

    monkeypatch.setattr(
        hierarchy_masking_module,
        "_track_bar_span_candidates",
        reversed_candidates,
    )
    reordered = _plan(
        oracle_graph,
        oracle_piece,
        TRACK_BAR_PITCH_SPAN,
        epoch=17,
        requested_mask_rate=0.30,
        config=config,
    )

    assert reordered == expected
    assert _actual_span(reordered) == _actual_span(expected)


def test_span_sequence_is_stable_across_fresh_processes() -> None:
    program = """
import json
from music_critic.ssl.hierarchy_fixture import build_phase8a_hierarchy_fixture
from music_critic.ssl.hierarchical_masking import (
    TRACK_BAR_PITCH_SPAN,
    HierarchyMaskPolicyConfig,
    build_hierarchy_mask_plan,
)
fixture = build_phase8a_hierarchy_fixture()
piece = fixture.supplemental_piece
graph = fixture.raw_samples("train")[-1].raw_graph
config = HierarchyMaskPolicyConfig.create(
    weights={TRACK_BAR_PITCH_SPAN: 1.0},
    min_span_bars=1,
    max_span_bars=1,
)
sequence = []
for epoch in range(64):
    plan = build_hierarchy_mask_plan(
        graph,
        dataset_id=piece.dataset_name,
        piece_id=piece.piece_id,
        policy=TRACK_BAR_PITCH_SPAN,
        global_seed=42,
        epoch=epoch,
        requested_mask_rate=0.30,
        stage="train",
        policy_config=config,
    )
    sequence.append([
        plan.selected_local_track_index,
        list(plan.selected_local_unit_indices),
        list(plan.selected_local_note_indices),
    ])
print(json.dumps(sequence, separators=(",", ":")))
"""

    def run() -> str:
        completed = subprocess.run(
            [sys.executable, "-c", program],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    first = run()
    assert first == run()
    assert len(
        {
            json.dumps(row, separators=(",", ":"))
            for row in json.loads(first)
        }
    ) > 1


def test_batch_order_does_not_change_per_piece_resolution(
    bounded_fixture: Phase7ABoundedFixture,
) -> None:
    samples = bounded_fixture.raw_samples("train")[:2]
    config = _single_policy_config(TRACK_BAR_PITCH_SPAN)
    reverse_samples = tuple(reversed(samples))

    def evidence(ordered_samples):
        graph_batch = Batch.from_data_list(
            [sample.raw_graph for sample in ordered_samples]
        )
        result = {
            (sample.dataset_id, sample.piece_id): []
            for sample in ordered_samples
        }
        for epoch in range(16):
            resolutions = build_batched_hierarchy_mask_resolutions(
                graph_batch,
                dataset_ids=tuple(
                    sample.dataset_id for sample in ordered_samples
                ),
                piece_ids=tuple(
                    sample.piece_id for sample in ordered_samples
                ),
                global_seed=71,
                epoch=epoch,
                requested_mask_rate=0.30,
                policy_config=config,
            )
            for resolution in resolutions:
                plan = resolution.plan
                assert isinstance(plan, HierarchicalMaskPlan)
                result[resolution.sample_identity].append(
                    (
                        resolution.to_dict(),
                        _actual_span(plan),
                    )
                )
        return {
            identity: tuple(sequence)
            for identity, sequence in result.items()
        }

    assert evidence(samples) == evidence(reverse_samples)


def test_portable_plan_and_resolution_serialization_is_repeatable(
    source_graph,
    source_piece,
) -> None:
    config = _single_policy_config(BEAT_PITCH_DESCENDANTS)
    batch = Batch.from_data_list([source_graph])
    resolution = build_batched_hierarchy_mask_resolutions(
        batch,
        dataset_ids=(source_piece.dataset_name,),
        piece_ids=(source_piece.piece_id,),
        global_seed=42,
        epoch=0,
        policy_config=config,
    )[0]
    plan = resolution.plan
    assert isinstance(plan, HierarchicalMaskPlan)

    plan_json = json.dumps(
        plan.to_dict(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    resolution_json = json.dumps(
        resolution.to_dict(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    assert plan_json == json.dumps(
        pickle.loads(pickle.dumps(plan)).to_dict(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    assert resolution_json == json.dumps(
        pickle.loads(pickle.dumps(resolution)).to_dict(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    assert plan.fingerprint in resolution_json
    assert "target" not in plan_json.lower()
    assert "provenance" not in plan_json.lower()
    assert "tensor" not in plan_json.lower()


@pytest.mark.parametrize("enabled_policy", HIERARCHY_MASK_POLICIES)
def test_each_policy_can_be_enabled_alone_and_all_others_are_explicitly_disabled(
    source_graph,
    source_piece,
    enabled_policy: str,
) -> None:
    config = _single_policy_config(enabled_policy)
    resolution = build_batched_hierarchy_mask_resolutions(
        Batch.from_data_list([source_graph]),
        dataset_ids=(source_piece.dataset_name,),
        piece_ids=(source_piece.piece_id,),
        global_seed=42,
        epoch=0,
        policy_config=config,
    )[0]
    eligibility = {
        item.policy: item for item in resolution.eligibility
    }

    assert resolution.resolved_policy == enabled_policy
    assert resolution.eligible_policies == (enabled_policy,)
    assert resolution.eligible_normalized_weights == (
        (enabled_policy, 1.0),
    )
    assert eligibility[enabled_policy].eligible
    for policy in set(HIERARCHY_MASK_POLICIES) - {enabled_policy}:
        assert not eligibility[policy].eligible
        assert eligibility[policy].unavailable_reason is not None
        assert (
            eligibility[policy].unavailable_reason.code
            == "policy_disabled"
        )


def test_mixture_resolution_and_eligible_weight_renormalization_are_explicit(
    source_graph,
    source_piece,
) -> None:
    config = HierarchyMaskPolicyConfig.create(
        weights={
            ONSET_PITCH_DESCENDANTS: 1.0,
            BEAT_PITCH_DESCENDANTS: 3.0,
        }
    )
    resolution = build_batched_hierarchy_mask_resolutions(
        Batch.from_data_list([source_graph]),
        dataset_ids=(source_piece.dataset_name,),
        piece_ids=(source_piece.piece_id,),
        global_seed=19,
        epoch=2,
        policy_config=config,
    )[0]

    assert resolution.eligible_policies == (
        ONSET_PITCH_DESCENDANTS,
        BEAT_PITCH_DESCENDANTS,
    )
    assert resolution.eligible_normalized_weights == (
        (ONSET_PITCH_DESCENDANTS, 0.25),
        (BEAT_PITCH_DESCENDANTS, 0.75),
    )
    assert resolution.resolved_policy in resolution.eligible_policies
    assert resolution.plan is not None
    assert resolution.plan.mask_policy == resolution.resolved_policy
    assert resolution == build_batched_hierarchy_mask_resolutions(
        Batch.from_data_list([source_graph]),
        dataset_ids=(source_piece.dataset_name,),
        piece_ids=(source_piece.piece_id,),
        global_seed=19,
        epoch=2,
        policy_config=config,
    )[0]


def test_resolution_rejects_forged_weights_policy_and_plan_context(
    source_graph,
    source_piece,
) -> None:
    config = HierarchyMaskPolicyConfig.create(
        weights={
            ONSET_PITCH_DESCENDANTS: 1.0,
            BEAT_PITCH_DESCENDANTS: 3.0,
        }
    )
    resolution = build_batched_hierarchy_mask_resolutions(
        Batch.from_data_list([source_graph]),
        dataset_ids=(source_piece.dataset_name,),
        piece_ids=(source_piece.piece_id,),
        global_seed=19,
        epoch=2,
        policy_config=config,
    )[0]
    seed = StableSeed(
        resolution.stable_seed,
        resolution.stable_seed_sha256,
    )
    common = {
        "dataset_id": resolution.dataset_id,
        "piece_id": resolution.piece_id,
        "stage": resolution.stage,
        "epoch": resolution.epoch,
        "encoder_view_index": resolution.encoder_view_index,
        "global_seed": resolution.global_seed,
        "requested_mask_rate": resolution.requested_mask_rate,
        "relevant_structure_fingerprint": (
            resolution.relevant_structure_fingerprint
        ),
        "config": config,
        "eligibility": resolution.eligibility,
        "stable_seed": seed,
    }

    with pytest.raises(
        HierarchyMaskContractError,
        match="normalized_weights_non_canonical",
    ):
        HierarchyMaskResolution.create(
            **common,
            eligible_normalized_weights=(
                (ONSET_PITCH_DESCENDANTS, 0.5),
                (BEAT_PITCH_DESCENDANTS, 0.5),
            ),
            resolved_policy=resolution.resolved_policy,
            plan=resolution.plan,
        )

    other_policy = (
        BEAT_PITCH_DESCENDANTS
        if resolution.resolved_policy == ONSET_PITCH_DESCENDANTS
        else ONSET_PITCH_DESCENDANTS
    )
    other_plan = _plan(
        source_graph,
        source_piece,
        other_policy,
        global_seed=19,
        epoch=2,
        config=config,
    )
    with pytest.raises(
        HierarchyMaskContractError,
        match="resolved_policy_non_deterministic",
    ):
        HierarchyMaskResolution.create(
            **common,
            eligible_normalized_weights=(
                resolution.eligible_normalized_weights
            ),
            resolved_policy=other_policy,
            plan=other_plan,
        )

    wrong_context_plan = _plan(
        source_graph,
        source_piece,
        resolution.resolved_policy,
        global_seed=999,
        epoch=2,
        requested_mask_rate=0.75,
        config=config,
    )
    with pytest.raises(
        HierarchyMaskContractError,
        match="resolved_plan_context_mismatch",
    ):
        HierarchyMaskResolution.create(
            **common,
            eligible_normalized_weights=(
                resolution.eligible_normalized_weights
            ),
            resolved_policy=resolution.resolved_policy,
            plan=wrong_context_plan,
        )


@pytest.mark.parametrize("policy", _HIERARCHICAL_POLICIES)
def test_singleton_piece_returns_structured_unavailable(
    source_piece,
    policy: str,
) -> None:
    singleton_piece = replace(
        source_piece,
        notes=(source_piece.notes[0],),
        annotations=(),
        targets=(),
        quality_flags=(),
    )
    graph = build_raw_graph(singleton_piece)
    plan = _plan(graph, singleton_piece, policy)

    assert isinstance(plan, HierarchicalMaskPlan)
    assert not plan.available
    assert plan.unavailable_reason is not None
    assert plan.unavailable_reason.code == "fewer_than_two_pitched_notes"
    assert plan.selected_local_note_indices == ()
    assert plan.primary_masked_count == 0
    assert plan.visible_pitched_note_count == 1
    assert plan.collateral_feature_masks == ()


@pytest.mark.parametrize(
    "policy",
    (ONSET_PITCH_DESCENDANTS, BEAT_PITCH_DESCENDANTS),
)
def test_empty_unit_piece_returns_structured_unavailable(
    source_piece,
    policy: str,
) -> None:
    empty_piece = replace(
        source_piece,
        notes=(),
        annotations=(),
        targets=(),
        quality_flags=(),
    )
    plan = _plan(
        build_raw_graph(empty_piece),
        empty_piece,
        policy,
    )

    assert isinstance(plan, HierarchicalMaskPlan)
    assert not plan.available
    assert plan.unavailable_reason is not None
    assert (
        plan.unavailable_reason.code
        == "no_nonempty_hierarchy_units"
    )
    assert plan.unavailable_reason.candidate_count == 0


@pytest.mark.parametrize(
    ("policy", "reason"),
    (
        (CONTIGUOUS_BAR_PITCH_SPAN, "no_valid_span"),
        (TRACK_BAR_PITCH_SPAN, "no_valid_track_span"),
    ),
)
def test_no_valid_bounded_span_is_structured_unavailable(
    source_graph,
    source_piece,
    policy: str,
    reason: str,
) -> None:
    config = _single_policy_config(
        policy,
        min_span_bars=4,
        max_span_bars=4,
    )
    plan = _plan(
        source_graph,
        source_piece,
        policy,
        config=config,
    )

    assert isinstance(plan, HierarchicalMaskPlan)
    assert not plan.available
    assert plan.unavailable_reason is not None
    assert plan.unavailable_reason.code == reason
    assert plan.unavailable_reason.candidate_count == 0


@pytest.mark.parametrize("policy", _HIERARCHICAL_POLICIES)
def test_zero_mask_rate_is_structured_unavailable(
    source_graph,
    source_piece,
    policy: str,
) -> None:
    plan = _plan(
        source_graph,
        source_piece,
        policy,
        requested_mask_rate=0.0,
    )

    assert isinstance(plan, HierarchicalMaskPlan)
    assert not plan.available
    assert plan.unavailable_reason is not None
    assert plan.unavailable_reason.code == "zero_requested_mask_rate"
    assert plan.requested_hidden_note_count == 0


def test_batch_api_raises_structured_error_when_no_policy_is_eligible(
    source_piece,
) -> None:
    singleton_piece = replace(
        source_piece,
        notes=(source_piece.notes[0],),
        annotations=(),
        targets=(),
        quality_flags=(),
    )
    graph = build_raw_graph(singleton_piece)
    batch = Batch.from_data_list([graph])
    kwargs = {
        "dataset_ids": (singleton_piece.dataset_name,),
        "piece_ids": (singleton_piece.piece_id,),
        "global_seed": 42,
        "epoch": 0,
        "policy_config": HierarchyMaskPolicyConfig(),
    }
    resolution = build_batched_hierarchy_mask_resolutions(
        batch,
        **kwargs,
    )[0]

    assert resolution.plan is None
    assert resolution.resolved_policy is None
    assert resolution.eligible_policies == ()
    with pytest.raises(HierarchyMaskUnavailableError) as caught:
        build_batched_hierarchy_mask_plans(batch, **kwargs)
    assert caught.value.resolutions == (resolution,)


@pytest.mark.parametrize(
    ("min_span_bars", "max_span_bars", "message"),
    (
        (0, 1, "min_span_bars_invalid"),
        (True, 1, "min_span_bars_invalid"),
        (3, 2, "span_bounds_reversed"),
        (1, MAX_SPAN_BARS + 1, "max_span_exceeds_contract_bound"),
    ),
)
def test_span_bounds_fail_closed(
    min_span_bars: object,
    max_span_bars: object,
    message: str,
) -> None:
    with pytest.raises(HierarchyMaskContractError, match=message):
        HierarchyMaskPolicyConfig.create(
            weights={CONTIGUOUS_BAR_PITCH_SPAN: 1.0},
            min_span_bars=min_span_bars,
            max_span_bars=max_span_bars,
        )


@pytest.mark.parametrize(
    ("pool_size", "slack", "message"),
    (
        (0, 1, "span_selection_pool_size_invalid"),
        (True, 1, "span_selection_pool_size_invalid"),
        (
            MAX_SPAN_SELECTION_POOL_SIZE + 1,
            1,
            "span_selection_pool_size_exceeds_contract_bound",
        ),
        (4, -1, "span_budget_error_slack_invalid"),
        (4, True, "span_budget_error_slack_invalid"),
        (
            4,
            MAX_SPAN_BUDGET_ERROR_SLACK + 1,
            "span_budget_error_slack_exceeds_contract_bound",
        ),
    ),
)
def test_span_pool_configuration_fails_closed(
    pool_size: object,
    slack: object,
    message: str,
) -> None:
    with pytest.raises(HierarchyMaskContractError, match=message):
        HierarchyMaskPolicyConfig.create(
            weights={TRACK_BAR_PITCH_SPAN: 1.0},
            span_selection_pool_size=pool_size,
            span_budget_error_slack=slack,
        )


def test_span_selection_contract_versions_defaults_and_fingerprints() -> None:
    config = HierarchyMaskPolicyConfig()

    assert HIERARCHICAL_MASK_PLAN_CONTRACT_VERSION == "1.1.0"
    assert HIERARCHY_MASK_POLICY_VERSION == "1.1.0"
    assert HIERARCHY_POLICY_CONFIG_CONTRACT_VERSION == "1.1.0"
    assert HIERARCHY_SELECTION_EVIDENCE_CONTRACT_VERSION == "1.1.0"
    assert HIERARCHY_PREPARED_BINDING_PROFILE_VERSION == "1.1.0"
    assert config.span_selection_pool_size == 4
    assert config.span_budget_error_slack == 1
    assert config.fingerprint == (
        "2e53d771d67a33d2db426033850ca57bccf0d6e284954141ccdeb28e8af3d760"
    )
    assert HIERARCHY_MASK_POLICY_CONTRACT_FINGERPRINT == (
        "a2ad4fdd4c283413a1a7050a7471ea7fe86f29c95f17bf011cf4948f72547954"
    )


def test_all_disabled_and_unknown_policy_configurations_fail_closed() -> None:
    with pytest.raises(
        HierarchyMaskContractError,
        match="all_policies_disabled",
    ):
        HierarchyMaskPolicyConfig.create(weights={})
    with pytest.raises(
        HierarchyMaskContractError,
        match="unknown_policy",
    ):
        HierarchyMaskPolicyConfig.create(weights={"voice_span": 1.0})


def test_policy_config_is_deeply_immutable_and_canonical() -> None:
    canonical = HierarchyMaskPolicyConfig()
    with pytest.raises(
        HierarchyMaskContractError,
        match="policy_weights_not_immutable_tuple",
    ):
        HierarchyMaskPolicyConfig(
            policy_weights=list(canonical.policy_weights),  # type: ignore[arg-type]
        )
    with pytest.raises(
        HierarchyMaskContractError,
        match="policy_weight_invalid",
    ):
        HierarchyMaskPolicyConfig.create(
            weights={ONSET_PITCH_DESCENDANTS: -0.0}
        )
    with pytest.raises(
        HierarchyMaskContractError,
        match="policy_weight_not_normalizable",
    ):
        HierarchyMaskPolicyConfig.create(
            weights={
                ONSET_PITCH_DESCENDANTS: 5e-324,
                BEAT_PITCH_DESCENDANTS: float.fromhex(
                    "0x1.fffffffffffffp+1023"
                ),
            }
        )


def test_explicit_disabled_policy_fails_closed(
    source_graph,
    source_piece,
) -> None:
    with pytest.raises(
        HierarchyMaskContractError,
        match="policy_disabled:onset_pitch_descendants",
    ):
        _plan(
            source_graph,
            source_piece,
            ONSET_PITCH_DESCENDANTS,
            config=_single_policy_config(BEAT_PITCH_DESCENDANTS),
        )


def _set_forward_and_reverse(graph, edge_type, edge_index: Tensor) -> None:
    reverse = {
        _ONSET_STARTS_NOTE: ("note", "in_onset", "onset"),
        _BAR_CONTAINS_BEAT: ("beat", "belongs_to_bar", "bar"),
        _BAR_CONTAINS_NOTE: ("note", "belongs_to_bar", "bar"),
        _TRACK_CONTAINS_NOTE: ("note", "belongs_to_track", "track"),
    }[edge_type]
    graph[edge_type].edge_index = edge_index
    graph[reverse].edge_index = edge_index.flip(0)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("duplicate_track_owner", "note_owner_duplicate"),
        ("missing_onset_owner", "note_onset_owner_missing"),
        ("start_bar_mismatch", "start_bar_composition_mismatch"),
        (
            "beat_bar_mismatch",
            "onset_beat_bar_composition_mismatch",
        ),
        ("unknown_store_attribute", "raw_graph_invalid"),
    ),
)
def test_malformed_hierarchy_ownership_and_store_data_are_rejected(
    source_graph,
    source_piece,
    mutation: str,
    message: str,
) -> None:
    graph = deepcopy(source_graph)
    if mutation == "duplicate_track_owner":
        edge = graph[_TRACK_CONTAINS_NOTE].edge_index
        extra = torch.tensor([[1], [0]], dtype=torch.long)
        _set_forward_and_reverse(
            graph,
            _TRACK_CONTAINS_NOTE,
            torch.cat((edge, extra), dim=1),
        )
    elif mutation == "missing_onset_owner":
        edge = graph[_ONSET_STARTS_NOTE].edge_index[:, 1:]
        _set_forward_and_reverse(graph, _ONSET_STARTS_NOTE, edge)
    elif mutation == "start_bar_mismatch":
        edge = graph[_BAR_CONTAINS_NOTE].edge_index.clone()
        edge[0, 0] = 1
        _set_forward_and_reverse(graph, _BAR_CONTAINS_NOTE, edge)
    elif mutation == "beat_bar_mismatch":
        edge = graph[_BAR_CONTAINS_BEAT].edge_index.clone()
        edge[0, 0] = 1
        _set_forward_and_reverse(graph, _BAR_CONTAINS_BEAT, edge)
    else:
        graph["note"].theory_target = torch.zeros(
            int(graph["note"].num_nodes),
            dtype=torch.long,
        )

    with pytest.raises(HierarchyMaskContractError, match=message):
        _plan(graph, source_piece, ONSET_PITCH_DESCENDANTS)


def test_cross_sample_descendant_edge_is_rejected(
    bounded_fixture: Phase7ABoundedFixture,
) -> None:
    samples = bounded_fixture.raw_samples("train")[:2]
    batch = Batch.from_data_list(
        [deepcopy(sample.raw_graph) for sample in samples]
    )
    edge = batch[_ONSET_STARTS_NOTE].edge_index.clone()
    edge[0, 0] = batch["onset"].ptr[1]
    _set_forward_and_reverse(batch, _ONSET_STARTS_NOTE, edge)

    with pytest.raises(HierarchyMaskContractError):
        build_batched_hierarchy_mask_resolutions(
            batch,
            dataset_ids=tuple(
                sample.dataset_id for sample in samples
            ),
            piece_ids=tuple(sample.piece_id for sample in samples),
            global_seed=42,
            epoch=0,
            policy_config=_single_policy_config(
                ONSET_PITCH_DESCENDANTS
            ),
        )


def test_portable_plan_rejects_wrong_requested_count_and_overlay_rebuilds_descendants(
    oracle_graph,
    oracle_piece,
) -> None:
    plan = _plan(
        oracle_graph,
        oracle_piece,
        ONSET_PITCH_DESCENDANTS,
        global_seed=20,
        requested_mask_rate=0.23,
        config=_single_policy_config(
            ONSET_PITCH_DESCENDANTS,
            min_span_bars=1,
            max_span_bars=1,
        ),
    )
    assert isinstance(plan, HierarchicalMaskPlan)
    with pytest.raises(
        HierarchyMaskContractError,
        match="requested_count_inconsistent",
    ):
        _recreate_hierarchical_plan(
            plan,
            requested_hidden_note_count=999,
        )

    forged_selection = SelectedHierarchyUnits.create(
        policy=plan.resolved_policy,
        selected_local_unit_indices=(1,),
        selected_local_note_descendants=(
            plan.selected_local_note_indices
        ),
        total_valid_candidate_count=(
            plan.selection.total_valid_candidate_count
        ),
    )
    forged = _recreate_hierarchical_plan(
        plan,
        selection=forged_selection,
    )
    assert forged.fingerprint != plan.fingerprint
    with pytest.raises(
        HierarchyMaskContractError,
        match="graph_plan_non_canonical",
    ):
        build_feature_mask_overlay(oracle_graph, forged)


def test_coherent_raw_pitch_mutation_does_not_change_hierarchy_or_overlay(
    bounded_fixture: Phase7ABoundedFixture,
) -> None:
    piece = bounded_fixture.train_pieces[0]
    source_graph = bounded_fixture.raw_samples("train")[0].raw_graph
    mutation = mutate_piece_pitch_group(
        piece,
        (0, 3, 8, 11),
    )
    mutated_graph = mutation.mutated_raw_graph
    source = _plan(
        source_graph,
        piece,
        BEAT_PITCH_DESCENDANTS,
    )
    changed = _plan(
        mutated_graph,
        mutation.mutated_piece,
        BEAT_PITCH_DESCENDANTS,
    )

    assert graph_fingerprint(source_graph) != graph_fingerprint(mutated_graph)
    assert source == changed
    assert source.to_dict() == changed.to_dict()
    assert build_feature_mask_overlay(
        source_graph,
        source,
    ) == build_feature_mask_overlay(mutated_graph, changed)


def test_target_provenance_and_diagnostic_sidecars_do_not_change_plan_or_graph(
    source_piece,
    source_graph,
) -> None:
    changed_piece = replace(
        source_piece,
        source_path="ignored/phase8a-sidecar.mid",
        targets=(
            TargetArray(
                target_id="target:phase8a-inert",
                task="quality.overall",
                annotation_view_id=None,
                alignment_type="piece",
                entity_ids=(source_piece.piece_id,),
                value_type="scalar",
                class_labels=None,
                values=(0.75,),
                mask=(True,),
                confidence=(1.0,),
                source=("synthetic",),
                provenance=(
                    source_piece.provenance[0].provenance_id,
                ),
            ),
        ),
        provenance=(
            replace(
                source_piece.provenance[0],
                source="phase8a_test_provenance_mutation",
                details=(("diagnostic", "changed"),),
            ),
        ),
        quality_flags=(
            QualityFlag(
                code="phase8a.test.diagnostic",
                severity="info",
                message="target-blind hierarchy diagnostic mutation",
                entity_ids=(source_piece.piece_id,),
                provenance_id=source_piece.provenance[0].provenance_id,
            ),
        ),
    )
    changed_graph = build_raw_graph(changed_piece)
    source = _plan(
        source_graph,
        source_piece,
        ONSET_PITCH_DESCENDANTS,
    )
    changed = _plan(
        changed_graph,
        changed_piece,
        ONSET_PITCH_DESCENDANTS,
    )

    assert graph_fingerprint(changed_graph) == graph_fingerprint(source_graph)
    assert changed == source
    assert build_feature_mask_overlay(
        changed_graph,
        changed,
    ) == build_feature_mask_overlay(source_graph, source)


def test_hierarchy_overlay_closes_peer_and_owner_track_pitch_leakage(
    source_graph,
    source_piece,
) -> None:
    before = graph_fingerprint(source_graph)
    plan = _plan(
        source_graph,
        source_piece,
        TRACK_BAR_PITCH_SPAN,
    )
    overlay = build_feature_mask_overlay(source_graph, plan)
    slots = {
        (slot.node_type, slot.kind, slot.feature_name): slot
        for slot in overlay.slot_masks
    }
    selected = plan.selected_local_note_indices
    owner_track_notes = _adjacency(
        source_graph,
        _TRACK_CONTAINS_NOTE,
    )[plan.selected_local_track_index]

    for feature in ("pitch", "pitch_class", "octave"):
        assert slots[
            ("note", "categorical", feature)
        ].global_node_indices == selected
    assert slots[
        ("note", "continuous", "track_relative_pitch")
    ].global_node_indices == owner_track_notes
    for feature in ("mean_pitch", "pitch_std", "min_pitch", "max_pitch"):
        assert slots[
            ("track", "continuous", feature)
        ].global_node_indices == (plan.selected_local_track_index,)

    token = torch.tensor([-3.0, -2.0, -1.0, 0.0])
    bound = overlay.bind(token)
    rows = int(source_graph["note"].num_nodes)
    values = torch.arange(rows * 4, dtype=torch.float32).reshape(rows, 4)
    availability = values + 1000.0
    masked_values, masked_availability = (
        bound.replace_feature_contributions(
            node_type="note",
            kind="continuous",
            feature_name="track_relative_pitch",
            value_contribution=values,
            availability_contribution=availability,
        )
    )
    row_mask = overlay.feature_row_mask(
        node_type="note",
        kind="continuous",
        feature_name="track_relative_pitch",
    )

    assert tuple(row_mask.nonzero(as_tuple=False).flatten().tolist()) == (
        owner_track_notes
    )
    assert torch.equal(
        masked_values[row_mask],
        token.expand(int(row_mask.count_nonzero()), -1),
    )
    assert torch.equal(
        masked_availability[row_mask],
        torch.zeros_like(masked_availability[row_mask]),
    )
    assert graph_fingerprint(source_graph) == before
    assert tuple(
        mask.reason for mask in plan.collateral_feature_masks
    ) == (
        NOTE_PITCH_GROUP.peer_note_collateral_reason,
        NOTE_PITCH_GROUP.collateral_reason,
    )


def test_phase7a_control_plan_overlay_and_prepared_binding_remain_exact(
    bounded_fixture: Phase7ABoundedFixture,
) -> None:
    sample = bounded_fixture.raw_samples("train")[0]
    piece = bounded_fixture.train_pieces[0]
    direct = build_hierarchy_mask_plan(
        sample.raw_graph,
        dataset_id=sample.dataset_id,
        piece_id=sample.piece_id,
        policy=INDEPENDENT_NOTE_PITCH,
        global_seed=42,
        epoch=0,
        requested_mask_rate=0.30,
        stage="train",
        policy_config=_single_policy_config(INDEPENDENT_NOTE_PITCH),
    )
    from music_critic.ssl.masking import build_mask_plan

    legacy = build_mask_plan(
        sample.raw_graph,
        dataset_id=sample.dataset_id,
        piece_id=sample.piece_id,
        global_seed=42,
        epoch=0,
        requested_mask_rate=0.30,
        stage="train",
    )

    assert type(direct) is MaskPlan
    assert direct == legacy
    assert direct.to_dict() == legacy.to_dict()
    assert direct.fingerprint == (
        "f07c83364859e4f28b499d821985f9fb20c3be866c4d5e6f4bea237d3e16647c"
    )
    assert build_feature_mask_overlay(
        sample.raw_graph,
        direct,
    ) == build_feature_mask_overlay(sample.raw_graph, legacy)

    batch = collate_ssl_samples((sample,))
    legacy_binding = prepare_mask_binding(
        batch,
        global_seed=42,
        epoch=0,
        requested_mask_rate=0.30,
        stage="train",
    )
    hierarchy_control_binding = prepare_hierarchy_mask_binding(
        batch,
        policy_config=_single_policy_config(INDEPENDENT_NOTE_PITCH),
        global_seed=42,
        epoch=0,
        requested_mask_rate=0.30,
        stage="train",
    )

    assert hierarchy_control_binding.to_dict() == legacy_binding.to_dict()
    assert hierarchy_control_binding.fingerprint == legacy_binding.fingerprint
    assert hierarchy_control_binding.mask_plans == legacy_binding.mask_plans
    assert (
        hierarchy_control_binding.feature_overlay
        == legacy_binding.feature_overlay
    )
