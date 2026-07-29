#!/usr/bin/env python3
"""Bounded Phase 8A masking mechanics; never speed or quality acceptance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import Any, Callable, TypeVar

import torch
from torch_geometric.data import Batch

from music_critic.graph import MANDATORY_EDGE_TYPES, MANDATORY_NODE_TYPES
from music_critic.models import HierarchicalBaselineConfig
from music_critic.ssl.contracts import MaskPlan
from music_critic.ssl.data import SSLBatch, collate_ssl_samples
from music_critic.ssl.hierarchy_fixture import (
    build_phase8a_hierarchy_fixture,
)
from music_critic.ssl.hierarchical_masking import (
    BEAT_PITCH_DESCENDANTS,
    CONTIGUOUS_BAR_PITCH_SPAN,
    HIERARCHICAL_MASK_PLAN_CONTRACT_VERSION,
    HIERARCHY_MASK_POLICIES,
    HIERARCHY_MASK_POLICY_VERSION,
    HIERARCHY_POLICY_CONFIG_CONTRACT_VERSION,
    HIERARCHY_POLICY_MIXTURE_CONTRACT_VERSION,
    HIERARCHY_PREPARED_BINDING_PROFILE_VERSION,
    INDEPENDENT_NOTE_PITCH,
    ONSET_PITCH_DESCENDANTS,
    TRACK_BAR_PITCH_SPAN,
    HierarchicalMaskPlan,
    HierarchyMaskPolicyConfig,
    build_batched_hierarchy_mask_resolutions,
)
from music_critic.ssl.masking import (
    PREPARED_HIERARCHY_MASK_BINDING_CONTRACT_VERSION,
    move_ssl_batch_with_prepared_binding,
    prepare_hierarchy_mask_binding,
)
from music_critic.ssl.model import (
    PHASE8A_HIERARCHY_SSL_OUTPUT_CONTRACT_VERSION,
    MaskedGraphSSLConfig,
    MaskedGraphSSLModel,
)
from music_critic.ssl.views import build_feature_mask_overlay
PHASE8A_MASKING_BENCHMARK_CONTRACT_VERSION = "1.0.0"

_ONSET_STARTS_NOTE = ("onset", "starts_note", "note")
_BEAT_CONTAINS_ONSET = ("beat", "contains_onset", "onset")
_BAR_CONTAINS_BEAT = ("bar", "contains_beat", "beat")
_BAR_CONTAINS_ONSET = ("bar", "contains_onset", "onset")
_TRACK_CONTAINS_NOTE = ("track", "contains_note", "note")
_RELEVANT_RELATIONS = (
    _ONSET_STARTS_NOTE,
    _BEAT_CONTAINS_ONSET,
    _BAR_CONTAINS_BEAT,
    _BAR_CONTAINS_ONSET,
    _TRACK_CONTAINS_NOTE,
)

_T = TypeVar("_T")


def _canonical_json_bytes(value: object) -> int:
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _timed(
    operation: Callable[[], _T],
    *,
    repeats: int,
) -> tuple[_T, dict[str, float]]:
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
        raise ValueError("repeats must be a positive integer")
    elapsed: list[float] = []
    result: _T | None = None
    for _ in range(repeats):
        started = perf_counter()
        result = operation()
        elapsed.append(perf_counter() - started)
    assert result is not None
    return result, {
        "mean": fmean(elapsed),
        "min": min(elapsed),
        "max": max(elapsed),
    }


def make_phase8a_benchmark_batch() -> SSLBatch:
    """Return only the accepted bounded raw-only fixture batch."""

    fixture = build_phase8a_hierarchy_fixture()
    return collate_ssl_samples(fixture.raw_samples("train"))


def make_phase8a_benchmark_model(
    *,
    seed: int = 811,
) -> MaskedGraphSSLModel:
    """Build a small existing-objective model with no Phase 8B heads."""

    torch.manual_seed(seed)
    return MaskedGraphSSLModel(
        HierarchicalBaselineConfig(
            hidden_dim=8,
            local_gnn_layers=1,
            transformer_layers=1,
            attention_heads=2,
            ffn_multiplier=2,
            dropout=0.0,
        ),
        MaskedGraphSSLConfig(
            mask_rate=0.30,
            decoder_views=1,
            decoder_remask_probability=0.0,
            decoder_hidden_dim=8,
            projector_hidden_dim=8,
        ),
    )


def single_policy_config(policy: str) -> HierarchyMaskPolicyConfig:
    if policy not in HIERARCHY_MASK_POLICIES:
        raise ValueError(f"unknown Phase 8A policy {policy!r}")
    return HierarchyMaskPolicyConfig.create(
        weights={policy: 1.0},
        min_span_bars=1,
        max_span_bars=2,
    )


def _ptr(graph: Batch, node_type: str) -> tuple[int, ...]:
    return tuple(
        int(value)
        for value in graph[node_type].ptr.detach().tolist()
    )


def _edge_pairs(
    graph: Batch,
    edge_type: tuple[str, str, str],
) -> tuple[tuple[int, int], ...]:
    rows = graph[edge_type].edge_index.detach().tolist()
    return tuple(
        (int(source), int(target))
        for source, target in zip(rows[0], rows[1], strict=True)
    )


def _append_index(
    pairs: tuple[tuple[int, int], ...],
) -> dict[int, tuple[int, ...]]:
    mutable: dict[int, list[int]] = {}
    for source, target in pairs:
        mutable.setdefault(source, []).append(target)
    return {
        source: tuple(sorted(targets))
        for source, targets in mutable.items()
    }


def _build_relation_index(graph: Batch) -> dict[str, object]:
    """Build a benchmark-only sparse index from the same raw relations."""

    onset_notes = _append_index(_edge_pairs(graph, _ONSET_STARTS_NOTE))
    beat_onsets = _append_index(
        _edge_pairs(graph, _BEAT_CONTAINS_ONSET)
    )
    bar_onsets = _append_index(
        _edge_pairs(graph, _BAR_CONTAINS_ONSET)
    )
    note_owner_track: dict[int, int] = {}
    for track, note in _edge_pairs(graph, _TRACK_CONTAINS_NOTE):
        if note in note_owner_track:
            raise ValueError("benchmark encountered duplicate note ownership")
        note_owner_track[note] = track
    return {
        "ptr": {
            node_type: _ptr(graph, node_type)
            for node_type in ("track", "bar", "beat", "onset", "note")
        },
        "onset_notes": onset_notes,
        "beat_onsets": beat_onsets,
        "bar_onsets": bar_onsets,
        "note_owner_track": note_owner_track,
    }


def _notes_from_onsets(
    onset_indices: set[int],
    onset_notes: dict[int, tuple[int, ...]],
) -> set[int]:
    return {
        note
        for onset in onset_indices
        for note in onset_notes.get(onset, ())
    }


def _resolve_selected_descendants(
    plans: tuple[MaskPlan | HierarchicalMaskPlan, ...],
    index: dict[str, object],
) -> tuple[tuple[int, ...], ...]:
    """Re-resolve selected evidence to benchmark descendant work separately."""

    ptr = index["ptr"]
    assert isinstance(ptr, dict)
    onset_notes = index["onset_notes"]
    beat_onsets = index["beat_onsets"]
    bar_onsets = index["bar_onsets"]
    note_owner_track = index["note_owner_track"]
    assert isinstance(onset_notes, dict)
    assert isinstance(beat_onsets, dict)
    assert isinstance(bar_onsets, dict)
    assert isinstance(note_owner_track, dict)
    result: list[tuple[int, ...]] = []
    for sample_index, plan in enumerate(plans):
        note_start = ptr["note"][sample_index]
        if isinstance(plan, MaskPlan):
            local_notes = plan.selected_local_node_indices
        else:
            units = plan.selected_local_unit_indices
            policy = plan.resolved_policy
            if policy == ONSET_PITCH_DESCENDANTS:
                onset_start = ptr["onset"][sample_index]
                global_notes = _notes_from_onsets(
                    {onset_start + unit for unit in units},
                    onset_notes,
                )
            elif policy == BEAT_PITCH_DESCENDANTS:
                beat_start = ptr["beat"][sample_index]
                global_onsets = {
                    onset
                    for unit in units
                    for onset in beat_onsets.get(beat_start + unit, ())
                }
                global_notes = _notes_from_onsets(
                    global_onsets,
                    onset_notes,
                )
            elif policy in {
                CONTIGUOUS_BAR_PITCH_SPAN,
                TRACK_BAR_PITCH_SPAN,
            }:
                bar_start = ptr["bar"][sample_index]
                global_onsets = {
                    onset
                    for unit in units
                    for onset in bar_onsets.get(bar_start + unit, ())
                }
                global_notes = _notes_from_onsets(
                    global_onsets,
                    onset_notes,
                )
                if policy == TRACK_BAR_PITCH_SPAN:
                    selected_track = plan.selected_local_track_index
                    assert selected_track is not None
                    owner = ptr["track"][sample_index] + selected_track
                    global_notes = {
                        note
                        for note in global_notes
                        if note_owner_track.get(note) == owner
                    }
            else:  # pragma: no cover - exact contracts reject this first.
                raise ValueError(f"unsupported hierarchy policy {policy!r}")
            local_notes = tuple(
                sorted(note - note_start for note in global_notes)
            )
        if local_notes != plan.selected_local_node_indices:
            raise ValueError(
                "benchmark descendant resolution differs from the plan"
            )
        result.append(local_notes)
    return tuple(result)


def _graph_counts(graph: Batch) -> dict[str, object]:
    node_counts = {
        node_type: int(graph[node_type].num_nodes)
        for node_type in MANDATORY_NODE_TYPES
    }
    edge_counts = {
        "|".join(edge_type): int(
            graph[edge_type].edge_index.shape[1]
        )
        for edge_type in MANDATORY_EDGE_TYPES
    }
    relevant_edge_counts = {
        "|".join(edge_type): edge_counts["|".join(edge_type)]
        for edge_type in _RELEVANT_RELATIONS
    }
    return {
        "sample_count": int(graph.num_graphs),
        "node_counts": node_counts,
        "edge_counts": edge_counts,
        "total_nodes": sum(node_counts.values()),
        "total_edges": sum(edge_counts.values()),
        "relevant_edge_counts": relevant_edge_counts,
        "total_relevant_edges": sum(relevant_edge_counts.values()),
    }


def benchmark_phase8a_policy(
    policy: str,
    *,
    repeats: int = 3,
    batch: SSLBatch | None = None,
) -> dict[str, object]:
    """Measure one policy without imposing any timing acceptance threshold."""

    if batch is None:
        batch = make_phase8a_benchmark_batch()
    graph = batch.raw_graph_batch
    if not isinstance(graph, Batch):
        raise TypeError("Phase 8A benchmark requires a PyG Batch")
    config = single_policy_config(policy)
    planner_arguments = {
        "dataset_ids": batch.dataset_ids,
        "piece_ids": batch.piece_ids,
        "global_seed": 42,
        "epoch": 0,
        "requested_mask_rate": 0.30,
        "stage": "train",
        "policy_config": config,
    }
    resolutions, plan_timing = _timed(
        lambda: build_batched_hierarchy_mask_resolutions(
            graph,
            **planner_arguments,
        ),
        repeats=repeats,
    )
    if any(resolution.plan is None for resolution in resolutions):
        raise ValueError(
            "bounded benchmark policy unexpectedly unavailable"
        )
    plans = tuple(
        resolution.plan
        for resolution in resolutions
        if resolution.plan is not None
    )
    relation_index, index_timing = _timed(
        lambda: _build_relation_index(graph),
        repeats=repeats,
    )
    descendants, descendant_timing = _timed(
        lambda: _resolve_selected_descendants(plans, relation_index),
        repeats=repeats,
    )
    overlay, overlay_timing = _timed(
        lambda: build_feature_mask_overlay(graph, plans),
        repeats=repeats,
    )
    binding, prepared_binding_timing = _timed(
        lambda: prepare_hierarchy_mask_binding(
            batch,
            policy_config=config,
            global_seed=42,
            epoch=0,
            requested_mask_rate=0.30,
            stage="train",
        ),
        repeats=repeats,
    )
    moved_batch, moved_binding = move_ssl_batch_with_prepared_binding(
        batch,
        binding,
        "cpu",
    )
    model = make_phase8a_benchmark_model(
        seed=811 + HIERARCHY_MASK_POLICIES.index(policy)
    ).cpu().eval()
    with torch.no_grad():
        output, forward_timing = _timed(
            lambda: model.forward_hierarchy(
                moved_batch,
                prepared_mask_binding=moved_binding,
            ),
            repeats=repeats,
        )
    if output.objective.total_loss is None:
        raise ValueError("bounded prepared forward has no finite objective")
    if not bool(torch.isfinite(output.objective.total_loss)):
        raise ValueError("bounded prepared forward objective is non-finite")

    plan_json_bytes = tuple(
        _canonical_json_bytes(plan.to_dict()) for plan in plans
    )
    resolution_json_bytes = tuple(
        _canonical_json_bytes(resolution.to_dict())
        for resolution in resolutions
    )
    candidate_count = sum(
        (
            plan.maskable_node_count
            if isinstance(plan, MaskPlan)
            else plan.selection.candidate_count
        )
        for plan in plans
    )
    primary_entries = sum(len(value) for value in descendants)
    collateral_note_entries = sum(
        len(mask.local_node_indices)
        for plan in plans
        for mask in plan.collateral_feature_masks
        if mask.node_type == "note"
    )
    collateral_track_entries = sum(
        len(mask.local_node_indices)
        for plan in plans
        for mask in plan.collateral_feature_masks
        if mask.node_type == "track"
    )
    overlay_entries = sum(
        len(slot.global_node_indices) for slot in overlay.slot_masks
    )
    return {
        "benchmark_contract_version": (
            PHASE8A_MASKING_BENCHMARK_CONTRACT_VERSION
        ),
        "scope": (
            "bounded CPU mechanics only; no speed threshold, throughput "
            "claim, GPU claim, or representation-quality claim"
        ),
        "policy": policy,
        "policy_config": config.to_dict(),
        "contract_versions": {
            "hierarchical_mask_plan": (
                HIERARCHICAL_MASK_PLAN_CONTRACT_VERSION
            ),
            "hierarchy_mask_policy": HIERARCHY_MASK_POLICY_VERSION,
            "hierarchy_policy_config": (
                HIERARCHY_POLICY_CONFIG_CONTRACT_VERSION
            ),
            "hierarchy_policy_mixture": (
                HIERARCHY_POLICY_MIXTURE_CONTRACT_VERSION
            ),
            "hierarchy_prepared_binding_profile": (
                HIERARCHY_PREPARED_BINDING_PROFILE_VERSION
            ),
            "hierarchy_prepared_binding": (
                PREPARED_HIERARCHY_MASK_BINDING_CONTRACT_VERSION
            ),
            "hierarchy_ssl_output": (
                PHASE8A_HIERARCHY_SSL_OUTPUT_CONTRACT_VERSION
            ),
        },
        "repeats": repeats,
        "timing_seconds": {
            "plan_construction": plan_timing,
            "relation_index_construction": index_timing,
            "selected_descendant_resolution": descendant_timing,
            "overlay_construction": overlay_timing,
            "prepared_binding_construction": (
                prepared_binding_timing
            ),
            "prepared_forward": forward_timing,
        },
        "measurement_boundary": {
            "plan_construction": (
                "public CPU planner including its validated sparse index"
            ),
            "relation_index_construction": (
                "benchmark-only sparse index used to independently "
                "re-resolve selected descendants"
            ),
            "selected_descendant_resolution": (
                "selected units to exact note descendants; no model compute"
            ),
            "overlay_construction": (
                "immutable row/field overlay; no encoder compute"
            ),
            "prepared_forward": (
                "attested target plus masked-online Phase 7A objective "
                "forward; no backward"
            ),
            "retained_plan_metadata": (
                "canonical compact JSON UTF-8 retained-state proxy; "
                "the batch sum is the peak simultaneously retained plan "
                "state in this call, not Python heap, temporary peak, RSS, "
                "CUDA allocation, or checkpoint size"
            ),
        },
        "counts": {
            **_graph_counts(graph),
            "candidate_units_or_control_notes": candidate_count,
            "primary_descendant_note_entries": primary_entries,
            "collateral_peer_note_entries": (
                collateral_note_entries
            ),
            "collateral_owner_track_entries": (
                collateral_track_entries
            ),
            "emitted_overlay_row_field_entries": overlay_entries,
        },
        "retained_metadata": {
            "plan_json_bytes_by_sample": list(plan_json_bytes),
            "resolution_json_bytes_by_sample": list(
                resolution_json_bytes
            ),
            "max_single_plan_metadata_json_bytes": max(
                plan_json_bytes,
                default=0,
            ),
            "peak_retained_batch_plan_metadata_json_bytes": sum(
                plan_json_bytes
            ),
            "batch_plan_metadata_json_bytes": sum(plan_json_bytes),
            "batch_resolution_metadata_json_bytes": sum(
                resolution_json_bytes
            ),
            "prepared_binding_public_json_bytes": (
                _canonical_json_bytes(binding.to_dict())
            ),
        },
        "fingerprints": {
            "plans": [plan.fingerprint for plan in plans],
            "resolutions": [
                resolution.fingerprint
                for resolution in resolutions
            ],
            "overlay": overlay.fingerprint,
            "prepared_binding": binding.fingerprint,
        },
        "resolved_policies": [
            resolution.resolved_policy
            for resolution in resolutions
        ],
        "finite_existing_objective_forward": True,
        "device": "cpu",
        "timing_acceptance_thresholds": None,
        "gpu_measurement": None,
    }


def benchmark_phase8a_suite(
    *,
    repeats: int = 3,
) -> dict[str, object]:
    batch = make_phase8a_benchmark_batch()
    policies = {
        policy: benchmark_phase8a_policy(
            policy,
            repeats=repeats,
            batch=batch,
        )
        for policy in HIERARCHY_MASK_POLICIES
    }
    return {
        "benchmark_contract_version": (
            PHASE8A_MASKING_BENCHMARK_CONTRACT_VERSION
        ),
        "scope": (
            "Phase 8A bounded deterministic masking mechanics; "
            "Phase 8B objectives and comparative training are absent"
        ),
        "policies": policies,
        "all_policies_measured": tuple(policies)
        == HIERARCHY_MASK_POLICIES,
        "timing_acceptance_thresholds": None,
        "gpu_measurement": None,
        "independent_control_policy": INDEPENDENT_NOTE_PITCH,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path; stdout is always emitted.",
    )
    arguments = parser.parse_args()
    report = benchmark_phase8a_suite(repeats=arguments.repeats)
    text = json.dumps(
        report,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    )
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            text + "\n",
            encoding="utf-8",
            newline="\n",
        )
    print(text)


if __name__ == "__main__":
    main()
