from __future__ import annotations

import torch

from music_critic.graph import build_raw_graph
from scripts.audit_dilemmadata_coverage_remediation import (
    _acyclic_next_in_track,
    _context_gate,
    _target_smoke_selection,
)
from tests.adapters.test_dilemmadata import CORPUS, _accepted, _discover


def test_remediation_audit_graph_gates_cover_context_and_cycles() -> None:
    accepted = _accepted(CORPUS, "an:training:same")
    graph = build_raw_graph(accepted.piece, assume_valid=True)
    context = _context_gate(graph)
    assert context["each_note_has_one_bar"] is True
    assert context["each_note_has_beat_context"] is True
    assert _acyclic_next_in_track(graph)

    cyclic = graph.clone()
    cyclic[("note", "next_in_track", "note")].edge_index = torch.tensor(
        [[0], [0]],
        dtype=torch.long,
    )
    assert not _acyclic_next_in_track(cyclic)


def test_target_smoke_selection_is_deterministic_and_avoids_source_test_splits() -> None:
    records = {record.record_id: record for record in _discover(CORPUS).records}
    accepted = set(records)
    reasons = {
        "an:training:same": "tie_predecessor_missing",
        "an:validation:same-alt": "ambiguous_measure_mapping",
        "dlc:demo:same": "unsupported_leading_partial_measure",
    }
    repairs = {
        "an:training:same": ("orphan_tie_continuation_promoted_to_attack",),
        "an:validation:same-alt": ("ambiguous_measure_mapping_selected",),
        "dlc:demo:same": ("leading_partial_measure_structural_padding",),
    }
    first = _target_smoke_selection(records, accepted, repairs, reasons)
    second = _target_smoke_selection(records, accepted, repairs, reasons)
    assert first == second
    assert "an:validation:same-alt" not in first
    assert first == ("an:training:same", "dlc:demo:same")
