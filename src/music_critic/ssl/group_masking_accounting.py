"""Semantic validation of a completed pilot update's compact accounting.

The schedule binds the compact batch evidence and per-head target population
as well as a content digest of the full prepared inputs. This checks arithmetic
and structural consistency. Authenticating exact losses, mask selections or
relabeling within a fixed target population requires replay of the inputs.
"""
from __future__ import annotations

import math

from music_critic.experiments.analysisgnn.corrected_training import ACTIVE_HEADS
from music_critic.experiments.analysisgnn.training_policy import (
    AUXILIARY_HEADS, GROUP_WEIGHTS, PRIMARY_HEADS,
)
from music_critic.ssl.objective import ANTI_COLLAPSE_DIAGNOSTICS_CONTRACT_VERSION


_POLICIES = ("independent_note_pitch", "beat_pitch_descendants")
_ROW_KEYS = {
    "update", "batch_binding", "batch_evidence", "encoder_forwards", "paired_masks",
    "optimizer_gradient_parameter_indices", "ssl_loss", "theory_loss", "total_loss",
    "counts", "collapse", "theory_heads",
}
_COUNT_KEYS = {
    "ssl_samples", "ssl_selected_notes", "theory_samples", "theory_valid_rows",
    "theory_masked_rows", "theory_unsupported_rows", "theory_active_heads",
}
_MASK_KEYS = {
    "piece", "plan_fingerprint", "hidden_notes", "eligible_notes", "hidden_fraction",
    "hidden_units", "unit", "candidate_units", "collateral_rows", "requested_fraction",
}
_DIAGNOSTIC_KEYS = {
    "contract_version", "aggregation_scope", "row_count", "embedding_dim",
    "source_dtype", "accumulation_dtype", "target_embedding_variance",
    "prediction_embedding_variance", "target_mean_norm", "prediction_mean_norm",
    "target_zero_norm_count", "prediction_zero_norm_count",
    "target_mean_off_diagonal_cosine", "prediction_mean_off_diagonal_cosine",
    "unavailable_reason", "pairwise_unavailable_reason", "pairwise_policy",
}


def _require(condition, reason):
    if not condition:
        raise ValueError("group_pilot.resume_" + reason)


def _mapping(value, keys, reason="accounting_schema_invalid"):
    _require(type(value) is dict and set(value) == set(keys), reason)


def _integer(value, *, minimum=0, maximum=None):
    return type(value) is int and minimum <= value and (maximum is None or value <= maximum)


def _finite(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _sha(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _batch(value, config):
    if value is None:
        return
    _mapping(value, {"identities", "note_counts", "node_count", "edge_count"}, "batch_accounting_schema_invalid")
    identities, notes = value["identities"], value["note_counts"]
    _require(type(identities) is list and 1 <= len(identities) <= config.max_samples_per_batch,
             "batch_sample_count_invalid")
    _require(all(type(identity) is list and len(identity) == 2 and all(
        type(part) is str and bool(part) and part == part.strip() for part in identity
    ) for identity in identities), "batch_identity_invalid")
    # A draw may repeat an identity; the ordered occurrence is the binding.
    _require(type(notes) is list and len(notes) == len(identities) and all(
        _integer(n) for n in notes) and sum(notes) <= config.max_notes_per_batch,
        "batch_note_count_invalid")
    _require(_integer(value["node_count"], minimum=sum(notes), maximum=config.max_nodes_per_batch)
             and _integer(value["edge_count"], maximum=config.max_edges_per_batch),
             "batch_resource_count_invalid")


def _masks(value, evidence, config, policy):
    _mapping(value, _POLICIES, "paired_masks_invalid")
    for name in _POLICIES:
        plans = value[name]
        _require(type(plans) is list and len(plans) == len(evidence["identities"]),
                 "mask_sample_count_invalid")
        for plan, identity, notes in zip(plans, evidence["identities"], evidence["note_counts"], strict=True):
            _mapping(plan, _MASK_KEYS, "mask_schema_invalid")
            _require(type(plan["piece"]) is list and plan["piece"] == identity
                     and _sha(plan["plan_fingerprint"]), "mask_identity_invalid")
            _require(_integer(plan["eligible_notes"], minimum=2)
                     and plan["eligible_notes"] == notes
                     and _integer(plan["hidden_notes"], minimum=1, maximum=notes - 1),
                     "mask_note_count_invalid")
            _require(_finite(plan["hidden_fraction"])
                     and plan["hidden_fraction"] == plan["hidden_notes"] / notes
                     and _finite(plan["requested_fraction"])
                     and plan["requested_fraction"] == config.mask_rate, "mask_density_invalid")
            _require(_integer(plan["candidate_units"], minimum=1, maximum=notes)
                     and _integer(plan["hidden_units"], minimum=1,
                                  maximum=min(plan["candidate_units"], plan["hidden_notes"])),
                     "mask_unit_count_invalid")
            _require(plan["unit"] == ("note" if name == _POLICIES[0] else "beat"),
                     "mask_unit_invalid")
            if name == _POLICIES[0]:
                _require(plan["candidate_units"] == notes
                         and plan["hidden_units"] == plan["hidden_notes"]
                         and plan["hidden_notes"] == max(1, math.floor(notes * config.mask_rate)),
                         "independent_mask_count_invalid")
            else:
                # Nonempty beat units have disjoint descendants. At least one
                # unit remains visible; their exact sizes require raw structure.
                _require(plan["hidden_units"] < plan["candidate_units"], "group_mask_count_invalid")
            collateral = plan["collateral_rows"]
            _mapping(collateral, {"note", "track"}, "mask_collateral_schema_invalid")
            _require(_integer(collateral["note"], maximum=notes - plan["hidden_notes"])
                     and _integer(collateral["track"], minimum=1,
                                  maximum=min(notes, plan["hidden_notes"]))
                     and collateral["note"] + collateral["track"] <= evidence["node_count"],
                     "mask_collateral_count_invalid")
    return sum(plan["hidden_notes"] for plan in value[policy])


def _collapse(value, evidence, selected_notes):
    _mapping(value, {"note", "bar", "song"}, "collapse_schema_invalid")
    for level, diagnostic in value.items():
        _mapping(diagnostic, _DIAGNOSTIC_KEYS, "collapse_schema_invalid")
        rows = diagnostic["row_count"]
        _require(_integer(rows, minimum=1, maximum=evidence["node_count"])
                 and _integer(diagnostic["embedding_dim"], minimum=1), "collapse_dimensions_invalid")
        if level == "note":
            _require(rows == selected_notes, "collapse_note_count_invalid")
        elif level == "song":
            _require(rows == len(evidence["identities"]), "collapse_song_count_invalid")
        _require(diagnostic["contract_version"] == ANTI_COLLAPSE_DIAGNOSTICS_CONTRACT_VERSION
                 and diagnostic["aggregation_scope"] == "single_batch"
                 and diagnostic["pairwise_policy"] == "exact_linear_normalized_sum"
                 and diagnostic["source_dtype"] == "float32"
                 and diagnostic["accumulation_dtype"] == "input_dtype"
                 and diagnostic["unavailable_reason"] is None, "collapse_contract_invalid")
        for prefix in ("target", "prediction"):
            _require(_integer(diagnostic[prefix + "_zero_norm_count"], maximum=rows),
                     "collapse_zero_count_invalid")
            for suffix in ("embedding_variance", "mean_norm"):
                metric = diagnostic[prefix + "_" + suffix]
                _require(_finite(metric) and metric >= 0, "collapse_metric_invalid")
            cosine = diagnostic[prefix + "_mean_off_diagonal_cosine"]
            if rows == 1:
                _require(cosine is None and diagnostic["pairwise_unavailable_reason"] == "fewer_than_two_rows",
                         "collapse_pairwise_invalid")
            else:
                _require(_finite(cosine) and -1.000001 <= cosine <= 1.000001
                         and diagnostic["pairwise_unavailable_reason"] is None, "collapse_pairwise_invalid")


def _heads(value, counts, loss, target_rows):
    _mapping(value, ACTIVE_HEADS, "theory_head_schema_invalid")
    for task, head in value.items():
        _mapping(head, {"weighted_ce", "valid", "masked", "unsupported"}, "theory_head_schema_invalid")
        _require(all(_integer(head[key]) for key in ("valid", "masked", "unsupported")),
                 "theory_head_count_invalid")
        ce = head["weighted_ce"]
        _require((_finite(ce) and ce >= 0) if head["valid"] else ce is None,
                 "theory_head_loss_invalid")
        _require(sum(head[key] for key in ("valid", "masked", "unsupported")) == target_rows[task],
                 "theory_head_population_mismatch")
    for name in ("valid", "masked", "unsupported"):
        _require(counts["theory_" + name + "_rows"] == sum(h[name] for h in value.values()),
                 "theory_head_accounting_mismatch")
    active = sum(h["weighted_ce"] is not None for h in value.values())
    _require(active > 0 and counts["theory_active_heads"] == active, "theory_active_head_count_invalid")
    expected = 0.
    for names, scale in ((PRIMARY_HEADS, 1.), (AUXILIARY_HEADS, GROUP_WEIGHTS["auxiliary"])):
        terms = [value[name]["weighted_ce"] for name in names if value[name]["weighted_ce"] is not None]
        if terms:
            expected += scale * sum(terms) / len(terms)
    _require(math.isclose(loss, expected, rel_tol=1e-6, abs_tol=1e-6), "theory_head_loss_aggregate_invalid")


def validate_successful_row(row, *, config, policy, scheduled_binding, update_index) -> None:
    """Reject malformed or internally inconsistent successful update evidence.

    ``update_index`` is zero-based. Finite optimizer slots and exact parameter
    inventory are checked by the pilot, which owns those actual tensors.
    """
    from music_critic.ssl.group_masking_pilot import tree_fingerprint

    _mapping(row, _ROW_KEYS)
    _require(policy in _POLICIES and _integer(update_index, maximum=config.updates - 1)
             and _integer(row["update"], minimum=1) and row["update"] == update_index + 1,
             "schedule_prefix_mismatch")
    wrapper = row["batch_evidence"]
    _mapping(wrapper, {"content_fingerprint", "accounting"}, "batch_evidence_schema_invalid")
    _require(_sha(wrapper["content_fingerprint"]), "batch_content_digest_invalid")
    accounting = wrapper["accounting"]
    _mapping(accounting, {"ssl", "theory", "theory_target_rows"}, "batch_accounting_schema_invalid")
    for branch in ("ssl", "theory"):
        _batch(accounting[branch], config)
    target_rows = accounting["theory_target_rows"]
    if accounting["theory"] is None:
        _require(target_rows is None, "absent_theory_target_population")
    else:
        _mapping(target_rows, ACTIVE_HEADS, "theory_target_population_schema_invalid")
        _require(all(_integer(value) for value in target_rows.values()), "theory_target_population_count_invalid")
    _require(_sha(row["batch_binding"]) and _sha(scheduled_binding)
             and tree_fingerprint(wrapper) == row["batch_binding"] == scheduled_binding,
             "batch_evidence_binding_mismatch")
    ssl_active, theory_active = config.lambda_ssl > 0, config.lambda_theory > 0
    expected_forwards = 2 * int(ssl_active) + int(theory_active)
    _require(_integer(row["encoder_forwards"]) and row["encoder_forwards"] == expected_forwards,
             "forward_count_invalid")
    counts = row["counts"]
    _mapping(counts, _COUNT_KEYS, "count_schema_invalid")
    _require(all(_integer(value) for value in counts.values()), "count_range_invalid")
    total = 0.
    for branch, coefficient in (("ssl", config.lambda_ssl), ("theory", config.lambda_theory)):
        loss, evidence = row[branch + "_loss"], wrapper["accounting"][branch]
        if coefficient:
            _require(evidence is not None and counts[branch + "_samples"] == len(evidence["identities"]),
                     "active_batch_sample_count_invalid")
            _require(_finite(loss), "loss_invalid")
            total += coefficient * loss
        else:
            _require(loss is None and all(value == 0 for key, value in counts.items() if key.startswith(branch + "_")),
                     "inactive_loss_or_accounting")
    _require(_finite(row["total_loss"]) and math.isclose(row["total_loss"], total, rel_tol=1e-6, abs_tol=1e-6),
             "total_loss_invalid")
    if ssl_active:
        evidence = wrapper["accounting"]["ssl"]
        selected = _masks(row["paired_masks"], evidence, config, policy)
        _require(counts["ssl_selected_notes"] == selected, "selected_note_count_invalid")
        _collapse(row["collapse"], evidence, selected)
    else:
        _require(type(row["paired_masks"]) is dict and not row["paired_masks"]
                 and row["collapse"] is None, "inactive_ssl_accounting")
    if theory_active:
        _heads(row["theory_heads"], counts, row["theory_loss"], target_rows)
    else:
        _require(row["theory_heads"] is None, "inactive_theory_accounting")
    indices = row["optimizer_gradient_parameter_indices"]
    _require(type(indices) is list and bool(indices) and all(_integer(i) for i in indices)
             and indices == sorted(set(indices)), "gradient_inventory_invalid")
