"""Count-only context diagnostics; callers own artifact hashes and registration.

No model, dataset reader, training operation, or file write is reachable here.
POP source groups are the resampling units; Dilemma marginals cannot supply
event confusions or component uncertainty. These functions accept only the
explicit retained schemas and never infer softmax probabilities from counts.
"""

from collections import Counter, defaultdict
from fractions import Fraction
from random import Random


VIEWS = ("native", "metadata_removed")
TASK_CLASSES = {"root": 12, "quality": 17}
D7_CLASS_ID = 6


def _ratio(numerator, denominator, reason):
    return {"numerator": numerator, "denominator": denominator,
            "value": numerator / denominator if denominator else None,
            "reason": None if denominator else reason}


def _count(value):
    if type(value) is not int or value < 0:
        raise ValueError("context_artifacts.nonnegative_integer_count_required")
    return value


def marginal_metrics(rows, *, unknown_predicted_mass=0):
    """Derive exact ratios from explicit class_id/support/predicted_count/TP."""
    unknown_predicted_mass = _count(unknown_predicted_mass)
    result = []
    seen = set()
    for row in rows:
        class_id = _count(row["class_id"])
        if class_id in seen:
            raise ValueError("context_artifacts.duplicate_class")
        seen.add(class_id)
        support = _count(row["support"])
        predicted = None if row["predicted_count"] is None else _count(row["predicted_count"])
        tp = _count(row["true_positive"])
        if tp > support or (predicted is None and tp != 0) or (predicted is not None and tp > predicted):
            raise ValueError("context_artifacts.impossible_marginals")
        unknown = {"numerator": tp, "denominator": None, "value": None, "reason": "predicted_class_count_not_recoverable"}
        f1 = ({"numerator": 0, "denominator": None, "value": 0.0 if support else None,
               "reason": "zero_tp_with_gold_support" if support else "predicted_class_count_not_recoverable"}
              if predicted is None else _ratio(2 * tp, support + predicted, "no_true_or_predicted_support"))
        result.append({"class_id": class_id, "label": row.get("label"),
                       "support": support, "predicted_count": predicted,
                       "true_positive": tp, "false_positive": None if predicted is None else predicted - tp,
                       "false_negative": support - tp,
                       "precision": unknown if predicted is None else _ratio(tp, predicted, "no_predictions"),
                       "recall": _ratio(tp, support, "no_gold_support"),
                       "f1": f1})
    result.sort(key=lambda row: row["class_id"])
    total = sum(row["support"] for row in result)
    if total != sum(row["predicted_count"] or 0 for row in result) + unknown_predicted_mass:
        raise ValueError("context_artifacts.incomplete_class_marginals")
    if unknown_predicted_mass and all(row["predicted_count"] is not None for row in result):
        raise ValueError("context_artifacts.unallocated_mass_without_unknown_classes")
    included = [row for row in result if row["support"]]
    return {"per_class": result, "unallocated_predicted_count": unknown_predicted_mass,
            "accuracy": _ratio(sum(row["true_positive"] for row in result), total, "no_eligible_events"),
            "observed_true_class_macro_f1": _ratio(
                sum(row["f1"]["value"] for row in included), len(included), "no_observed_true_classes"),
            "included_class_ids": [row["class_id"] for row in included]}


def _task_summary(rows, task):
    count = TASK_CLASSES[task]
    confusion = [[0] * count for _ in range(count)]
    for row in rows:
        value = row[task]
        if value["reason"] is None:
            confusion[value["gold"]][value["prediction"]] += 1
    margins = [{"class_id": i, "support": sum(confusion[i]),
                "predicted_count": sum(row[i] for row in confusion),
                "true_positive": confusion[i][i]} for i in range(count)]
    return {**marginal_metrics(margins), "confusion_class_ids": list(range(count)),
            "confusion": confusion, "excluded_events": len(rows) - sum(map(sum, confusion))}


def _identity(row):
    return (row["piece_id"], row["event_id"])


def _fixed_event(row):
    return {key: value if key not in TASK_CLASSES else
            {k: v for k, v in value.items() if k != "prediction"}
            for key, value in row.items()}


def _validated_rows(report):
    if report.get("schema_version") != "Pop909ClTransferReport@1.0.0":
        raise ValueError("context_artifacts.unsupported_pop_schema")
    if report.get("split") not in ("train", "validation"):
        raise ValueError("context_artifacts.test_or_unknown_split_forbidden")
    pieces = {}
    for row in report["pieces"]:
        if row["piece_id"] in pieces or not row["piece_id"] or not row["group_id"]:
            raise ValueError("context_artifacts.duplicate_or_missing_piece_identity")
        pieces[row["piece_id"]] = row
    result = {}
    for view in VIEWS:
        rows = {}
        for row in report["views"][view]["event_observations"]:
            key = _identity(row)
            piece = pieces.get(row["piece_id"])
            if key in rows or not row["event_id"] or piece is None or row["group_id"] != piece["group_id"]:
                raise ValueError("context_artifacts.invalid_event_membership")
            if row.get("split", report["split"]) != report["split"]:
                raise ValueError("context_artifacts.event_split_mismatch")
            qn = row["start_qn"]
            if (set(qn) != {"num", "den"} or type(qn["num"]) is not int or
                    type(qn["den"]) is not int or qn["den"] <= 0 or qn["num"] < 0):
                raise ValueError("context_artifacts.exact_nonnegative_timing_required")
            if Fraction(qn["num"], qn["den"]).denominator != qn["den"]:
                raise ValueError("context_artifacts.reduced_timing_required")
            for task, count in TASK_CLASSES.items():
                item = row[task]
                for field in ("gold", "prediction"):
                    if item[field] is not None and (type(item[field]) is not int or not 0 <= item[field] < count):
                        raise ValueError("context_artifacts.class_out_of_range")
                if item["reason"] is None and (item["gold"] is None or item["prediction"] is None or
                                                item["label_reason"] is not None or row["alignment_reason"] is not None):
                    raise ValueError("context_artifacts.invalid_eligible_event")
            rows[key] = row
        if Counter(row["piece_id"] for row in rows.values()) != Counter(
                {piece_id: _count(piece["event_count"]) for piece_id, piece in pieces.items() if piece["event_count"]}):
            raise ValueError("context_artifacts.event_count_mismatch")
        result[view] = rows
    return result


def _paired_rows(c0, c2):
    if c0.get("split") != c2.get("split") or c0.get("protocol") != c2.get("protocol"):
        raise ValueError("context_artifacts.paired_protocol_or_split_mismatch")
    if sorted(c0["pieces"], key=lambda r: r["piece_id"]) != sorted(c2["pieces"], key=lambda r: r["piece_id"]):
        raise ValueError("context_artifacts.paired_piece_or_target_mismatch")
    reports = {"C0": _validated_rows(c0), "C2": _validated_rows(c2)}
    reference = reports["C0"]["native"]
    for model_rows in reports.values():
        for rows in model_rows.values():
            if rows.keys() != reference.keys() or any(_fixed_event(rows[k]) != _fixed_event(reference[k]) for k in rows):
                raise ValueError("context_artifacts.paired_event_gold_mask_or_timing_mismatch")
    for view in VIEWS:
        def bindings(report):
            items = report["views"][view]["input_bindings"]
            result = {row["piece_id"]: {k: v for k, v in row.items() if k != "logits_fingerprints"} for row in items}
            if len(result) != len(items) or set(result) != {row["piece_id"] for row in report["pieces"]}:
                raise ValueError("context_artifacts.input_binding_membership_mismatch")
            for key, item in result.items():
                candidates = item["candidate_ids"]
                piece = next(row for row in report["pieces"] if row["piece_id"] == key)
                if item["group_id"] != piece["group_id"]:
                    raise ValueError("context_artifacts.input_group_mismatch")
                if len(set(candidates)) != len(candidates):
                    raise ValueError("context_artifacts.duplicate_candidate")
                if any(row["beat_id"] is not None and row["beat_id"] not in candidates
                       for event, row in reference.items() if event[0] == key):
                    raise ValueError("context_artifacts.event_candidate_mismatch")
            return result
        if bindings(c0) != bindings(c2):
            raise ValueError("context_artifacts.paired_raw_input_mismatch")
    native_candidates = {row["piece_id"]: row["candidate_ids"] for row in c0["views"]["native"]["input_bindings"]}
    if native_candidates != {row["piece_id"]: row["candidate_ids"] for row in c0["views"]["metadata_removed"]["input_bindings"]}:
        raise ValueError("context_artifacts.paired_view_candidate_mismatch")
    return reports


def _false_d7(rows):
    predicted = [r for r in rows if r["quality"]["prediction"] == D7_CLASS_ID]
    eligible = [r for r in predicted if r["quality"]["reason"] is None]
    false = [r for r in eligible if r["quality"]["gold"] != D7_CLASS_ID]
    root_status = Counter("unavailable" if r["root"]["reason"] is not None else
                          "correct" if r["root"]["gold"] == r["root"]["prediction"] else "wrong" for r in false)
    return {"eligible_predictions": len(eligible), "true_positive": len(eligible) - len(false),
            "false_positive": len(false),
            "false_by_gold_quality": {str(k): v for k, v in sorted(Counter(r["quality"]["gold"] for r in false).items())},
            "false_by_root_status": {k: root_status[k] for k in ("correct", "wrong", "unavailable")},
            "false_by_group": dict(sorted(Counter(r["group_id"] for r in false).items())),
            "false_by_piece": dict(sorted(Counter(r["piece_id"] for r in false).items())),
            "excluded_prediction_count": len(predicted) - len(eligible),
            "excluded_by_reason": dict(sorted(Counter(r["quality"]["reason"] for r in predicted if r["quality"]["reason"] is not None).items())),
            "probabilities_available": False}


def _joint_by_quality(rows):
    result = []
    for class_id in range(17):
        eligible = [row for row in rows if row["quality"]["reason"] is None and row["root"]["reason"] is None
                    and row["quality"]["gold"] == class_id]
        correct = sum(all(row[task]["gold"] == row[task]["prediction"] for task in TASK_CLASSES) for row in eligible)
        result.append({"class_id": class_id, **_ratio(correct, len(eligible), "no_jointly_eligible_gold_class_events")})
    return result


def _changes(native, removed):
    result = {}
    for task in TASK_CLASSES:
        counts = Counter()
        for key, row in native.items():
            if row[task]["reason"] is not None:
                counts["unavailable"] += 1
                continue
            before = row[task]["prediction"] == row[task]["gold"]
            after = removed[key][task]["prediction"] == row[task]["gold"]
            counts[{(True, True): "correct_to_correct", (True, False): "correct_to_wrong",
                    (False, True): "wrong_to_correct", (False, False): "wrong_to_wrong"}[before, after]] += 1
            if row[task]["prediction"] != removed[key][task]["prediction"]:
                counts["prediction_changed"] += 1
        result[task] = {k: counts[k] for k in ("correct_to_correct", "correct_to_wrong", "wrong_to_correct",
                                               "wrong_to_wrong", "unavailable", "prediction_changed")}
    return result


def _group_counts(rows):
    groups = defaultdict(lambda: [0] * 9)
    for row in rows:
        count = groups[row["group_id"]]
        r, q = row["root"], row["quality"]
        if r["reason"] is None:
            count[0] += r["prediction"] == r["gold"]
            count[1] += 1
        if q["reason"] is None:
            count[2] += q["prediction"] == q["gold"]
            count[3] += 1
            count[6] += q["prediction"] == q["gold"] == D7_CLASS_ID
            count[7] += q["prediction"] == D7_CLASS_ID
            count[8] += q["gold"] == D7_CLASS_ID
        if r["reason"] is None and q["reason"] is None:
            count[4] += r["prediction"] == r["gold"] and q["prediction"] == q["gold"]
            count[5] += 1
    return groups


_RATIOS = {"root_accuracy": (0, 1), "quality_accuracy": (2, 3),
           "joint_accuracy": (4, 5), "d7_precision": (6, 7), "d7_recall": (6, 8)}


def _quantile(values, q):
    ordered = sorted(values)
    at = (len(ordered) - 1) * q
    lower = int(at)
    return ordered[lower] + (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]) * (at - lower)


def _bootstrap(reports, groups, draws, seed):
    if type(draws) is not int or not 1 <= draws <= 200 or type(seed) is not int:
        raise ValueError("context_artifacts.bootstrap_requires_1_to_200_draws_and_integer_seed")
    groups = sorted(groups)
    counters = {(model, view): _group_counts(rows.values()) for model, views in reports.items() for view, rows in views.items()}
    samples = defaultdict(list)
    rng = Random(seed)
    for _ in range(draws):
        multiplicities = Counter(rng.choices(groups, k=len(groups)))
        values = {}
        for key, grouped in counters.items():
            total = [sum(grouped.get(group, [0] * 9)[i] * n for group, n in multiplicities.items()) for i in range(9)]
            values[key] = {name: total[num] / total[den] if total[den] else None for name, (num, den) in _RATIOS.items()}
            for name, value in values[key].items():
                samples[(key[0], key[1], name)].append(value)
        for view in VIEWS:
            for name in _RATIOS:
                left, right = values["C0", view][name], values["C2", view][name]
                samples[("C2_minus_C0", view, name)].append(None if left is None or right is None else right - left)
    intervals = {}
    for (model, view, name), values in sorted(samples.items()):
        finite = [v for v in values if v is not None]
        intervals.setdefault(model, {}).setdefault(view, {})[name] = {
            "lower": _quantile(finite, .025) if finite else None,
            "upper": _quantile(finite, .975) if finite else None,
            "defined_resamples": len(finite), "undefined_resamples": draws - len(finite),
            "reason": None if finite else "no_defined_resamples"}
    return {"unit": "source_group_id", "group_count": len(groups), "draws": draws, "seed": seed,
            "method": "paired_group_resampling_with_replacement_percentile_95_linear_interpolation",
            "estimand": "pooled_event_ratios_with_group_resampling",
            "rare_class_policy": "undefined_denominator_resamples_excluded_and_counted",
            "limitations": ["200_or_fewer_draws_is_a_bounded_exploratory_interval", "reused_validation_not_independent_replication"],
            "intervals": intervals}


def summarize_pop_pair(c0_report, c2_report, *, bootstrap_draws=200, seed=17):
    """Summarize verified saved POP reports without re-running the model."""
    paired = _paired_rows(c0_report, c2_report)
    models = {}
    for model, views in paired.items():
        models[model] = {view: {"tasks": {task: _task_summary(list(rows.values()), task) for task in TASK_CLASSES},
                                "d7": _false_d7(list(rows.values())),
                                "joint_by_gold_quality": _joint_by_quality(list(rows.values()))} for view, rows in views.items()}
        models[model]["native_to_metadata_removed"] = _changes(views["native"], views["metadata_removed"])
    return {"schema_version": "ContextSavedPopDiagnostics@1.0.0", "split": c0_report["split"],
            "models": models, "probabilities_available": False,
            "uncertainty": _bootstrap(paired, {p["group_id"] for p in c0_report["pieces"]}, bootstrap_draws, seed),
            "claim_boundary": "saved_classification_decisions_do_not_establish_contextual_reasoning_or_music_quality"}


def summarize_dilemma_marginals(audit):
    """Adapt the verified arithmetic-audit marginals; never invent a matrix."""
    if audit.get("schema") != "ValidationArithmeticAudit@1.0.0" or audit.get("raw_predictions_logits_full_confusions_present") is not False:
        raise ValueError("context_artifacts.unsupported_dilemma_marginal_schema")
    heads = {}
    for task in ("cadence", "metrical_strength"):
        source = audit["final_class_details"][task]
        if source.get("missing_full_confusion") is not True:
            raise ValueError("context_artifacts.explicit_missing_confusion_required")
        heads[task] = {}
        for model in ("C0", "C2"):
            prefix = model.lower()
            rows = [{"class_id": row["class_id"], "label": row["label"], "support": row["support"],
                     "predicted_count": row[prefix + "_predicted_support_recovered"],
                     "true_positive": row[prefix + "_tp_recovered"]} for row in source["rows"]]
            heads[task][model] = {**marginal_metrics(rows, unknown_predicted_mass=source["unknown_zero_tp_predicted_margin_mass"][model]), "confusion": None,
                                  "confusion_reason": "not_retained_in_source", "count_provenance": "parent_audit_recovered_from_saved_fp32_fractions"}
    return {"schema_version": "ContextSavedDilemmaDiagnostics@1.0.0", "checkpoint_selection": "final_120000_updates",
            "heads": heads, "probabilities_available": False,
            "uncertainty": {"value": None, "reason": "per_group_predictions_not_retained"},
            "off_diagonal_event_routes": None,
            "claim_boundary": "class_marginals_cannot_identify_IAC_to_PAC_or_contextual_cause"}
