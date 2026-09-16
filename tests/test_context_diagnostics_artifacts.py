"""Independent tiny count oracles; no corpus or checkpoint fixtures."""

from copy import deepcopy

import pytest

from music_critic.experiments.context_diagnostics_artifacts import (
    marginal_metrics, summarize_dilemma_marginals, summarize_pop_pair,
)


def _event(piece, event, gold, prediction, *, root_gold=0, root_prediction=0, reason=None):
    return {"piece_id": piece, "group_id": "g:" + piece, "event_id": event,
            "start_qn": {"num": 0, "den": 1}, "beat_id": "b", "alignment_reason": None,
            "quality": {"gold": gold, "prediction": prediction, "reason": reason,
                        "label_reason": reason, "source_label": "label"},
            "root": {"gold": root_gold, "prediction": root_prediction,
                     "reason": None if root_gold is not None else "ambiguous",
                     "label_reason": None if root_gold is not None else "ambiguous", "source_label": root_gold}}


def _report():
    rows = [_event("a", "0", 6, 6), _event("a", "1", 0, 6),
            _event("b", "0", 0, 0, root_prediction=1),
            _event("b", "1", None, 6, reason="ambiguous")]
    pieces = [{"piece_id": name, "group_id": "g:" + name, "event_count": 2,
               "target_fingerprint": "target-" + name} for name in ("a", "b")]
    bindings = [{"piece_id": name, "group_id": "g:" + name, "candidate_ids": ["b"],
                 "model_input_fingerprint": "input-" + name, "logits_fingerprints": {"quality": "x"}}
                for name in ("a", "b")]
    return {"schema_version": "Pop909ClTransferReport@1.0.0", "split": "validation",
            "protocol": {"frozen": True}, "pieces": pieces,
            "views": {view: {"event_observations": deepcopy(rows), "input_bindings": deepcopy(bindings)}
                      for view in ("native", "metadata_removed")}}


def _dilemma():
    classes = [{"class_id": i, "label": str(i), "support": support,
                "c0_predicted_support_recovered": pred, "c0_tp_recovered": tp,
                "c2_predicted_support_recovered": pred, "c2_tp_recovered": tp}
               for i, support, pred, tp in ((0, 2, 1, 1), (1, 1, 2, 1))]
    return {"schema": "ValidationArithmeticAudit@1.0.0",
            "raw_predictions_logits_full_confusions_present": False,
            "final_class_details": {task: {"missing_full_confusion": True, "rows": deepcopy(classes),
                                            "unknown_zero_tp_predicted_margin_mass": {"C0": 0, "C2": 0}}
                                    for task in ("cadence", "metrical_strength")}}


def _seal_pop_report(payload):
    """Independent oracle for transfer.py: ASCII JSON + exactly one newline."""
    from hashlib import sha256
    import json
    def fingerprint(value):
        return sha256((json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
                                  allow_nan=False) + "\n").encode("utf-8")).hexdigest()
    value = deepcopy(payload)
    value["runtime"] = {"wall_seconds": 1.25}
    value["provenance"] = {"fixture_name": "cadence café"}
    value["scientific_fingerprint"] = fingerprint({k: v for k, v in value.items() if k != "runtime"})
    value["report_fingerprint"] = fingerprint(value)
    return value


def test_hand_computed_class_oracle_and_masks():
    report = _report()
    result = summarize_pop_pair(report, deepcopy(report), bootstrap_draws=5)
    quality = result["models"]["C2"]["native"]["tasks"]["quality"]
    assert quality["accuracy"]["value"] == pytest.approx(2 / 3)
    assert quality["confusion"][0][6] == 1
    assert quality["confusion"][6][6] == 1
    assert sum(map(sum, quality["confusion"])) == 3
    d7 = quality["per_class"][6]
    assert d7["precision"]["value"] == .5
    assert d7["recall"]["value"] == 1
    assert d7["f1"]["value"] == pytest.approx(2 / 3)
    assert quality["per_class"][8]["recall"]["value"] is None
    assert quality["excluded_events"] == 1
    false = result["models"]["C2"]["native"]["d7"]
    assert false["false_by_gold_quality"] == {"0": 1}
    assert false["false_by_root_status"] == {"correct": 1, "wrong": 0, "unavailable": 0}
    assert false["excluded_prediction_count"] == 1
    assert false["probabilities_available"] is False
    joint = result["models"]["C2"]["native"]["joint_by_gold_quality"]
    assert joint[0]["numerator"] == 0 and joint[0]["denominator"] == 2
    assert joint[6]["value"] == 1


def test_paired_view_buckets_and_independent_root_mask():
    report = _report()
    report["views"]["metadata_removed"]["event_observations"][1]["quality"]["prediction"] = 0
    report["views"]["metadata_removed"]["event_observations"][0]["quality"]["prediction"] = 0
    result = summarize_pop_pair(report, deepcopy(report), bootstrap_draws=5)
    bucket = result["models"]["C0"]["native_to_metadata_removed"]["quality"]
    assert bucket == {"correct_to_correct": 1, "correct_to_wrong": 1, "wrong_to_correct": 1,
                      "wrong_to_wrong": 0, "unavailable": 1, "prediction_changed": 2}


def test_bootstrap_resamples_groups_not_events_and_preserves_pairing():
    report = _report()
    result = summarize_pop_pair(report, deepcopy(report), bootstrap_draws=200, seed=12)
    uncertainty = result["uncertainty"]
    assert uncertainty["group_count"] == 2
    assert uncertainty["unit"] == "source_group_id"
    for views in uncertainty["intervals"]["C2_minus_C0"].values():
        for value in views.values():
            assert value["lower"] == value["upper"] == 0
    precision = uncertainty["intervals"]["C2"]["native"]["d7_precision"]
    assert precision["undefined_resamples"] > 0
    assert precision["lower"] == precision["upper"] == .5


def test_bootstrap_iteration_invariance_and_no_input_mutation():
    report = _report()
    original = deepcopy(report)
    expected = summarize_pop_pair(report, deepcopy(report), bootstrap_draws=20, seed=5)
    reordered = deepcopy(report)
    reordered["pieces"].reverse()
    for view in reordered["views"].values():
        view["event_observations"].reverse()
        view["input_bindings"].reverse()
    assert summarize_pop_pair(reordered, deepcopy(reordered), bootstrap_draws=20, seed=5) == expected
    assert report == original


@pytest.mark.parametrize("draws", [0, 201, -1, True, 1.5])
def test_bootstrap_budget_gate(draws):
    with pytest.raises(ValueError, match="bootstrap"):
        summarize_pop_pair(_report(), _report(), bootstrap_draws=draws)


@pytest.mark.parametrize("split", ["test", "TEST", None, "development"])
def test_only_trainval_reports(split):
    report = _report()
    report["split"] = split
    with pytest.raises(ValueError, match="split"):
        summarize_pop_pair(report, deepcopy(report))


@pytest.mark.parametrize("field,value", [("event_id", "different"), ("group_id", "other"),
                                         ("start_qn", {"num": 1, "den": 2}), ("beat_id", "new")])
def test_event_identity_or_timing_drift_rejected(field, value):
    changed = _report()
    changed["views"]["native"]["event_observations"][0][field] = value
    with pytest.raises(ValueError):
        summarize_pop_pair(_report(), changed)


@pytest.mark.parametrize("field,value", [("gold", 1), ("reason", "missing"), ("label_reason", "missing")])
def test_gold_and_mask_drift_rejected(field, value):
    changed = _report()
    changed["views"]["native"]["event_observations"][0]["quality"][field] = value
    with pytest.raises(ValueError):
        summarize_pop_pair(_report(), changed)


def test_raw_input_drift_and_duplicate_events_rejected():
    changed = _report()
    changed["views"]["native"]["input_bindings"][0]["model_input_fingerprint"] = "different"
    with pytest.raises(ValueError, match="raw_input"):
        summarize_pop_pair(_report(), changed)
    changed = _report()
    changed["views"]["native"]["event_observations"].append(deepcopy(changed["views"]["native"]["event_observations"][0]))
    with pytest.raises(ValueError, match="membership"):
        summarize_pop_pair(_report(), changed)


def test_unreduced_or_float_time_and_class_values_rejected():
    for value in ({"num": 0, "den": 2}, {"num": 0.0, "den": 1}):
        changed = _report()
        changed["views"]["native"]["event_observations"][0]["start_qn"] = value
        with pytest.raises(ValueError, match="timing"):
            summarize_pop_pair(changed, deepcopy(changed))
    changed = _report()
    changed["views"]["native"]["event_observations"][0]["quality"]["prediction"] = 17
    with pytest.raises(ValueError, match="class_out_of_range"):
        summarize_pop_pair(changed, deepcopy(changed))


def test_dilemma_marginals_are_not_event_confusions():
    result = summarize_dilemma_marginals(_dilemma())
    head = result["heads"]["cadence"]["C0"]
    assert head["per_class"][0]["precision"]["value"] == 1
    assert head["per_class"][0]["recall"]["value"] == .5
    assert head["accuracy"]["value"] == pytest.approx(2 / 3)
    assert head["confusion"] is None
    assert result["off_diagonal_event_routes"] is None
    assert result["uncertainty"]["value"] is None


def test_unknown_zero_tp_margins_are_not_zero_predictions():
    rows = [{"class_id": 0, "support": 2, "predicted_count": 1, "true_positive": 1},
            {"class_id": 1, "support": 1, "predicted_count": None, "true_positive": 0}]
    result = marginal_metrics(rows, unknown_predicted_mass=2)
    assert result["unallocated_predicted_count"] == 2
    unknown = result["per_class"][1]
    assert unknown["predicted_count"] is unknown["false_positive"] is unknown["precision"]["value"] is None
    assert unknown["f1"]["value"] == 0
    assert unknown["recall"]["value"] == 0


@pytest.mark.parametrize("changes", [{"true_positive": 3}, {"support": -1},
                                     {"predicted_count": 2}, {"class_id": True}])
def test_impossible_marginal_oracles(changes):
    row = {"class_id": 0, "support": 1, "predicted_count": 1, "true_positive": 1, **changes}
    with pytest.raises(ValueError):
        marginal_metrics([row])


@pytest.fixture
def artifact_cli():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "scripts" / "context_diagnostics_artifacts.py"
    spec = importlib.util.spec_from_file_location("context_artifact_cli_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _planned_cli_record(module):
    return {"status": "planned", "experiment_id": "EXP-CTX-C2-FIXTURE-001", "record_fingerprint": "fixture",
            "data_access": {"declared_policy": {"validation_allowed": True, "test_inputs_allowed": False,
                                                 "test_targets_allowed": False, "test_metrics_allowed": False,
                                                 "test_selection_allowed": False}},
            "protocol": {"views": ["native", "metadata_removed"], "bootstrap": {"draws": 2, "seed": 17},
                         "source_reports": {name: module.SOURCE_BINDINGS[name] for name in ("C0", "C2")},
                         "dilemma": {"source_report": module.SOURCE_BINDINGS["Dilemma"]}}}


@pytest.mark.parametrize("change", ["terminal", "test_allowed", "binding"])
def test_cli_access_guard_precedes_all_source_reads(artifact_cli, monkeypatch, tmp_path, change):
    record = deepcopy(_planned_cli_record(artifact_cli))
    if change == "terminal":
        record["status"] = "completed"
    elif change == "test_allowed":
        record["data_access"]["declared_policy"]["test_targets_allowed"] = True
    else:
        record["protocol"]["source_reports"]["C0"]["sha256"] = "bad"
    monkeypatch.setattr(artifact_cli, "load_record", lambda _: record)
    monkeypatch.setattr(artifact_cli, "_load_verified", lambda *_: pytest.fail("source read before guard"))
    with pytest.raises(ValueError):
        artifact_cli.materialize(source_root=tmp_path / "sources", experiment_record=tmp_path / "record", output=tmp_path / "out")


def test_cli_hash_and_path_escape_guards(artifact_cli, tmp_path):
    from hashlib import sha256
    root = tmp_path / "root"
    root.mkdir()
    raw = b"{}"
    (root / "source.json").write_bytes(raw)
    binding = {"locator": "source.json", "sha256": "wrong", "size_bytes": len(raw)}
    with pytest.raises(ValueError, match="hash_or_size"):
        artifact_cli._load_verified(root, binding)
    outside = tmp_path / "outside.json"
    outside.write_bytes(raw)
    binding = {"locator": "../outside.json", "sha256": sha256(raw).hexdigest(), "size_bytes": len(raw)}
    with pytest.raises(ValueError, match="path_escape"):
        artifact_cli._load_verified(root, binding)


def test_cli_materializes_only_new_output_and_keeps_sources(artifact_cli, monkeypatch, tmp_path):
    from hashlib import sha256
    import json
    root = tmp_path / "sources"
    root.mkdir()
    bindings = {}
    for name, value in {"C0": _seal_pop_report(_report()), "C2": _seal_pop_report(_report()), "Dilemma": _dilemma()}.items():
        raw = json.dumps(value).encode()
        (root / (name + ".json")).write_bytes(raw)
        bindings[name] = {"locator": name + ".json", "sha256": sha256(raw).hexdigest(), "size_bytes": len(raw)}
    monkeypatch.setattr(artifact_cli, "SOURCE_BINDINGS", bindings)
    record = _planned_cli_record(artifact_cli)
    monkeypatch.setattr(artifact_cli, "load_record", lambda _: record)
    out = tmp_path / "result.json"
    result = artifact_cli.materialize(source_root=root, experiment_record=tmp_path / "record.json", output=out)
    assert result["access_observations"]["model_forwards"] == 0
    assert json.loads(out.read_text()) == result
    with pytest.raises(FileExistsError):
        artifact_cli.materialize(source_root=root, experiment_record=tmp_path / "record.json", output=out)
    with pytest.raises(ValueError, match="read_only"):
        artifact_cli.materialize(source_root=root, experiment_record=tmp_path / "record.json", output=root / "new.json")
    for binding in bindings.values():
        assert sha256((root / binding["locator"]).read_bytes()).hexdigest() == binding["sha256"]


@pytest.mark.parametrize("change", [None, "runtime", "scientific", "missing", "registry_hash"])
def test_cli_uses_exact_production_report_hash_convention(artifact_cli, tmp_path, change):
    from hashlib import sha256
    import json
    from music_critic.experiments.registry import canonical_json_bytes
    value = _seal_pop_report(_report())
    if change == "runtime":
        value["runtime"]["wall_seconds"] = 2.5
    elif change == "scientific":
        value["scientific_fingerprint"] = "0" * 64
    elif change == "missing":
        del value["scientific_fingerprint"]
        del value["report_fingerprint"]
    elif change == "registry_hash":
        value["report_fingerprint"] = sha256(canonical_json_bytes({k: v for k, v in value.items()
                                                                  if k != "report_fingerprint"})).hexdigest()
    raw = json.dumps(value).encode()
    (tmp_path / "report.json").write_bytes(raw)
    binding = {"locator": "report.json", "sha256": sha256(raw).hexdigest(), "size_bytes": len(raw)}
    if change is None:
        assert artifact_cli._load_verified(tmp_path, binding) == value
    else:
        with pytest.raises(ValueError, match="internal_report_fingerprint"):
            artifact_cli._load_verified(tmp_path, binding)
