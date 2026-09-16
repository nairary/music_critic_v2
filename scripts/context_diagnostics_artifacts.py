#!/usr/bin/env python3
"""Materialize one registered count-only diagnostic from fixed saved reports."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from music_critic.experiments.context_diagnostics_artifacts import (
    summarize_dilemma_marginals, summarize_pop_pair,
)
from music_critic.experiments.pop909_cl_frozen.contracts import fingerprint as pop_report_fingerprint
from music_critic.experiments.registry import canonical_json_bytes, load_record


SOURCE_BINDINGS = {
    "C0": {"locator": "outputs/pop909_transfer/20260908/c0-001/report.json",
           "sha256": "74a796b450d1483d0a23ed582724224f5a8f416f1ec33e2de96725727b76f8b4", "size_bytes": 20359289},
    "C2": {"locator": "outputs/pop909_transfer/20260908/c2-001/report.json",
           "sha256": "933dbba2a196800dd7ef42aa57f69d3bcb4ae269c54389089629188aa5f25b90", "size_bytes": 20359372},
    "Dilemma": {"locator": "outputs/phase9eb5k/EXP-9EB5K-C0-C2-AUDIT-001/recomputed_metrics/metrics_recomputed.json",
                "sha256": "ba7d082f0f69874d76c970bbf22aec4d3b76a14a89a52509602b6dc06123fb3c", "size_bytes": 63383},
}


def _guard_record(path):
    record = load_record(path)
    if record["status"] != "planned" or not record["experiment_id"].startswith("EXP-CTX-C2-"):
        raise ValueError("context_artifacts.planned_context_record_required")
    policy = record["data_access"]["declared_policy"]
    if policy.get("validation_allowed") is not True or any(policy.get(key) is not False for key in (
            "test_inputs_allowed", "test_targets_allowed", "test_metrics_allowed", "test_selection_allowed")):
        raise ValueError("context_artifacts.closed_test_validation_policy_required")
    protocol = record["protocol"]
    if protocol.get("views") != ["native", "metadata_removed"]:
        raise ValueError("context_artifacts.frozen_views_required")
    for name, expected in SOURCE_BINDINGS.items():
        declared = protocol["dilemma"]["source_report"] if name == "Dilemma" else protocol["source_reports"][name]
        if {k: declared.get(k) for k in expected} != expected:
            raise ValueError("context_artifacts.registered_source_binding_mismatch")
    bootstrap = protocol["bootstrap"]
    if type(bootstrap["draws"]) is not int or not 1 <= bootstrap["draws"] <= 200 or type(bootstrap["seed"]) is not int:
        raise ValueError("context_artifacts.registered_bootstrap_budget_invalid")
    return record


def _load_verified(source_root, binding):
    root = source_root.resolve(strict=True)
    path = (root / binding["locator"]).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("context_artifacts.source_path_escape")
    with path.open("rb") as handle:
        raw = handle.read(binding["size_bytes"] + 1)
    if len(raw) != binding["size_bytes"] or sha256(raw).hexdigest() != binding["sha256"]:
        raise ValueError("context_artifacts.source_hash_or_size_mismatch")
    payload = json.loads(raw)
    if payload.get("schema_version") == "Pop909ClTransferReport@1.0.0":
        # The frozen POP contract hashes ASCII-escaped canonical JSON plus a
        # newline; registry records use a different byte convention. Reuse the
        # owning production function instead of substituting the record hash.
        expected = pop_report_fingerprint({k: v for k, v in payload.items() if k != "report_fingerprint"})
        scientific = pop_report_fingerprint({k: v for k, v in payload.items()
                                              if k not in ("report_fingerprint", "scientific_fingerprint", "runtime")})
        if payload.get("report_fingerprint") != expected or payload.get("scientific_fingerprint") != scientific:
            raise ValueError("context_artifacts.internal_report_fingerprint_mismatch")
    return payload


def materialize(*, source_root, experiment_record, output):
    """Explicit local artifacts only; a terminal record cannot authorize replay."""
    record = _guard_record(experiment_record)
    if not source_root.is_absolute() or not output.is_absolute():
        raise ValueError("context_artifacts.absolute_runtime_paths_required")
    if output.exists():
        raise FileExistsError(output)
    if output.resolve().is_relative_to(source_root.resolve()):
        raise ValueError("context_artifacts.source_root_is_read_only")
    sources = {name: _load_verified(source_root, binding) for name, binding in SOURCE_BINDINGS.items()}
    bootstrap = record["protocol"]["bootstrap"]
    result = {"schema_version": "ContextSavedArtifactDiagnostics@1.0.0", "experiment_id": record["experiment_id"],
              "planned_record_fingerprint": record["record_fingerprint"], "source_bindings": SOURCE_BINDINGS,
              "pop": summarize_pop_pair(sources["C0"], sources["C2"], bootstrap_draws=bootstrap["draws"], seed=bootstrap["seed"]),
              "dilemma": summarize_dilemma_marginals(sources["Dilemma"]),
              "access_observations": {"saved_validation_artifacts_read": 3, "raw_corpus_reads": 0,
                                      "model_forwards": 0, "applied_updates": 0,
                                      "test_inputs_accessed": False, "test_targets_accessed": False,
                                      "test_metrics_computed": False, "test_used_for_selection": False}}
    result["report_fingerprint"] = sha256(canonical_json_bytes(result)).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(canonical_json_bytes(result))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--experiment-record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = materialize(source_root=args.source_root, experiment_record=args.experiment_record, output=args.output)
    print(json.dumps({"experiment_id": result["experiment_id"], "report_fingerprint": result["report_fingerprint"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
