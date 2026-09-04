from __future__ import annotations

import copy
from io import BytesIO
import importlib.util
import json
from pathlib import Path
import tarfile

import pytest

from music_critic.experiments.registry import (
    ACCESS_POLICY_FIELDS,
    DATA_ACCESS_FIELDS,
    ExperimentRegistryError,
    SCHEMA_VERSION,
    inspect_archive,
    load_registry,
    record_fingerprint,
    render_markdown_ledger,
    seal_record,
    store_archive,
    validate_record,
    write_record_once,
)


CLI_PATH = Path(__file__).resolve().parents[2] / "scripts/experiment_registry.py"
REPOSITORY_ROOT = CLI_PATH.parent.parent
CLI_SPEC = importlib.util.spec_from_file_location("experiment_registry_cli", CLI_PATH)
assert CLI_SPEC is not None and CLI_SPEC.loader is not None
cli = importlib.util.module_from_spec(CLI_SPEC)
CLI_SPEC.loader.exec_module(cli)


def _record(
    *, experiment_id: str = "phase9eb5d-seed17-comparison", status: str = "completed"
) -> dict[str, object]:
    results: dict[str, object] = (
        {"primary_validation_score": 0.3548871111124754}
        if status == "completed"
        else {}
    )
    validity: dict[str, object] = {"valid": status != "invalid"}
    if status in {"failed", "aborted", "invalid"}:
        validity["reason"] = f"{status} for a recorded reason"
    access_evidence_status = "verified_primary_evidence"
    access_values: dict[str, bool | None] = {
        "validation_accessed": True,
        "test_inputs_accessed": False,
        "test_targets_accessed": False,
        "test_metrics_computed": False,
        "test_used_for_selection": False,
    }
    if status in {"planned", "running"}:
        access_evidence_status = (
            "not_applicable_planning_record"
            if status == "planned"
            else "pending_primary_evidence"
        )
        access_values = {field: None for field in DATA_ACCESS_FIELDS}
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "kind": "paired_validation_comparison",
        "phase": "9E-B5D",
        "title": "Seed-17 corrected AnalysisGNN comparison",
        "status": status,
        "question": "Does C1 outperform C0 on the frozen validation metric?",
        "hypothesis": "C1 improves the corrected primary macro score.",
        "decision_use": "Select the corrected AnalysisGNN baseline.",
        "repository": {"git_commit": "a" * 40, "dirty": False},
        "seeds": [17],
        "budget": {"applied_updates_per_arm": 10_000},
        "data_access": {
            "access_evidence_status": access_evidence_status,
            **access_values,
            "declared_policy": {
                "validation_allowed": True,
                "test_inputs_allowed": False,
                "test_targets_allowed": False,
                "test_metrics_allowed": False,
                "test_selection_allowed": False,
                "source": "test fixture protocol",
            },
        },
        "results": results,
        "validity": validity,
        "claims": {
            "supported": ["one-seed validation observation"],
            "not_supported": ["test quality", "statistical superiority"],
        },
        "artifacts": [
            {
                "role": "compact_result",
                "sha256": "b" * 64,
                "size_bytes": 321,
                "availability": "tracked",
                "locator": "tests/fixtures/analysisgnn/result.json",
            }
        ],
        "import_provenance": {
            "mode": "historical_import",
            "source_locator": "sha256/bb/archive.tar.gz",
        },
    }
    return seal_record(payload)


def _tar_bytes(entries: list[tuple[tarfile.TarInfo, bytes | None]]) -> bytes:
    output = BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for info, payload in entries:
            archive.addfile(info, None if payload is None else BytesIO(payload))
    return output.getvalue()


def _file_info(name: str, payload: bytes) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    return info


def _write_tar(path: Path, entries: list[tuple[tarfile.TarInfo, bytes | None]]) -> None:
    path.write_bytes(_tar_bytes(entries))


def test_record_fingerprint_and_ledger_are_deterministic() -> None:
    first = _record(experiment_id="z-last")
    reordered = dict(reversed(list(first.items())))
    assert record_fingerprint(first) == record_fingerprint(reordered)
    assert validate_record(reordered) == reordered

    second = _record(experiment_id="a-first")
    forward = render_markdown_ledger([first, second])
    reverse = render_markdown_ledger([second, first])
    assert forward == reverse
    assert forward.index("a-first") < forward.index("z-last")
    assert "no/no/no/no" in forward
    assert forward.endswith("\n")


def test_immutable_record_write_is_compact_and_bound_to_filename(tmp_path: Path) -> None:
    record = _record()
    destination = tmp_path / f"{record['experiment_id']}.json"
    write_record_once(destination, record)
    payload = destination.read_bytes()
    assert payload.endswith(b"\n")
    assert b'": ' not in payload
    assert load_registry(tmp_path) == (record,)
    with pytest.raises(ExperimentRegistryError, match="immutable_collision"):
        write_record_once(destination, record)
    with pytest.raises(ExperimentRegistryError, match="filename_identity_mismatch"):
        validate_record(record, filename="another-experiment.json")

    pretty_root = tmp_path / "pretty"
    pretty_root.mkdir()
    pretty = pretty_root / destination.name
    pretty.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ExperimentRegistryError, match="noncanonical_encoding"):
        load_registry(pretty_root)


@pytest.mark.parametrize("status", ["planned", "running", "failed", "aborted", "invalid"])
def test_non_completed_statuses_permit_empty_results(status: str) -> None:
    validate_record(_record(status=status))


def test_status_results_and_terminal_reason_fail_closed() -> None:
    record = _record()
    record["status"] = "unknown"
    record = seal_record(record)
    with pytest.raises(ExperimentRegistryError, match="status_invalid"):
        validate_record(record)

    record = _record()
    record["results"] = {}
    record = seal_record(record)
    with pytest.raises(ExperimentRegistryError, match="completed_results_required"):
        validate_record(record)

    record = _record(status="failed")
    validity = record["validity"]
    assert isinstance(validity, dict)
    validity.pop("reason")
    record = seal_record(record)
    with pytest.raises(ExperimentRegistryError, match="validity.reason"):
        validate_record(record)


@pytest.mark.parametrize("field", DATA_ACCESS_FIELDS)
def test_data_access_observations_are_required(field: str) -> None:
    missing = _record()
    access = missing["data_access"]
    assert isinstance(access, dict)
    access.pop(field)
    missing = seal_record(missing)
    with pytest.raises(ExperimentRegistryError, match=field):
        validate_record(missing)


@pytest.mark.parametrize("field", DATA_ACCESS_FIELDS)
def test_data_access_observations_accept_bool_or_explicit_null(field: str) -> None:
    partial = _record()
    access = partial["data_access"]
    assert isinstance(access, dict)
    access["access_evidence_status"] = "partial_primary_evidence"
    access[field] = None
    validate_record(seal_record(partial))

    invalid = _record()
    access = invalid["data_access"]
    assert isinstance(access, dict)
    access[field] = 0
    invalid = seal_record(invalid)
    with pytest.raises(ExperimentRegistryError, match=field):
        validate_record(invalid)


@pytest.mark.parametrize(
    "evidence_status",
    ["pending_primary_evidence", "not_applicable_planning_record"],
)
@pytest.mark.parametrize("observed", [False, True])
def test_unverified_access_observations_must_remain_null(
    evidence_status: str, observed: bool
) -> None:
    record = _record(status="running")
    access = record["data_access"]
    assert isinstance(access, dict)
    access["access_evidence_status"] = evidence_status
    access["test_inputs_accessed"] = observed
    with pytest.raises(
        ExperimentRegistryError, match="data_access_unverified_values_must_be_null"
    ):
        validate_record(seal_record(record))


@pytest.mark.parametrize(
    "evidence_status",
    ["verified_primary_evidence", "partial_primary_evidence"],
)
def test_negative_access_observations_require_evidence(evidence_status: str) -> None:
    record = _record()
    access = record["data_access"]
    assert isinstance(access, dict)
    access["access_evidence_status"] = evidence_status
    if evidence_status == "partial_primary_evidence":
        access["validation_accessed"] = None
    validated = validate_record(seal_record(record))
    validated_access = validated["data_access"]
    assert isinstance(validated_access, dict)
    assert validated_access["test_inputs_accessed"] is False


def test_partial_access_evidence_requires_known_and_unknown_observations() -> None:
    all_known = _record()
    access = all_known["data_access"]
    assert isinstance(access, dict)
    access["access_evidence_status"] = "partial_primary_evidence"
    with pytest.raises(
        ExperimentRegistryError, match="data_access_partial_values_required"
    ):
        validate_record(seal_record(all_known))

    all_unknown = _record(status="running")
    access = all_unknown["data_access"]
    assert isinstance(access, dict)
    access["access_evidence_status"] = "partial_primary_evidence"
    with pytest.raises(
        ExperimentRegistryError, match="data_access_partial_values_required"
    ):
        validate_record(seal_record(all_unknown))


@pytest.mark.parametrize("policy_field", ACCESS_POLICY_FIELDS)
def test_declared_access_policy_fields_are_required_booleans(
    policy_field: str,
) -> None:
    missing = _record()
    access = missing["data_access"]
    assert isinstance(access, dict)
    policy = access["declared_policy"]
    assert isinstance(policy, dict)
    policy.pop(policy_field)
    with pytest.raises(ExperimentRegistryError, match=policy_field):
        validate_record(seal_record(missing))

    invalid = _record()
    access = invalid["data_access"]
    assert isinstance(access, dict)
    policy = access["declared_policy"]
    assert isinstance(policy, dict)
    policy[policy_field] = 0
    with pytest.raises(ExperimentRegistryError, match=policy_field):
        validate_record(seal_record(invalid))


@pytest.mark.parametrize(
    ("actual_field", "policy_field"),
    [
        ("validation_accessed", "validation_allowed"),
        ("test_inputs_accessed", "test_inputs_allowed"),
        ("test_targets_accessed", "test_targets_allowed"),
        ("test_metrics_computed", "test_metrics_allowed"),
        ("test_used_for_selection", "test_selection_allowed"),
    ],
)
def test_observed_access_cannot_violate_declared_policy(
    actual_field: str, policy_field: str
) -> None:
    record = _record()
    access = record["data_access"]
    assert isinstance(access, dict)
    access[actual_field] = True
    policy = access["declared_policy"]
    assert isinstance(policy, dict)
    policy[policy_field] = False
    with pytest.raises(ExperimentRegistryError, match=f"policy_violation:{actual_field}"):
        validate_record(seal_record(record))


@pytest.mark.parametrize(
    ("policy_overrides", "message"),
    [
        (
            {
                "test_inputs_allowed": False,
                "test_targets_allowed": True,
                "test_metrics_allowed": True,
            },
            "test_metrics_require_inputs_and_targets",
        ),
        (
            {
                "test_inputs_allowed": True,
                "test_targets_allowed": True,
                "test_metrics_allowed": False,
                "test_selection_allowed": True,
            },
            "test_selection_requires_metrics",
        ),
    ],
)
def test_internally_inconsistent_declared_policies_are_rejected(
    policy_overrides: dict[str, bool], message: str
) -> None:
    record = _record()
    access = record["data_access"]
    assert isinstance(access, dict)
    policy = access["declared_policy"]
    assert isinstance(policy, dict)
    policy.update(policy_overrides)
    with pytest.raises(ExperimentRegistryError, match=message):
        validate_record(seal_record(record))


def test_unknown_access_is_rendered_explicitly_and_separately_from_policy() -> None:
    record = _record(status="running")
    ledger = render_markdown_ledger([record])
    assert (
        "| unknown; unknown/unknown/unknown/unknown | pending_primary_evidence | "
        "allowed; forbidden/forbidden/forbidden/forbidden |"
    ) in ledger
    assert "Observed access (VALIDATION; TEST inputs/targets/metrics/selection)" in ledger
    assert "Access evidence" in ledger
    assert "Declared policy" in ledger


def test_absolute_locators_and_bad_fingerprints_are_rejected() -> None:
    record = _record()
    artifact = record["artifacts"]
    assert isinstance(artifact, list) and isinstance(artifact[0], dict)
    artifact[0]["locator"] = "/Users/example/output.json"
    record = seal_record(record)
    with pytest.raises(ExperimentRegistryError, match="absolute_locator"):
        validate_record(record)

    record = _record()
    repository = record["repository"]
    assert isinstance(repository, dict)
    repository["checkout"] = "/private/repository"
    record = seal_record(record)
    with pytest.raises(ExperimentRegistryError, match="absolute_locator"):
        validate_record(record)

    record = _record()
    provenance = record["import_provenance"]
    assert isinstance(provenance, dict)
    provenance["source_locator"] = r"C:\\evidence\\archive.tar.gz"
    record = seal_record(record)
    with pytest.raises(ExperimentRegistryError, match="absolute_locator"):
        validate_record(record)

    record = _record()
    record["title"] = "tampered"
    with pytest.raises(ExperimentRegistryError, match="fingerprint_mismatch"):
        validate_record(record)


def test_safe_archive_and_nested_archive_are_inspected_without_extraction(
    tmp_path: Path,
) -> None:
    inner_payload = b'{"metric":0.5}\n'
    inner = _tar_bytes([(_file_info("result.json", inner_payload), inner_payload)])
    outer_payload = b"experiment evidence\n"
    archive = tmp_path / "evidence.tar.gz"
    _write_tar(
        archive,
        [
            (_file_info("README.txt", outer_payload), outer_payload),
            (_file_info("nested/results.tgz", inner), inner),
        ],
    )
    report = inspect_archive(archive)
    assert report["safe"] is True
    assert report["member_count"] == 3
    assert report["regular_file_count"] == 2
    assert report["nested_archive_count"] == 1
    paths = [row["path"] for row in report["members"]]
    assert paths == [
        "README.txt",
        "nested/results.tgz",
        "nested/results.tgz!/result.json",
    ]
    assert not (tmp_path / "README.txt").exists()
    assert not (tmp_path / "result.json").exists()


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        ([(_file_info("/absolute.txt", b"x"), b"x")], "unsafe_path"),
        ([(_file_info("../escape.txt", b"x"), b"x")], "unsafe_path"),
        ([(_file_info("bad\nname.txt", b"x"), b"x")], "control_character"),
        ([(_file_info("bad\u0085name.txt", b"x"), b"x")], "control_character"),
        (
            [
                (_file_info("same.txt", b"a"), b"a"),
                (_file_info("same.txt", b"b"), b"b"),
            ],
            "duplicate_member",
        ),
    ],
)
def test_unsafe_archive_names_are_rejected(
    tmp_path: Path,
    entries: list[tuple[tarfile.TarInfo, bytes | None]],
    message: str,
) -> None:
    archive = tmp_path / "unsafe.tar.gz"
    _write_tar(archive, entries)
    with pytest.raises(ExperimentRegistryError, match=message):
        inspect_archive(archive)


def test_links_special_members_and_unsafe_nested_tar_are_rejected(
    tmp_path: Path,
) -> None:
    symlink = tarfile.TarInfo("link")
    symlink.type = tarfile.SYMTYPE
    symlink.linkname = "target"
    archive = tmp_path / "link.tar.gz"
    _write_tar(archive, [(symlink, None)])
    with pytest.raises(ExperimentRegistryError, match="link_member"):
        inspect_archive(archive)

    fifo = tarfile.TarInfo("pipe")
    fifo.type = tarfile.FIFOTYPE
    archive = tmp_path / "special.tar.gz"
    _write_tar(archive, [(fifo, None)])
    with pytest.raises(ExperimentRegistryError, match="special_member"):
        inspect_archive(archive)

    unsafe_inner = _tar_bytes([(_file_info("../../escape", b"x"), b"x")])
    archive = tmp_path / "nested.tar.gz"
    _write_tar(
        archive,
        [(_file_info("evidence/inner.tar.gz", unsafe_inner), unsafe_inner)],
    )
    with pytest.raises(ExperimentRegistryError, match="unsafe_path"):
        inspect_archive(archive)


def test_content_addressed_store_is_idempotent_and_refuses_collision(
    tmp_path: Path,
) -> None:
    payload = b"compact evidence\n"
    archive = tmp_path / "source.tgz"
    _write_tar(archive, [(_file_info("evidence.txt", payload), payload)])
    root = tmp_path / "store"

    first = store_archive(archive, root)
    second = store_archive(archive, root)
    assert first == second
    assert set(first) == {"sha256", "size_bytes", "locator"}
    stored = root / str(first["locator"])
    assert stored.read_bytes() == archive.read_bytes()
    assert stored.name == f"{first['sha256']}.tar.gz"
    assert not str(first["locator"]).startswith("/")

    stored.chmod(0o644)
    corrupted = bytearray(stored.read_bytes())
    corrupted[0] ^= 1
    stored.write_bytes(corrupted)
    with pytest.raises(ExperimentRegistryError, match="content_collision"):
        store_archive(archive, root)


def test_content_addressed_store_rejects_non_gzip_tar(tmp_path: Path) -> None:
    archive = tmp_path / "evidence.tar"
    payload = b"evidence\n"
    with tarfile.open(archive, mode="w") as handle:
        handle.addfile(_file_info("result.txt", payload), BytesIO(payload))
    assert inspect_archive(archive)["safe"] is True
    with pytest.raises(ExperimentRegistryError, match="gzip_tar_required"):
        store_archive(archive, tmp_path / "store")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {
                "test_inputs_accessed": False,
                "test_targets_accessed": True,
                "test_metrics_computed": True,
                "test_used_for_selection": False,
            },
            "test_metrics_require_inputs_and_targets",
        ),
        (
            {
                "test_inputs_accessed": True,
                "test_targets_accessed": True,
                "test_metrics_computed": False,
                "test_used_for_selection": True,
            },
            "test_selection_requires_metrics",
        ),
        (
            {
                "test_inputs_accessed": None,
                "test_targets_accessed": True,
                "test_metrics_computed": True,
                "test_used_for_selection": False,
            },
            "test_metrics_require_inputs_and_targets",
        ),
        (
            {
                "test_inputs_accessed": True,
                "test_targets_accessed": True,
                "test_metrics_computed": None,
                "test_used_for_selection": True,
            },
            "test_selection_requires_metrics",
        ),
    ],
)
def test_impossible_test_access_states_are_rejected(
    overrides: dict[str, bool | None], message: str
) -> None:
    record = _record()
    access = record["data_access"]
    assert isinstance(access, dict)
    access.update(overrides)
    if any(value is None for value in overrides.values()):
        access["access_evidence_status"] = "partial_primary_evidence"
    policy = access["declared_policy"]
    assert isinstance(policy, dict)
    policy.update(
        {
            "test_inputs_allowed": True,
            "test_targets_allowed": True,
            "test_metrics_allowed": True,
            "test_selection_allowed": True,
        }
    )
    record = seal_record(record)
    with pytest.raises(ExperimentRegistryError, match=message):
        validate_record(record)


def test_cli_exposes_check_render_inspect_and_store(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    records = tmp_path / "records"
    record = _record()
    write_record_once(records / f"{record['experiment_id']}.json", record)
    ledger = tmp_path / "EXPERIMENTS.md"

    assert cli.main(["check", "--records", str(records)]) == 0
    assert json.loads(capsys.readouterr().out)["record_count"] == 1
    assert cli.main(
        ["render", "--records", str(records), "--output", str(ledger)]
    ) == 0
    capsys.readouterr()
    assert cli.main(
        [
            "render",
            "--records",
            str(records),
            "--output",
            str(ledger),
            "--check",
        ]
    ) == 0
    capsys.readouterr()

    payload = b"evidence\n"
    archive = tmp_path / "evidence.tar.gz"
    _write_tar(archive, [(_file_info("result.txt", payload), payload)])
    assert cli.main(["inspect-archive", str(archive)]) == 0
    assert json.loads(capsys.readouterr().out)["safe"] is True
    assert cli.main(
        ["store-archive", str(archive), "--root", str(tmp_path / "store")]
    ) == 0
    stored = json.loads(capsys.readouterr().out)
    assert (tmp_path / "store" / stored["locator"]).is_file()


def test_schema_and_artifact_binding_are_required() -> None:
    record = _record()
    record["schema_version"] = "2.0.0"
    record = seal_record(record)
    with pytest.raises(ExperimentRegistryError, match="schema_version"):
        validate_record(record)

    for field in ("role", "sha256", "size_bytes", "availability", "locator"):
        record = _record()
        artifacts = record["artifacts"]
        assert isinstance(artifacts, list) and isinstance(artifacts[0], dict)
        artifacts[0].pop(field)
        record = seal_record(record)
        with pytest.raises(ExperimentRegistryError, match="artifact_fields_missing"):
            validate_record(record)


def test_sealing_does_not_mutate_input() -> None:
    record = _record()
    unsealed = copy.deepcopy(record)
    unsealed.pop("record_fingerprint")
    before = copy.deepcopy(unsealed)
    sealed = seal_record(unsealed)
    assert unsealed == before
    assert sealed["record_fingerprint"] == record_fingerprint(sealed)
    assert json.loads(json.dumps(sealed)) == sealed


def test_committed_registry_and_generated_ledger_are_in_sync() -> None:
    records = load_registry(REPOSITORY_ROOT / "docs/experiments/records")
    identities = {str(record["experiment_id"]) for record in records}
    assert {
        "EXP-9EB5D-C0-001",
        "EXP-9EB5D-C1-001",
        "EXP-9EB5H-C2-001",
        "EXP-9EB5K-C0-120K-001",
        "PLAN-9EB5K-C0-120K",
    }.issubset(identities)
    assert (REPOSITORY_ROOT / "docs/EXPERIMENT_LEDGER.md").read_text(
        encoding="utf-8"
    ) == render_markdown_ledger(records)


def test_running_c0_access_is_unknown_while_declared_policy_stays_closed() -> None:
    records = load_registry(REPOSITORY_ROOT / "docs/experiments/records")
    by_id = {str(record["experiment_id"]): record for record in records}
    record = by_id["EXP-9EB5K-C0-120K-001"]
    assert record["status"] == "running"

    access = record["data_access"]
    assert isinstance(access, dict)
    assert access["access_evidence_status"] == "pending_primary_evidence"
    assert all(access[field] is None for field in DATA_ACCESS_FIELDS)

    policy = access["declared_policy"]
    assert isinstance(policy, dict)
    assert policy["validation_allowed"] is True
    assert all(
        policy[field] is False
        for field in (
            "test_inputs_allowed",
            "test_targets_allowed",
            "test_metrics_allowed",
            "test_selection_allowed",
        )
    )
    assert isinstance(policy["source"], str) and policy["source"].strip()

    ledger = render_markdown_ledger([record])
    assert "unknown; unknown/unknown/unknown/unknown" in ledger
    assert "forbidden/forbidden/forbidden/forbidden" in ledger


def test_seed17_import_manifest_is_bound_to_registered_records() -> None:
    manifest = json.loads(
        (
            REPOSITORY_ROOT
            / "docs/experiments/imports/seed17-transfer-2a340242.json"
        ).read_text(encoding="utf-8")
    )
    records = load_registry(REPOSITORY_ROOT / "docs/experiments/records")
    by_id = {str(record["experiment_id"]): record for record in records}
    expected_ids = set(manifest["registered_experiments"]) | set(
        manifest["registered_plans"]
    )
    assert expected_ids <= by_id.keys()

    archive = manifest["archive"]
    assert archive == {
        "availability": "local_ignored",
        "locator": (
            "artifacts/experiments/sha256/2a/"
            "2a340242cdd7b917eec4def9c8644bb2c10330e72ec9c8cbcf9d2291acfe9823"
            ".tar.gz"
        ),
        "media_type": "application/gzip",
        "sha256": (
            "2a340242cdd7b917eec4def9c8644bb2c10330e72ec9c8cbcf9d2291acfe9823"
        ),
        "size_bytes": 22_237_876,
    }
    for experiment_id in expected_ids:
        artifacts = by_id[experiment_id]["artifacts"]
        assert isinstance(artifacts, list)
        assert any(
            item["sha256"] == archive["sha256"]
            and item["size_bytes"] == archive["size_bytes"]
            and item["locator"] == archive["locator"]
            for item in artifacts
        )
