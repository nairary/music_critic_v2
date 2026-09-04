"""Immutable, source-free experiment evidence registry primitives.

The registry deliberately stores only compact JSON evidence.  Heavy archives are
kept in a content-addressed store and are referenced by relative locators so a
record never depends on one developer's filesystem layout.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from hashlib import sha256
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import tarfile
import tempfile
import unicodedata
from typing import BinaryIO


SCHEMA_VERSION = "1.0.0"
STATUS_VALUES = frozenset(
    {"planned", "running", "completed", "failed", "aborted", "invalid"}
)
TERMINAL_NON_COMPLETED_STATUSES = frozenset({"failed", "aborted", "invalid"})
REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "experiment_id",
        "kind",
        "phase",
        "title",
        "status",
        "question",
        "hypothesis",
        "decision_use",
        "repository",
        "seeds",
        "budget",
        "data_access",
        "results",
        "validity",
        "claims",
        "artifacts",
        "import_provenance",
        "record_fingerprint",
    }
)
TEST_ACCESS_FIELDS = (
    "test_inputs_accessed",
    "test_targets_accessed",
    "test_metrics_computed",
    "test_used_for_selection",
)
DATA_ACCESS_FIELDS = ("validation_accessed", *TEST_ACCESS_FIELDS)
ACCESS_EVIDENCE_STATUSES = frozenset(
    {
        "verified_primary_evidence",
        "partial_primary_evidence",
        "pending_primary_evidence",
        "not_applicable_planning_record",
    }
)
ACCESS_POLICY_FIELDS = (
    "validation_allowed",
    "test_inputs_allowed",
    "test_targets_allowed",
    "test_metrics_allowed",
    "test_selection_allowed",
)
ARTIFACT_FIELDS = frozenset(
    {"role", "sha256", "size_bytes", "availability", "locator"}
)

_EXPERIMENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NESTED_ARCHIVE_SUFFIXES = (".tar.gz", ".tgz")
_MAX_NESTED_ARCHIVE_DEPTH = 16
_LOCATOR_FIELD_TOKENS = frozenset(
    {"path", "locator", "directory", "root", "checkout", "worktree", "cwd"}
)


class ExperimentRegistryError(ValueError):
    """Stable fail-closed error for experiment evidence operations."""


def _json_path(parent: str, key: object) -> str:
    if isinstance(key, str) and key.isidentifier():
        return f"{parent}.{key}"
    return f"{parent}[{json.dumps(key, ensure_ascii=False)}]"


def _validate_json_value(value: object, *, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ExperimentRegistryError(
                f"experiment_registry.json.non_finite:{path}"
            )
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ExperimentRegistryError(
                    f"experiment_registry.json.non_string_key:{path}"
                )
            _validate_json_value(item, path=_json_path(path, key))
        return
    raise ExperimentRegistryError(
        f"experiment_registry.json.unsupported_type:{path}:{type(value).__name__}"
    )


def canonical_json_bytes(value: object) -> bytes:
    """Return deterministic compact UTF-8 JSON bytes."""

    _validate_json_value(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_json(value: object) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def record_fingerprint(record: Mapping[str, object]) -> str:
    """Fingerprint a record while excluding its self-referential field."""

    payload = {
        key: value for key, value in record.items() if key != "record_fingerprint"
    }
    return sha256(canonical_json_bytes(payload)).hexdigest()


def seal_record(record: Mapping[str, object]) -> dict[str, object]:
    """Return a new record with a deterministic fingerprint."""

    sealed = dict(record)
    sealed.pop("record_fingerprint", None)
    sealed["record_fingerprint"] = record_fingerprint(sealed)
    return sealed


def _require_nonempty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExperimentRegistryError(
            f"experiment_registry.record.nonempty_string_required:{field}"
        )
    return value


def _is_absolute_filesystem_locator(value: str) -> bool:
    lowered = value.lower()
    if lowered.startswith("file:") or value.startswith(("~/", "~\\")):
        return True
    if PurePosixPath(value).is_absolute():
        return True
    windows = PureWindowsPath(value)
    return windows.is_absolute() or bool(windows.drive)


def _reject_absolute_locators(
    value: object,
    *,
    path: str = "$",
    locator_context: bool = False,
) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = key.lower()
            field_tokens = frozenset(
                token for token in re.split(r"[^a-z0-9]+", lowered) if token
            )
            child_locator_context = (
                locator_context
                or bool(field_tokens & _LOCATOR_FIELD_TOKENS)
                or lowered == "path"
                or lowered.endswith("_path")
                or lowered == "locator"
                or lowered.endswith("_locator")
                or lowered == "directory"
                or lowered.endswith("_directory")
                or lowered == "root"
                or lowered.endswith("_root")
            )
            _reject_absolute_locators(
                item,
                path=_json_path(path, key),
                locator_context=child_locator_context,
            )
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_absolute_locators(
                item,
                path=f"{path}[{index}]",
                locator_context=locator_context,
            )
        return
    if (
        locator_context
        and isinstance(value, str)
        and _is_absolute_filesystem_locator(value)
    ):
        raise ExperimentRegistryError(
            f"experiment_registry.record.absolute_locator_forbidden:{path}"
        )


def _validate_artifacts(value: object) -> None:
    if not isinstance(value, list):
        raise ExperimentRegistryError(
            "experiment_registry.record.artifacts_must_be_list"
        )
    for index, artifact in enumerate(value):
        if not isinstance(artifact, dict):
            raise ExperimentRegistryError(
                f"experiment_registry.record.artifact_invalid:{index}"
            )
        missing = ARTIFACT_FIELDS - artifact.keys()
        if missing:
            raise ExperimentRegistryError(
                "experiment_registry.record.artifact_fields_missing:"
                + ",".join(sorted(missing))
            )
        _require_nonempty_string(artifact["role"], field=f"artifacts[{index}].role")
        digest = artifact["sha256"]
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ExperimentRegistryError(
                f"experiment_registry.record.artifact_sha256_invalid:{index}"
            )
        size = artifact["size_bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ExperimentRegistryError(
                f"experiment_registry.record.artifact_size_invalid:{index}"
            )
        _require_nonempty_string(
            artifact["availability"], field=f"artifacts[{index}].availability"
        )
        locator = _require_nonempty_string(
            artifact["locator"], field=f"artifacts[{index}].locator"
        )
        if _is_absolute_filesystem_locator(locator):
            raise ExperimentRegistryError(
                f"experiment_registry.record.absolute_locator_forbidden:"
                f"$.artifacts[{index}].locator"
            )


def validate_record(
    record: Mapping[str, object],
    *,
    filename: str | Path | None = None,
) -> dict[str, object]:
    """Validate and return a detached experiment record.

    ``filename`` may be a basename or a full path.  When supplied, its basename
    is bound to the record identity.
    """

    if not isinstance(record, Mapping):
        raise ExperimentRegistryError("experiment_registry.record.object_required")
    result = dict(record)
    _validate_json_value(result)
    missing = REQUIRED_FIELDS - result.keys()
    if missing:
        raise ExperimentRegistryError(
            "experiment_registry.record.fields_missing:" + ",".join(sorted(missing))
        )
    if result["schema_version"] != SCHEMA_VERSION:
        raise ExperimentRegistryError(
            "experiment_registry.record.schema_version_invalid"
        )

    experiment_id = _require_nonempty_string(
        result["experiment_id"], field="experiment_id"
    )
    if _EXPERIMENT_ID.fullmatch(experiment_id) is None:
        raise ExperimentRegistryError(
            "experiment_registry.record.experiment_id_invalid"
        )
    if filename is not None and Path(filename).name != f"{experiment_id}.json":
        raise ExperimentRegistryError(
            "experiment_registry.record.filename_identity_mismatch"
        )

    for field in (
        "kind",
        "phase",
        "title",
        "question",
        "hypothesis",
        "decision_use",
    ):
        _require_nonempty_string(result[field], field=field)

    status = result["status"]
    if not isinstance(status, str) or status not in STATUS_VALUES:
        raise ExperimentRegistryError("experiment_registry.record.status_invalid")

    for field in (
        "repository",
        "budget",
        "data_access",
        "results",
        "validity",
        "claims",
        "import_provenance",
    ):
        if not isinstance(result[field], dict):
            raise ExperimentRegistryError(
                f"experiment_registry.record.object_required:{field}"
            )
    seeds = result["seeds"]
    if not isinstance(seeds, list) or any(
        isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds
    ):
        raise ExperimentRegistryError("experiment_registry.record.seeds_invalid")

    access = result["data_access"]
    assert isinstance(access, dict)
    evidence_status = access.get("access_evidence_status")
    if evidence_status not in ACCESS_EVIDENCE_STATUSES:
        raise ExperimentRegistryError(
            "experiment_registry.record.data_access_evidence_status_invalid"
        )
    missing_access_fields = [
        field for field in DATA_ACCESS_FIELDS if field not in access
    ]
    if missing_access_fields:
        raise ExperimentRegistryError(
            "experiment_registry.record.data_access_field_required:"
            f"{missing_access_fields[0]}"
        )
    for field in DATA_ACCESS_FIELDS:
        value = access[field]
        if value is not None and type(value) is not bool:
            raise ExperimentRegistryError(
                f"experiment_registry.record.data_access_bool_or_null_required:{field}"
            )
    observed_values = [access[field] for field in DATA_ACCESS_FIELDS]
    if evidence_status == "verified_primary_evidence" and any(
        type(value) is not bool for value in observed_values
    ):
        raise ExperimentRegistryError(
            "experiment_registry.record.data_access_verified_values_required"
        )
    if evidence_status in {
        "pending_primary_evidence",
        "not_applicable_planning_record",
    } and any(value is not None for value in observed_values):
        raise ExperimentRegistryError(
            "experiment_registry.record.data_access_unverified_values_must_be_null"
        )
    if evidence_status == "partial_primary_evidence" and (
        all(value is None for value in observed_values)
        or all(type(value) is bool for value in observed_values)
    ):
        raise ExperimentRegistryError(
            "experiment_registry.record.data_access_partial_values_required"
        )

    declared_policy = access.get("declared_policy")
    if not isinstance(declared_policy, dict):
        raise ExperimentRegistryError(
            "experiment_registry.record.data_access_declared_policy_required"
        )
    for field in ACCESS_POLICY_FIELDS:
        if type(declared_policy.get(field)) is not bool:
            raise ExperimentRegistryError(
                f"experiment_registry.record.data_access_policy_bool_required:{field}"
            )
    _require_nonempty_string(
        declared_policy.get("source"), field="data_access.declared_policy.source"
    )

    actual_to_policy = {
        "validation_accessed": "validation_allowed",
        "test_inputs_accessed": "test_inputs_allowed",
        "test_targets_accessed": "test_targets_allowed",
        "test_metrics_computed": "test_metrics_allowed",
        "test_used_for_selection": "test_selection_allowed",
    }
    for actual_field, policy_field in actual_to_policy.items():
        if access[actual_field] is True and declared_policy[policy_field] is False:
            raise ExperimentRegistryError(
                "experiment_registry.record.data_access_policy_violation:"
                f"{actual_field}"
            )
    if access["test_metrics_computed"] is True and not (
        access["test_inputs_accessed"] is True
        and access["test_targets_accessed"] is True
    ):
        raise ExperimentRegistryError(
            "experiment_registry.record.data_access_inconsistent:"
            "test_metrics_require_inputs_and_targets"
        )
    if (
        access["test_used_for_selection"] is True
        and access["test_metrics_computed"] is not True
    ):
        raise ExperimentRegistryError(
            "experiment_registry.record.data_access_inconsistent:"
            "test_selection_requires_metrics"
        )
    if declared_policy["test_metrics_allowed"] and not (
        declared_policy["test_inputs_allowed"]
        and declared_policy["test_targets_allowed"]
    ):
        raise ExperimentRegistryError(
            "experiment_registry.record.data_access_policy_inconsistent:"
            "test_metrics_require_inputs_and_targets"
        )
    if (
        declared_policy["test_selection_allowed"]
        and not declared_policy["test_metrics_allowed"]
    ):
        raise ExperimentRegistryError(
            "experiment_registry.record.data_access_policy_inconsistent:"
            "test_selection_requires_metrics"
        )

    validity = result["validity"]
    assert isinstance(validity, dict)
    if status in TERMINAL_NON_COMPLETED_STATUSES:
        _require_nonempty_string(
            validity.get("reason"), field="validity.reason"
        )
    results = result["results"]
    assert isinstance(results, dict)
    if status == "completed" and not results:
        raise ExperimentRegistryError(
            "experiment_registry.record.completed_results_required"
        )

    _validate_artifacts(result["artifacts"])
    _reject_absolute_locators(result)

    observed = result["record_fingerprint"]
    expected = record_fingerprint(result)
    if not isinstance(observed, str) or observed != expected:
        raise ExperimentRegistryError(
            "experiment_registry.record.fingerprint_mismatch"
        )
    return result


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ExperimentRegistryError(
                f"experiment_registry.json.duplicate_key:{key}"
            )
        result[key] = value
    return result


def load_record(path: str | Path) -> dict[str, object]:
    source = Path(path)
    try:
        raw = source.read_bytes()
        text = raw.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ExperimentRegistryError(
                    f"experiment_registry.json.non_finite_token:{token}"
                )
            ),
        )
    except ExperimentRegistryError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExperimentRegistryError(
            f"experiment_registry.record.read_invalid:{source.name}"
        ) from exc
    if not isinstance(value, dict):
        raise ExperimentRegistryError("experiment_registry.record.object_required")
    validated = validate_record(value, filename=source.name)
    if raw != canonical_json_bytes(validated) + b"\n":
        raise ExperimentRegistryError(
            f"experiment_registry.record.noncanonical_encoding:{source.name}"
        )
    return validated


def write_record_once(path: str | Path, record: Mapping[str, object]) -> str:
    """Atomically create one compact record and never replace existing bytes."""

    destination = Path(path)
    validated = validate_record(record, filename=destination.name)
    payload = canonical_json_bytes(validated) + b"\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ExperimentRegistryError(
            f"experiment_registry.record.immutable_collision:{destination.name}"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise ExperimentRegistryError(
                f"experiment_registry.record.immutable_collision:{destination.name}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    return sha256(payload).hexdigest()


def load_registry(records: str | Path) -> tuple[dict[str, object], ...]:
    root = Path(records)
    if root.is_file():
        paths = (root,)
    elif root.is_dir():
        paths = tuple(sorted(root.glob("*.json"), key=lambda path: path.name))
    else:
        raise ExperimentRegistryError(
            f"experiment_registry.records.not_found:{root}"
        )
    loaded = tuple(load_record(path) for path in paths)
    identities = [record["experiment_id"] for record in loaded]
    if len(identities) != len(set(identities)):
        raise ExperimentRegistryError(
            "experiment_registry.records.duplicate_experiment_id"
        )
    return loaded


def validate_registry(records: str | Path) -> tuple[dict[str, object], ...]:
    return load_registry(records)


def registry_fingerprint(records: Iterable[Mapping[str, object]]) -> str:
    rows = sorted(
        (
            {
                "experiment_id": record["experiment_id"],
                "record_fingerprint": record["record_fingerprint"],
            }
            for record in records
        ),
        key=lambda row: str(row["experiment_id"]),
    )
    return sha256(canonical_json_bytes(rows)).hexdigest()


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _access_observation(value: object) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "unknown"


def _access_policy(value: object) -> str:
    return "allowed" if value is True else "forbidden"


def render_markdown_ledger(
    records: Iterable[Mapping[str, object]],
) -> str:
    """Render a deterministic, human-readable ledger from validated records."""

    validated = [validate_record(record) for record in records]
    validated.sort(key=lambda record: str(record["experiment_id"]))
    identities = [record["experiment_id"] for record in validated]
    if len(identities) != len(set(identities)):
        raise ExperimentRegistryError(
            "experiment_registry.records.duplicate_experiment_id"
        )
    lines = [
        "# Experiment evidence ledger",
        "",
        "Generated deterministically from schema 1.0.0 records. Do not edit by hand.",
        "",
        (
            "| Experiment | Phase | Kind | Status | Title | Decision use | "
            "Observed access (VALIDATION; TEST inputs/targets/metrics/selection) | "
            "Access evidence | Declared policy "
            "(VALIDATION; TEST inputs/targets/metrics/selection) | "
            "Artifacts | Fingerprint |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | --- |",
    ]
    for record in validated:
        access = record["data_access"]
        assert isinstance(access, dict)
        observed_access = (
            _access_observation(access["validation_accessed"])
            + "; "
            + "/".join(
                _access_observation(access[field]) for field in TEST_ACCESS_FIELDS
            )
        )
        declared_policy = access["declared_policy"]
        assert isinstance(declared_policy, dict)
        declared_access = (
            _access_policy(declared_policy["validation_allowed"])
            + "; "
            + "/".join(
                _access_policy(declared_policy[field])
                for field in (
                    "test_inputs_allowed",
                    "test_targets_allowed",
                    "test_metrics_allowed",
                    "test_selection_allowed",
                )
            )
        )
        artifacts = record["artifacts"]
        assert isinstance(artifacts, list)
        lines.append(
            "| "
            + " | ".join(
                _markdown_cell(value)
                for value in (
                    record["experiment_id"],
                    record["phase"],
                    record["kind"],
                    record["status"],
                    record["title"],
                    record["decision_use"],
                    observed_access,
                    access["access_evidence_status"],
                    declared_access,
                    len(artifacts),
                    record["record_fingerprint"],
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            f"Registry fingerprint: `{registry_fingerprint(validated)}`",
            "",
        ]
    )
    return "\n".join(lines)


def render_registry(records: str | Path) -> str:
    return render_markdown_ledger(load_registry(records))


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_tar_member_name(name: str) -> str:
    if not name or any(
        unicodedata.category(character) == "Cc" for character in name
    ):
        raise ExperimentRegistryError(
            "experiment_registry.archive.control_character_in_name"
        )
    normalized_separators = name.replace("\\", "/")
    posix = PurePosixPath(normalized_separators)
    windows = PureWindowsPath(name)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or ".." in posix.parts
    ):
        raise ExperimentRegistryError(
            f"experiment_registry.archive.unsafe_path:{name!r}"
        )
    canonical = posix.as_posix()
    if canonical in {"", "/"}:
        raise ExperimentRegistryError(
            "experiment_registry.archive.empty_member_name"
        )
    return canonical


def _hash_stream(stream: BinaryIO) -> str:
    digest = sha256()
    while chunk := stream.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def _inspect_tar_handle(
    handle: tarfile.TarFile,
    *,
    container: str,
    depth: int,
) -> list[dict[str, object]]:
    if depth > _MAX_NESTED_ARCHIVE_DEPTH:
        raise ExperimentRegistryError(
            "experiment_registry.archive.nesting_limit_exceeded"
        )
    seen: set[str] = set()
    rows: list[dict[str, object]] = []
    for member in handle:
        canonical_name = _safe_tar_member_name(member.name)
        if canonical_name in seen:
            raise ExperimentRegistryError(
                f"experiment_registry.archive.duplicate_member:{container}!{canonical_name}"
            )
        seen.add(canonical_name)
        display_name = (
            canonical_name if not container else f"{container}!/{canonical_name}"
        )
        if member.issym() or member.islnk():
            raise ExperimentRegistryError(
                f"experiment_registry.archive.link_member:{display_name}"
            )
        if member.isdir():
            rows.append(
                {
                    "path": display_name,
                    "kind": "directory",
                    "size_bytes": 0,
                    "depth": depth,
                }
            )
            continue
        if not member.isfile():
            raise ExperimentRegistryError(
                f"experiment_registry.archive.special_member:{display_name}"
            )
        source = handle.extractfile(member)
        if source is None:
            raise ExperimentRegistryError(
                f"experiment_registry.archive.member_unreadable:{display_name}"
            )
        is_nested = canonical_name.lower().endswith(_NESTED_ARCHIVE_SUFFIXES)
        if is_nested:
            with tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024) as nested:
                digest = sha256()
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
                    nested.write(chunk)
                nested.seek(0)
                row = {
                    "path": display_name,
                    "kind": "nested_archive",
                    "size_bytes": member.size,
                    "sha256": digest.hexdigest(),
                    "depth": depth,
                }
                rows.append(row)
                try:
                    with tarfile.open(fileobj=nested, mode="r:*") as nested_handle:
                        rows.extend(
                            _inspect_tar_handle(
                                nested_handle,
                                container=display_name,
                                depth=depth + 1,
                            )
                        )
                except ExperimentRegistryError:
                    raise
                except (OSError, tarfile.TarError, EOFError) as exc:
                    raise ExperimentRegistryError(
                        f"experiment_registry.archive.nested_invalid:{display_name}"
                    ) from exc
        else:
            rows.append(
                {
                    "path": display_name,
                    "kind": "file",
                    "size_bytes": member.size,
                    "sha256": _hash_stream(source),
                    "depth": depth,
                }
            )
    return rows


def inspect_archive(path: str | Path) -> dict[str, object]:
    """Recursively inspect a tar archive without extracting any member."""

    source = Path(path)
    if not source.is_file():
        raise ExperimentRegistryError(
            f"experiment_registry.archive.not_found:{source}"
        )
    try:
        with tarfile.open(source, mode="r:*") as handle:
            members = _inspect_tar_handle(handle, container="", depth=0)
    except ExperimentRegistryError:
        raise
    except (OSError, tarfile.TarError, EOFError) as exc:
        raise ExperimentRegistryError(
            f"experiment_registry.archive.invalid:{source.name}"
        ) from exc
    kinds = [member["kind"] for member in members]
    return {
        "schema_version": SCHEMA_VERSION,
        "safe": True,
        "sha256": file_sha256(source),
        "size_bytes": source.stat().st_size,
        "member_count": len(members),
        "regular_file_count": kinds.count("file"),
        "directory_count": kinds.count("directory"),
        "nested_archive_count": kinds.count("nested_archive"),
        "members": members,
    }


inspect_tar_archive = inspect_archive


def _validate_existing_store_object(
    destination: Path, *, expected_sha256: str, expected_size: int
) -> None:
    if (
        destination.is_symlink()
        or not destination.is_file()
        or destination.stat().st_size != expected_size
        or file_sha256(destination) != expected_sha256
    ):
        raise ExperimentRegistryError(
            f"experiment_registry.store.content_collision:{destination.name}"
        )


def store_archive(path: str | Path, root: str | Path) -> dict[str, object]:
    """Validate and atomically copy a gzip tar into a SHA-256 store."""

    source = Path(path)
    try:
        with source.open("rb") as handle:
            gzip_magic = handle.read(2)
    except OSError as exc:
        raise ExperimentRegistryError(
            f"experiment_registry.archive.not_found:{source}"
        ) from exc
    if gzip_magic != b"\x1f\x8b":
        raise ExperimentRegistryError(
            "experiment_registry.store.gzip_tar_required"
        )
    inspection = inspect_archive(source)
    digest = str(inspection["sha256"])
    size = int(inspection["size_bytes"])
    locator = PurePosixPath("sha256", digest[:2], f"{digest}.tar.gz").as_posix()
    destination = Path(root) / Path(locator)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        _validate_existing_store_object(
            destination, expected_sha256=digest, expected_size=size
        )
        return {"sha256": digest, "size_bytes": size, "locator": locator}

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{digest}.", suffix=".partial", dir=destination.parent
    )
    temporary = Path(temporary_name)
    copied_digest = sha256()
    copied_size = 0
    try:
        with source.open("rb") as input_handle, os.fdopen(
            descriptor, "wb"
        ) as output_handle:
            while chunk := input_handle.read(1024 * 1024):
                output_handle.write(chunk)
                copied_digest.update(chunk)
                copied_size += len(chunk)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        if copied_digest.hexdigest() != digest or copied_size != size:
            raise ExperimentRegistryError(
                "experiment_registry.store.source_changed_during_copy"
            )
        try:
            os.link(temporary, destination)
            destination.chmod(0o444)
        except FileExistsError:
            _validate_existing_store_object(
                destination, expected_sha256=digest, expected_size=size
            )
    finally:
        temporary.unlink(missing_ok=True)
    return {"sha256": digest, "size_bytes": size, "locator": locator}


store_archive_content_addressed = store_archive


__all__ = [
    "ACCESS_EVIDENCE_STATUSES",
    "ACCESS_POLICY_FIELDS",
    "ARTIFACT_FIELDS",
    "DATA_ACCESS_FIELDS",
    "ExperimentRegistryError",
    "REQUIRED_FIELDS",
    "SCHEMA_VERSION",
    "STATUS_VALUES",
    "TEST_ACCESS_FIELDS",
    "canonical_json",
    "canonical_json_bytes",
    "file_sha256",
    "inspect_archive",
    "inspect_tar_archive",
    "load_record",
    "load_registry",
    "record_fingerprint",
    "registry_fingerprint",
    "render_markdown_ledger",
    "render_registry",
    "seal_record",
    "store_archive",
    "store_archive_content_addressed",
    "validate_record",
    "validate_registry",
    "write_record_once",
]
