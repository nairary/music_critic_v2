from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
import shutil

import pytest
import torch
from torch.utils.data import DataLoader

from music_critic.adapters import (
    DILEMMADATA_ADAPTER_VERSION,
    DILEMMADATA_CORPUS_IDENTITY_VERSION,
    DILEMMADATA_GROUPING_VERSION,
    DILEMMADATA_RAW_PROJECTION_VERSION,
    DilemmadataAccepted,
    DilemmadataCorpusIdentity,
    DilemmadataCorpusIdentityError,
    DilemmadataQuarantine,
    convert_dilemmadata_record,
    discover_dilemmadata_corpus,
    iter_dilemmadata_corpus,
)
from music_critic.data import dumps_piece, loads_piece
from music_critic.graph import (
    build_raw_graph,
    dumps_graph,
    graph_fingerprint,
    model_input_fingerprint,
)
from music_critic.ssl.data import IndexedSSLRawDataset, collate_ssl_samples
from music_critic.tasks import (
    CorpusCacheConfig,
    IndexedMultiSourceDataset,
    MultiCorpusDataset,
    build_dilemmadata_corpus_cache,
    plan_group_hash_split,
    validate_split_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "dilemmadata"
CORPUS = FIXTURES / "corpus"


def _fixture_identity(root: Path) -> DilemmadataCorpusIdentity:
    observed = discover_dilemmadata_corpus(root, require_valid=False)
    return DilemmadataCorpusIdentity(
        release_version=observed.release_version or "v1.0",
        installation_file_count=observed.installation_file_count,
        content_fingerprint=observed.content_fingerprint,
        primary_record_count=len(observed.records),
        an_record_count=sum(row.dialect == "an_joint" for row in observed.records),
        dlc_record_count=sum(row.dialect == "dlc" for row in observed.records),
    )


def _discover(root: Path):
    return discover_dilemmadata_corpus(root, identity=_fixture_identity(root))


def _record(root: Path, record_id: str):
    return next(row for row in _discover(root).records if row.record_id == record_id)


def _accepted(root: Path, record_id: str) -> DilemmadataAccepted:
    outcome = convert_dilemmadata_record(_record(root, record_id))
    assert isinstance(outcome, DilemmadataAccepted)
    return outcome


def _read_tsv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    return rows[0], rows[1:]


def _write_tsv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _set_cell(path: Path, row: int, column: str, value: str) -> None:
    header, rows = _read_tsv(path)
    rows[row][header.index(column)] = value
    _write_tsv(path, header, rows)


def _raw_evidence(outcome: DilemmadataAccepted) -> tuple[str, str, str, str]:
    graph = build_raw_graph(outcome.piece, assume_valid=True)
    return (
        dumps_piece(outcome.piece),
        dumps_graph(graph),
        graph_fingerprint(graph),
        model_input_fingerprint(graph),
    )


def test_public_versions_and_pinned_identity_gate() -> None:
    assert DILEMMADATA_ADAPTER_VERSION == "1.0.0"
    assert DILEMMADATA_CORPUS_IDENTITY_VERSION == "1.0.0"
    assert DILEMMADATA_RAW_PROJECTION_VERSION == "1.0.0"
    assert DILEMMADATA_GROUPING_VERSION == "1.0.0"

    with pytest.raises(DilemmadataCorpusIdentityError):
        discover_dilemmadata_corpus(CORPUS)

    discovery = _discover(CORPUS)
    assert discovery.is_valid
    assert [row.record_id for row in discovery.records] == [
        "an:training:same",
        "an:validation:same-alt",
        "dlc:demo:same",
    ]
    assert discovery.component_count == 1
    assert discovery.multi_record_component_count == 1
    assert discovery.explicit_overlap_count == 1
    assert discovery.suggested_split_conflict_count == 1


def test_both_dialects_are_raw_only_exact_and_source_neutral() -> None:
    outcomes = tuple(
        iter_dilemmadata_corpus(CORPUS, identity=_fixture_identity(CORPUS))
    )
    assert len(outcomes) == 3
    assert all(isinstance(row, DilemmadataAccepted) for row in outcomes)
    assert {row.record.dialect for row in outcomes} == {"an_joint", "dlc"}

    for outcome in outcomes:
        assert isinstance(outcome, DilemmadataAccepted)
        piece = outcome.piece
        assert piece.annotations == ()
        assert piece.targets == ()
        assert len(piece.tracks) == 1
        assert piece.tracks[0].name is None
        assert piece.tracks[0].instrument_name is None
        assert piece.tracks[0].is_percussion is False
        assert piece.split is None
        assert not Path(piece.source_path).is_absolute()
        assert outcome.validation_report.errors == ()
        payload = dumps_piece(piece)
        assert loads_piece(payload) == piece
        assert dumps_piece(loads_piece(payload)) == payload
        assert graph_fingerprint(build_raw_graph(piece, assume_valid=True))
    assert any(
        outcome.statistics.incomplete_bar_count > 0
        for outcome in outcomes
        if isinstance(outcome, DilemmadataAccepted)
    )


@pytest.mark.parametrize(
    ("record_id", "columns"),
    [
        ("an:training:same", ("s_step", "s_alter", "s_part_id", "s_voice_id")),
        ("dlc:demo:same", ("step", "alter", "staff", "voice")),
    ],
    ids=["an", "dlc"],
)
def test_optional_spelling_staff_and_voice_dropout_keeps_source_neutral_topology(
    tmp_path: Path, record_id: str, columns: tuple[str, ...]
) -> None:
    copied = tmp_path / "corpus"
    shutil.copytree(CORPUS, copied)
    before = _accepted(copied, record_id)
    source = before.record.path
    header, rows = _read_tsv(source)
    keep = [index for index, name in enumerate(header) if name not in columns]
    _write_tsv(
        source,
        [header[index] for index in keep],
        [[row[index] for index in keep] for row in rows],
    )

    after = _accepted(copied, record_id)
    assert len(after.piece.tracks) == 1
    assert all(
        note.spelling_step is None
        and note.spelling_alter is None
        and note.staff is None
        and note.voice is None
        for note in after.piece.notes
    )
    before_graph = build_raw_graph(before.piece, assume_valid=True)
    after_graph = build_raw_graph(after.piece, assume_valid=True)
    assert before_graph.node_types == after_graph.node_types
    assert before_graph.edge_types == after_graph.edge_types
    assert {
        node_type: int(before_graph[node_type].num_nodes)
        for node_type in before_graph.node_types
    } == {
        node_type: int(after_graph[node_type].num_nodes)
        for node_type in after_graph.node_types
    }
    assert {
        edge_type: int(before_graph[edge_type].edge_index.shape[1])
        for edge_type in before_graph.edge_types
    } == {
        edge_type: int(after_graph[edge_type].edge_index.shape[1])
        for edge_type in after_graph.edge_types
    }


@pytest.mark.parametrize(
    "variant",
    [
        "all_target_values",
        "alt_label",
        "valid_gate",
        "delete",
        "reorder",
        "analyst_metadata",
    ],
)
def test_theory_column_mutations_do_not_change_raw_or_model_evidence(
    tmp_path: Path, variant: str
) -> None:
    copied = tmp_path / "corpus"
    shutil.copytree(CORPUS, copied)
    source = copied / "pitch_arrays" / "AN" / "training" / "same_joint.tsv"
    before_record = _record(copied, "an:training:same")
    before = convert_dilemmadata_record(before_record)
    assert isinstance(before, DilemmadataAccepted)
    header, rows = _read_tsv(source)
    raw = {
        "onset_div",
        "duration_div",
        "s_offset_frac",
        "s_duration_frac",
        "s_midi",
        "s_isOnset",
        "s_step",
        "s_alter",
        "ks_fifths",
        "ts_beats",
        "ts_beat_type",
        "s_part_id",
        "s_voice_id",
        "s_measure",
        "measureNumberWithSuffix",
        "mn_onset",
    }
    theory_indices = [index for index, name in enumerate(header) if name not in raw]
    physical_source_changes = True
    if variant == "all_target_values":
        for row_number, row in enumerate(rows):
            for index in theory_indices:
                row[index] = f"mutated-target-{row_number}-{index}"
    elif variant == "alt_label":
        header.append("alt_label")
        for row_number, row in enumerate(rows):
            row.append(f"alternate-{row_number}")
    elif variant == "valid_gate":
        gate = header.index("valid_chord_label")
        for row in rows:
            row[gate] = "False" if row[gate] == "True" else "True"
    elif variant == "delete":
        keep = [index for index in range(len(header)) if index not in theory_indices]
        header = [header[index] for index in keep]
        rows = [[row[index] for index in keep] for row in rows]
    elif variant == "reorder":
        raw_indices = [index for index in range(len(header)) if index not in theory_indices]
        order = [*raw_indices, *reversed(theory_indices)]
        header = [header[index] for index in order]
        rows = [[row[index] for index in order] for row in rows]
    else:
        summary = copied / "pitch_arrays" / "AN" / "dataset_summary.tsv"
        summary_payload = summary.read_text(encoding="utf-8")
        summary.write_text(
            summary_payload.replace("Analyst A", "Different Analyst").replace(
                "Proofreader A", "Different Proofreader"
            ),
            encoding="utf-8",
        )
        physical_source_changes = False
    if variant != "analyst_metadata":
        _write_tsv(source, header, rows)

    after_record = _record(copied, "an:training:same")
    after = convert_dilemmadata_record(after_record)
    assert isinstance(after, DilemmadataAccepted)
    assert (
        before_record.physical_source_sha256 != after_record.physical_source_sha256
    ) is physical_source_changes
    assert before_record.raw_projection_sha256 == after_record.raw_projection_sha256
    assert before_record.raw_equivalence_id == after_record.raw_equivalence_id
    assert before_record.grouping_fingerprint == after_record.grouping_fingerprint
    assert before_record.source_group_id == after_record.source_group_id
    assert _raw_evidence(before) == _raw_evidence(after)


@pytest.mark.parametrize(
    ("columns", "values", "expected_quarantine"),
    [
        (("s_midi",), ("61",), False),
        (("s_offset_frac", "onset_div"), ("1/2", "240"), False),
        (("s_duration_frac", "duration_div"), ("1/2", "240"), False),
        (("ts_beats",), ("3",), False),
        (("s_isOnset",), ("False",), True),
        (("s_voice_id",), ("2",), False),
    ],
    ids=["pitch", "onset", "duration", "meter", "tie", "voice"],
)
def test_raw_mutations_change_canonical_evidence_or_quarantine(
    tmp_path: Path,
    columns: tuple[str, ...],
    values: tuple[str, ...],
    expected_quarantine: bool,
) -> None:
    copied = tmp_path / "corpus"
    shutil.copytree(CORPUS, copied)
    source = copied / "pitch_arrays" / "AN" / "training" / "same_joint.tsv"
    before_record = _record(copied, "an:training:same")
    before = convert_dilemmadata_record(before_record)
    assert isinstance(before, DilemmadataAccepted)
    for column, value in zip(columns, values, strict=True):
        _set_cell(source, 0, column, value)

    after_record = _record(copied, "an:training:same")
    after = convert_dilemmadata_record(after_record)
    assert before_record.raw_projection_sha256 != after_record.raw_projection_sha256
    if expected_quarantine:
        assert isinstance(after, DilemmadataQuarantine)
        assert "dilemmadata.tie_predecessor_missing" in after.categories
    else:
        assert isinstance(after, DilemmadataAccepted)
        assert _raw_evidence(before) != _raw_evidence(after)


def test_tie_merge_and_grace_policies_are_exact(tmp_path: Path) -> None:
    copied = tmp_path / "corpus"
    shutil.copytree(CORPUS, copied)
    source = copied / "pitch_arrays" / "AN" / "training" / "same_joint.tsv"
    header, rows = _read_tsv(source)
    index = {name: header.index(name) for name in header}
    continuation = rows[0].copy()
    continuation[index["onset_div"]] = "480"
    continuation[index["s_offset_frac"]] = "1"
    continuation[index["s_isOnset"]] = "False"
    grace = rows[1].copy()
    grace[index["onset_div"]] = "960"
    grace[index["duration_div"]] = "0"
    grace[index["s_offset_frac"]] = "2"
    grace[index["s_duration_frac"]] = "0"
    grace[index["s_midi"]] = "67"
    grace[index["s_step"]] = "G"
    rows = [rows[0], continuation, grace]
    _write_tsv(source, header, rows)

    outcome = _accepted(copied, "an:training:same")
    assert outcome.statistics.source_note_row_count == 3
    assert outcome.statistics.tie_continuation_row_count == 1
    assert outcome.statistics.tie_merge_count == 1
    assert outcome.statistics.canonical_note_count == 2
    assert outcome.statistics.grace_note_count == 1
    merged = next(note for note in outcome.piece.notes if note.pitch == 60)
    assert merged.duration_qn.num == 2
    assert merged.duration_qn.den == 1
    assert any(note.is_grace and note.duration_qn.num == 0 for note in outcome.piece.notes)


def test_dlc_zero_duration_is_retained_as_grace_without_invented_time(
    tmp_path: Path,
) -> None:
    copied = tmp_path / "corpus"
    shutil.copytree(CORPUS, copied)
    source = copied / "pitch_arrays" / "DLC" / "demo" / "same.tsv"
    _set_cell(source, 0, "duration", "0")
    _set_cell(source, 0, "duration_div", "0")

    outcome = _accepted(copied, "dlc:demo:same")
    grace = next(note for note in outcome.piece.notes if note.pitch == 60)
    assert grace.is_grace
    assert grace.duration_qn.num == 0
    assert outcome.statistics.grace_note_count == 1


def test_zero_duration_tie_continuation_is_quarantined(tmp_path: Path) -> None:
    copied = tmp_path / "corpus"
    shutil.copytree(CORPUS, copied)
    source = copied / "pitch_arrays" / "DLC" / "demo" / "same.tsv"
    _set_cell(source, 0, "duration", "0")
    _set_cell(source, 0, "duration_div", "0")
    _set_cell(source, 0, "is_note_onset", "False")

    outcome = convert_dilemmadata_record(_record(copied, "dlc:demo:same"))
    assert isinstance(outcome, DilemmadataQuarantine)
    assert "dilemmadata.grace_conflict" in outcome.categories


def test_post_discovery_raw_mutation_fails_fingerprint_binding(
    tmp_path: Path,
) -> None:
    copied = tmp_path / "corpus"
    shutil.copytree(CORPUS, copied)
    record = _record(copied, "an:training:same")
    _set_cell(record.path, 0, "s_midi", "61")

    outcome = convert_dilemmadata_record(record)
    assert isinstance(outcome, DilemmadataQuarantine)
    assert "dilemmadata.raw_fingerprint_mismatch" in outcome.categories


def test_pickup_meter_change_and_measure_anchors_are_reconstructed(tmp_path: Path) -> None:
    copied = tmp_path / "corpus"
    shutil.copytree(CORPUS, copied)
    source = copied / "pitch_arrays" / "AN" / "training" / "same_joint.tsv"
    header, template_rows = _read_tsv(source)
    header.extend(["s_measure", "measureNumberWithSuffix", "mn_onset"])
    original_header = header[:-3]
    original_index = {name: original_header.index(name) for name in original_header}

    def row(onset: int, duration: int, measure: str, mn_onset: str, meter: int) -> list[str]:
        value = template_rows[0].copy()
        value[original_index["onset_div"]] = str(onset * 480)
        value[original_index["duration_div"]] = str(duration * 480)
        value[original_index["s_offset_frac"]] = str(onset)
        value[original_index["s_duration_frac"]] = str(duration)
        value[original_index["ts_beats"]] = str(meter)
        return [*value, measure, measure, mn_onset]

    rows = [
        row(0, 1, "0", "0", 4),
        row(1, 4, "1", "0", 4),
        row(5, 3, "2", "0", 3),
    ]
    _write_tsv(source, header, rows)

    outcome = _accepted(copied, "an:training:same")
    assert [(event.onset_qn.num, event.numerator) for event in outcome.piece.meter_events] == [
        (0, 4),
        (5, 3),
    ]
    assert len(outcome.piece.bars) == 3
    assert outcome.piece.bars[0].is_pickup
    assert outcome.piece.bars[0].duration_qn.num == 1
    assert not outcome.piece.bars[1].is_incomplete
    assert not outcome.piece.bars[2].is_incomplete


def test_inconsistent_measure_anchor_is_quarantined(tmp_path: Path) -> None:
    copied = tmp_path / "corpus"
    shutil.copytree(CORPUS, copied)
    source = copied / "pitch_arrays" / "AN" / "training" / "same_joint.tsv"
    header, rows = _read_tsv(source)
    header.extend(["s_measure", "measureNumberWithSuffix", "mn_onset"])
    rows[0].extend(["1", "1", "0"])
    rows[1].extend(["1", "1", "0"])
    _write_tsv(source, header, rows)

    outcome = convert_dilemmadata_record(_record(copied, "an:training:same"))
    assert isinstance(outcome, DilemmadataQuarantine)
    assert "dilemmadata.bar_reconstruction_failed" in outcome.categories


def test_inconsistent_simultaneous_meter_is_quarantined(tmp_path: Path) -> None:
    copied = tmp_path / "corpus"
    shutil.copytree(CORPUS, copied)
    source = copied / "pitch_arrays" / "AN" / "training" / "same_joint.tsv"
    _set_cell(source, 1, "s_offset_frac", "0")
    _set_cell(source, 1, "onset_div", "0")
    _set_cell(source, 1, "ts_beats", "3")

    outcome = convert_dilemmadata_record(_record(copied, "an:training:same"))
    assert isinstance(outcome, DilemmadataQuarantine)
    assert "dilemmadata.meter_conflict" in outcome.categories


@pytest.mark.parametrize(
    ("fixture", "category"),
    [
        ("an_missing_field_joint.tsv", "dilemmadata.missing_required_raw_field"),
        ("dlc_bad_width.tsv", "dilemmadata.row_width_mismatch"),
    ],
)
def test_malformed_records_have_stable_quarantine_categories(
    tmp_path: Path, fixture: str, category: str
) -> None:
    copied = tmp_path / "corpus"
    shutil.copytree(CORPUS, copied)
    if fixture.startswith("an_"):
        target = copied / "pitch_arrays" / "AN" / "training" / "broken_joint.tsv"
    else:
        target = copied / "pitch_arrays" / "DLC" / "broken" / "bad.tsv"
        target.parent.mkdir()
    shutil.copy2(FIXTURES / "malformed" / fixture, target)
    discovery = _discover(copied)
    record = next(row for row in discovery.records if row.path == target.resolve())
    outcome = convert_dilemmadata_record(record)
    assert isinstance(outcome, DilemmadataQuarantine)
    assert category in outcome.categories


def test_target_only_cache_reuses_artifacts_but_raw_mutation_misses(
    tmp_path: Path,
) -> None:
    copied = tmp_path / "corpus"
    shutil.copytree(CORPUS, copied)
    cache = CorpusCacheConfig(tmp_path / "cache")
    first_index, first_report = build_dilemmadata_corpus_cache(
        copied, cache_config=cache, identity=_fixture_identity(copied)
    )
    assert first_report.cache_miss_count == 3
    source = copied / "pitch_arrays" / "AN" / "training" / "same_joint.tsv"
    _set_cell(source, 0, "a_romanNumeral", "target-only-mutation")
    second_index, second_report = build_dilemmadata_corpus_cache(
        copied, cache_config=cache, identity=_fixture_identity(copied)
    )
    assert second_report.cache_hit_count == 3
    first_split = plan_group_hash_split((first_index,), seed=37, ratios={"train": 1.0})
    second_split = plan_group_hash_split((second_index,), seed=37, ratios={"train": 1.0})
    assert {
        (row.dataset_id, row.piece_id): row.split for row in first_split.assignments
    } == {
        (row.dataset_id, row.piece_id): row.split for row in second_split.assignments
    }
    by_piece_first = {row.piece_id: row for row in first_index.records}
    by_piece_second = {row.piece_id: row for row in second_index.records}
    for piece_id in by_piece_first:
        assert by_piece_first[piece_id].cache_key == by_piece_second[piece_id].cache_key
        assert (
            by_piece_first[piece_id].canonical_relative_path
            == by_piece_second[piece_id].canonical_relative_path
        )
        assert by_piece_first[piece_id].canonical_sha256 == by_piece_second[piece_id].canonical_sha256

    _set_cell(source, 0, "s_midi", "61")
    third_index, third_report = build_dilemmadata_corpus_cache(
        copied, cache_config=cache, identity=_fixture_identity(copied)
    )
    assert third_report.cache_hit_count == 2
    assert third_report.cache_miss_count == 1
    changed_piece_id = _record(copied, "an:training:same").piece_id
    by_piece_third = {row.piece_id: row for row in third_index.records}
    assert by_piece_second[changed_piece_id].cache_key != by_piece_third[changed_piece_id].cache_key
    assert (
        by_piece_second[changed_piece_id].canonical_relative_path
        != by_piece_third[changed_piece_id].canonical_relative_path
    )


def test_cache_dataset_ssl_and_group_safe_split_round_trip(tmp_path: Path) -> None:
    cache = CorpusCacheConfig(tmp_path / "cache")
    index, report = build_dilemmadata_corpus_cache(
        CORPUS, cache_config=cache, identity=_fixture_identity(CORPUS)
    )
    assert report.accepted_count == 3
    assert report.raw_only_piece_count == 3
    dataset = IndexedMultiSourceDataset(index, cache_config=cache)
    assert tuple(dataset[row].piece_id for row in range(len(dataset))) == tuple(
        record.piece_id for record in index.records
    )
    ssl_dataset = IndexedSSLRawDataset(index, cache_config=cache)
    ssl_batch = collate_ssl_samples([ssl_dataset[0], ssl_dataset[1]])
    assert ssl_batch.sample_count == 2
    assert ssl_batch.dataset_ids == ("dilemmadata", "dilemmadata")

    manifest = plan_group_hash_split(
        (index,), seed=19, ratios={"train": 1.0}
    )
    validate_split_manifest(manifest, (index,))
    assert {row.split for row in manifest.assignments} == {"train"}
    assert len({row.component_fingerprint for row in manifest.assignments}) == 1
    mixed = MultiCorpusDataset((dataset,), manifest, split="train")
    assert len(mixed) == 3


def _ssl_loader_evidence(index, cache: CorpusCacheConfig, *, workers: int):
    dataset = IndexedSSLRawDataset(index, cache_config=cache)
    generator = torch.Generator().manual_seed(31)
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_ssl_samples,
        generator=generator,
        multiprocessing_context="spawn" if workers else None,
    )
    return tuple(
        (
            batch.dataset_ids,
            batch.piece_ids,
            batch.sample_count,
            batch.node_count,
            batch.edge_count,
        )
        for batch in loader
    )


def test_dilemmadata_ssl_dataloader_workers_zero_and_spawn_have_parity(
    tmp_path: Path,
) -> None:
    cache = CorpusCacheConfig(tmp_path / "cache")
    index, _report = build_dilemmadata_corpus_cache(
        CORPUS, cache_config=cache, identity=_fixture_identity(CORPUS)
    )
    assert _ssl_loader_evidence(index, cache, workers=0) == _ssl_loader_evidence(
        index, cache, workers=2
    )


def test_identity_type_can_be_replaced_without_changing_release_commit() -> None:
    identity = _fixture_identity(CORPUS)
    altered = replace(identity, primary_record_count=identity.primary_record_count + 1)
    assert altered.release_commit == identity.release_commit
    with pytest.raises(DilemmadataCorpusIdentityError):
        discover_dilemmadata_corpus(CORPUS, identity=altered)
