from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from music_critic.adapters import (
    convert_pop909_cl_file,
    discover_pop909_cl_corpus,
)
from music_critic.graph import build_raw_graph, graph_fingerprint
from music_critic.tasks import (
    load_corpus_index,
    load_split_manifest,
    validate_split_manifest,
)
from scripts.accept_pop909_cl_adapter import (
    _load_expectations,
    build_acceptance_report,
)


def _root() -> Path:
    return Path(
        os.environ.get(
            "MUSIC_CRITIC_POP909_CL_ROOT",
            "data/pop909-cl",
        )
    )


@pytest.mark.skipif(
    os.environ.get("MUSIC_CRITIC_RUN_POP909_CL_PRODUCTION_ACCEPTANCE") != "1",
    reason="set MUSIC_CRITIC_RUN_POP909_CL_PRODUCTION_ACCEPTANCE=1",
)
def test_complete_pop909_cl_production_acceptance() -> None:
    root = _root()
    manifest = Path("tests/fixtures/pop909_cl/production_manifest.json")
    report = build_acceptance_report(root, _load_expectations(manifest))
    assert report["ready"], json.dumps(
        {
            "mismatches": report["mismatches"],
            "fatal_failure_count": report["fatal_failure_count"],
            "fatal_failure_samples": report["fatal_failure_samples"],
        },
        sort_keys=True,
    )


@pytest.mark.skipif(
    os.environ.get("MUSIC_CRITIC_RUN_POP909_CL_PRODUCTION_ACCEPTANCE") != "1",
    reason="set MUSIC_CRITIC_RUN_POP909_CL_PRODUCTION_ACCEPTANCE=1",
)
def test_real_543_553_raw_equivalence_and_target_difference() -> None:
    discovery = discover_pop909_cl_corpus(_root())
    records = {
        record.song_id: record
        for record in discovery.records
        if record.song_id in {"543", "553"}
    }
    left = convert_pop909_cl_file(records["543"])
    right = convert_pop909_cl_file(records["553"])

    assert left.record.sha256 == (
        "7dc63700fb5e58d2d12b580aa53614413317232caa151920d6079ad2440b662b"
    )
    assert right.record.sha256 == (
        "618b99761e750edfaffb4053cc3ad073661fd5c969bfea840481f466a03ec07a"
    )
    assert left.score_projection_sha256 == right.score_projection_sha256
    assert left.record.source_group_id == right.record.source_group_id
    assert left.piece.piece_id != right.piece.piece_id
    assert graph_fingerprint(
        build_raw_graph(left.piece)
    ) == graph_fingerprint(build_raw_graph(right.piece))
    left_targets = {target.task: target for target in left.piece.targets}
    right_targets = {
        target.task: target for target in right.piece.targets
    }
    assert left_targets["pop909_cl.chord.boundary"] == right_targets[
        "pop909_cl.chord.boundary"
    ]
    assert left_targets["pop909_cl.chord.no_chord"] == right_targets[
        "pop909_cl.chord.no_chord"
    ]
    assert left_targets["pop909_cl.chord.bass"].mask == right_targets[
        "pop909_cl.chord.bass"
    ].mask
    for task_id in (
        "pop909_cl.chord.root",
        "pop909_cl.chord.quality",
        "pop909_cl.chord.inversion",
    ):
        assert left_targets[task_id].mask != right_targets[task_id].mask
        assert left_targets[task_id].values != right_targets[task_id].values
    assert left_targets["pop909_cl.chord.bass"].values != right_targets[
        "pop909_cl.chord.bass"
    ].values
    for task_id in (
        "pop909_cl.chord.root",
        "pop909_cl.chord.quality",
        "pop909_cl.chord.bass",
        "pop909_cl.chord.inversion",
    ):
        assert left_targets[task_id] != right_targets[task_id]


@pytest.mark.skipif(
    os.environ.get("MUSIC_CRITIC_RUN_POP909_CL_FULL_CACHE_ACCEPTANCE") != "1",
    reason="set MUSIC_CRITIC_RUN_POP909_CL_FULL_CACHE_ACCEPTANCE=1",
)
def test_full_cache_and_joint_split_keep_raw_duplicates_atomic() -> None:
    hook = load_corpus_index("data/cache/hooktheory.index.json")
    pop = load_corpus_index("data/cache/pop909_cl.index.json")
    manifest = load_split_manifest("data/cache/global.split.json")
    validate_split_manifest(manifest, (hook, pop))

    assert len(pop.records) == 908
    assert len({row.piece_id for row in pop.records}) == 908
    assert len({row.source_group_id for row in pop.records}) == 907
    duplicates = tuple(
        row
        for row in pop.records
        if row.source_identity in {"543", "553"}
    )
    assert len(duplicates) == 2
    assert len({row.source_group_id for row in duplicates}) == 1
    assignments = {
        (row.dataset_id, row.piece_id): row
        for row in manifest.assignments
    }
    duplicate_splits = {
        assignments[(row.dataset_id, row.piece_id)].split
        for row in duplicates
    }
    assert len(duplicate_splits) == 1
