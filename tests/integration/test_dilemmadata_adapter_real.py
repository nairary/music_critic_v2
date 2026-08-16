from __future__ import annotations

from collections import Counter
import os
from pathlib import Path

import pytest

from music_critic.adapters import (
    DILEMMADATA_AN_RECORD_COUNT,
    DILEMMADATA_DLC_RECORD_COUNT,
    DILEMMADATA_PRIMARY_RECORD_COUNT,
    DilemmadataAccepted,
    DilemmadataCorpusIdentity,
    DilemmadataQuarantine,
    convert_dilemmadata_record,
    discover_dilemmadata_corpus,
)
from music_critic.graph import build_raw_graph, graph_fingerprint


def _corpus_root() -> Path:
    value = os.environ.get("MUSIC_CRITIC_DILEMMADATA_ROOT")
    if not value:
        pytest.skip("MUSIC_CRITIC_DILEMMADATA_ROOT is not configured")
    return Path(value)


def test_pinned_full_corpus_streams_one_outcome_per_primary_record() -> None:
    discovery = discover_dilemmadata_corpus(
        _corpus_root(), identity=DilemmadataCorpusIdentity()
    )
    assert len(discovery.records) == DILEMMADATA_PRIMARY_RECORD_COUNT
    assert Counter(row.dialect for row in discovery.records) == {
        "an_joint": DILEMMADATA_AN_RECORD_COUNT,
        "dlc": DILEMMADATA_DLC_RECORD_COUNT,
    }

    statuses: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    for record in discovery.records:
        outcome = convert_dilemmadata_record(record)
        statuses[outcome.status] += 1
        if isinstance(outcome, DilemmadataAccepted):
            assert outcome.piece.targets == ()
            assert outcome.piece.annotations == ()
            assert not outcome.validation_report.errors
            assert graph_fingerprint(build_raw_graph(outcome.piece, assume_valid=True))
        else:
            assert isinstance(outcome, DilemmadataQuarantine)
            categories.update(outcome.categories)

    assert sum(statuses.values()) == DILEMMADATA_PRIMARY_RECORD_COUNT
    assert statuses["accepted"] + statuses["quarantined"] == (
        DILEMMADATA_PRIMARY_RECORD_COUNT
    )
    assert all(category.startswith("dilemmadata.") for category in categories)
