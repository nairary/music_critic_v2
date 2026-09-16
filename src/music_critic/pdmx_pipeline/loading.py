"""Split-guarded lazy raw windows, compatible with the existing SSL collator."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from music_critic.data.schema import CanonicalPiece
from music_critic.graph.builder import build_raw_graph
from music_critic.graph.serialization import graph_fingerprint
from music_critic.ssl.data import SSLRawSample
from music_critic.tasks.corpus import (
    CorpusCacheConfig, CorpusIndex, load_cached_piece, validate_current_corpus_index,
)
from music_critic.tasks.loading import SplitManifest, validate_split_manifest

from .manifest import PdmxPipelineError, fingerprint, guard_split
from .pipeline import read_publication, validate_pdmx_contract
from music_critic.adapters.pdmx import PDMX_ADAPTER_VERSION


class PdmxWindowDataset:
    """Construction loads metadata only; indexing reads exactly one window.

    `reader` is an injection point for instrumentation. Guard and index checks
    execute before it. Neither targets nor target sidecars enter the graph.
    """

    def __init__(self, index: CorpusIndex, split_manifest: SplitManifest, *,
                 cache_config: CorpusCacheConfig, split: str,
                 reader: Callable[..., CanonicalPiece] | None = None) -> None:
        guard_split(split)
        validate_current_corpus_index(index)
        if index.header.adapter_name != "pdmx" or index.header.adapter_version != PDMX_ADAPTER_VERSION:
            raise PdmxPipelineError("pdmx.cached_adapter_stale")
        for assignment in split_manifest.assignments:
            guard_split(assignment.split)
        for row in index.records:
            guard_split(row.suggested_split)
        contract = validate_pdmx_contract(cache_config.root, split=split)
        if index.header.adapter_config_fingerprint != fingerprint({"contract": contract}):
            raise PdmxPipelineError("pdmx.cached_index_contract_mismatch")
        validate_split_manifest(split_manifest, (index,))
        if index.header.dataset_id != "pdmx":
            raise PdmxPipelineError("pdmx.dataset_identity_invalid")
        assignments = {row.piece_id: row.split for row in split_manifest.assignments}
        if any(assignments[row.piece_id] != row.suggested_split for row in index.records):
            raise PdmxPipelineError("pdmx.cached_split_mismatch")
        self.records = tuple(row for row in index.records if assignments[row.piece_id] == split)
        self.cache_config = cache_config
        self.split = split
        self.reader = load_cached_piece if reader is None else reader

    @classmethod
    def from_output(cls, root: Path, *, split: str) -> PdmxWindowDataset:
        guard_split(split)
        index, manifest, _ = read_publication(root, split=split)
        return cls(index, manifest, cache_config=CorpusCacheConfig(root, "canonical"), split=split)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, position: int) -> SSLRawSample:
        guard_split(self.split)
        if type(position) is not int or not 0 <= position < len(self):
            raise IndexError("PDMX window index out of range")
        record = self.records[position]
        guard_split(record.suggested_split)
        piece = self.reader(record, self.cache_config)
        if piece.split != self.split or piece.targets or piece.annotations:
            raise PdmxPipelineError("pdmx.cached_raw_boundary_invalid")
        if (piece.piece_id, piece.dataset_name, piece.source_group_id) != (
            record.piece_id, "pdmx", record.source_group_id,
        ):
            raise PdmxPipelineError("pdmx.cached_identity_invalid")
        graph = build_raw_graph(piece)
        return SSLRawSample(graph, graph_fingerprint(graph), "pdmx", piece.piece_id)
