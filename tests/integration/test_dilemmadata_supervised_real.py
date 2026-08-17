from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from music_critic.models import (
    DILEMMADATA_ACTIVE_TASK_IDS,
    DilemmadataHierarchicalConfig,
    DilemmadataHierarchicalModel,
)
from music_critic.tasks import (
    CorpusCacheConfig,
    DilemmadataTargetCacheConfig,
    IndexedMultiSourceDataset,
    collate_multisource_samples,
    load_corpus_index,
    load_dilemmadata_target_cache_index,
)


RUN_REAL = os.environ.get("MUSIC_CRITIC_RUN_REAL_DILEMMADATA_TESTS") == "1"

@pytest.mark.skipif(
    not RUN_REAL,
    reason="set MUSIC_CRITIC_RUN_REAL_DILEMMADATA_TESTS=1 with cache paths",
)
def test_real_bounded_batch_has_two_an_two_dlc_and_four_head_forward() -> None:
    raw_index_path = Path(os.environ["MUSIC_CRITIC_DILEMMADATA_RAW_INDEX"])
    raw_cache_root = Path(os.environ["MUSIC_CRITIC_DILEMMADATA_RAW_CACHE"])
    target_index_path = Path(os.environ["MUSIC_CRITIC_DILEMMADATA_TARGET_INDEX"])
    target_cache_root = Path(os.environ["MUSIC_CRITIC_DILEMMADATA_TARGET_CACHE"])
    raw_index = load_corpus_index(raw_index_path)
    target_index = load_dilemmadata_target_cache_index(target_index_path)
    dataset = IndexedMultiSourceDataset(
        raw_index,
        cache_config=CorpusCacheConfig(raw_cache_root),
        target_cache_index=target_index,
        target_cache_config=DilemmadataTargetCacheConfig(target_cache_root),
        require_target_sidecars=True,
    )
    selected = []
    counts = {"an": 0, "dlc": 0}
    for index, record in enumerate(raw_index.records):
        dialect = "an" if record.source_identity.startswith("an:") else "dlc"
        if counts[dialect] < 2:
            selected.append(dataset[index])
            counts[dialect] += 1
        if counts == {"an": 2, "dlc": 2}:
            break
    assert counts == {"an": 2, "dlc": 2}
    batch = collate_multisource_samples(tuple(selected))
    model = DilemmadataHierarchicalModel(
        DilemmadataHierarchicalConfig(
            hidden_dim=16,
            local_gnn_layers=1,
            transformer_layers=1,
            attention_heads=4,
            ffn_multiplier=2,
            dropout=0.0,
        )
    )
    output = model(batch)
    assert tuple(row.task_id for row in output.predictions) == (
        DILEMMADATA_ACTIVE_TASK_IDS
    )
    assert output.harmonic_loss.total_loss is not None
    assert torch.isfinite(output.harmonic_loss.total_loss)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable on this host"
)
def test_optional_cuda_amp_smoke_is_finite() -> None:
    from tests.models.test_dilemmadata_heads import _batch

    device = torch.device("cuda")
    from music_critic.training.device import move_multisource_batch

    batch = move_multisource_batch(_batch(), device)
    model = DilemmadataHierarchicalModel(
        DilemmadataHierarchicalConfig(
            hidden_dim=16,
            local_gnn_layers=1,
            transformer_layers=1,
            attention_heads=4,
            ffn_multiplier=2,
            dropout=0.0,
        )
    ).to(device)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        output = model(batch)
        loss = output.harmonic_loss.total_loss
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
