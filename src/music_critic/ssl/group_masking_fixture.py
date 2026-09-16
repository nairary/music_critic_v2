"""Generated TRAIN-only engineering fixture; never a real-data factory."""
from __future__ import annotations

import torch

from music_critic.experiments.analysisgnn.contracts import fingerprint
from music_critic.experiments.analysisgnn.corrected_training import ACTIVE_HEADS, FrozenClassWeights
from music_critic.experiments.analysisgnn.multitask_contract import TASK_BY_ID, get_vocabulary
from music_critic.ssl.bounded_fixture import build_phase7a_bounded_fixture
from music_critic.ssl.data import collate_ssl_samples
from music_critic.ssl.group_masking_pilot import PilotBatch


def build_group_pilot_fixture():
    samples = build_phase7a_bounded_fixture().raw_samples("train")[:2]
    batch = collate_ssl_samples(samples)
    sizes = {task: TASK_BY_ID[task].class_count for task in ACTIVE_HEADS}
    weights = FrozenClassWeights(
        {task: torch.ones(size) for task, size in sizes.items()},
        {task: torch.ones(size, dtype=torch.bool) for task, size in sizes.items()},
        fingerprint({"provenance": "synthetic", "uniform_class_counts": sizes}))
    sidecars = []
    for sample in samples:
        graph = sample.raw_graph
        numerator, denominator = map(int, graph["onset"].entity_id[0].removeprefix("onset:").split("_"))
        entities = []
        ids = {"harmonic_event": "harmonic-event:group-fixture", "onset": "onset-sidecar:group-fixture",
               "note": graph["note"].entity_id[0]}
        for level, entity_id in ids.items():
            targets = {}
            for task in ACTIVE_HEADS:
                if TASK_BY_ID[task].prediction_level == level:
                    label = get_vocabulary(TASK_BY_ID[task].vocabulary_id).labels[1]
                    targets[task] = {"available": True, "masked": False, "canonical_value": label,
                                     "source_value": label, "missing_reason": None,
                                     "provenance": {"origin": "synthetic"}}
            entities.append({"entity_type": level, "canonical_entity_id": entity_id,
                "onset_qn": {"num": numerator, "den": denominator}, "targets": targets})
        sidecar = {"schema_version": "analysisgnn-source-native-target-sidecar-v1",
                   "record_id": "synthetic:" + sample.piece_id,
                   "source_component_id": "synthetic:" + sample.piece_id, "dialect": "dlc",
                   "entities": entities, "relations": [{"relation": "harmonic_event_to_beat",
                       "source_entity_id": ids["harmonic_event"], "target_entity_id": graph["beat"].entity_id[0]}],
                   "provenance": "synthetic"}
        sidecar["fingerprint"] = fingerprint(sidecar)
        sidecars.append(sidecar)
    return PilotBatch(batch, batch, tuple(sidecars), weights)
