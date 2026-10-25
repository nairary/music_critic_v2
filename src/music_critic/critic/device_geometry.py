"""Forward-local constants, packed before sequential learned operations.

Only raw geometry is materialized here. No learned operations are batched and
no device tensors are retained by the module or a cross-forward cache.
"""
from dataclasses import dataclass

import torch

from .geometry_plan import GeometryPlan


@dataclass(frozen=True)
class PoolConstants:
    log_counts: torch.Tensor
    one: torch.Tensor

    def scalars(self, count):
        if not 0 <= count < len(self.log_counts):
            raise ValueError('pool count exceeds geometry bound')
        return self.log_counts[count:count + 1], self.one


@dataclass(frozen=True)
class DeviceGeometry:
    _indices: dict
    _timings: dict
    _weights: dict
    pool: PoolConstants
    storages: tuple[torch.Tensor, ...]

    def indices(self, values):
        return self._indices[values]

    def timing(self, values):
        return self._timings[values]

    def weights(self, values):
        return self._weights[values]


def prepare_constants(plan: GeometryPlan, zero: torch.Tensor) -> DeviceGeometry:
    """Use a fixed number of host/device materializations, preserving row order."""
    index_values = {(): None}
    def remember(values):
        index_values.setdefault(values, None)
    remember(plan.active_tracks)
    for row in plan.notes:
        remember(row.slices)
    for row in plan.events:
        remember(row.notes)
        remember(row.slices)
    for row in plan.slices:
        for values in (row.attacks, row.holds, row.releases, row.beats, row.onsets,
                       row.bars, *row.spans, row.notes, row.events):
            remember(values)
    for row in plan.bars:
        remember(row.events)
    for row in plan.tracks:
        remember(row.notes)
        remember(row.events)

    flat_indices = []
    offsets = {}
    for values in index_values:
        start = len(flat_indices)
        flat_indices.extend(values)
        offsets[values] = (start, len(flat_indices))
    indices = torch.tensor(flat_indices, dtype=torch.long, device=zero.device)
    index_views = {key: indices[a:b] for key, (a, b) in offsets.items()}

    timings = dict.fromkeys(row.timing for row in (*plan.notes, *plan.events, *plan.slices, *plan.bars))
    weights = dict.fromkeys([*(row.weights for row in plan.bars), *plan.beat_weights])
    floats = []
    def append(values):
        start = len(floats)
        floats.extend(values)
        return start, len(floats)
    time_offsets = {key: append(key) for key in timings}
    weight_offsets = {key: append(key.dense()) for key in weights}
    values = zero.new_tensor(floats)
    time_views = {key: values[a:b] for key, (a, b) in time_offsets.items()}
    weight_views = {key: values[a:b] for key, (a, b) in weight_offsets.items()}

    # This upper bound also covers every concatenated pool (song, notes/events).
    maximum = (len(plan.notes) + len(plan.events) + len(plan.slices)
               + len(plan.bars) + len(plan.tracks) + 2)
    # Overlapping beat/onset parents may outnumber the graph rows above.
    maximum = max(maximum, max(map(len, index_values)))
    # Convert integer counts to the same dtype before the original torch log1p.
    counts = zero.new_tensor(range(maximum + 1)).log1p()
    one = zero.new_ones(1)
    return DeviceGeometry(index_views, time_views, weight_views,
                          PoolConstants(counts, one), (indices, values, counts, one))
