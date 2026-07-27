from __future__ import annotations

import ast
import inspect
import textwrap

from music_critic.models.reconstruction import reconstruction_loss
from music_critic.training.device import _validate_moved_structure
from music_critic.training.engine import _losses, _optimize_batch
from music_critic.training.metrics import EpochMetricAccumulator


def _attribute_calls(value) -> tuple[str, ...]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(value)))
    return tuple(
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
    )


def test_normal_engine_and_device_paths_have_no_tensor_host_conversion(
) -> None:
    forbidden = {"item", "tolist", "cpu", "numpy"}
    for value in (
        _validate_moved_structure,
        _losses,
        _optimize_batch,
    ):
        assert forbidden.isdisjoint(_attribute_calls(value))


def test_joint_reconstruction_has_no_per_family_host_predicate() -> None:
    calls = _attribute_calls(reconstruction_loss)
    assert "any" not in calls
    assert {"item", "tolist", "cpu", "numpy"}.isdisjoint(calls)


def test_metric_add_has_one_explicit_packed_transfer_site() -> None:
    calls = _attribute_calls(EpochMetricAccumulator.add)
    assert calls.count("tolist") == 1
    assert "item" not in calls
    assert "cpu" not in calls
