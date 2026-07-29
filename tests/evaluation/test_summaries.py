from __future__ import annotations

from music_critic.evaluation.contracts import metric_value
from music_critic.evaluation.engine import build_macro_summaries


def _task(kind: str, macro_f1: dict[str, object]) -> dict[str, object]:
    names = (
        (
            "top1_accuracy",
            "top3_accuracy",
            "balanced_accuracy",
            "macro_f1",
            "micro_f1",
        )
        if kind == "closed_categorical"
        else (
            "micro_precision",
            "micro_recall",
            "micro_f1",
            "macro_precision",
            "macro_recall",
            "macro_f1",
        )
    )
    return {
        "model": {
            "kind": kind,
            **{
                name: (
                    macro_f1 if name == "macro_f1" else metric_value(0.5)
                )
                for name in names
            },
        }
    }


def test_macro_summaries_are_unweighted_with_explicit_undefined_counts() -> None:
    undefined = metric_value(
        None,
        category="no_defined_class_f1",
        reason="fixture task has no defined class F1",
    )
    summaries = build_macro_summaries(
        {
            "hooktheory": {
                "theory.cat.zero": _task(
                    "closed_categorical", metric_value(0.0)
                ),
                "theory.cat.one": _task(
                    "closed_categorical", metric_value(1.0)
                ),
                "theory.cat.undefined": _task(
                    "closed_categorical", undefined
                ),
                "theory.multi": _task(
                    "closed_multilabel", metric_value(0.75)
                ),
            },
            "pop909_cl": {
                "pop909_cl.cat": _task(
                    "closed_categorical", metric_value(0.25)
                )
            },
        }
    )
    groups = {
        (group["dataset_id"], group["encoding_kind"]): group
        for group in summaries["groups"]
    }
    hook_categorical = groups[
        ("hooktheory", "closed_categorical_index")
    ]["metrics"]["macro_f1"]

    assert hook_categorical["value"] == 0.5
    assert hook_categorical["included_task_ids"] == [
        "theory.cat.one",
        "theory.cat.zero",
    ]
    assert hook_categorical["undefined_task_ids"] == [
        "theory.cat.undefined"
    ]
    assert hook_categorical["defined_task_count"] == 2
    assert hook_categorical["undefined_task_count"] == 1
    assert "unweighted arithmetic mean" in hook_categorical[
        "aggregation_rule"
    ]


def test_macro_summaries_never_cross_dataset_or_encoding_kind() -> None:
    summaries = build_macro_summaries(
        {
            "hooktheory": {
                "theory.cat": _task(
                    "closed_categorical", metric_value(0.0)
                ),
                "theory.multi": _task(
                    "closed_multilabel", metric_value(1.0)
                ),
            },
            "pop909_cl": {
                "pop909_cl.cat": _task(
                    "closed_categorical", metric_value(0.5)
                )
            },
        }
    )
    groups = summaries["groups"]

    assert {
        (group["dataset_id"], group["encoding_kind"])
        for group in groups
    } == {
        ("hooktheory", "closed_categorical_index"),
        ("hooktheory", "closed_multilabel"),
        ("pop909_cl", "closed_categorical_index"),
    }
    assert summaries["cross_dataset_aggregation"] is False
    assert summaries["cross_encoding_aggregation"] is False
    for group in groups:
        assert "nll" not in group["metrics"]
        assert "bce_nll" not in group["metrics"]
        assert all(
            item["category"] == "scientifically_incomparable"
            for item in group["omitted_metrics"].values()
        )
