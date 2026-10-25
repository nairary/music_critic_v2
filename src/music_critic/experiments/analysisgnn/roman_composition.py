"""TRAIN-only diagnostics and decoding of the source-native simple numeral.

The lookup does not claim compositional generalization: unseen tuples abstain,
and collisions are reported. Target availability is never a negative label.
This module does not alter model heads, training loss, or checkpoint selection.
"""
from collections import Counter, defaultdict
import math


FEATURE_GROUPS = {
    "degree_quality": ("primary_degree", "quality"),
    "degree_quality_inversion": ("primary_degree", "quality", "inversion"),
    "key_degree_quality_inversion": ("local_key", "primary_degree", "quality", "inversion"),
    "degrees_quality": ("primary_degree", "secondary_degree", "quality"),
    "degrees_quality_inversion": ("primary_degree", "secondary_degree", "quality", "inversion"),
    "paper_joint": ("local_key", "primary_degree", "secondary_degree", "quality", "inversion"),
    "all_harmonic": ("local_key", "tonicized_key", "root", "bass", "primary_degree",
                     "secondary_degree", "quality", "inversion", "pitch_class_set"),
}


def harmonic_rows(record, sidecar):
    """Preserve source-scoped identity, masks and raw label evidence."""
    if record["split"] not in ("train", "validation"):
        raise ValueError("TEST is forbidden")
    if (sidecar["record_id"] != record["record_id"] or
            sidecar["source_component_id"] != record["source_component_id"]):
        raise ValueError("sidecar identity mismatch")
    seen = set()
    for entity in sidecar["entities"]:
        if entity["entity_type"] != "harmonic_event":
            continue
        identity = entity["canonical_entity_id"]
        if identity in seen:
            raise ValueError("duplicate harmonic identity")
        seen.add(identity)
        labels, unavailable, raw = {}, {}, {}
        for task, state in entity["targets"].items():
            valid = state.get("available") is True and state.get("masked") is False
            value = state.get("canonical_value")
            if valid and not isinstance(value, str):
                raise ValueError("available target without string value")
            labels[task] = value if valid else None
            if not valid:
                unavailable[task] = state.get("missing_reason")
            raw[task] = state.get("source_value")
        yield dict(record_id=record["record_id"], split=record["split"],
                   component_id=record["source_component_id"], dialect=sidecar["dialect"],
                   entity_id=identity, source_entity_id=entity["source_entity_id"],
                   onset_qn=entity.get("onset_qn"), labels=labels,
                   unavailable=unavailable, source_values=raw)


def component_key(labels, features):
    values = tuple(labels.get(task) for task in features)
    return values if all(isinstance(value, str) for value in values) else None


def fit_lookup(rows, features):
    table = defaultdict(Counter)
    for row in rows:
        if row["split"] != "train":
            raise ValueError("only TRAIN may fit a decoder")
        key = component_key(row["labels"], features)
        target = row["labels"].get("roman_numeral")
        if key is not None and target is not None:
            table[key][target] += 1
    return dict(table)


def lookup_predict(table, key):
    counts = table.get(key)
    if not counts:
        return None
    # Explicit train-frequency baseline; deterministic ties. Not an exact rule.
    return min(counts, key=lambda value: (-counts[value], value))


def score_lookup(rows, features, table):
    counts = Counter()
    for row in rows:
        target = row["labels"].get("roman_numeral")
        if target is None:
            counts["target_missing"] += 1
            continue
        counts["target_available"] += 1
        key = component_key(row["labels"], features)
        if key is None:
            counts["components_missing"] += 1
            continue
        counts["complete"] += 1
        choices = table.get(key)
        if choices is None:
            counts["unseen_tuple"] += 1
            continue
        counts["seen_tuple"] += 1
        counts["ambiguous_tuple" if len(choices) > 1 else "unique_tuple"] += 1
        counts["correct_mode"] += lookup_predict(table, key) == target
        counts["correct_unique"] += len(choices) == 1 and target in choices
        counts["target_in_candidates"] += target in choices
    result = dict(sorted(counts.items()))
    for name, numerator, denominator in (
        ("mode_accuracy_complete", "correct_mode", "complete"),
        ("mode_accuracy_seen", "correct_mode", "seen_tuple"),
        ("complete_fraction", "complete", "target_available"),
        ("unique_correct_fraction_all_targets", "correct_unique", "target_available"),
    ):
        result[name] = counts[numerator] / counts[denominator] if counts[denominator] else None
    return result


def audit(rows):
    train = [r for r in rows if r["split"] == "train"]
    validation = [r for r in rows if r["split"] == "validation"]
    if len(train) + len(validation) != len(rows):
        raise ValueError("TEST is forbidden")
    if {r["component_id"] for r in train} & {r["component_id"] for r in validation}:
        raise ValueError("component crosses split")
    groups = {}
    for name, features in FEATURE_GROUPS.items():
        table = fit_lookup(train, features)
        example_index = defaultdict(dict)
        for row in train:
            key = component_key(row["labels"], features)
            target = row["labels"].get("roman_numeral")
            if key in table and len(table[key]) > 1 and target in table[key]:
                example_index[key].setdefault(target, {k: row[k] for k in (
                    "record_id", "entity_id", "source_entity_id")})
        collisions = []
        for key, counts in table.items():
            if len(counts) > 1:
                collisions.append(dict(components=list(key), numerals=dict(counts), examples=example_index[key]))
        groups[name] = dict(features=list(features), train=score_lookup(train, features, table),
                            validation=score_lookup(validation, features, table), tuple_count=len(table),
                            collision_count=len(collisions), collisions=sorted(
                                collisions, key=lambda row: -sum(row["numerals"].values())))
    inventory = {}
    for split, subset in (("train", train), ("validation", validation)):
        classes = defaultdict(lambda: {"events": 0, "records": set(), "components": set()})
        for row in subset:
            label = row["labels"].get("roman_numeral")
            if label is not None:
                value = classes[label]
                value["events"] += 1
                value["records"].add(row["record_id"])
                value["components"].add(row["component_id"])
        inventory[split] = dict(events=len(subset), records=len({r["record_id"] for r in subset}),
                               classes={k: {n: len(v) if isinstance(v, set) else v for n, v in d.items()}
                                        for k, d in sorted(classes.items())})
    return dict(groups=groups, inventory=inventory, model_inference=False,
                interpretation="Oracle components and TRAIN lookup; not model accuracy")


def classification_metrics(targets, predictions, vocabulary):
    """Fixed-vocabulary macro F1, including unsupported classes as zero.

    None is an abstention (a false negative, no class false positive).
    """
    if len(targets) != len(predictions) or not targets:
        raise ValueError("nonempty paired predictions required")
    allowed = set(vocabulary)
    if any(t not in allowed for t in targets) or any(p is not None and p not in allowed for p in predictions):
        raise ValueError("unknown class")
    tp, predicted, support = Counter(), Counter(), Counter(targets)
    for target, pred in zip(targets, predictions):
        if pred is not None:
            predicted[pred] += 1
        if pred == target:
            tp[target] += 1
    scores = {label: (2 * tp[label] / (support[label] + predicted[label])
                      if support[label] + predicted[label] else 0.) for label in vocabulary}
    return dict(support=len(targets), accuracy=sum(tp.values()) / len(targets),
                macro_f1=sum(scores.values()) / len(vocabulary),
                macro_f1_present=sum(scores[k] for k in support) / len(support),
                abstentions=sum(p is None for p in predictions),
                per_class={k: dict(f1=scores[k], support=support[k], predicted=predicted[k], tp=tp[k])
                           for k in vocabulary})


def compatibility_scores(table, component_probabilities, vocabulary):
    """Marginalize TRAIN tuple frequencies conditional on each simple numeral.

    Returns an unnormalized compatibility score, not a joint posterior.
    All numeral classes have equal prior; classes absent from TRAIN score zero.
    No VALIDATION target, target masks, or true component enters decoding.
    """
    total, score = Counter(), Counter()
    for key, counts in table.items():
        mass = math.prod(probs.get(value, 0.) for probs, value in zip(component_probabilities, key, strict=True))
        for label, count in counts.items():
            total[label] += count
            score[label] += count * mass
    return [score[label] / total[label] if total[label] else 0. for label in vocabulary]
