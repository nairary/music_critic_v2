"""Source-independent auxiliary event targets; never encoder input features.

Vocabularies use typed canonical JSON values: false, zero and a missing label
are distinct. Build a vocabulary once on admitted TRAIN, then freeze it for
all consumers. Unseen values remain masked, with an explicit OOV report.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from fractions import Fraction
import gzip
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable, Mapping


SCHEMA = "EventTheoryEnrichment@1"
ORIGINS = frozenset({"source_native", "source_derived", "astra_individual_context", "model_individual_context",
                     "algorithmic_candidate", "unknown"})
MODEL_ANNOTATOR_SETTINGS = (("gpt-6-astra", "ultra"), ("gpt-6.1-sol", "high"))
MODEL_AUTHORED_ORIGINS = frozenset({"astra_individual_context", "model_individual_context"})


class EnrichmentError(ValueError):
    """Identity, geometry, mask or provenance does not satisfy the contract."""


def require(condition, message):
    if not condition:
        raise EnrichmentError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def fraction(value):
    require(isinstance(value, dict) and set(value) == {"num", "den"}, "exact rational required")
    require(type(value["num"]) is int and type(value["den"]) is int and value["den"] > 0,
            "invalid rational integers")
    result = Fraction(value["num"], value["den"])
    require(result.numerator == value["num"] and result.denominator == value["den"],
            "rational must be reduced")
    return result


def _field(field):
    require(isinstance(field, dict), "field must be object")
    required = {"value", "available", "origin", "confidence", "evidence", "unknown_reason"}
    require(required <= field.keys(), "field provenance incomplete")
    require(type(field["available"]) is bool, "availability must be boolean")
    require(field["origin"] in ORIGINS, "unknown origin")
    canonical(field["value"])
    confidence = field["confidence"]
    require(confidence is None or (type(confidence) in (int, float) and 0 <= confidence <= 1),
            "invalid confidence")
    if field["available"]:
        require(field["value"] is not None, "available target cannot be null")
        require(field["origin"] not in ("unknown", "algorithmic_candidate"), "candidate cannot be target")
        require(bool(field["evidence"]), "available target requires evidence")
        require(field["unknown_reason"] is None, "available target has unknown reason")
    else:
        require(field["value"] is None, "unavailable target must be null")
        require(bool(field["unknown_reason"]), "unavailable target needs reason")


def _fields(fields):
    require(isinstance(fields, dict), "fields must be object")
    for name, field in fields.items():
        require(isinstance(name, str) and bool(name), "field name required")
        _field(field)


def validate_enrichment(piece, bundle: Mapping, *, require_note_coverage=True):
    """Validate IDs and exact geometry against an already admitted raw piece."""
    require(bundle.get("schema") == SCHEMA, "unsupported enrichment schema")
    require((bundle.get("dataset_id"), bundle.get("piece_id")) == (piece.dataset_name, piece.piece_id),
            "piece identity mismatch")
    require(bundle.get("current_split") == "train" and piece.split in (None, "train"), "TRAIN required")
    notes = {note.note_id: note for note in piece.notes}
    require(len(notes) == len(piece.notes), "duplicate canonical notes")
    duration=piece.duration_qn.to_fraction()
    geometry={n.note_id:(n.onset_qn.to_fraction(),
                        n.onset_qn.to_fraction()+n.duration_qn.to_fraction()) for n in piece.notes}
    def bounded(row, label):
        left,right=fraction(row['start_qn']),fraction(row['end_qn'])
        require(0 <= left <= right <= duration, label+' outside piece or reversed interval')
        return left,right
    harmonies = {}
    ids = set(notes)
    for harmony in bundle.get("harmonies", []):
        identity = harmony.get("id")
        require(isinstance(identity, str) and identity and identity not in ids, "duplicate harmony identity")
        ids.add(identity)
        left, right = bounded(harmony,'harmony')
        require(left < right, "harmony interval must be positive")
        _fields(harmony["fields"])
        harmonies[identity] = (left, right)
        geometry[identity] = (left,right)
    spans = defaultdict(list)
    for row in bundle.get("note_contexts", []):
        identity = row.get("id")
        require(isinstance(identity, str) and identity and identity not in ids, "duplicate context identity")
        ids.add(identity)
        require(row.get("note_id") in notes, "unknown canonical note")
        note = notes[row["note_id"]]
        left, right = fraction(row["start_qn"]), fraction(row["end_qn"])
        start = note.onset_qn.to_fraction()
        end = start + note.duration_qn.to_fraction()
        require(start <= left <= right <= end, "context outside canonical note")
        require(left < right or start == end or row.get("applicability") == "source_note_annotation",
                "point context needs explicit applicability")
        hid = row.get("harmony_id")
        if hid is not None:
            require(hid in harmonies, "unknown harmony")
            hleft, hright = harmonies[hid]
            require(hleft <= left and right <= hright and (left < hright), "context outside harmony")
        _fields(row["fields"])
        spans[row["note_id"]].append((left, right))
        geometry[identity]=(left,right)
    if require_note_coverage:
        require(set(spans) == set(notes), "not all notes accounted for")
        # Alternate analytic views may overlap. Coverage is their union, not
        # an assertion that every field or every view is available everywhere.
        for identity, note in notes.items():
            cursor = note.onset_qn.to_fraction()
            for left, right in sorted(spans[identity]):
                require(left <= cursor, "unaccounted context gap")
                cursor = max(cursor, right)
            require(cursor == note.onset_qn.to_fraction() + note.duration_qn.to_fraction(),
                    "unaccounted context tail")
    for obj in bundle.get("objects", []):
        require(isinstance(obj.get("id"), str) and obj['id'] and obj["id"] not in ids, "duplicate analysis object")
        geometry[obj['id']]=bounded(obj,'object')
        ids.add(obj["id"])
    for obj in bundle.get('objects',[]):
        members=obj.get('member_ids',[])
        require(isinstance(members,list) and len(members)==len(set(members))
                and set(members) <= ids and obj['id'] not in members, 'invalid object members')
        left,right=geometry[obj['id']]
        require(all(left <= geometry[i][0] <= geometry[i][1] <= right for i in members),
                'object does not contain member scope')
    relation_ids = set()
    for relation in bundle.get("relations", []):
        identity = relation.get("id")
        require(isinstance(identity, str) and identity and identity not in ids | relation_ids,
                "duplicate relation identity")
        relation_ids.add(identity)
        require(isinstance(relation.get("type"), str) and relation["type"], "relation type required")
        for name in ("source_ids", "target_ids"):
            participants = relation.get(name)
            require(isinstance(participants, list) and participants and len(set(participants)) == len(participants),
                    "relation participants required")
            require(set(participants) <= ids, "unknown relation participant")
        left,right=bounded(relation,'relation')
        require(all(left <= geometry[i][0] <= geometry[i][1] <= right
                    for i in relation['source_ids']+relation['target_ids']),
                'relation does not contain participant scope')
        _field(relation)
    return bundle


def load_enrichment(path, *, expected_sha256=None):
    """Authenticate bytes before decoding; caller owns source/split admission."""
    path = Path(path)
    payload = path.read_bytes()
    if expected_sha256 is not None:
        require(sha256(payload).hexdigest() == expected_sha256, "enrichment hash mismatch")
    if path.suffix == ".gz":
        payload = gzip.decompress(payload)
    return json.loads(payload)


def load_bound_piece(binding):
    from music_critic.data import loads_piece
    payload = Path(binding["path"]).read_bytes()
    require(len(payload) == binding["size_bytes"] and sha256(payload).hexdigest() == binding["sha256"],
            "canonical piece hash/size mismatch")
    return loads_piece(payload.decode("utf-8"))


def merge_interpretations(piece, bundle, overlay, *, base_sha256):
    """Attach a separately authored weak layer without overwriting native facts.

    The caller supplies the digest of the authenticated base file. Overlay
    evidence is not proof of musical correctness; origin remains explicit.
    """
    validate_enrichment(piece,bundle)
    require(overlay.get("schema")=="EventTheoryInterpretations@1", "interpretation schema mismatch")
    require(overlay.get("piece_id")==piece.piece_id and overlay.get("base_sha256")==base_sha256,
            "interpretation base binding mismatch")
    require((overlay.get("model_requested"), overlay.get("reasoning_effort_requested")) in MODEL_ANNOTATOR_SETTINGS
            and overlay.get("human_gold") is False, "authored model provenance required")
    result=deepcopy(bundle)
    def authored(field):
        require(field.get("origin") in MODEL_AUTHORED_ORIGINS | {"unknown"}
                and field.get("confidence") is None, "authored weak provenance required")
        _field(field)
        if overlay["model_requested"] == "gpt-6.1-sol" and field["available"]:
            require(field["origin"] == "model_individual_context", "Sol origin must retain model provenance")
    for layer,rows_key,id_key in (("harmony_fields","harmonies","harmony_id"),
                                   ("note_fields","note_contexts","context_id")):
        rows={r["id"]:r for r in result[rows_key]}
        for update in overlay.get(layer,[]):
            require(update.get(id_key) in rows, "unknown interpretation target")
            row=rows[update[id_key]]
            for name,field in update["fields"].items():
                require(name.startswith("analysis."), "authored fields require analysis namespace")
                previous=row["fields"].get(name)
                require(previous is None or (previous["available"] is False
                        and previous["origin"]=="unknown" and previous["value"] is None),
                        "cannot overwrite prior field")
                authored(field)
                row["fields"][name]=deepcopy(field)
    for relation in overlay.get("relations",[]):
        authored(relation)
    for obj in overlay.get("objects", []):
        for field in obj.get("fields", {}).values():
            authored(field)
    result.setdefault("objects",[]).extend(deepcopy(overlay.get("objects",[])))
    result["relations"].extend(deepcopy(overlay.get("relations",[])))
    result.setdefault("provenance",{})["individual_analysis"]={
        "base_sha256":base_sha256,"model_requested":overlay["model_requested"],
        "reasoning_effort_requested":overlay["reasoning_effort_requested"],"actual_backend_version":overlay.get("actual_backend_version"),
        "human_gold":False,"review_scope":overlay.get("review_scope"),
        "annotation_id":overlay.get("annotation_id"),
    }
    validate_enrichment(piece,result)
    return result


def build_vocabulary(bundles: Iterable[Mapping]):
    """Build deterministic field vocabularies on admitted TRAIN bundles only."""
    values = {scope: defaultdict(set) for scope in ("harmonies", "note_contexts")}
    relation_types = set()
    relation_values = defaultdict(set)
    for bundle in bundles:
        require(bundle.get("schema") == SCHEMA and bundle.get("current_split") == "train", "TRAIN bundle required")
        for scope in values:
            for row in bundle.get(scope, []):
                _fields(row["fields"])
                for field, target in row["fields"].items():
                    # Keep unavailable-only fields visible with an empty vocab.
                    values[scope][field]
                    if target["available"]:
                        values[scope][field].add(canonical(target["value"]))
        for relation in bundle.get("relations", []):
            _field(relation)
            if relation["available"]:
                relation_types.add(relation["type"])
                relation_values[relation["type"]].add(canonical(relation["value"]))
    result = {"schema": "EventTheoryVocabulary@1", "unknown_value": -1,
              "fields": {scope: {name: [json.loads(v) for v in sorted(vs)]
                                  for name, vs in sorted(fields.items())}
                         for scope, fields in values.items()},
              "relation_types": sorted(relation_types),
              "relation_values": {name:[json.loads(v) for v in sorted(vs)]
                                  for name,vs in sorted(relation_values.items())}}
    result["fingerprint"] = sha256(canonical(result).encode()).hexdigest()
    return result


def _vocabulary_maps(vocabulary):
    require(vocabulary.get("schema") == "EventTheoryVocabulary@1", "vocabulary schema mismatch")
    identity = {k:v for k,v in vocabulary.items() if k != "fingerprint"}
    require(sha256(canonical(identity).encode()).hexdigest() == vocabulary.get("fingerprint"),
            "vocabulary fingerprint mismatch")
    return {scope: {name: {canonical(value):index for index,value in enumerate(vs)}
                   for name,vs in fields.items()}
            for scope,fields in vocabulary["fields"].items()}


def encode_enrichment(piece, bundle, vocabulary):
    validate_enrichment(piece, bundle)
    mappings = _vocabulary_maps(vocabulary)
    note_indices = {n.note_id:i for i,n in enumerate(piece.notes)}
    result = {"schema":"EventTheoryNumericTargets@1", "piece_id":piece.piece_id,
              "raw_event_train_eligible":bool(piece.notes),
              "vocabulary_fingerprint":vocabulary["fingerprint"], "oov_counts":{}}
    for scope, fields in mappings.items():
        rows = bundle.get(scope, [])
        encoded = {"ids":[r["id"] for r in rows],
                   "intervals":[[r["start_qn"],r["end_qn"]] for r in rows],
                   "view_ids":[r.get('view_id',r.get('harmony_view')) for r in rows],
                   "values":{}, "masks":{}}
        if scope == "note_contexts":
            encoded["note_indices"] = [note_indices[r["note_id"]] for r in rows]
            encoded["harmony_ids"] = [r.get("harmony_id") for r in rows]
            encoded["applicability"] = [r.get("applicability", "note_harmonic_context") for r in rows]
        for name, lookup in fields.items():
            vals, masks, oov = [], [], 0
            for row in rows:
                field = row["fields"].get(name)
                available = bool(field and field["available"])
                value = lookup.get(canonical(field["value"]), -1) if available else -1
                oov += int(available and value == -1)
                vals.append(value)
                masks.append(available and value != -1)
            encoded["values"][name], encoded["masks"][name] = vals, masks
            if oov:
                result["oov_counts"][scope+":"+name] = oov
        result[scope] = encoded
    types = {v:i for i,v in enumerate(vocabulary["relation_types"])}
    relations = bundle.get("relations", [])
    participants = [{"id":n.note_id,"kind":"note","canonical_note_index":i}
                    for i,n in enumerate(piece.notes)]
    for scope,kind in (("harmonies","harmony"),("note_contexts","context"),("objects","object")):
        for row in bundle.get(scope,[]):
            participant=dict(id=row['id'],kind=kind,interval=[row['start_qn'],row['end_qn']])
            if kind=='object':
                participant['object_type']=row.get('object_type',row.get('type',row.get('kind')))
                participant['member_ids']=row.get('member_ids',[])
            participants.append(participant)
    participant_indices = {obj["id"]:i for i,obj in enumerate(participants)}
    for obj in participants:
        if obj['kind']=='object':
            obj['member_indices']=[participant_indices[i] for i in obj['member_ids']]
    relation_values = {name:{canonical(v):i for i,v in enumerate(vs)}
                       for name,vs in vocabulary["relation_values"].items()}
    encoded_values = [relation_values.get(r["type"],{}).get(canonical(r["value"]),-1)
                      if r["available"] else -1 for r in relations]
    result["participants"] = participants
    result["relations"] = {
        "ids":[r["id"] for r in relations],
        "types":[types.get(r["type"],-1) if r["available"] else -1 for r in relations],
        "values":encoded_values,
        "masks":[r["available"] and r["type"] in types and v != -1
                 for r,v in zip(relations,encoded_values)],
        "source_ids":[r["source_ids"] for r in relations],
        "target_ids":[r["target_ids"] for r in relations],
        "source_indices":[[participant_indices[i] for i in r["source_ids"]] for r in relations],
        "target_indices":[[participant_indices[i] for i in r["target_ids"]] for r in relations],
        "intervals":[[r["start_qn"],r["end_qn"]] for r in relations],
    }
    oov_relations = sum(r["available"] and v == -1 for r,v in zip(relations,encoded_values))
    if oov_relations:
        result["oov_counts"]["relations"] = oov_relations
    return result
