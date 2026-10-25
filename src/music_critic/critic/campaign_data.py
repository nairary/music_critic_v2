"""Explicit TRAIN-source preparation for weak intervention critic campaigns.

Roles belong to generation-side provenance, never to predictor features. The
pilot policy assigns notated parts; it does not establish expert melody/bass.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, is_dataclass, replace
from fractions import Fraction
from hashlib import sha256
import json
import math
from pathlib import Path, PurePosixPath
import tarfile
import time
from typing import Mapping

from music_critic.data import CanonicalPiece, ProvenanceRecord, QualityFlag, dumps_piece, loads_piece, validate_or_raise
from music_critic.experiments.genre_interventions import build_genre_interventions, verify_genre_variant
from music_critic.experiments.music_interventions import Intervention, apply_interventions
from .contracts import ASPECTS
from .targets import InterventionTargetBuilder, PairTargets, VerifiedPatch

PILOT_ROLE_POLICY = "source-staff-upper-part-pilot-v1"
SOURCES_SCHEMA = "CriticCampaignSources@1.0.0"


@dataclass(frozen=True)
class CampaignSource:
    source_id: str
    work_id: str
    component_id: str
    piece: CanonicalPiece
    roles: Mapping[str, frozenset[str]]
    role_provenance: Mapping[str, object]
    genre: str = "classical"
    source_split: str = "train"


@dataclass(frozen=True)
class CampaignDataConfig:
    sources_manifest: str = ""
    sources_manifest_sha256: str = ""
    seed: int = 17
    head_dev_fraction: float = 0.2
    max_sources: int = 400
    max_bars: int = 16
    allow_development_sources: bool = False

    def __post_init__(self):
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("nonnegative campaign seed required")
        if not math.isfinite(self.head_dev_fraction) or not 0 < self.head_dev_fraction < 1:
            raise ValueError("head_dev_fraction must lie in (0,1)")
        if any(type(v) is not int or v < 1 for v in (self.max_sources, self.max_bars)) or self.max_bars > 16:
            raise ValueError("positive source cap and at most 16 bars required")
        if type(self.allow_development_sources) is not bool:
            raise ValueError("development source policy must be boolean")


@dataclass(frozen=True)
class RawCampaignPair:
    pair_id: str
    work_id: str
    component_id: str
    split: str
    original: CanonicalPiece
    variant: CanonicalPiece
    targets: PairTargets
    operator: str
    source_id: str


@dataclass(frozen=True)
class CampaignDataset:
    pairs: tuple[RawCampaignPair, ...]
    split_assignments: Mapping[str, str]
    omissions: tuple[Mapping[str, object], ...]
    source_summary: Mapping[str, object]
    fingerprint: str


def build_campaign_dataset(sources, config=CampaignDataConfig()):
    config = _config(config)
    sources = tuple(sorted(sources, key=lambda s: s.source_id))
    if not sources or len(sources) > config.max_sources or len({s.source_id for s in sources}) != len(sources):
        raise ValueError("campaign sources empty, duplicated or exceed explicit cap")
    for source in sources:
        _validate_source(source, config)
    assignments, components = _split_sources(sources, config)
    # Membership is now fixed. No generator or materialized variant was consulted.
    pairs, omissions = [], []
    for source in sources:
        original = replace(source.piece, annotations=(), targets=(), quality_flags=())
        roles = {key: set(values) for key, values in source.roles.items()}
        candidates, omitted = build_genre_interventions(original, roles, source.genre)
        omissions.extend({"source_id": source.source_id, **row} for row in omitted)
        for candidate in sorted(candidates, key=lambda row: row["id"]):
            if candidate["aspect"] not in ASPECTS:
                omissions.append({"source_id": source.source_id, "id": candidate["id"],
                                  "aspect": candidate["aspect"], "reason": "diagnostic_only_not_training"})
                continue
            if candidate["aspect"] == "melody" and not _monophonic_melody_support(original, roles, candidate):
                omissions.append({"source_id": source.source_id, "id": candidate["id"], "aspect": "melody",
                                  "reason": "designated_melody_not_monophonic_in_operator_support"})
                continue
            edited = apply_interventions(original, (Intervention(candidate["id"], candidate["patches"], candidate["context"]),)).piece
            report = verify_genre_variant(original, edited, candidate, roles)
            check = candidate["check"]
            boundaries = ()
            if check["kind"] == "bar_order_swap":
                boundaries = tuple(Fraction(*value) for value in (check["window"][0], check["midpoint_qn"], check["window"][1]))
            proof = VerifiedPatch(original.piece_id, edited.piece_id, source.work_id,
                candidate["patches"], (candidate["aspect"],), check["kind"],
                support_ids=tuple(check["support_ids"]), swap_boundaries=boundaries,
                provenance={"operator_verification": report, "generation_roles": dict(source.role_provenance),
                            "source_id": source.source_id, "source_split": source.source_split,
                            "campaign_generation": SOURCES_SCHEMA})
            targets = InterventionTargetBuilder().build(original, edited, proof)
            pair_id = "critic-pair:" + _fingerprint([source.source_id, candidate["id"], original.piece_id, edited.piece_id])
            pairs.append(RawCampaignPair(pair_id, source.work_id, components[source.source_id],
                assignments[source.source_id], original, edited, targets, candidate["id"], source.source_id))
    by_split = {}
    for split in ("train", "head_dev"):
        selected = [pair for pair in pairs if pair.split == split]
        by_split[split] = {"pairs": len(selected), "works": len({p.work_id for p in selected}),
            "components": len({p.component_id for p in selected}),
            "aspects": {aspect: sum(p.targets.pair_mask[i] for p in selected) for i, aspect in enumerate(ASPECTS)}}
    summary = {"sources": len(sources), "splits": by_split, "source_splits": dict(Counter(s.source_split for s in sources)),
               "role_policies": dict(Counter(str(s.role_provenance.get("policy", "explicit_descriptor")) for s in sources)),
               "claim": "weak synthetic intervention ranking; no human quality preference labels"}
    identity = {"schema": SOURCES_SCHEMA, "config": {k: v for k, v in asdict(config).items() if k != "sources_manifest"},
        "sources": [{"id": s.source_id, "work": s.work_id, "component": components[s.source_id],
                     "split": assignments[s.source_id], "raw": observation_fingerprint(s.piece),
                     "roles": s.roles, "role_provenance": s.role_provenance} for s in sources],
        "pairs": [{"pair_id": p.pair_id, "targets": p.targets} for p in pairs], "omissions": omissions}
    return CampaignDataset(tuple(pairs), assignments, tuple(omissions), summary, _fingerprint(identity))


def _monophonic_melody_support(piece, roles, candidate):
    notes = {n.note_id: n for n in piece.notes}
    support = [notes[p.note_id] for p in candidate["patches"]]
    lo = min(n.onset_qn for n in support)
    hi = max(n.onset_qn + n.duration_qn for n in support)
    if "window" in candidate["check"]:
        lo, hi = (Fraction(*value) for value in candidate["check"]["window"])
        # Canonical rational timing compares via exact fractions below.
    else:
        lo, hi = Fraction(lo.num, lo.den), Fraction(hi.num, hi.den)
    if any(n.note_id in roles["melody"] and n.is_grace
           and lo <= Fraction(n.onset_qn.num, n.onset_qn.den) < hi for n in piece.notes):
        return False
    intervals = sorted((Fraction(n.onset_qn.num, n.onset_qn.den),
                        Fraction(n.onset_qn.num, n.onset_qn.den) + Fraction(n.duration_qn.num, n.duration_qn.den))
                       for n in piece.notes if n.note_id in roles["melody"] and n.duration_qn.num > 0)
    intervals = [(start, end) for start, end in intervals if start < hi and end > lo]
    return all(left[1] <= right[0] for left, right in zip(intervals, intervals[1:]))


def inspect_manifest(config):
    """Verify only the explicit metadata manifest, without opening raw pieces."""
    config, payload = _read_manifest(config)
    return {"schema": SOURCES_SCHEMA, "sources_manifest_sha256": config.sources_manifest_sha256,
            "sources": len(payload["sources"]), "source_splits": dict(Counter(r["source_split"] for r in payload["sources"])),
            "piece_payloads_accessed": False}


def load_campaign_dataset(config):
    config, payload = _read_manifest(config)
    sources = []
    for row in payload["sources"]:
        binding = row["piece"]
        path = Path(binding["path"])
        if not path.is_absolute():
            path = Path(config.sources_manifest).parent / path
        raw = path.read_bytes()
        if len(raw) != binding["size_bytes"] or sha256(raw).hexdigest() != binding["sha256"]:
            raise ValueError("campaign piece integrity mismatch")
        piece = loads_piece(raw.decode("utf-8"))
        sources.append(CampaignSource(row["source_id"], row["work_id"], row["component_id"], piece,
            {key: frozenset(values) for key, values in row["roles"].items()}, row["role_provenance"],
            row["genre"], row["source_split"]))
    return build_campaign_dataset(sources, config)


def _read_manifest(config):
    config = _config(config)
    raw = Path(config.sources_manifest).read_bytes()
    if sha256(raw).hexdigest() != config.sources_manifest_sha256:
        raise ValueError("campaign source manifest integrity mismatch")
    payload = json.loads(raw)
    if payload.get("schema") != SOURCES_SCHEMA or not isinstance(payload.get("sources"), list):
        raise ValueError("campaign source manifest schema mismatch")
    sources = payload["sources"]
    if not sources or len(sources) > config.max_sources or len({r["source_id"] for r in sources}) != len(sources):
        raise ValueError("campaign manifest source count invalid")
    permitted = {"train", "development"} if config.allow_development_sources else {"train"}
    # Every split is checked before any piece path is opened, including later rows.
    if any(row["source_split"] not in permitted for row in sources):
        raise ValueError("campaign source split is not permitted TRAIN")
    return config, payload


def _config(config):
    return CampaignDataConfig(**config) if isinstance(config, Mapping) else config


def _plain(value):
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, Fraction):
        return [value.numerator, value.denominator]
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(_plain(v) for v in value)
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _fingerprint(value):
    return sha256(json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def observation_fingerprint(piece):
    """Identify duplicate raw observations even when piece/note IDs are aliases."""
    # Canonical ordering uses IDs to break musical ties. Sort the observations
    # themselves and conservatively group track permutations as the same source.
    note_groups = defaultdict(list)
    for n in piece.notes:
        note_groups[n.track_id].append((n.pitch, n.onset_qn, n.duration_qn,
            n.velocity, n.channel, n.program, n.is_percussion, n.is_grace))
    tracks = [(t.program, t.channel, t.is_percussion,
               sorted(note_groups[t.track_id], key=_fingerprint)) for t in piece.tracks]
    return _fingerprint({"tracks": sorted(tracks, key=_fingerprint),
        "bars": [(b.start_qn, b.duration_qn, b.metric_offset_qn, b.is_pickup, b.is_incomplete) for b in piece.bars],
        "beats": [(b.start_qn, b.duration_qn, b.position_in_bar_qn, b.is_downbeat, b.strength) for b in piece.beats],
        "meter": [(m.onset_qn, m.numerator, m.denominator) for m in piece.meter_events],
        "tempo": [(t.onset_qn, t.microseconds_per_quarter) for t in piece.tempo_events],
        "key": [(k.onset_qn, k.fifths, k.mode) for k in piece.key_signature_events], "duration": piece.duration_qn})


def _validate_source(source, config):
    if any(not isinstance(v, str) or not v.strip() for v in (source.source_id, source.work_id, source.component_id)):
        raise ValueError("known source/work/component identity required")
    permitted = {"train", "development"} if config.allow_development_sources else {"train"}
    if source.source_split not in permitted or source.piece.split != "train":
        raise ValueError("campaign source split is not permitted TRAIN")
    if len(source.piece.bars) > config.max_bars:
        raise ValueError("source clip exceeds bar budget")
    if any(source.roles.values()) and (not source.role_provenance.get("basis") or not source.role_provenance.get("confidence")):
        raise ValueError("role eligibility requires explicit basis and confidence provenance")
    validate_or_raise(source.piece)


def _split_sources(sources, config):
    parents = list(range(len(sources)))
    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    seen = {}
    for i, source in enumerate(sources):
        for key in (("work", source.work_id), ("component", source.component_id),
                    ("source_group", source.piece.source_group_id),
                    ("piece", source.piece.piece_id), ("raw", observation_fingerprint(source.piece))):
            if key in seen:
                parents[root(i)] = root(seen[key])
            seen[key] = i
    groups = defaultdict(list)
    for i, source in enumerate(sources):
        groups[root(i)].append(source)
    units = [("critic-component:" + _fingerprint(sorted(s.component_id for s in group)), group)
             for group in groups.values()]
    if len(units) < 2:
        raise ValueError("at least two disjoint work/components required before variants")
    units.sort(key=lambda row: _fingerprint([config.seed, row[0]]))
    dev_count = max(1, min(len(units) - 1, round(len(units) * config.head_dev_fraction)))
    assignments, components = {}, {}
    for i, (component, group) in enumerate(units):
        for source in group:
            assignments[source.source_id] = "head_dev" if i < dev_count else "train"
            components[source.source_id] = component
    return assignments, components


def pilot_roles(piece: CanonicalPiece):
    validate_or_raise(piece)
    pitched = [n for n in piece.notes if not n.is_percussion]
    if not pitched or any(n.staff is None or n.voice is None for n in pitched):
        raise ValueError("pilot roles require observed staff and voice on every pitched note")
    melody = {n.note_id for n in pitched if n.staff == 1 and n.voice == 1}
    accompaniment = {n.note_id for n in pitched} - melody
    if len({n.staff for n in pitched}) < 2 or not melody or not accompaniment:
        raise ValueError("pilot roles require distinct source staff upper part and accompaniment")
    lowest_staff = max(n.staff for n in pitched if n.note_id in accompaniment)
    groups = defaultdict(list)
    for note in pitched:
        if note.staff == lowest_staff and note.note_id in accompaniment:
            groups[note.onset_qn].append(note)
    bass = {n.note_id for notes in groups.values() for n in notes if n.pitch == min(v.pitch for v in notes)}
    roles = {"melody": frozenset(melody), "accompaniment": frozenset(accompaniment),
             "bass": frozenset(bass), "drums": frozenset(n.note_id for n in piece.notes if n.is_percussion)}
    provenance = {"policy": PILOT_ROLE_POLICY, "gold_roles": False,
        "basis": "observed staff1/voice1 designated upper part; other pitched notes accompaniment; lowest attack on largest staff bass",
        "confidence": "source staff/voice observed; musical roles are generation-only pilot inference",
        "bass_staff": lowest_staff,
        "melody_monophony": "not assumed; octave-contour operator verifies four isolated consecutive attacks"}
    return roles, provenance


def make_pilot_clip(piece: CanonicalPiece, *, max_bars=16, max_notes=1024, window_limit=8):
    from music_critic.pdmx_pipeline.windowing import WindowConfig, iter_bar_windows
    if piece.split != "train":
        raise ValueError("pilot clip requires explicit TRAIN source")
    if type(max_bars) is not int or not 2 <= max_bars <= 16 or type(window_limit) is not int or window_limit < 1:
        raise ValueError("invalid pilot clip budget")
    raw = replace(piece, annotations=(), targets=(), quality_flags=())
    config = WindowConfig(bar_count=max_bars, stride=max_bars, max_notes_per_window=max_notes)
    reasons = []
    for index, window in enumerate(iter_bar_windows(raw, config)):
        if index >= window_limit:
            break
        if window.piece is None:
            reasons.append(window.reason)
            continue
        if sum(not b.is_incomplete and not b.is_pickup for b in window.piece.bars) < 2:
            reasons.append("fewer_than_two_complete_bars")
            continue
        try:
            roles, provenance = pilot_roles(window.piece)
        except ValueError as exc:
            reasons.append(str(exc))
            continue
        if len(roles["melody"]) < 4 or len(roles["accompaniment"]) < 4:
            reasons.append("fewer_than_four_role_notes")
            continue
        return window.piece, roles, provenance, {
            "source_piece_id": piece.piece_id, "source_bar_start": window.source_bar_start,
            "source_bar_stop": window.source_bar_stop, "global_offset_qn": _plain(window.global_offset_qn),
            "clipped_note_count": window.clipped_note_count, "window_policy": "exact-source-bar-window-v1",
            "earlier_window_refusals": reasons}
    raise ValueError("no eligible pilot clip: " + "; ".join(reasons))


def prepare_frozen_train_sources(manifest_binding, receipt_binding, output, *, max_sources=400,
                                 max_inputs=1000, seed=17, max_bars=16, max_notes=1024):
    """Stream only bound DLC TRAIN piece members; never read sidecar/held-out payloads.

    The caller registers the experiment before real use. Every source is admitted
    from source-only staff/window evidence before any intervention is generated.
    All read attempts and refusals are retained. Existing output is never reused.
    """
    if type(max_inputs) is not int or max_inputs < max_sources or max_inputs > 1000:
        raise ValueError("at most 1000 raw inputs and sources <= inputs required")
    CampaignDataConfig(seed=seed, max_sources=max_sources, max_bars=max_bars)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"status": "running", "manifest_binding": dict(manifest_binding), "archive_receipt_binding": dict(receipt_binding),
        "train_piece_payloads_read": 0, "validation_payloads_read": 0, "test_payloads_read": 0,
        "target_sidecar_payloads_read": 0, "refusals": [], "source_access": [], "verified_archives": [],
        "selection_policy": "seeded_component_rank_then_source_only_staff_window_eligibility",
        "role_policy": PILOT_ROLE_POLICY, "max_sources": max_sources, "max_inputs": max_inputs,
        "max_bars": max_bars, "max_notes": max_notes, "seed": seed}
    try:
        manifest = json.loads(_bound_bytes(manifest_binding))
        receipt = json.loads(_bound_bytes(receipt_binding))
        if receipt.get("test_payload_packaged") is not False:
            raise ValueError("archive receipt must explicitly exclude TEST payloads")
        by_component = defaultdict(list)
        for row in manifest["records"]:
            if row["split"] == "train" and row["record_id"].startswith("dlc:"):
                by_component[row["source_component_id"]].append(row)
        candidates = [min(rows, key=lambda r: _fingerprint([seed, r["record_id"]])) for rows in by_component.values()]
        candidates.sort(key=lambda r: _fingerprint([seed, r["source_component_id"]]))
        candidates = candidates[:max_inputs]
        expected = {}
        for row in candidates:
            name = str(PurePosixPath(row["piece"]["path"]).relative_to(PurePosixPath(receipt["destination"])))
            _safe_member_name(name)
            if name in expected:
                raise ValueError("duplicate bound source member")
            expected[name] = row
        eligible, seen = [], set()
        (output / "pieces").mkdir()
        for part in receipt["parts"]:
            _check_deadline(started)
            _safe_member_name(part["path"])
            archive_path = Path(receipt_binding["path"]).parent / part["path"]
            _verify_file(archive_path, part, started=started)
            _check_deadline(started)
            report["verified_archives"].append(dict(part))
            with tarfile.open(archive_path, "r|gz") as archive:
                for member in archive:
                    _check_deadline(started)
                    _safe_member_name(member.name)
                    if not (member.isfile() or member.isdir()):
                        raise ValueError("nonregular archive member refused")
                    if member.name not in expected:
                        continue
                    if not member.isfile() or member.name in seen:
                        raise ValueError("duplicate or nonregular selected member")
                    row = expected[member.name]
                    seen.add(member.name)
                    raw = archive.extractfile(member).read()
                    report["train_piece_payloads_read"] += 1
                    access = {"record_id": row["record_id"], "split": "train",
                        "source_component_id": row["source_component_id"], "member": member.name,
                        "archive": part["path"], "sha256": sha256(raw).hexdigest(), "size_bytes": len(raw)}
                    report["source_access"].append(access)
                    _check_deadline(started)
                    if len(raw) != row["piece"]["size_bytes"] or sha256(raw).hexdigest() != row["piece"]["sha256"]:
                        raise ValueError("TRAIN source payload integrity mismatch")
                    source = loads_piece(raw.decode("utf-8"))
                    access["observed_embedded_split"] = source.split
                    if source.split not in (None, "train"):
                        if source.split == "validation":
                            report["validation_payloads_read"] += 1
                        if source.split == "test":
                            report["test_payloads_read"] += 1
                        raise ValueError("embedded source split contradicts authenticated TRAIN assignment")
                    source, admission = _admit_bound_train_source(source, row, manifest_binding["sha256"])
                    access["assigned_split"] = "train"
                    try:
                        clip, roles, role_provenance, window_receipt = make_pilot_clip(source, max_bars=max_bars, max_notes=max_notes)
                    except ValueError as exc:
                        report["refusals"].append({"record_id": row["record_id"], "reason": str(exc)})
                        _check_deadline(started)
                        continue
                    _check_deadline(started)
                    admission = replace(admission, parents=tuple(sorted(p.provenance_id for p in clip.provenance)))
                    clip = replace(clip, provenance=(*clip.provenance, admission), quality_flags=tuple(sorted((*clip.quality_flags,
                        QualityFlag("critic.frozen_train_admission", "info", "TRAIN assigned from authenticated frozen manifest; embedded source split retained in provenance.",
                                    (clip.piece_id,), admission.provenance_id)), key=lambda f: (f.code, f.entity_ids, f.message))))
                    validate_or_raise(clip)
                    payload = dumps_piece(clip).encode()
                    relative = "pieces/" + _fingerprint(row["record_id"]) + ".json"
                    with (output / relative).open("xb") as stream:
                        stream.write(payload)
                    eligible.append({"source_id": row["record_id"], "work_id": row["record_id"],
                        "component_id": row["source_component_id"], "source_split": "train", "genre": "classical",
                        "piece": {"path": relative, "sha256": sha256(payload).hexdigest(), "size_bytes": len(payload)},
                        "roles": {k: sorted(v) for k, v in roles.items()}, "role_provenance": role_provenance,
                        "raw_source_binding": row["piece"], "window": window_receipt})
                    if report["train_piece_payloads_read"] % 25 == 0:
                        print(f"TRAIN source payloads {report['train_piece_payloads_read']}; eligible clips {len(eligible)}", flush=True)
                    _check_deadline(started)
            if len(seen) == len(expected):
                break
        if len(seen) != len(expected):
            raise ValueError("selected TRAIN members missing from authenticated archives")
        eligible.sort(key=lambda row: _fingerprint([seed, row["component_id"]]))
        selected = eligible[:max_sources]
        if len(selected) < 2:
            raise ValueError("fewer than two eligible TRAIN source components")
        manifest_path = output / "sources.json"
        _write_json_once(manifest_path, {"schema": SOURCES_SCHEMA, "sources": selected})
        config = asdict(CampaignDataConfig(sources_manifest=str(manifest_path.resolve()),
            sources_manifest_sha256=sha256(manifest_path.read_bytes()).hexdigest(), seed=seed, max_sources=max_sources, max_bars=max_bars))
        _write_json_once(output / "dataset-config.json", config)
        report.update(status="completed", selected_sources=len(selected), eligible_sources=len(eligible),
            selected_source_ids=[r["source_id"] for r in selected], selected_source_manifest=config["sources_manifest_sha256"],
            source_collections=dict(Counter(r["source_id"].split(":")[1] for r in selected)))
        return config
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        _write_json_once(output / "prepare-receipt.json", report)


def _safe_member_name(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError("unsafe archive member name")


def _admit_bound_train_source(source, row, manifest_sha256):
    if row["split"] != "train" or source.split not in (None, "train"):
        raise ValueError("embedded source split contradicts authenticated TRAIN assignment")
    if source.source_group_id != row["source_component_id"]:
        raise ValueError("canonical source component disagrees with authenticated manifest")
    provenance = ProvenanceRecord(
        provenance_id="prov:critic-train-" + _fingerprint([manifest_sha256, row["record_id"]]),
        kind="derivation", source="music_critic.critic.frozen_train_admission", record_id=row["record_id"],
        uri=None, version="1.0.0", checksum_sha256=row["piece"]["sha256"], created_at=None, parents=(),
        details=tuple(sorted((("assigned_split", "train"), ("observed_embedded_split", source.split),
                 ("frozen_manifest_sha256", manifest_sha256), ("source_component_id", row["source_component_id"])))))
    return replace(source, split="train"), provenance


def _bound_bytes(binding):
    raw = Path(binding["path"]).read_bytes()
    if len(raw) != binding["size_bytes"] or sha256(raw).hexdigest() != binding["sha256"]:
        raise ValueError("source metadata integrity mismatch")
    return raw


def _verify_file(path, binding, *, started):
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 << 20):
            _check_deadline(started)
            digest.update(block)
    if path.stat().st_size != binding["size_bytes"] or digest.hexdigest() != binding["sha256"]:
        raise ValueError("archive integrity mismatch")


def _check_deadline(started):
    if time.monotonic() - started > 7200:
        raise RuntimeError("two-hour source preparation budget exhausted")


def _write_json_once(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(_plain(value), stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
