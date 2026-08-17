# Dilemmadata Phase 9B.2A Target Sidecars

Phase 9B.2A adds production, source-native theory supervision for the raw-only
Dilemmadata corpus path. It does not add theory heads, losses, supervised
training, or a scientific result.

## Boundary

`DilemmadataAdapter@1.0.1` remains the only producer of model input. Its
`CanonicalPiece` has empty `annotations` and `targets`; the accepted raw cache,
raw graph, graph fingerprint, model-input fingerprint, source grouping, and
split manifest remain target-independent.

The target adapter receives the accepted raw result and reads only the
Phase 9A-evidenced theory, validity-gate, `alt_label`, and analyst/reviewer
metadata fields. It emits `TargetBundle@1.0.0` outside the canonical cache.
Attaching that bundle to an `IndexedMultiSourceDataset` sample reuses the exact
already-bound raw graph. Target spans are supplied only to the alignment and
loss/evaluation path.

Phase 9B.1 did not retain enough information to prove which source rows were
merged into each canonical tied note. The narrow blocking addition is
`DilemmadataRawTargetAlignmentEvidence@1.0.0`: a target-neutral seal over every
row ordinal, exact rational onset, tie-continuation state, and canonical note
ID. It changes neither raw semantics nor any raw adapter/cache/graph version.

## Contracts

| Contract | Version |
|---|---:|
| Dilemmadata target adapter | `1.0.0` |
| external target sidecar | `1.0.0` |
| generic `TargetBundle` | `1.0.0` |
| raw-to-target alignment evidence | `1.0.0` |
| target-only analyst metadata index | `1.0.0` |
| source-native family registry | `1.0.0` |
| source-native encoding registry | `1.0.0` |
| exact alignment rules | `1.0.0` |
| target audit report / manifest | `1.0.0` / `1.0.0` |

The Dilemmadata registry is an explicit extension to, not a mutation of, the
18-task HookTheory/POP909-CL core registry. Therefore the accepted Phase 9B.1
cache/index and split fingerprints do not change. A sample with a Dilemmadata
sidecar exposes all 40 availability rows: the 18 core tasks plus the complete
22-task extension. Families belonging to the other Dilemmadata dialect are
present as absent families, not fabricated labels.

## Source-native inventory

`closed` below means a frozen full-corpus vocabulary and stable categorical
index. `PU` means positive-unlabeled event supervision: absence is not a
negative class. `open CPU` preserves a source string and is deferred from model
heads because no safe vocabulary or crosswalk exists.

| Task | Encoding | Exact source coordinate |
|---|---|---|
| `dilemmadata.an.key.local` | open CPU | annotation run |
| `dilemmadata.an.chord.boundary` | closed PU (`present`) | exact onset point |
| `dilemmadata.an.harmony.roman_numeral` | open CPU | AN annotation run |
| `dilemmadata.an.chord.root` | open CPU, crosswalk deferred | AN annotation run |
| `dilemmadata.an.chord.quality` | closed, 64 classes | AN annotation run |
| `dilemmadata.an.chord.bass` | open CPU, crosswalk deferred | AN annotation run |
| `dilemmadata.an.chord.inversion` | closed, 4 classes | AN annotation run |
| `dilemmadata.an.harmony.applied` | open CPU | AN annotation run |
| `dilemmadata.an.note.scale_degree` | open CPU | canonical note identity |
| `dilemmadata.dlc.key.global` | open CPU | exact full-piece span |
| `dilemmadata.dlc.key.local` | open CPU | annotation run |
| `dilemmadata.dlc.chord.boundary` | closed PU (`present`) | exact onset point |
| `dilemmadata.dlc.harmony.roman_numeral` | open CPU | DLC harmony run |
| `dilemmadata.dlc.chord.root` | open CPU, crosswalk deferred | DLC harmony run |
| `dilemmadata.dlc.chord.quality` | closed, 15 classes | DLC harmony run |
| `dilemmadata.dlc.chord.bass` | open CPU, crosswalk deferred | DLC harmony run |
| `dilemmadata.dlc.chord.inversion` | closed, 6 classes | DLC harmony run |
| `dilemmadata.dlc.harmony.applied` | open CPU | DLC harmony run |
| `dilemmadata.dlc.cadence` | closed PU, 6 positive classes | exact onset point |
| `dilemmadata.dlc.phrase.boundary` | closed PU (`present`) | exact onset point |
| `dilemmadata.dlc.section.boundary` | closed PU (`present`) | exact onset point |
| `dilemmadata.dlc.note.scale_degree` | open CPU | canonical note identity |

There are 9 encodable/model-ready sidecar families and 13 deferred open-string
families. “Model-ready” means the sidecar can be deterministically tensorized;
Phase 9B.2A deliberately supplies no model head or loss for it.

No borrowed-harmony task exists because the source has no such field. Source
staff/voice is not a semantic voice-role label. Tonal region is not duplicated
from local key. AN and DLC tasks remain separate, and neither dialect is
crosswalked to HookTheory or POP909-CL. Root, bass, tonal-region, and
chord-quality cross-source mappings remain deferred.

## Target states and provenance

At source-row level, `available`, `masked`, `missing`, and `unsupported` are
mutually exclusive. `ambiguous` is recorded for a conflicting exact entity;
`alignment conflict` and `unaligned` are counted after entity construction and
exact alignment. Open-string available rows are also counted as `deferred`.

False or missing validity gates mask supervision. A true gate with no value is
missing. Malformed gates and values outside a frozen vocabulary are
unsupported. None becomes a negative class or a class ID. Runtime vocabulary
construction and Python-hash IDs are forbidden.

`alt_label` stays a diagnostic fingerprint/count, not another family and not a
rule for choosing a preferred analysis. AN analyst/proofreader and DLC
annotator/reviewer/source metadata are canonicalized in
`DilemmadataTargetMetadataIndex@1.0.0` and copied only into target provenance.
Conflicting duplicate metadata values are preserved deterministically and
flagged; they are never used for raw grouping or model input. Confidence is
`None` because the release provides no calibrated numeric confidence.

## Exact alignment

All time is `RationalTime`; floats, tolerances, nearest-neighbour snapping, and
node-type priority are forbidden.

- Note labels reference canonical note IDs. Every source row merged by the raw
  tie policy must be available and agree. Otherwise the canonical-note target
  is masked with an alignment-conflict diagnostic.
- Point events retain their exact observed onset. If there is no exact
  candidate, the available row survives with `entity_index=-1` and a false
  entity-index mask.
- Spans are exact half-open intervals, `start <= t < end`. A next observed
  source boundary may close a run; an unproved terminal end is not invented.
- Equal exact duplicates merge deterministically. Conflicting duplicates emit
  one masked entry and a distinct conflict diagnostic. Masked entries are not
  replicated across raw candidates.

One exact source span may legitimately align to multiple permitted raw node
types at the same coordinate. This is explicit multi-candidate supervision,
not snapping or priority selection.

## Alternative analyses and leakage

Every source record produces its own `analysis_view_id` and sidecar. No average,
vote, or “primary” analysis is selected. Phase 9A/9B.1 source components remain
split-atomic: 1,507 components, 126 multi-record components, 98 explicit AN/DLC
overlaps, 30 conservative same-input candidate groups, and five conflicting
release split-hint components. The committed Phase 9B.1 split membership and
fingerprint remain authoritative.

## Public path

The public API consists of `DilemmadataTargetAdapterConfig`, typed accepted and
quarantine outcomes, `build_dilemmadata_target_sidecar`,
`convert_dilemmadata_target_sidecar`, `iter_dilemmadata_target_sidecars`, the
target metadata-index loader, deterministic `TargetBundle` serialization and
fingerprinting, registry/encoding serialization and fingerprints, and
`attach_target_bundle` for an already loaded raw-cache sample.

The production path is:

```text
raw cache -> IndexedMultiSourceDataset -> attach TargetBundle
          -> exact alignment -> tensorizer -> existing collator
          -> MultiSourceBatch
```

The compact full-corpus evidence is
`tests/fixtures/dilemmadata/target_manifest.json`. The full generated report,
corpus, target artifacts, raw caches, predictions, and checkpoints are not
committed. GitHub CI checks deterministic bounded fixtures and the committed
manifest; the pinned full-corpus scan remains explicit local opt-in evidence.

## Pinned full-corpus evidence

The completed local opt-in scan accepted a sidecar for every one of the 719
Phase 9B.1 accepted raw records: 108 AN and 611 DLC, with zero target quarantine
and zero fatal target outcomes. The unchanged raw scan still reports 914 raw
quarantines. Across accepted records it observed 1,042,098 source rows,
1,918,235 available target entries, 757,181 masked entries, 1,654,221 target
alignment spans, 11,713 non-empty `alt_label` observations, and 4,476 retained
analyst/reviewer metadata fields.

Exact alignment emitted 6,705,009 aligned available rows and retained 3,950
available unaligned rows. It recorded 18,496 merged-tie agreements, 556
merged-tie conflicts, 556 total conflicts, and 6,391,335 deterministic equal
duplicate merges. Family-local source states, emitted/model-ready row counts,
value counts, and value fingerprints are in the manifest rather than repeated
here.

The real E2E cache/Dataset/sidecar/alignment/tensorizer/collator batch used
seven distinct records (4 AN, 3 DLC), including an accepted alternative pair,
four merged-tie records, and three records with cadence/phrase/section events.
It contained 4,656 raw nodes, 34,300 raw edges, 6,276 source target entries,
and 17,913 tensorized rows. Raw graph/model-input fingerprints and candidate
identities were unchanged; 9 closed tasks became tensors, 13 open tasks stayed
CPU, and retained CUDA/prediction tensor counts were zero.

Key fingerprints are:

- family registry:
  `4e5077fc4ed70ea9e39a56955ac3d0b9d701c1f48427c3f0c171b9a35ccfa3b4`;
- source-native encoding registry:
  `699920917d20f560408252f115048a80268cdae7ab3ccf1d30dc3f8be5103d7b`;
- combined target sidecars:
  `75a71bdeaab2df79182549a3222a3ae83f51f9c07359fd0ac10dc94ff0a7361b`;
- target metadata index:
  `41e15e1d2edb1c52ad3ca90acf782bec7c26bfb042fea51dc805d6f86b52d0a7`;
- raw-to-target alignment evidence:
  `52d44cb5d00d3ab62f9cf3af6d258b00c90b8050e68ad82d7ef641d616d1369e`;
- target audit manifest:
  `6e1813850253b18f81bc342144a71b5c8261f8e4ae4609504dd789c6989f6d5c`.

The accepted raw index remains
`c0451976b6b6eab88cb90aa6c47d6afdba1b81ce9b588f0f84daa846154adb0e`
and the split manifest remains
`58ac7720f65f7fd3102248fb39d89291a78d65c06fc2ab9a16d78a6ee1666a3e`.
