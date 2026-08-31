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
`DilemmadataRawTargetAlignmentEvidence@1.1.0`: target-neutral evidence over
every row ordinal, physical line, exact rational onset, tie-continuation state,
and canonical note ID. The embedded fingerprint is only a corruption check.
Before any target or metadata values are read, the target adapter independently
re-runs the closed raw parser/tie-merger from the pinned physical source,
requires byte-exact equality with the supplied canonical piece, and compares
the complete ordered evidence object. It changes neither raw semantics nor any
raw adapter/cache/graph version.

## Contracts

| Contract | Version |
|---|---:|
| Dilemmadata target adapter | `1.1.0` |
| external target sidecar | `1.0.0` |
| generic `TargetBundle` | `1.0.0` |
| raw-to-target alignment evidence | `1.1.0` |
| target-only analyst metadata index | `1.0.0` |
| source-native family registry | `1.0.0` |
| source-native encoding registry | `1.0.0` |
| exact alignment rules | `1.0.0` |
| target audit report / manifest | `1.1.0` / `1.1.0` |

### Phase 9E-B2 remediated compatibility path

The table above remains authoritative for every historical unmodified record.
For a raw record recovered by Phase 9E-B2 only, the raw adapter is `1.1.0`, the
raw-to-target alignment evidence is `1.2.0`, and the target adapter is `1.2.0`;
the external sidecar and generic `TargetBundle` contracts stay `1.0.0`.

Alignment `1.2.0` is bound to `DilemmadataRawRepairEvidence@1.0.0` and applies
the same exact raw-derived onset transform before target alignment. Repair
evidence itself is not target content. A `note` repair-mask scope affects only
entries bound to the repaired canonical note; an `all` scope affects only the
task entries whose structural coordinate cannot be proved. Missing one target
family remains ordinary per-task unavailability and never rejects a raw
record. The target adapter cannot guide tie, time, measure, bar, beat, note, or
graph reconstruction.

The production remediation smoke builds representative AN and DLC sidecars in
memory only and verifies transform equality, repair-fingerprint binding,
per-task availability, and local masking. It writes no historical sidecar.
All source sidecars and common projections for the previous 719 records remain
unchanged.

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

Evidence provenance is not established by its self-fingerprint. The
independent oracle reads only the closed raw-value field set declared by the
AN/DLC raw parser, never theory or analyst/reviewer metadata fields. It rebuilds
the exact canonical piece and row bindings from the pinned source and compares
ordered ordinal, physical line, rational onset, tie state, and canonical note
ID. Any difference fails closed as
`dilemmadata.target.alignment_binding_mismatch` before target parsing.

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
  `a3ff6ba2ea8f3b2f6062fdf30ce92d40d7c70dc505770a06419462a452dae080`;
- source-native encoding registry:
  `699920917d20f560408252f115048a80268cdae7ab3ccf1d30dc3f8be5103d7b`;
- combined target sidecars:
  `1d183ab63913084c94f62dc8777995b721ca415b20081e1ea907e69015ee5c72`;
- target metadata index:
  `41e15e1d2edb1c52ad3ca90acf782bec7c26bfb042fea51dc805d6f86b52d0a7`;
- raw-to-target alignment evidence:
  `18ee4971f33f5a208e944939949152b275e09928916880c134fda3d1fb2d189a`;
- target audit manifest:
  `a971ff0daf8d5a442beaa3365ec8c43ca9368f07baab4a1102927977f6ebdd05`.

The accepted raw index remains
`c0451976b6b6eab88cb90aa6c47d6afdba1b81ce9b588f0f84daa846154adb0e`
and the split manifest remains
`58ac7720f65f7fd3102248fb39d89291a78d65c06fc2ab9a16d78a6ee1666a3e`.

## Phase 9E-A common harmonic projection

Phase 9E-A adds a second, derived target sidecar without revising
`TargetBundle@1.0.0`. `DilemmadataCommonHarmonicProjection@1.0.0` binds the
exact source-bundle fingerprint and the frozen
`DilemmadataCommonHarmonicRegistry@1.0.0`; its six model-ready families are
common chord quality, ordinal inversion, root pitch class, bass pitch class,
factorized local key, and target-derived pitch-class set. Every entry retains
its source task/value/provenance, per-field mask, mapping state, diagnostic,
information loss, mapping evidence, and dependency entity IDs.

The quality registry covers all 64 AN and 15 DLC vocabulary rows. It contains
50 common classes: 65 rows are exact and 14 AN rows are explicitly coarsened
because they embed inversion or an enharmonic-equivalence qualification. No
lossy row is labelled exact. The pinned AnalysisGNN reference is commit
`e115182fb29b74bdcb6bf3547ed427d967580947` under MIT; 26 quality rows agree,
51 are not applicable to its frozen mapping, and DLC `+7`/`+M7` deliberately
diverge from its acknowledged collapse to `augmented triad`. Ten independently
declared interval templates permit pitch-class-set derivation; incomplete,
extended, augmented-sixth, or otherwise unproved qualities stay available as
quality supervision but have pitch-class-set masked.

The AnalysisGNN reference schema is `AnalysisGNNReferenceMapping@1.0.1` and
identifies inversion rows by source task plus source value. AN ordinal `2` and
DLC figured bass `2` are therefore distinct dialect-specific values: AN `2`
maps to `second`, while DLC `2` is shorthand for `4/2` and maps to `third`.
The latter matches the pinned AnalysisGNN conversion of `2`/`42` to third
inversion. All ten inversion rows agree with the common projection. The
combined frozen quality-plus-inversion parity table contains 36 agreements,
51 not-applicable rows, and two divergences, exclusively DLC quality `+7` and
`+M7`.

AN spelling and DLC line-of-fifths TPC use an exact target-only conversion to
chromatic pitch class. Enharmonic reduction is recorded as information loss.
DLC retained spellings/modes are supplemental target evidence reconstructed
only after the existing raw/target row binding passes. TPC/spelling conflicts
are invalid structured outcomes. Bass is never inferred from the lowest raw
note. Local tonic and mode have independent field masks; unknown mode is kept
unknown. Figured bass and AN ordinals map to `root/first/second/third`, while a
conflict with a proven chord cardinality is ambiguous and masked.

The full Phase 9E-A representation audit covered all 1,633 source records and
1,507 components, including 98 explicit AN/DLC overlaps, and built one common
projection for each of the unchanged 719 accepted annotation views. All 30
conservative same-input alternative-analysis groups were audited independently.
The accepted split remains 577/71/71. It observed 712,509 exact, 43 coarsened, 4
ambiguous, 997 unsupported, 281,938 masked common entries, and no accepted
invalid entry. The 997 unsupported rows are pitch-class-set derivations with
no proven template. Overlap evidence contains 4 exact agreements, 2
enharmonic-only agreements, 66 conflicts, and 702 unavailable comparisons;
conflicts remain evidence and never select a preferred view.

Canonical-piece, raw-graph, model-input, grouping, and source-target-bundle
fingerprints were unchanged. The raw index remains
`c0451976b6b6eab88cb90aa6c47d6afdba1b81ce9b588f0f84daa846154adb0e`,
the split remains
`58ac7720f65f7fd3102248fb39d89291a78d65c06fc2ab9a16d78a6ee1666a3e`,
the common registry is
`bb50920808b6ad3a19fb32b8315a417a837b2ab008efd7bee71e71d120e2ee2e`,
and the combined projection fingerprint is
`7bf051343b24d79530ee483f9d8b49fd13f10e0fa1db0c535cbdb23a00c18f77`.
The pinned AnalysisGNN reference fingerprint is
`1e6a713665eddabac8f98c67df990aa1a7a01fde5f0956b9f9207158cad611ba`,
the full report semantic fingerprint is
`a3d3c3ac869613787602e7239eb0af10dfb904621ac5b81b9d9431c33e3750a1`,
and the compact manifest fingerprint is
`4ce7b657d2003d2ce3aadcfe9de9e39c7f9a49b69e985a745a399ef02e056294`.
The committed compact manifest is
`tests/fixtures/dilemmadata/common_harmonic_manifest.json`; `--check` validates
it without corpus access. Locked-test targets were read only for the declared
representation/coverage audit—no model inference, metric, selection, or test
unlock occurred.
