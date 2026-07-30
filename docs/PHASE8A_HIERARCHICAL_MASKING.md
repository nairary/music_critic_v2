# Phase 8A deterministic hierarchical masking

## Status and scope

Phase 8A implements hierarchy-aware mask contracts, sparse planners,
model-side overlays, the shared prepared-input security path, and bounded
mechanics evidence. Draft PR #16 is pending final-head review, Required CI,
and an independent exact-final RTX 3090 CUDA/AMP acceptance run. It must remain
draft and must not be merged by this implementation task.

Phase 8B has not started. Phase 8A adds no onset, beat, bar, or track
objective head; no role/voice labels; no Dilemmadata, PDMX, or PLL path; no
preference critic; and no production or full-corpus SSL training. The
unchanged Phase 7A note/bar/song representation losses are used only as a
bounded integration smoke. Nothing here shows that one masking policy learns
better representations than another.

## Contract versions

The seed-ranked full-tolerance-set remediation changes portable plan
semantics. These contracts are `1.2.0`:

- hierarchical mask plan;
- hierarchy mask policy;
- policy configuration;
- selected hierarchy-unit and descendant evidence;
- prepared hierarchy binding profile;
- `PreparedHierarchyMaskBinding` envelope;
- bounded acceptance report;
- bounded benchmark report.

These unchanged Phase 8A contracts remain `1.0.0`:

- deterministic policy mixture;
- structured unavailable reason;
- `Phase8AHierarchySSLForwardOutput`;
- pitch-leakage audit;
- supplemental hierarchy fixture.

The separate optional
`Phase8ACudaAmpHardwareEvidence@1.1.0` artifact binds the remediated portable
contracts. A result for an earlier draft head is intermediate evidence only.
Hardware identity, timing, and peak allocated/reserved VRAM never enter the
portable deterministic CPU report fingerprint.

The existing Phase 7A `MaskPlan@1.0.0`, mask policy, feature overlay
`1.0.0`, `PreparedMaskBinding@1.1.0`, `SSLForwardOutput@1.2.0`, model and
checkpoint metadata, and Hydra configuration are unchanged. The accepted
post-hotfix contracts also remain unchanged: device transfer `1.0.2`,
representation/multi-view/objective FP32-under-AMP `1.0.1`, umbrella SSL
`1.2.2`, training report `1.2.2`, anti-collapse diagnostics `1.1.1`, and the
independent no-leakage and pitch-sensitive-reconstruction evidence contracts
`1.0.0`.

An independent-only configuration delegates directly to the Phase 7A binding
builder and preserves its type, dictionary, fingerprint, tensors, and
numerical outputs. Public `forward()` remains Phase 7A-only; hierarchy
integration is explicit through `forward_hierarchy()`. The remediated policy
contract fingerprint is
`2d39eb5e1ddf6ad53c626a18b364d0ffae0896663008a4e1422215c0c20fbdb1`;
the default all-policy configuration fingerprint is
`e38651e00726ce9681dc015634c5d1f48f11586d07e0faf3187e20bda9ffee67`.
The current model metadata fingerprint is
`7f98ff6e79fa7515986d22287c601723a1329e6b5ec294fc45cbdaae3e304bb7`.
Pre-remediation fingerprints are historical evidence only.

## Exact policies

`independent_note_pitch` is the control. It dispatches directly to Phase 7A
`uniform_note_without_replacement` without renaming, wrapping, or
re-fingerprinting the plan.

`onset_pitch_descendants` selects raw onset nodes and follows:

```text
onset --starts_note--> note
```

All notes in a polyphonic onset are indivisible descendants.

`beat_pitch_descendants` selects raw beat nodes and follows:

```text
beat --contains_onset--> onset --starts_note--> note
```

Empty beats are not eligible units.

`contiguous_bar_pitch_span` selects one inclusive contiguous raw-bar range.
Its length is between configured `min_span_bars` and `max_span_bars`, and its
descendants follow only:

```text
bar --contains_onset--> onset --starts_note--> note
```

`track_bar_pitch_span` selects one raw track and one such range. Its primary
notes are the exact intersection:

```text
track --contains_note--> note
∩
bar --contains_onset--> onset --starts_note--> note
```

No melody, bass, chord, voice, staff, theory, provenance, or target field is
read. Both span policies are start-anchored. A note beginning before the span
does not become primary because it remains sounding inside it; `active_at` and
`has_active_note` are never traversed by a planner.

## Visible and hidden evidence

All four hierarchy policies remain pitch-only. For every primary note the
overlay hides value and availability contributions of `pitch`, `pitch_class`,
`octave`, and `track_relative_pitch`. Rhythm, onset, duration, metric
position, velocity, raw ownership, and topology remain visible.

The Phase 7A collateral closure is unchanged:

- each unselected peer note in an affected owner track hides
  `track_relative_pitch` and its availability contribution;
- each affected track hides `mean_pitch`, `pitch_std`, `min_pitch`, and
  `max_pitch`, including availability.

Collateral rows are not reconstruction targets. The fail-closed leakage audit
classifies all 68 current raw registry fields as four primary note fields,
four unique owner-track collateral fields, and the exact ordered 60-field
visible remainder. Its unchanged fingerprint is
`27fc135b61649e5b892036dd0aacc92f679493ff671320c8235d33396a7c9949`,
and it pins raw feature-registry fingerprint
`567a5fdbb0d132010af4716c5988686c2bdf998cf6f1b2eec897f8af3ca8c0e2`.
A registry change fails the audit until the classification is reviewed.

Canonical ordering of simultaneous same-track notes can use pitch as a
tie-break, so visible temporal topology may expose relative rank. It does not
duplicate exact MIDI pitch. Phase 8A deliberately keeps topology visible and
does not claim it contains no pitch information.

## Sparse validation and deterministic unit selection

Planning runs on the fully validated CPU raw graph before device transfer. One
target-blind sparse index resolves notes by onset, onsets by beat/bar, beats by
bar, one raw owner track per note, start-descendant notes by bar, and occupied
track/bar cells.

The validator rejects duplicate or missing note-onset, onset-beat, onset-bar,
beat-bar, or note-track ownership; disagreement between an onset's direct bar
and its owning beat's bar; duplicate relevant forward edges; cross-sample
endpoints; disagreement between direct note-bar ownership and the composed
start relation; and malformed local `next_bar` chains. Portable structure
evidence uses only local counts and sorted endpoints, not batch-global offsets,
feature values, targets, entity IDs, provenance, or diagnostics.

The requested note mask rate defines a target hidden-note count. A positive
fractional rate uses `max(1, floor(note_count * rate))`; the target is not
silently capped when exact realization is impossible.

Onset and beat units begin in canonical local order and receive the versioned
linear SplitMix64/Fisher–Yates permutation derived from SHA-256 seed evidence.
Each unit is visited once and descendants are deduplicated. At the first
budget crossing, valid prefixes immediately before and after the crossing are
compared by absolute error; equal error uses stable SHA-256 evidence.

## Bounded near-optimal span selection

Span bounds satisfy:

```text
1 <= min_span_bars <= max_span_bars <= 8
```

Bar candidates enumerate only bounded contiguous ranges. Track/bar candidates
are generated around occupied sparse cells, never a dense tracks-by-bars
matrix. Selection method `bounded_near_optimal_seed_rank_v2` is:

1. Scan all valid candidates to find the best absolute hidden-note budget
   error.
2. Admit candidates with error at most
   `best_error + span_budget_error_slack`.
3. Compute `stable_seed_sha256_pool_membership_v1` independently for every
   tolerance-qualified candidate. The rank binds dataset/piece identity,
   canonical epoch, view, global seed, policy/version, configuration
   fingerprint, and the full canonical candidate identity.
4. Retain the `span_selection_pool_size` smallest membership ranks by a
   streaming bounded selector. The canonical
   track/start/end/descendants key is only the hash-collision fallback.
5. Select one retained candidate with the separate
   `stable_seed_sha256_final_choice_v1` domain. The membership hash is never
   reused as the final-choice hash.

Configuration `1.2.0` binds pool size in `[1, 8]` and integer slack in
`[0, 8]`. Defaults are pool size `4` and slack `1`. Pool size `1` now means a
one-entry seed-ranked pool over the complete tolerance set; it is not a
canonical prefix or exact-closest control. Slack `0` restricts admission to
exact-best error. Enumeration order cannot affect membership or choice, and
no full candidate sort is performed.

Selection evidence records total valid candidates, best error,
tolerance-qualified count, retained-pool count, configured pool limit and
slack, selected error/start/end/track, selected descendant count, realized
rate, the complete selected canonical identity, and separate overall,
membership-rank, and final-choice method identifiers.

Every available plan masks at least one primary note and leaves at least one
pitched note visible. Spans cannot cross sample or selected-track boundaries.
Zero-rate, one-note, empty-unit, and no-valid-span cases return a versioned
structured unavailable reason and never silently fall back.

A policy configuration records all five ordered non-negative weights, span
bounds, pool size, and error slack. Weight zero disables a policy. Mixture
resolution records every policy's eligibility/reason, eligible normalized
weights, resolution seed, and selected policy. Reports retain explicit
eligibility and realized-frequency denominators.

Validation canonicalizes epoch to zero. Train epoch, global seed,
dataset/piece identity, view index, relevant raw structure, and configuration
are the only varying plan inputs. Python `hash()`, global RNG, batch
position/order, worker/process count, targets, annotations, provenance, and
diagnostics are excluded.

## Diversity and budget audit

The default pool `4`/slack `1` was audited over train epochs `0..255`, seed
`42`, mask rate `0.30`, and the four bounded train identities (`t0`, `t1`,
`t2`, and supplemental oracle `o`). The table reports
`valid / best / tolerance / retained`, distinct actual selections, and the
selected-error histogram:

| Policy | Piece | Candidate evidence | Distinct | Selected errors |
|---|---:|---:|---:|---|
| bar span | t0 | 5 / 1 / 3 / 3 | 3 | `1:256` |
| bar span | t1 | 2 / 4 / 2 / 2 | 2 | `4:256` |
| bar span | t2 | 2 / 3 / 2 / 2 | 2 | `3:256` |
| bar span | o | 5 / 0 / 2 / 2 | 2 | `0:130, 1:126` |
| track/bar | t0 | 10 / 1 / 10 / 4 | 10 | `1:100, 2:156` |
| track/bar | t1 | 9 / 1 / 9 / 4 | 9 | `1:88, 2:168` |
| track/bar | t2 | 6 / 0 / 4 / 4 | 4 | `0:256` |
| track/bar | o | 10 / 0 / 9 / 4 | 9 | `0:63, 1:193` |

Every selected error is within `best + 1`, and every audited piece has actual
span diversity. This is bounded fixture justification for defaults, not a
quality or corpus-distribution result.

A crafted positional-bias oracle has 36 tolerance-qualified single-bar
track spans over three tracks and bars `0..11`: one unique error-0 candidate
and 35 error-1 candidates. Across epochs `0..255`, all 36 candidates enter a
retained pool and all 36 are actually selected. Selected start bars cover
`0..11`, all three tracks occur, and 224 selections escape the first four
candidates of the obsolete canonical order. The selected-error histogram is
`0:7, 1:249`; ordered replay fingerprint is
`4c02e60f26498e6a78633ede6631d313696ef2318ab054073a52842534f539a7`.
Fresh repeat, reverse enumeration, and a deterministic permutation are
bit-exact. Validation epochs 0 and 999 canonicalize to one actual selection;
slack `0` is the exact-best control. This proves removal of canonical
positional exclusion, not unbiased or uniform random sampling.

## Prepared-input security and model integration

`PreparedHierarchyMaskBinding@1.2.0` reuses the exact
`PreparedMaskBinding@1.1.0` runtime graph evidence, process-local HMAC, opaque
prepared token, transfer renewal, and private Phase 6 prepared encoder. It is
a distinct portable envelope over one shared security kernel, not a copied
validator, and there is no public boolean bypass.

The hierarchy fingerprint binds ordered sample identities and raw CPU
structure, stage/canonical epoch/seed/view/rate, policy configuration and
ordered resolutions, ordered plans and overlay, and exact compact global
descendant-note indices. Runtime evidence attests graph/store identity and
exact attribute sets, all model-facing tensors by strong reference, identity,
`_version`, shape, dtype, and device, and typed non-tensor metadata.
Post-prepare feature, relation, metadata, or store mutation fails before the
first encoder operation. Transfer re-attests the source and renews evidence
over destination objects. Prepared CPU/CUDA forward performs no graph-sized
accelerator-to-host `.cpu()`, `.tolist()`, or `.item()`.

The overlay does not mutate or copy a canonical piece/raw graph, remove
nodes/edges, change graph fingerprints, write mask metadata into PyG stores,
or enter canonical caches. Target, provenance, theory, and diagnostic
replacement cannot affect plans, descendants, overlays, or model input.

The model adds no parameter or head. `forward_hierarchy()` returns
`Phase8AHierarchySSLForwardOutput@1.0.0`; detached full-view note/bar/song
targets, decoder remasking, and existing losses are unchanged. Phase 8B owns
future onset/beat/bar/track objectives and comparative training.

## Compatibility and fixture boundary

The independent dispatcher is checked against direct Phase 7A for exact plan,
overlay, binding, compact indices, encoder tensors, predictions, and loss
tensors on CPU and, when available, CUDA with AMP. Phase 7A checkpoints,
resume, encoder transfer, bounded anchors, and the post-hotfix concrete-device
and FP32-under-AMP behavior remain unchanged. Correct-target preference stays
a signed non-gating diagnostic; no-leakage and pitch-sensitive reconstruction
remain separate evidence objects.

No graph schema, feature registry, canonical/cache/split contract, adapter,
ontology, target encoding, Phase 6 state, or numerical output changes. The
immutable Phase 7A fixture remains unchanged. Phase 8A adds a separate
target-free oracle; combined bounded composition remains 6 pieces, 14 tracks,
15 bars, 60 beats, 42 onsets, 93 notes, 39 polyphonic onsets, one multi-onset
beat, one cross-bar sustained note, and 34 occupied track/bar cells. Fixture
and leakage contracts remain `1.0.0`.

## Complexity and memory boundary

Let `C` be emitted valid span candidates and `S` their stored descendant
entries. Candidate generation retains `O(C + S)` sparse state. Span selection
uses a best-error pass followed by seed-ranked bounded insertion into at most
`K <= 8` entries and a separately ranked final choice:

```text
O(C * K) = O(C) time under the contract-fixed K bound
O(K) additional selection scratch
```

There is no unbounded/full candidate sort, dense node-to-unit or note-to-note
matrix, simultaneous-note clique, full `O(B²)` span set, dense
tracks-by-bars enumeration, or all-note-pairs loop. The no-threshold benchmark
records index/planning/descendant/overlay/binding/forward timing, sparse
counts, and retained compact plan bytes. Those bytes are a deterministic
retained-state proxy, not Python allocator, temporary, CUDA, or process peak
memory.

Anti-collapse diagnostics retain `O(D)` sufficient statistics, but current
`from_values` materializes a temporary float64 `N x D` `values64` tensor and
normalized `N x D` working temporaries. No `O(D)` peak-temporary-memory claim
is made. Their real CUDA cost is not measured. Production SSL on an RTX 3090
requires a separate profiler/optimization gate.

## CPU and optional CUDA acceptance

Portable deterministic acceptance runs on CPU and was rebuilt in two fresh
processes with byte-identical canonical JSON: 93,062 bytes each, SHA-256
`b21cf11e018130e7270abdfa47d56b0414a4a5a01ea14db973e125e8590c6fb1`.
Its single-policy fingerprints and bounded objective observations are:

| Policy | Config | Overlay | Prepared binding | Total loss | Grad tensors present/nonzero |
|---|---|---|---|---:|---:|
| independent | `24bc6c071aad797dcb6b40143546e7e6af968a63b44be235d73c863b40f4efa4` | `0dced56d924d8d6a78116817b12bd09927cdbdc349326e8bbffebb372e95a20f` | `d26ec7aab9074c1afe7ae6811a82716192a2d7d8ce5c2d9136e5d5a702cbf1aa` | 3.1845867634 | 380/361 |
| onset | `c5984122052d5ecb08e02a8e9ac3d8d76a7f32e6f308a31ddd4b2fb2045abbd6` | `7e3011644abc61989fc70281cc23fecc4732ffec3c6f85d5031822e87abd088e` | `7d09ae45e8640eac33da58d595528b1c3b25ad1eccde57d92262cc37132b12eb` | 3.0620796680 | 380/361 |
| beat | `da4e0f33328f0e3eafa5040741e74a93aa34e627f83eca4341f85a8465816bd4` | `8cfa89745238ac3d297dbdf24499d75984adcdce05d90db3e2926a351a12ec09` | `3dbf4e8abba7afc119e6b7634e9ef2f33743638d98b1ed7323a3c77c7f06faa5` | 3.1490781307 | 380/361 |
| bar span | `de33e468d4edccb9fe4d13280edb10a6e408e6b5fc7fda70a7dc62e1aa622b9e` | `1de24d9cfc414d09e549b0b092c61fdc96ca9165d2426733b0ec78d45410c2a7` | `bcc40f182e46cf15157a79f883e9591b6c21af9a857a197a9fc9a841d1fdc760` | 3.0673878193 | 380/361 |
| track/bar | `1355a41c7d40c3585fba05d588cd53f1899467f8c3624c3fa11b5c77fc659c54` | `6e22d5edb04e873a2a3a82f99a0ac55f09c293b16a29ccd65b0093e6adfbea7d` | `c2cd4cb05e55e70279a6b5240afd44feccf932f3575e5283dff6d128ae3fc6b1` | 3.2411322594 | 380/376 |

These losses are deterministic bounded mechanics observations, not training
or representation-quality results.

The optional CUDA/AMP path must use an explicit `cuda:0`, exercise all five
single policies plus the mixture, assert concrete-device bindings, finite
forward/loss/gradients, and prepared CPU-to-CUDA transfer, and record peak
allocated/reserved VRAM. It emits a separate hardware artifact. If CUDA is
unavailable, the comprehensive pytest path skips honestly; CPU output must
not be presented as GPU evidence. Independent exact-final RTX 3090 evidence
remains the pre-merge gate.

Focused and regression commands are:

```bash
.venv/bin/python -m pytest -q tests/ssl/test_hierarchy_fixture.py \
  tests/ssl/test_hierarchical_masking.py \
  tests/ssl/test_hierarchical_prepared_binding.py
.venv/bin/python -m pytest -q \
  tests/ssl/test_data_config.py::test_workers_zero_and_two_preserve_per_identity_inputs_and_plans
.venv/bin/python scripts/accept_phase8a_hierarchical_masking.py \
  --output /tmp/phase8a-cpu-acceptance-a.json
.venv/bin/python scripts/accept_phase8a_hierarchical_masking.py \
  --output /tmp/phase8a-cpu-acceptance-b.json
cmp /tmp/phase8a-cpu-acceptance-a.json \
  /tmp/phase8a-cpu-acceptance-b.json
sha256sum /tmp/phase8a-cpu-acceptance-a.json \
  /tmp/phase8a-cpu-acceptance-b.json
.venv/bin/python scripts/benchmark_phase8a_hierarchical_masking.py
.venv/bin/python -m scripts.accept_phase8a_cuda_amp \
  --device cuda:0 --amp --amp-dtype float16 \
  --expected-head "$(git rev-parse HEAD)" \
  --expected-device-name "NVIDIA GeForce RTX 3090" \
  --portable-report /tmp/phase8a-cpu-acceptance-a.json \
  --output /tmp/phase8a-cuda-amp-hardware.json
.venv/bin/python -m pytest -q \
  tests/ssl/test_phase8a_cuda_amp_acceptance.py -rs
.venv/bin/python -m pytest -q tests/ssl
.venv/bin/python -m pytest -q \
  tests/models tests/graph tests/test_device.py
.venv/bin/python -m pytest -q tests/training tests/evaluation
.venv/bin/python -m pytest -q \
  tests/ssl/test_checkpoint_training.py tests/ssl/test_transfer.py \
  tests/training/test_runner.py::test_epoch_boundary_resume_is_bit_exact_in_metrics \
  tests/training/test_runner.py::test_incompatible_resume_preserves_rng_and_all_artifacts
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q src scripts tests
git diff --check
git show --check
```

The benchmark and bounded acceptance establish deterministic mechanics only.
No HookTheory or POP909-CL corpus scan, Dilemmadata or PDMX integration,
production cache rebuild, production/full-corpus SSL training, PLL, critic
training, or quality/likelihood interpretation is part of Phase 8A.

Final local verification before commit:

- focused Phase 8A: `113 passed, 2 skipped, 2 warnings`;
- fresh-process plus workers `0/2` parity: `2 passed, 2 warnings`;
- complete SSL: `319 passed, 8 skipped, 8 warnings`;
- model/graph/device regressions: `165 passed, 1 skipped, 2 warnings`;
- training/evaluation regressions: `94 passed, 6 skipped, 4 warnings`;
- checkpoint/resume/transfer regressions: `21 passed, 2 warnings`;
- deterministic held-out plus repository audit: `7 passed, 2 warnings`;
- complete repository: `1187 passed, 29 skipped, 10 warnings`;
- bounded benchmark: all five policies, no timing threshold;
- `compileall` and `git diff --check`: passed.

CUDA was unavailable locally. The CUDA skips above are not GPU evidence.
Required GitHub CI and the independent exact-final RTX 3090 run remain
pre-merge gates.
