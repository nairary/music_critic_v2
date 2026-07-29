# Phase 8A deterministic hierarchical masking

## Status and scope

Phase 8A implements hierarchy-aware mask contracts, sparse planners,
model-side overlays, the shared prepared-input security path, and bounded
mechanics evidence. It does not complete all of Phase 8.

Phase 8B has not started. Phase 8A adds no onset, beat, bar, or track
objective head; no role/voice labels; no PDMX or PLL path; no preference
critic; and no production or full-corpus SSL training. The existing Phase 7A
note/bar/song representation losses are used only as a bounded integration
smoke. Nothing here is evidence that one masking policy learns better
representations than another.

## Contracts

All new Phase 8A contracts begin at `1.0.0`:

- hierarchical mask plan;
- hierarchy mask policy;
- policy configuration and deterministic mixture resolution;
- selected hierarchy-unit and descendant evidence;
- structured unavailable reason;
- prepared hierarchy binding envelope;
- prepared hierarchy binding profile;
- hierarchy SSL integration output envelope;
- pitch-leakage audit;
- supplemental hierarchy fixture;
- bounded acceptance report and benchmark.

The existing Phase 7A `MaskPlan@1.0.0`, mask policy, feature overlay
`1.0.0`, `PreparedMaskBinding@1.1.0`, decoder/objective contracts,
`SSLForwardOutput@1.2.0`, model/checkpoint metadata, and Hydra configuration
are unchanged. Hierarchy execution uses the distinct
`PreparedHierarchyMaskBinding@1.0.0` portable envelope and
`Phase8AHierarchySSLForwardOutput@1.0.0`. The hierarchy binding records the
shared Phase 7A attestation-kernel version `1.1.0` and a hierarchy profile
`1.0.0`; its payload additionally binds the policy configuration and ordered
resolution fingerprints. An independent-only configuration delegates
directly to the old binding builder and therefore returns the exact Phase 7A
binding type, dictionary, and fingerprint. The normal public model
`forward()` remains Phase 7A-only; Phase 8A integration is explicit through
`forward_hierarchy()`.

The exact checked-in hierarchy-policy contract fingerprint is
`b188e90a60d3ec6184dfdb3233ef37b1a0ea133cd5957a10fad3eddf58d77ccd`.
The all-policies default configuration fingerprint is
`32ef80c55f2b06f06a8da39d083f1d484dcbdd26134cc041df5adc667e9bfada`.
The pitch-leakage audit fingerprint is
`27fc135b61649e5b892036dd0aacc92f679493ff671320c8235d33396a7c9949`.

Portable plans and resolutions contain only immutable scalar/tuple evidence.
They contain no graph tensor, feature value, target, annotation, provenance,
diagnostic, process identity, HMAC, or device object.

## Exact policies

`independent_note_pitch` is the control name. It dispatches directly to
Phase 7A `uniform_note_without_replacement`; it does not rename, wrap, or
re-fingerprint that plan.

`onset_pitch_descendants` selects raw onset nodes and masks every note reached
through:

```text
onset --starts_note--> note
```

All notes in a polyphonic onset are indivisible descendants.

`beat_pitch_descendants` selects raw beat nodes and follows:

```text
beat --contains_onset--> onset --starts_note--> note
```

Empty beats are not eligible units.

`contiguous_bar_pitch_span` selects one inclusive, contiguous raw-bar range
`[start_bar, end_bar]`, with length between configured `min_span_bars` and
`max_span_bars`. Descendants follow only:

```text
bar --contains_onset--> onset --starts_note--> note
```

`track_bar_pitch_span` selects one raw track and one such bar range. Primary
notes are the exact intersection of raw track ownership with the
start-descendants of the range:

```text
track --contains_note--> note
∩
bar --contains_onset--> onset --starts_note--> note
```

It never reads melody, bass, chord, voice, staff, or any other semantic role.

Both span policies are start-anchored. A note that begins before a span and
remains sounding inside it through `active_at` is not a primary descendant.
`active_at` and `has_active_note` are never traversed by a mask planner.

## Visible and hidden evidence

All four new policies are pitch-only. For each primary note the overlay hides
the value and availability contributions of:

- `pitch`;
- `pitch_class`;
- `octave`;
- `track_relative_pitch`.

Rhythm, onset, duration, metric position, velocity, raw track membership, and
topology stay visible. The Phase 7A collateral closure is applied unchanged:

- every unselected peer note in an affected owner track also hides
  `track_relative_pitch` and its availability contribution;
- each affected track hides `mean_pitch`, `pitch_std`, `min_pitch`, and
  `max_pitch`, including availability.

Collateral rows are not reconstruction targets.

The fail-closed leakage audit serializes an exhaustive classification of all
68 current raw registry fields: four primary note fields, four unique
owner-track collateral fields, and the exact ordered 60-field visible
remainder. The peer-note collateral identity is
`note.continuous.track_relative_pitch`, already one of the four primary field
identities but applied to a different row population. The audit pins raw
feature-registry fingerprint
`567a5fdbb0d132010af4716c5988686c2bdf998cf6f1b2eec897f8af3ca8c0e2`.
Those eight note/track fields are the only current exact pitch values,
duplicates, or aggregates. No song/bar/beat/onset field derives from pitch.
If the raw registry changes, the Phase 8A audit fails until the classification
and contract are reviewed.

One deliberate boundary remains: canonical ordering of simultaneous notes in
one track can use pitch as a tie-break, and visible temporal topology can
therefore expose a relative rank. It does not duplicate an exact MIDI pitch,
and Phase 8A explicitly keeps topology visible. The documentation does not
claim topology is free of all pitch information.

## Sparse hierarchy validation

Planning runs on the fully validated CPU raw graph before device transfer. A
single sparse index resolves:

- notes by onset;
- onsets by beat;
- onsets and beats by bar;
- one raw owner track per note;
- start-descendant notes by bar and sparse track/bar cell.

The Phase 8A validator additionally rejects duplicate/missing note-onset,
onset-beat, onset-bar, beat-bar, or note-track ownership; disagreement between
an onset's direct bar and its owning beat's bar; duplicate relevant forward
edges; cross-sample endpoints; disagreement between direct
`bar --contains_note--> note` and the composed start relation; and a malformed
local `next_bar` chain. The relevant per-piece structure fingerprint uses
local counts and sorted local endpoints, never batch-global offsets or feature
values.

## Deterministic budget and mixture behavior

The requested note mask rate defines a target hidden-note count. A positive
fractional rate uses `max(1, floor(note_count * rate))`; the true target is not
silently capped when it cannot be achieved.

For onset and beat policies, eligible units begin in canonical local order and
receive a versioned linear SplitMix64/Fisher–Yates permutation derived from
SHA-256 seed evidence. Units are visited once. Descendants are accumulated
without duplicate counting. At the first budget crossing, the valid prefix
before and after the crossing are compared by absolute target distance; an
equal-distance choice uses stable SHA-256 evidence.

Span lengths satisfy:

```text
1 <= min_span_bars <= max_span_bars <= 8
```

The fixed hard maximum keeps enumeration bounded. Bar spans enumerate
`O(B * K)` candidates. Track-bar candidates are created only around occupied
sparse `(track, bar)` cells, not over a dense tracks-by-bars matrix. The valid
candidate whose descendant count is closest to the requested count wins;
ties use the plan seed and then canonical local indices.

Every available hierarchy plan has at least one primary note and leaves at
least one pitched note visible. Units and spans cannot cross a sample, and a
track span cannot cross its selected track. Zero-rate, one-note, empty-unit,
and no-valid-span cases return a versioned structured unavailable reason.
They never fall back silently to the independent policy.

A policy configuration records all five ordered non-negative weights and the
span bounds. Weight zero disables a policy. Mixture resolution records every
policy's eligibility, structured reason, eligible normalized weights,
resolution seed, and selected policy. Renormalization is only over positive,
piece-eligible policies and is deterministic. Reports record eligibility
counts, resolved counts, and realized frequency with an explicit denominator.

Validation canonicalizes epoch to zero. Train epoch, global seed,
dataset/piece identity, view index, relevant raw structure, and versioned
configuration are the only varying plan inputs. Python `hash()`, global RNG,
batch position/order, workers, targets, target availability, annotations,
provenance, and diagnostics are excluded.

## Prepared-input security and model integration

`PreparedHierarchyMaskBinding@1.0.0` reuses the exact
`PreparedMaskBinding@1.1.0` runtime graph evidence implementation, process-local
HMAC, opaque prepared token, transfer evidence renewal, and private Phase 6
prepared encoder. This is a separate portable envelope/type over one shared
security kernel, not a copied validator, and there is no boolean bypass.

For hierarchy profiles, the HMAC-covered portable binding fingerprint binds:

- ordered sample identities and raw CPU structure evidence;
- stage, canonical epoch, seed, view, and requested rate;
- policy-configuration and ordered resolution fingerprints;
- ordered plan and overlay fingerprints;
- exact compact global descendant-note indices.

Runtime evidence still attests graph/store identity and exact attribute sets,
all model-facing raw tensors by identity, version, shape, dtype, and device,
and typed non-tensor metadata. Post-prepare feature, relation, metadata, or
store mutation fails before the first encoder operation. Transfer renews that
evidence over destination objects. Prepared CPU/CUDA forward performs no
graph-sized accelerator-to-host `.cpu()`, `.tolist()`, or `.item()`.

The model-side overlay neither mutates nor copies a canonical piece or raw
graph. It does not remove nodes/edges, change graph fingerprints, write mask
metadata into PyG stores, or enter canonical caches. Target/provenance/
diagnostic replacement cannot affect plans, descendants, overlays, or model
input.

The Phase 7A model has no new parameters or heads. Its normal `forward()` and
`SSLForwardOutput@1.2.0` remain restricted to the old binding. The explicit
`forward_hierarchy()` integration method returns
`Phase8AHierarchySSLForwardOutput@1.0.0`. Its detached full-view
note/bar/song targets, decoder remasking, and existing losses are unchanged.
Phase 8B owns any future onset/beat/bar/track objective and comparative
training contract.

## Compatibility evidence

The control dispatcher is tested against the direct Phase 7A path for exact
plan dictionaries/fingerprints, overlay dictionaries/fingerprints, prepared
binding dictionaries/fingerprints, compact indices, encoder tensors, decoder
predictions, and every loss tensor. The existing Phase 7A bounded anchors,
checkpoint metadata, resume path, and encoder transfer therefore remain
compatible. No graph schema, raw feature registry, canonical/cache/split
contract, adapter, ontology, target encoding, or Phase 6 numerical output was
changed.

The immutable Phase 7A bounded fixture also remains unchanged. Phase 8A wraps
its five pieces and adds a separate target-free oracle piece. Combined bounded
composition is:

- 6 pieces with disjoint train/validation identities;
- 14 tracks, 15 bars, 60 beats, 42 onsets, and 93 notes;
- 39 polyphonic onsets;
- one beat with multiple onsets;
- one cross-bar sustained note;
- 34 non-empty track/bar cells.

The supplemental piece has 2 tracks, 3 bars, 12 beats, 6 onsets, and 9 notes.
Its exact relation and per-policy descendant/collateral oracles live in the
versioned fixture contract and tests. In particular, the bar-1 oracle masks
notes `(4, 5, 6)` but not sustained note `3`, which began in bar 0.

Pinned fixture evidence is:

- Phase 8A wrapper:
  `ffd0d4c7db80323b8f1f8d72c1e4b7e530151c1b95dd68033e1a30273dd98a1b`;
- supplemental oracle composition:
  `75885a66c8f131711650c20ba7180033f2c8074dbffff82fd6021c8aef1e9359`;
- supplemental raw graph:
  `8061c20f6394b3689a179d4d3ba7f3e418071b14a08bf4183cc8f22f1975d18c`;
- supplemental canonical piece:
  `c92c2a16b14b224c88227e7922f1300dcbfb14230d64a8623e08f65d85c1ea90`;
- unchanged bound Phase 7A fixture:
  `9f959d91d6805101983711511abcf89450e24b1886417632ea37fd0dc96ba922`.

## Bounded acceptance evidence

The final bounded acceptance uses seed `42`, train epoch `0`, requested note
rate `0.30`, and the four train identities abbreviated below as `t0`, `t1`,
`t2`, and `o` (the supplemental oracle). Canonical compact report SHA-256 is
`e6915779f21784a1907c930da7967d2d6c1dae4cfd72fbb0ed5c24bec37cc03a`
over 32,229 bytes. Rebuilding the report is byte-exact.

Exact per-piece mechanics are:

| Policy | Piece | Units (track for track-span) | Primary notes | Visible notes | Rate | Peer/track collateral |
|---|---:|---|---|---|---:|---:|
| independent | t0 | `[1,11,13,14,15]` | `[1,11,13,14,15]` | `[0,2,3,4,5,6,7,8,9,10,12,16,17]` | 5/18 | 13/2 |
| independent | t1 | `[0,5,10,13,14]` | `[0,5,10,13,14]` | `[1,2,3,4,6,7,8,9,11,12,15,16,17]` | 5/18 | 13/3 |
| independent | t2 | `[3,10,11]` | `[3,10,11]` | `[0,1,2,4,5,6,7,8,9]` | 3/12 | 9/2 |
| independent | o | `[4,5]` | `[4,5]` | `[0,1,2,3,6,7,8]` | 2/9 | 7/2 |
| onset | t0 | `[3,4,7]` | `[6,7,8,9,14,15]` | `[0,1,2,3,4,5,10,11,12,13,16,17]` | 6/18 | 12/2 |
| onset | t1 | `[0,1]` | `[0,1,2,3,4,5]` | `[6,7,8,9,10,11,12,13,14,15,16,17]` | 6/18 | 12/3 |
| onset | t2 | `[0,1]` | `[0,1,2,3]` | `[4,5,6,7,8,9,10,11]` | 4/12 | 8/2 |
| onset | o | `[0]` | `[0,1]` | `[2,3,4,5,6,7,8]` | 2/9 | 7/2 |
| beat | t0 | `[0,1,3]` | `[0,1,2,3,4,5]` | `[6,7,8,9,10,11,12,13,14,15,16,17]` | 6/18 | 12/2 |
| beat | t1 | `[0,1]` | `[0,1,2,3,4,5]` | `[6,7,8,9,10,11,12,13,14,15,16,17]` | 6/18 | 12/3 |
| beat | t2 | `[5,7]` | `[8,9,10,11]` | `[0,1,2,3,4,5,6,7]` | 4/12 | 8/2 |
| beat | o | `[8]` | `[7,8]` | `[0,1,2,3,4,5,6]` | 2/9 | 7/2 |
| bar span | t0 | `[2]` | `[12,13,14,15,16,17]` | `[0,1,2,3,4,5,6,7,8,9,10,11]` | 6/18 | 12/2 |
| bar span | t1 | `[1]` | `[9,10,11,12,13,14,15,16,17]` | `[0,1,2,3,4,5,6,7,8]` | 9/18 | 9/3 |
| bar span | t2 | `[1]` | `[6,7,8,9,10,11]` | `[0,1,2,3,4,5]` | 6/12 | 6/2 |
| bar span | o | `[2]` | `[7,8]` | `[0,1,2,3,4,5,6]` | 2/9 | 7/2 |
| track/bar | t0 | `[0,1]` (track 0) | `[0,2,4,6,8,10]` | `[1,3,5,7,9,11,12,13,14,15,16,17]` | 6/18 | 3/1 |
| track/bar | t1 | `[0,1]` (track 1) | `[1,4,7,10,13,16]` | `[0,2,3,5,6,8,9,11,12,14,15,17]` | 6/18 | 0/1 |
| track/bar | t2 | `[0]` (track 1) | `[1,3,5]` | `[0,2,4,6,7,8,9,10,11]` | 3/12 | 3/1 |
| track/bar | o | `[1,2]` (track 0) | `[4,7]` | `[0,1,2,3,5,6,8]` | 2/9 | 3/1 |

The corresponding ordered plan fingerprints are:

- independent:
  `f07c83364859e4f28b499d821985f9fb20c3be866c4d5e6f4bea237d3e16647c`,
  `3b5c90bc0016a528cb840ee9c3a3214e52cbd2d0eafbad2aa6ded52e0729da5d`,
  `42da3df81221b200303fd9184097e59bc7d4b85eca94a26ac7648f14bc120751`,
  `ecc80ed0e421b668ab81c7ed0b659da51c7501af35e77e589c616f8cc2b01a26`;
- onset:
  `1dfd70eb35af84642911d769eaf42a2567ccc33df2431831d73aa35fab76731c`,
  `471d0f422453d68e541f3168a981067fa8e32ae5ee115f7e1a944917e25b3b8e`,
  `1ed0a00b73114579469dc2a2cad5de18b02ab2cdf7b50cc25cb67babf7287c02`,
  `ee41d447262a718784bba4c69704a0e01cfdc0682862da7bea269d5459bad7d0`;
- beat:
  `559c18f6a01d2ee172e5e25ef9eede6a249eaf94fecabf5ad81693e383f93797`,
  `a6abf342d9e0aa0a556cad7600b9ac35d5e018d85916581dd4fbdfde1ad01b31`,
  `7d656cfc50d8d74c7937d49a35af6915bda52ecec9fa8aab1003fe3df148af38`,
  `cac0d74ded68ebde1faff7685be7d64714ab66c27929fa8396d497c4b195ac2b`;
- bar span:
  `f6a9ab7e72f3de5448192e32a607cdcf9c9fb2de662cfd073f77829f2607e5ff`,
  `bbe982ca3bb59e18af6b85370f12c5af2610f844f03efcec4e62a7d164aef27b`,
  `e92236756fd96b9df151fbb36089b88dbe6dd7ebe7ba67bd7d0c2d87b1a2e573`,
  `c83b780bcb351dbf7eea86d06b083072420b981d0b08ca4a616bf45fb69eb8f0`;
- track/bar:
  `830143606949191dc5b0ababa6f36e59db4f6a8ac2708cb930a29925d6f43ce5`,
  `16a979ef3b2dd551184908931c7b01dd0f99e4dcecb4fd7d002c905bd3cb0ac6`,
  `d4b04f645f06b35c61382b45f8691c2c2238964fbf408910b25aab8a0f4801c9`,
  `6086a0152e59acbbc958066048e1b6fb6a0caa7361d7a7dd3611c0e7aa27623f`.

Per-policy aggregate execution evidence is:

| Policy | Overlay fingerprint | Prepared binding fingerprint | Finite total loss | Gradient tensors |
|---|---|---|---:|---|
| independent | `0dced56d924d8d6a78116817b12bd09927cdbdc349326e8bbffebb372e95a20f` | `d26ec7aab9074c1afe7ae6811a82716192a2d7d8ce5c2d9136e5d5a702cbf1aa` | 3.1845867634 | 380/474 present and finite; 361 nonzero; mask token nonzero |
| onset | `26db9acecb944c1f0f7b47ed99d5d927c0e8f89f919e6bb1500bea7dd5ed6afe` | `82a29dc145278fe73935fa8c16bf184b2dc5f1bbaab699145ac5c9bc5446493f` | 3.0467743874 | 380/474 present and finite; 361 nonzero; mask token nonzero |
| beat | `6f6869d0f2c18f583ba92ac7a02e588c88aaa5170b70013ea29f8edbbd326b5c` | `f34a4409a5219a4b201c79c6a3efa2f57e30d2d3fdc2dd591bc1664c6b6d3956` | 3.0418236256 | 380/474 present and finite; 361 nonzero; mask token nonzero |
| bar span | `2b42a6133681397d4d985591653c5b3c28d6e2d81cdc818b1e95fdbd9c6e69e4` | `00d6d3a9f57aabdb9b265688c71f407c4009b921c5b3a395ff234aa764751f3c` | 3.1029169560 | 380/474 present and finite; 361 nonzero; mask token nonzero |
| track/bar | `14d06184e271b2a1a6b829fe13dce851c1678e53c495b329ae8312c3b373c4f2` | `1de58b57fceb790a4428a653d42dac08c9761b8bc0b70c522497f9b2c22d62cf` | 3.2086246014 | 380/474 present and finite; 376 nonzero; mask token nonzero |

All 380 present gradients contain 19,008 finite elements. Absence on the
remaining trainable parameter tensors is expected for this one bounded batch;
this is mechanics evidence, not an optimization or quality result.

Each single-policy acceptance has frequency `4/4`. The separate all-policies
mixture smoke on the same four identities resolves track/bar `1/4`,
independent `2/4`, and onset `1/4`; all five policies were eligible for every
piece. These four draws demonstrate explicit denominator/accounting, not an
expected-frequency claim.

## Complexity and benchmark boundary

With contract-fixed `K = max_span_bars <= 8`, index construction, hierarchy
resolution, and planning are:

```text
O(nodes + relevant edges + emitted candidate/mask entries)
```

The implementation creates no dense node-to-unit or note-to-note matrix, no
simultaneous-note clique, no all-pairs note loop, no full `O(B²)` span set,
and no dense tracks-by-bars enumeration.

The Phase 8A benchmark separately records plan/index construction, descendant
resolution, overlay construction, prepared forward, raw node/edge/candidate/
emitted counts, and canonical serialized bytes retained by portable plan
metadata. `peak_retained_batch_plan_metadata_json_bytes` is the sum of the
four simultaneously retained compact plan serializations; it is a
deterministic retained-state proxy, not a Python allocator or temporary-memory
measurement. The benchmark has no speed threshold. Serialized retained bytes
are not a claim about Python allocator peak, CUDA peak, or total process
memory.
No Phase 8A GPU performance claim is made; CUDA tests run only when CUDA is
actually available and otherwise report an honest skip.

The bounded CPU benchmark (`repeats=3`, 147 nodes, 920 edges, 208 relevant
edges) recorded:

| Policy | Plan mean s | Index mean s | Descendant mean s | Overlay mean s | Binding mean s | Forward mean s | Candidates | Primary/peer/track/overlay entries | Peak retained plan JSON bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| independent | 0.063186 | 0.000078 | 0.000003 | 0.032556 | 0.120794 | 0.066883 | 57 | 15/42/9/138 | 5,604 |
| onset | 0.040429 | 0.000075 | 0.000010 | 0.051167 | 0.176619 | 0.050294 | 27 | 18/39/9/147 | 10,621 |
| beat | 0.040737 | 0.000088 | 0.000010 | 0.033545 | 0.170797 | 0.068648 | 26 | 18/39/9/147 | 10,612 |
| bar span | 0.037638 | 0.000091 | 0.000011 | 0.035470 | 0.156378 | 0.057900 | 14 | 23/34/9/162 | 10,555 |
| track/bar | 0.035645 | 0.000075 | 0.000016 | 0.039298 | 0.179825 | 0.060720 | 35 | 17/9/4/93 | 10,445 |

Times are observations from one bounded CPU run, with no acceptance threshold
or throughput claim. CUDA was not available and was not measured.

## Acceptance commands and claim boundary

Focused and regression evidence is produced with:

```bash
.venv/bin/python -m pytest -q tests/ssl/test_hierarchy_fixture.py \
  tests/ssl/test_hierarchical_masking.py \
  tests/ssl/test_hierarchical_prepared_binding.py
.venv/bin/python -m pytest -q \
  tests/ssl/test_data_config.py::test_workers_zero_and_two_preserve_per_identity_inputs_and_plans
.venv/bin/python scripts/accept_phase8a_hierarchical_masking.py
.venv/bin/python scripts/benchmark_phase8a_hierarchical_masking.py
.venv/bin/python -m pytest -q tests/ssl
.venv/bin/python -m pytest -q tests/models tests/graph
.venv/bin/python -m pytest -q \
  tests/ssl/test_checkpoint_training.py tests/ssl/test_transfer.py
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q src scripts tests
git diff --check
git show --check
```

Final local results on the implementation tree were:

- focused Phase 8A: `84 passed, 1 skipped`;
- workers `0/2` parity: `1 passed`;
- complete SSL: `241 passed, 3 skipped`;
- unchanged Phase 6 model/graph/leakage regressions:
  `146 passed, 1 skipped`;
- checkpoint/resume/encoder-transfer regressions: `19 passed`;
- full repository: `1073 passed, 22 skipped`;
- two independent compact acceptance builds: byte-identical 32,229 bytes
  with SHA-256
  `e6915779f21784a1907c930da7967d2d6c1dae4cfd72fbb0ed5c24bec37cc03a`.

CUDA was unavailable. The Phase 8A CUDA path and other repository CUDA-only or
explicit local-real-data tests skipped; no GPU or corpus-scan evidence is
claimed.

The benchmark and bounded acceptance establish deterministic mechanics only.
No HookTheory full scan, POP909-CL 909-file acceptance, production cache
rebuild, PDMX projection, production/full-corpus SSL training, PLL, critic
training, or quality/likelihood interpretation is part of Phase 8A.
