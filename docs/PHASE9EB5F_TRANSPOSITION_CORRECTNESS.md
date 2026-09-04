# Phase 9E-B5F transposition correctness

## Outcome

Status: `implementation_or_contract_defect`.

The B5A forward transformation and B5C runtime routing agree, but the physical
raw-graph round trip is not invertible for `shift_pc=6`. The corrected orbit
chooses `SIGNED_BY_SHIFT_PC[6] = +6`. The prescribed inverse PC is
`(-6) mod 12 = 6`, which chooses `+6` again. Thus a non-drum MIDI pitch `p`
follows `p -> p+6 -> p+12`, rather than returning to `p`. Semantic tritone
labels use an involutive within-vocabulary pairing and do return correctly, so
the defect creates a raw/target inverse asymmetry.

B5F records the defect and does not repair production code. C0 remains the
current baseline, C1 remains `experimental_deferred`, and
`ready_for_soft_augmentation=false`.

The defect is confined to the B5F inverse diagnostic: historical B5D/C1
training called only the canonical forward transform. Its `shift_pc=6` view
was the intended `+6` view, so the completed C1 result and its negative
directional comparison remain valid. B5G adds the missing directed inverse;
it does not rewrite this historical B5F evidence fixture.

## Executable matrix

The only transformation contract is B5A
`src/music_critic/experiments/analysisgnn/transposition.py`.

| behavior | heads |
|---|---|
| absolute equivariant | `local_key`, `tonicized_key`, `root`, `bass` |
| pitch-class-set equivariant | `pitch_class_set` |
| relative invariant | `primary_degree`, `secondary_degree`, `quality`, `inversion`, `roman_numeral`, `note_degree` |
| structural invariant | `harmonic_rhythm`, `cadence`, `phrase`, `section`, `metrical_strength` |
| boolean invariant | `pedal`, `chord_tone`, `is_root`, `is_bass` |

The audit compares this expected matrix to `transformation_registry()` rather
than maintaining another runtime mapping.

## Independent checks

For non-drum notes the oracle computes pitch by signed integer addition,
pitch class modulo 12, octave by floor division, and track-relative pitch from
the shifted track distribution. Categorical fields are exact; recomputed
continuous values use deterministic absolute tolerance `1e-6`. Only
`note.pitch`, `note.pitch_class`, `note.octave`, and
`note.track_relative_pitch` may change.

Absolute labels are independently parsed to pitch class; key mode must remain
stable. Pitch-class sets are independently shifted and sorted. Relative,
structural, and boolean semantic values, availability, masks, missing reasons,
entity IDs, relations, topology, and provenance remain exact. Missing relation
context is emitted as `not_checkable`.

The source-free regression passes transformed graphs through
`transpose_raw_graph_batch`, model forward, and
`align_target_sidecars_after_prediction`. It proves graph-before-forward and
target-after-logit ordering for all 18 active routed heads. `phrase` and
`section` remain classified deferred metadata heads.

## Corpus and artifacts

The production audit streams every TRAIN and VALIDATION record through all 12
shift-PC values:

```text
TRAIN       1295
VALIDATION   162
total       1457 records
pairs      17484 record-shift diagnostics
```

The ignored production directory is:

```text
outputs/phase9eb5f/analysisgnn-transposition-correctness/
```

It contains `record_shift_diagnostics.jsonl` and `audit_summary.json`. The
repository fixture contains only compact summaries and fingerprints. No TEST
record is loaded or evaluated, and no checkpoint, dataset, cache, or large log
is committed.

| shift PC | eligible | invalid | round-trip failures | runtime mismatches |
|---:|---:|---:|---:|---:|
| 0 | 1457 | 0 | 0 | 0 |
| 1 | 1430 | 27 | 0 | 0 |
| 2 | 1452 | 5 | 0 | 0 |
| 3 | 1446 | 11 | 0 | 0 |
| 4 | 1438 | 19 | 0 | 0 |
| 5 | 1455 | 2 | 0 | 0 |
| 6 | 1439 | 18 | 1439 | 0 |
| 7 | 1453 | 4 | 0 | 0 |
| 8 | 1442 | 15 | 0 | 0 |
| 9 | 1445 | 12 | 0 | 0 |
| 10 | 1448 | 9 | 0 | 0 |
| 11 | 1411 | 46 | 0 | 0 |

All 1,457 physical shift-6 graph views exhibit the `+12` raw round-trip
mismatch; 18 are already ineligible for target/vocabulary reasons, leaving
1,439 scientific round-trip failures among 1,439 eligible pairs. The real
TRAIN runtime regression covers shifts 0 and 1 for
`dlc:corelli:op03n04c`, all 18 active routed heads, and has fingerprint
`b517f197278161623cf8b87c74edd81e6ecafccc85e92f8ff8601ff783114b49`.
All executable cross-head checks have zero failures. Context absent from a
record is explicitly counted `not_checkable`: absolute heads 12,
degree/Roman-with-key/root 912, inversion-with-root/bass 84,
note-degree-with-pitch/key 228, pitch-class-set 13,453, and target vocabulary
closure 168 record-shift cases. The full summary fingerprint is
`3a724ab26fd9d1a149d67eda9795e261d600173d556e6a9591fbb39b0dd5cfb6`;
compact evidence and fixture fingerprints are
`3b6656dbde15f5f826a8d4e874a890e0e3a6b6144aa51df02e811a976006d3fc`
and `406e2f2c6dfc05c747f2b90c7fa4a1ba163b01fe8a6e7e9d62cad81899a71aa7`.

## Seed-17 C1 schedule

The production deterministic sampler reproduces 20,000 draws at seed 17, batch
size 2, and 10,000 applied updates. C0 and C1 record order is identical.

| shift PC | signed | draws |
|---:|---:|---:|
| 0 | 0 | 1539 |
| 1 | +1 | 1624 |
| 2 | +2 | 1630 |
| 3 | +3 | 1682 |
| 4 | +4 | 1690 |
| 5 | +5 | 1650 |
| 6 | +6 | 1706 |
| 7 | -5 | 1747 |
| 8 | -4 | 1636 |
| 9 | -3 | 1641 |
| 10 | -2 | 1781 |
| 11 | -1 | 1674 |

The identity fraction is `0.07695`, normalized shift entropy is
`0.9997385632110344`, and 64 TRAIN records have fewer than 12 valid shifts.
Fingerprints:

- record schedule: `67f4401806f2d5419bb849449aef811fd54dfbca62588c5a1543dbbe6c1b63f8`;
- C0 shifts: `af937f0ece2ffc459a093b5d8a19be815c4159653b545059eee723c3bc71bb2b`;
- C1 shifts: `745aef3bf213228635bbd4926a5f9d61f4dc26a425434b3757535eeccae4ef4a`.

## Checkpoint diagnostics

Local Torch is CPU-only and the sealed update-10,000 B5D C0/C1 checkpoints are
not present. Therefore `checkpoint_diagnostics_run=false` and
`shift0_metrics_reproduced=false`; per-shift performance values are not
fabricated. The checkpoint runner itself is exercised over both profiles, all
12 shifts, and all 18 active routed heads with its in-memory source-free smoke:

```bash
.venv/bin/python scripts/run_phase9eb5f_analysisgnn_shift_diagnostics.py \
  --smoke --device cpu --output-dir outputs/phase9eb5f/smoke
```

On the RTX server, from the repository root and with the original B5D outputs:

```bash
cd "$HOME/music_critic_v2"
git fetch origin phase/9eb5f-analysisgnn-transposition-correctness
git switch phase/9eb5f-analysisgnn-transposition-correctness
git pull --ff-only origin phase/9eb5f-analysisgnn-transposition-correctness

export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

.venv/bin/python scripts/run_phase9eb5f_analysisgnn_shift_diagnostics.py \
  --c0-checkpoint outputs/phase9eb5d/c0-seed17-full-u10000/last.ckpt \
  --c1-checkpoint outputs/phase9eb5d/c1-seed17-full-u10000/last.ckpt \
  --device cuda \
  --output-dir outputs/phase9eb5f/checkpoint-shift-diagnostics
```

The runner fails closed unless profile, seed, applied update, full-training
contract, record schedule, architecture, model state, and TEST-lock metadata
match B5D/B5E. It evaluates the same 162 VALIDATION records under each shift;
these views do not multiply independent support. Shift zero must reproduce the
B5E primary and joint metrics within absolute tolerance `1e-7` before the
artifact is valid. The results are diagnostic and cannot select a model.

## Reproduction

```bash
.venv/bin/python scripts/audit_phase9eb5f_analysisgnn_transposition_correctness.py --check

.venv/bin/python -m pytest -q \
  tests/experiments/test_phase9eb5a_transposition.py \
  tests/experiments/test_phase9eb5c_corrected_training.py \
  tests/experiments/test_phase9eb5d_full_training.py \
  tests/experiments/test_phase9eb5f_transposition_correctness.py \
  tests/audit/test_phase9eb5f_transposition_correctness_audit.py \
  tests/test_repository_contract.py
```

The full corpus audit is rerun only into a new empty output directory. It does
not train a model and never opens TEST targets.

The targeted suite passes `74 passed, 2 warnings in 21.84s`. The complete
local repository suite passes
`1901 passed, 59 skipped, 12 warnings in 678.76s`; warnings are unchanged
upstream Torch JIT and multiprocessing-fork deprecations. Audit `--check`,
source-free checkpoint-runner smoke, compileall, and `git diff --check` pass.
