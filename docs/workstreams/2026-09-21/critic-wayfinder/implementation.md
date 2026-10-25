# Critic v1 implementation closure — 22 September 2026

Engineering implementation of [spec.md](spec.md) is complete. Native admission and
scientific validation are not complete. [Public interfaces](../../../CRITIC_V1.md).

## Scope and source identity

Base: `main`, HEAD `287bf856643f9e555352e1661a7bf0563772178a`; current uncommitted
working tree, including untracked files. The index was not changed. Pre-existing
work is retained; the earlier DLC semantic repair and other experiments are not
claimed as part of this critic implementation. No legacy logic inspected/reused.

Code/test inventory fingerprint (SHA256 of sorted compact JSON path→SHA256):
`03a3250ac4a4663d6fe9593eb6aa952d8b45f9bd4245735bfd30fe70d09e5aaa`. Individual current-file identities:

| File | SHA256 |
|---|---|
| `src/music_critic/critic/__init__.py` | `14a6b0a07397221be039e688e1517a3398c2c3b4ea35a6fd5bde0b633c7516c8` |
| `src/music_critic/critic/baselines.py` | `65ea30cc73e3573b54ab9f18402b1773f0ccc86fd5ce6650d4c66959d3de2809` |
| `src/music_critic/critic/cache.py` | `fef647b665dde3b38afa50215bcaa0244cf2fa3c22baaf011300998b7bad28eb` |
| `src/music_critic/critic/contracts.py` | `06161122aaac82493ffa8812f23178bae3ff7379877f1a1d5fc850ac905d0fdf` |
| `src/music_critic/critic/features.py` | `0b5e5763d76f66c84093087164dbe0f68114e10021b69b79b65db2742c1ce494` |
| `src/music_critic/critic/inference.py` | `4bc9c8ac916f8e6971a9bd9c3b4325289c570749cec04cd9ba6e82f89a93b728` |
| `src/music_critic/critic/loss.py` | `b110e3e79f7f97f630af38e1913b00afad936b03cd17b38bda018b944290fdf6` |
| `src/music_critic/critic/model.py` | `35f1493802f1d5210a0ee5984ebbd2f6f1aabc5f79f30084fb4c3a20ca5e91a1` |
| `src/music_critic/critic/scoring.py` | `14610b0f65b1e2027690e2458b13a31344481bc2a9e79e961c2c1ed7ea7de493` |
| `src/music_critic/critic/targets.py` | `07a21834d9508f4d6dff7d1be59fa7f86d2a4ec9c6ea0efd1d08bcae8861f48c` |
| `src/music_critic/critic/training.py` | `3de2e0de91382c70aa6df0fe244d40f42468ee086ab2649a0049fcb6e18d41fa` |
| `src/music_critic/experiments/analysisgnn/temporal_slices/model.py` | `2b4c55fe19172b924990c5d70fc6b09910d981769091eb45fa53bf7d6be632e8` |
| `tests/critic/test_baselines.py` | `fb3dda384dd61074da9a64a81d4356a2f776ed865c1b514c63e9d3f4ea3c6f12` |
| `tests/critic/test_cache.py` | `f87668d12061c107d7a1ee821c3a9df6922ea9c42010e08499d4b8b501f254d6` |
| `tests/critic/test_contracts.py` | `f78d14a6574460d602c956abf03b729ccf8976d6cbb1b5a397eb3a75a71aa1bf` |
| `tests/critic/test_features.py` | `35d5cdd10109add080334fbffd089070e69ad5e2589c7e52bbbd7edbe684e7e1` |
| `tests/critic/test_inference.py` | `67941cbdf53c7c713cd7ca59fdcc311548f0099b2e054147a0dca5986511d8d1` |
| `tests/critic/test_integration.py` | `97d9bb306646230da832378d9e98a7a09977fe70472021504b9fa5b608dac396` |
| `tests/critic/test_loss.py` | `8b35cc5f01d47d8c784dfdc01ceb55ceb7583e35afb08a5af53c5bbf6e0ee207` |
| `tests/critic/test_model.py` | `34df76892cda15d90a54ccc8d2e766f52a410b188dda2ced990c59d42d10f4e8` |
| `tests/critic/test_scoring.py` | `449f8f3ee6dc9f23665372af9ccd6cd47b7b8b224b325f20f004a45e0148e7ca` |
| `tests/critic/test_targets.py` | `e3947474a3ff4b2aa0cd8807548ea4cb0f9d60264521bffa3aebcf4ab4cc3598` |
| `tests/critic/test_training.py` | `b8c56128d1d9a94a973ccb89bb5946675711aff69da0d52edad78a5d04ef418c` |

## Behavior and verification

Same-pass C2/temporal export, exact geometry and bound cache feed three recurrent
context paths, gated multiscale refinement, four aspect readouts and local
intervention heads. Verified sidecars and masked losses stay outside predictor
inputs. The trainer owns only critic parameters, samples work/aspect/pair and
persists optimizer/RNG/sampler/selection. Reference scaling binds actual predictor
weights and encoder features. One-MIDI inference exports exact coordinates and
explicit unavailable values. Three global controls use the same pair sidecars.

Final combined command:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 /tmp/music-critic-pcset-v2-env/bin/python -m pytest \
  tests/critic \
  tests/experiments/test_temporal_slices.py \
  tests/experiments/test_temporal_slice_checkpoint.py \
  tests/experiments/test_phase9eb5c_corrected_model.py \
  tests/graph tests/adapters/test_midi.py \
  tests/experiments/test_genre_interventions.py \
  tests/experiments/test_music_interventions.py -q
```

**335 passed, 1 warning, 15.21 seconds**:93new critic tests and242existing affected
checks. The warning is an existing Torch JIT deprecation, not a critic failure.
No dependencies were installed; the existing temporary environment was reused.

Test-first cycles recorded missing modules/behavior before implementation, then
green checks. Material regression cycles included incomplete-bar spans, empty
track pooling, malformed metadata, no-op targets, zero-duration deletions,
global-only controls with local sidecars, stale reference scales, raw bank
mutation, head-dev overlap/cohort changes and inference-mode tensor loading.

Integration checks fit a single synthetic documentation-fixture pair through the
actual frozen encoder/new critic/targets/trainer and verify lower loss and expected
pair ordering after70updates. Saved CPU continuation reproduces the next step
and weights exactly. Encoder weights remain unchanged. A12bar synthetic example
exercises all2/4/8span gates/queries. These establish mechanics, not music quality.

## Independent review

- **Standards:** no remaining actionable findings in feature/cache/inference,
  scoring/training and their tests. Exact TRAIN sample/hash overlap and changing
  head-dev cohort findings were fixed; three focused regressions independently passed.
- **Spec:** no remaining required findings in non-owned critic modules and their
  integration. Predictor/reference identity, no-op pairs, common pair scope,
  lambda-zero controls, separate local metrics and CPU RNG resume were checked;
  nine focused regressions independently passed. Own feature/cache/export code
  was excluded from this reviewer's independent verdict and covered by Standards.
- Root integrated the slices, verified the full combined suite and resolved the
  additional bundle-under-inference-mode regression. Reviews read WIP/untracked
  content directly, against the base above; unrelated dirty files were excluded.

## Remaining boundaries and next phase

No corpus records, retained real checkpoints, real-data inference/training, TEST
payloads, native GPU run, cloud operations, commit or push were performed.
Untrained prototype weights in tests are not a delivered trained critic.
Local intervention evidence is not validated musical-error localization.
The raw track is not a semantic voice; current structure scope is local order.
CPU tests do not establish CUDA memory, determinism or throughput.

Next: bind a retained main checkpoint/source, inventory role-eligible TRAIN
operators/works, freeze work/component splits and reference horizon, cap the
feature bank and registered pilot, then admit the selected native environment.
Changes, PLL, backbone unfreezing and human-quality calibration remain separate.
