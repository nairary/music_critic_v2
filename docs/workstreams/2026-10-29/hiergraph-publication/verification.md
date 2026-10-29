# HIERGRAPH publication review — 29 October 2026

The user requested result description and a Git push. The topic branch
`codex/hiergraph-full-e3-20261029` starts at published main
`645922d4db282ff0c3a06c1cf82ecc9d0594f5c7`. A separate clean checkout holds
this publication. The original main checkout and detached research WIP retain
their working files and index. No force push or merge is included.

## Scope

Whole-piece HIERGRAPH E3 implementation, its target projection/data consumers,
earlier relation/H-member and E1–E5 dependencies and source history, tests,
configs, research/workstream reports and 176 new compact immutable records.
The registry contains 549 records; all previously published records remain
byte-identical. `scope.json` records file hashes, sizes and explicit exclusions.

Published CUDA lifecycle fixes, critic tests, hash monkeypatch scope,
multisource fixture checksum, CI workflow and release-navigation history are
preserved from main. README adds the current baseline and links. STATUS retains
the published history. Ledger and project catalog are regenerated. Datasets,
weights, MIDI/audio, caches, outputs, environment files and local agent state
are excluded. Artifact locators in records describe external evidence roots;
publishing records does not upload or locally verify remote weights.

## Results and known limitations

500 applied / 500 attempted / 0 skipped, 5 h 26 min, seed 17, selected update 50.
Dev family macro: H recovery 43.057%, joint H+F 13.525%, h recovery 10.337%,
P recovery 72.807%, participants recall 9.312%. Directed scoring coverage is
0 / 3638 known positives. These are weak source/model labels, exploratory dev
inside global TRAIN, one seed, unknown raw pretraining overlap. TEST remained
closed. Old window updates are not a matched comparison with whole-work updates.

The parent record remains immutable `failed`: after saving result.json and dev
metrics, strict registry sealing rejected tuple metric pairs. The source retains
this known finalization defect. Publication neither repairs nor relabels the run.
The final report distinguishes completed compute from failed finalization and
zero scoring coverage from a measured relation-head classification score.

## Verification on the publication checkout

- HIERGRAPH focused suite: 340 passed, 1 CUDA-only skipped, 118 subtests passed.
- Wider H-member/relation/supervision/shared-target/catalog suite: 211 passed,
  249 subtests passed. Total: 551 passed, 1 skipped, 367 subtests passed.
- All 101 changed Python files parsed in independent review. New absolute
  `music_critic` imports resolve in the publication tree.
- All 549 registry records validate; fingerprint
  `adf5a33c3668168757c3119c4876a57166a7162f5283cccb875f80e7988db704`.
- Ledger and project catalog generated and checked. Staged whitespace checked.
- A bounded scan of all publication paths found no private-key headers,
  GitHub/OpenAI tokens, credential URLs or literal password assignments.
  This is a bounded content check, not an exhaustive security guarantee.

Tests ran on macOS ARM Python 3.12. The original temporary test environment lacked
Hydra; a separate temporary environment installed the already-declared training
extra dependency. Wider tests initially rejected the macOS temporary-directory
symlink alias. With `TMPDIR=/private/tmp`, all passed without changing admission
checks. No package dependency declarations were changed. The existing TorchScript
deprecation warning remains. Linux full-suite CI is pending publication; prior
main CI is not a test result for this branch.

Focused command:

```sh
PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/experiments/test_hiergraph_*.py tests/test_hiergraph_full_data.py tests/test_hiergraph_supervision.py tests/tasks/test_hiergraph_projection.py tests/tasks/test_chord_structure_projection.py tests/tasks/test_chord_interval_projection.py --tb=short
```

Wider command:

```sh
TMPDIR=/private/tmp PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/experiments/test_hmember_*.py tests/experiments/test_supervision_stream.py tests/test_hmember_cache.py tests/test_relation_probe_*.py tests/test_harmonic_event_reasoner_model.py tests/test_tasks_lazy_facade.py tests/tasks/test_dense_harmonic_targets.py tests/tasks/test_dilemmadata_common.py tests/test_project_catalog.py --tb=short
```

## Independent review

Standards: no publication blocker. Focused inspection covered contracts,
authenticated admission, raw-only model forward, masks, unsupported geometry,
capacity rejection, immutable evidence and preservation of main fixes.

Spec: no publication blocker. Primary train/review records support the reported
counts, selection and dev metrics. No superiority or causal claim was found.
A minor stale implementation-status paragraph was addressed with an explicit
historical-snapshot label and a pointer to the final results.

Both reviewers inspected the staged inventory against the same main base. Their
coverage was focused, not exhaustive neural-model verification. The only changes
after review are publication/STATUS documentation, the historical pointer and
trailing blank-line cleanup. No implementation repair, legacy reuse, corpus
read or new train/inference run was performed for publication. Next engineering
work remains terminal JSON normalization and relation/address coverage diagnosis.
