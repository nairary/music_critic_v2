# HIERGRAPH PR47 integration — 30 October 2026

The user explicitly requests promoting PR47 to main and updating the repository.
Work is performed in the clean publication checkout; the original main contains
older uncommitted relation-probe work and will be preserved before fast-forward.
No force push, test bypass, new training, inference, corpus access or metric
recomputation is included. Legacy files are not inspected or reused.

## Failure reproduced

PR head f62bfb6c5b2e8c5007a670c877d54d7177f9544a failed Linux CI run
40948059509: 2 failed, 6963 passed, 66 skipped, 335 errors, 1991 subtests passed.
The primary audit failures were direct CUDA peak resets outside the shared
lifecycle helper and an unregistered heavy import in an opt-in relation-probe
runtime. A C4 CUDA test patched backend descriptors individually; monkeypatch
undo attempted to delete an attribute not present in the backend module
namespace. Undo stopped before restoring its deterministic-algorithms stub,
causing the following autouse fixture failures.

Local red checks reproduced both audit failures and the teardown exception.
Two new H-member preflight cases failed specifically on bypassing the shared
CUDA lifecycle, for abstract cuda and concrete cuda:1 requests.

## Bounded repair

- H-member training resolves the requested device and calls the shared lifecycle.
- Relation-probe training and both HIERGRAPH native gates use the concrete model
  device for that lifecycle. Peak allocation and gate synchronization use its
  logical integer index. GPU naming also resolves the configured device.
- The C4 preflight test patches whole backend namespaces instead of descriptors,
  and its deterministic stub accepts restoration keyword arguments.
- The import audit permits Torch/PyG only in relation_probe/model.py and runtime.py;
  a fresh subprocess rejects those heavy dependencies during public/data imports.

The source-wide audit still requires exactly one direct peak-reset call, inside
cuda_memory.py. The source snapshot and conditions used for finished experiments,
immutable terminal records and scientific results are unchanged. The known
full-model tuple finalization defect and directed-link coverage failure remain.
The October29 scope manifest remains the identity of the initial publication;
this follow-up is the explicit integration delta.

## Verification

Independent Standards and Spec reviewers found no blocker in the six-file code
and test delta. Each ran the CUDA-boundary/repository-contract suite: 15 passed.
Broader local regression and Linux CI results are recorded below. macOS testing
uses the existing publication Python3.12 environment and TMPDIR=/private/tmp.
The initial sandbox run could not invoke ps for synthetic child-process cleanup;
that permission limitation is not a model/runtime failure.

Broader local regression passed: 264 tests, 2 hardware-only skips, 109 subtests,
6.20 seconds; one existing TorchScript deprecation warning. It covered CUDA
lifecycle/API boundaries, repository contracts, H-member training/reproducibility,
relation-probe suites and all test_hiergraph_* experiment modules. Synthetic
process cleanup checks required ps permission outside the sandbox. Full Linux
CI is pending the repaired head; a merge requires its successful completion.
