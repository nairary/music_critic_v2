# Minimal Torch2.2 compatibility handoff005

Scope: fix the concrete native004 test/setup failures; preserve sealed004,
all training/evaluation protocols and historical experiment fingerprints.

The coordinator's native004 log records697passed/4failed/39warnings in137.63s.
Three tests called the absent Torch2.2 `torch.amp.GradScaler` API (four call
sites); one tried to monkeypatch absent serialization safe-global hooks.
Those tests now use the existing production `_scaler()` factory and
`raising=False` for the forbidden optional API hooks. No production scaler,
optimizer or checkpoint-loading behavior changed.

The metadata setup failed `fusion_pilot.runtime_identity_required`.
`torch.__version__` is a `TorchVersion` string subclass; the runtime validator
deliberately accepts plain JSON strings. `runtime_observations` now normalizes
this one observed value to `str` when it already is a string instance. Its
contents and canonical runtime fingerprint remain identical. Missing, empty
and non-string identity still fail. GPU identity was not observed empty;
the earlier fallback hypothesis was withdrawn and no `nvidia-smi` fallback,
fake value or admission bypass was added.

The new regressions cover the complete mocked Torch2.2-shaped
`capture_runtime → validate_runtime_policy` path, unchanged fingerprint and
rejection of invalid identity. CPU-only fixtures do not establish native005
acceptance. Exact final tests and fingerprints are in the handoff README.
Local final acceptance: **706 passed,0failures/skips/deselections,
1 existing PyG warning,115.65seconds**. The focused nine-case regression,
compileall, diff check and48-record registry/ledger checks also pass.

Apply `incremental.patch` only to sealed004, or deploy the verified complete
`source.tar.gz` into `/home/humtech/music_critic_research_20260913/fusion-005-source`.
Place the005 handoff in the sibling `fusion-005-handoff` directory. The README
contains the unchanged setup/reference/pair commands with these005 paths.
Regenerate the source manifest and requests from the005 source; do not reuse
a request pinned to004 source bytes. All original004 package files remain
unchanged. Failed native004 test/setup logs remain with the coordinator and
are hashed in `review005-acceptance/coordinator-native004-inputs.json`.

No legacy logic was inspected/adapted, no main checkout was changed, and no
SSH, training, actual CUDA execution, real payload/checkpoint tensor read or
TEST access occurred here. No new gate abstraction or experiment was added.
Next action: coordinator reruns the bounded native CPU/setup checks and then
the already agreed registered reference gate and P512 pair.
