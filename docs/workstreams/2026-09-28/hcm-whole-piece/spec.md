# Whole-piece HCM implementation

Status: implemented with synthetic verification and independent review; see
[completion](completion.md). Motif scales include 1, 2, 4, 8, 16 and 32 bars.
Scope is implementation and synthetic verification.
No new real-data training budget or TEST access is implied. Source baseline:
main HEAD 287bf856643f9e555352e1661a7bf0563772178a, pre-existing dirty/untracked
work retained under ignored outputs/hcm_whole_implementation_20260928/baseline.

## Behaviour and boundaries

- Whole supplied source pieces form the observed context. Never silently crop
  to the old 16/32-bar windows. Metric 1/2/4-bar mask targets remain unchanged.
- Explicit resource profiles reject unsupported pieces without truncating them.
  Initial maximum is 512 bars/65536 notes, with inherited per-tile/codec limits.
- Full-song provenance is distinct from complete ingestion of an excerpt.
  Training defaults to evidenced full pieces; excerpts/unknown remain explicit
  exclusions unless a separate opt-in manifest policy admits them.
- Support source-bound canonical candidates from Dilemmadata, HookTheory,
  POP909-CL and PDMX. Preserve exact source geometry and grouping. Do not rebuild
  supposed complete songs from clip unions or invent missing music/labels.
- Mask raw target events before any graph, aggregate, motif key or selection.
  Preserve exact rational onset ownership, carry-in and unknown-versus-rest.
  TEST remains closed, split/group metadata is checked before payload reads.
- Retain current event projection/EOS and normalized conditional decoder.
  Neither new likelihood nor successful reconstruction establishes preference.

## Module seams and ownership

`whole_contracts.py`: immutable WholePieceConfig, effective context profile and
`split_whole_piece(piece, query, *, whole_config, context_profile)` producing
the existing MaskedExample with full observed source coverage. Strict config
validation/fingerprint; no neural modules in configuration loading.

`whole_data.py`: WholeDataManifest, WholeMaskSampler and bounded registered
preparation from canonical candidates with full/excerpt/unknown evidence.
Reuse immutable manifest hash, work-balanced sampling, draw/RNG state and
strict TRAIN boundaries. A new `prepare_pll_whole_sources.py` exposes preparation.

`whole_memory.py`: WholeSongMemory creates learned positional bar slots,
track/coverage-aware local states, 8-bar block summaries and whole-song memory.
Output local/global ContextMemory branches plus pre-global [bars,slots,width]
features, raw fine metadata and query states. Core ownership excludes halo
duplicates. Initial width is ModelConfig.width (256), four bar/block slots,
eight song slots, two block/global layers; all are explicit profile values.

`motifs.py`: overlapping bar candidates at 1/2/4/8/16/32 and short beat motifs,
pre-global content keys and observed-context query; all-candidate soft+null
attention supplies differentiable retrieval. At most four nonredundant
candidates expose 64 detail tokens each through positional alignment. Empty
bank remains finite. Candidate and selection diagnostics expose every scale
for later ablation; no hard-wired preference for or guarantee of long motifs.

`whole_model.py` / `whole_decoder.py`: opt-in WholePieceHCM, fully frozen local
encoder option, separate local/global/motif attention paths, existing field
codec and typed relations. Whole mode gets a distinct model contract. Existing
HCM1 and defaults remain available. Scales/global/motifs can be disabled through
bound configurations for future matched comparisons, not by rewriting results.

CLI dispatch selects strict whole manifest/sampler and model for a versioned
whole configuration. Checkpoint binding includes the complete new architecture
and sampler profile. Strict resume is not old-HCM-to-new-architecture transfer.
If weight-only warm start is provided it must be explicit, hash-bound, validate
every reused key/shape before mutation and start a fresh optimizer/schedule.

## Acceptance checks

1. A synthetic piece >32 bars remains whole; over-budget pieces fail explicitly.
2. Replacement/deletion of hidden targets leaves complete context, bank,
   selected candidates and encoder memory unchanged in eval mode.
3. 16/32-bar motifs occur when geometry allows; missing support is reported.
   All-short-scale configuration is a separately fingerprinted ablation.
4. Positional bar aggregation distinguishes order; distant visible changes can
   influence whole memory; no cropped-prefix or duplicate-halo aggregation.
5. Unknown metric cells differ from observed rest; carry-in sustains remain.
6. Soft retrieval/key/query and selected detail paths receive finite gradients;
   all-masked/empty-bank inputs remain well defined.
7. Real in-process synthetic C3→whole memory→motifs→decoder loss/backward and
   generation work; hidden-future target attributes cannot leak to predictions.
8. All four source adapters' canonical candidates preserve raw-only inputs,
   source/split/coverage provenance; malicious paths, changed hashes and
   heldout component declarations are rejected before music access.
9. Registered synthetic train/save/resume reproduces the next update including
   RNG and sampler; changing motif scales/profile invalidates resume.
10. Existing PLL tests remain green, plus independent Standards and Spec review
    against WIP file snapshots. No new dependencies or legacy runtime imports.

## Completion versus next phase

Implementation completion requires the interfaces, CLI wiring, documentation
and tests above. Native 3090 resource admission, complete-corpus availability,
cross-source deduplication, real training and 16/32-bar usefulness remain
separate registered work. Do not infer them from synthetic tests. Update
ARCHITECTURE, DATA_CONTRACT, DECISIONS, HCM documentation, roadmap and STATUS.
