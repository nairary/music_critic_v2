# Completion of the existing three research pilots — 2026-09-13

This review records the authorized completion work, before selecting a new
research direction. Historical C0/C2 checkpoints, data and training environments
are unchanged. New execution uses an isolated server source/environment and
immutable evidence under `outputs/research_resume_20260913/`. TEST is closed.

## Current scientific outcomes

| Workstream | Verified execution | Interpretation |
| --- | --- | --- |
| Context diagnostics | Synthetic20 frozen forwards and real32-piece local-control comparison completed | Descriptive context sensitivity and comparison, not a causal context or music-quality test |
| Logit fusion | All512updates and three full evaluations per arm completed; original process aborted; retained metrics and four exports checked separately | Final0.492768 control /0.468709 fusion; both best at update0; no improvement in this short pilot |
| Structural SSL | Both arms completed200 mixed plus100 recovery updates; all five full evaluations independently checked | Neither endpoint exceeds original C2; beat-group masking does not outperform random masking in this short seed17 pilot |

## Frozen C2 versus a simple local control

`EXP-CTX-C2-LOCAL-001`: fixed128 TRAIN pieces fit six nearest-centroid classifiers
on exact one-quarter-note raw features;32 VALIDATION pieces from31 official
components receive one frozen C2 forward each. Only the selected VALIDATION
observations shared by the two models enter each metric. No optimization of C2,
validation-based selection or TEST access occurs.

| Task | Local macro-F1 | C2 macro-F1 | Local accuracy | C2 accuracy | Observations |
| --- | ---: | ---: | ---: | ---: | ---: |
| Local key | 0.135170 | 0.425676 | 0.188715 | 0.789472 | 25,626 |
| Root | 0.237967 | 0.587355 | 0.397806 | 0.803526 | 25,525 |
| Quality | 0.089480 | 0.417442 | 0.183958 | 0.775392 | 25,658 |
| Primary degree | 0.035675 | 0.335861 | 0.086435 | 0.737463 | 25,684 |
| Cadence | 0.226259 | 0.294089 | 0.362676 | 0.711268 | 284 |
| Metrical strength | 0.101046 | 0.601345 | 0.239662 | 0.993557 | 49,357 |

C2 exceeds this specific local control on all six tasks. The experiment cannot
attribute that difference to long-range context: C2 had1295 training pieces,
whereas the local model used128, and their features and capacity differ. The
local control is deliberately simple, not a strong competing architecture.
Metrical accuracy also must not be read as broad rhythmic understanding.

The200 paired component bootstrap draws retain undefined rare-class results.
Macro-F1 intervals are conditional on only40–120 defined draws per task; they
are not unconditional95% confidence intervals over the population. Cadence
has labels in15 components and its conditional macro-F1 difference interval
crosses zero. No significance or general superiority claim is made.

Independent verification:47 retained artifact hashes; result/report fingerprints;
all12 accuracy/macro-F1 point metrics recomputed directly from observations;
32 completed forwards,0 unknown,160 raw/target records,6 local fits,0 updates.
Elapsed124.43s under the600s external guard; sampled whole-device peak1424MiB.
The prospective16384-note bound was fixed before launch using the completed raw
sizing audit; it did not change the selected population.

## Engineering acceptance and retained failures

- Native Linux ownership tests49 passed; the generic pilot wrapper and ownership
  tests62 passed without skips. The wrapper retains the child leader until owned
  group cleanup, then reaps it. Resource sampling is separate from scientific
  acceptance and does not claim a kernel-enforced GPU allocation limit.
- Fusion CUDA002 completed24 updates/48 draws with exact same-process fresh-object
  state restoration in both arms. All216886 fusion scalars changed;12 output
  heads have nonzero residuals. This does not establish fresh-process restoration
  or real-data efficacy. CUDA001 failed before Torch import because the coordinator
  supplied a record with a mismatching filename; that failed attempt is retained.
- SSL CUDA001 failed after C2 load but before any encoder forward/update because
  the historical Torch2.2.2 API differs from the newer autocast inspection API.
  The compatibility-only retry002 passed:8 updates,24 forwards,0 unknown,
  identical initialization and exact replay in both arms. Both final checkpoint
  hashes and durable attempt counts were verified. Total11.43s, peak1422MiB.
- Independent local tests:249 new context tests,58 compatibility tests and126
  scientific SSL tests passed. Native Torch2.2.2 compatibility tests58 passed.
  These establish only their tested scope; historical Torch2.2.2 remains below
  the repository's declared minimum, which was not changed.
- All1457 permitted cache entries were independently hash-bound. The original
  conversion from corpus files was not rerun, so annotation semantics and
  historical source-conversion provenance are not inferred from cache integrity.

## Integrated implementation and actual SSL input preparation

The developer checkout now includes93 accepted source/test/config/script files
and13 workstream documents without conflicting changes. Sealed server packages
remain separate execution identities; their pinned plans are not reused against
the newly combined source tree. General status, decisions and terminal records
were not overwritten by child worktrees. No dependencies or default model changed.

The combined targeted selection produced1596passing tests,18platform/CUDA skips
and27setup errors from a missing historical planning-record fixture. Moving that
immutable example to tests/fixtures removes the live-registry dependency; all28
affected module tests then passed. All1623 runnable selected tests are therefore
covered by passing runs. Native fusion005 separately passed706tests in143.26s.

SSL PREPARE002 completed2457raw reads and762sidecar reads in1804.81s.
There are1174jointly eligible TRAIN pieces;121 exceed the prospective4096-note
per-record bound. All162VALIDATION pieces and both policies remain available.
The preparation had zero model forwards/updates/TEST reads. This is a conservative
training subset, not a new representative full-corpus claim.

The actual SSL baseline evaluation completed162batches/810encoder forwards and
reproduced the historical C2 best primary macro-F1 exactly:0.49666884914040565.
The subsequent fixed200+100updates per arm run under a prospective7200s external
wall bound; scientific data, schedules, objectives and update budget are unchanged.
Completed run identity:EXP-SSL-GROUP-SCIENTIFIC-PILOT-001.

| Masking policy | Initial primary macro-F1 | After200 mixed updates | After100 further theory-recovery updates |
| --- | ---: | ---: | ---: |
| Independent note pitch | 0.496669 | 0.496348 | 0.494803 |
| Beat pitch descendants | 0.496669 | 0.493336 | 0.493686 |

Neither policy improves the initial primary score at either fixed endpoint.
Beat-group minus random masking is−0.003012 after mixed updates and−0.001117
after recovery. This short single-seed result does not support promoting
beat-group masking or these SSL endpoints as a new baseline. It does not
establish a general failure of SSL or of structured masking.

Independent verification checked600updates/605contiguous journal operations,
1400training plus4050validation encoder forwards,22referenced artifacts,
all five full162-piece memberships,18-head metric aggregation and checkpoint-
to-evaluation model fingerprints. Model/optimizer tensors are finite, with
no replay or unknown update completion. Total supervised wall time5072.73s.
The completed compact record retains69local evidence artifacts. A record-only
absolute-locator rejection was corrected before publication; no model run or
scientific artifact was repeated or changed.

This comparison has no separate matched theory-only continuation arm. It can
compare the two masking policies under the specified mixed objective, but
cannot isolate the causal contribution of SSL relative to ordinary continued
supervised training. Falling reconstruction loss is also insufficient: latent
targets move with the shared encoder. The first arm's song-projection mean
cosine increases from0.98800 to0.99933 after mixed updates; preserve this
homogenization diagnostic without claiming that the whole backbone collapsed.

## Completion boundary

No new melody, voice, rhythm, section, arrangement, shifted-window architecture,
PDMX-scale training or120k repetition is introduced in this task. The completed
bounded fusion result is now reviewed alongside these results; future work
awaits discussion. No legacy logic, commit, push or merge.

## Fusion report-assembly abort

Both arms completed512updates,1024TRAIN draws and486VALIDATION record reads.
Independent local verification checked all10104hash-chained journal events,
contiguous completed updates, identical draw events and six full evaluation
completion counts. The supervisor intentionally stopped the process later:
3777.06s total, swap6300→269112KiB, growth262812KiB exceeded the256MiB limit.
GPU peak2598MiB and minimum available hostRAM14615704KiB stayed inside their
bounds. Owned-process cleanup completed; this is not evidence of a server crash.

Original `EXP-FUSION-P512-CONTROL-001` and `EXP-FUSION-P512-ENABLED-001` are
terminal `aborted`. `run/report.json` and `run/artifact-manifest.json` were never
published. Both arm summaries, all checkpoints, four model exports, comparison
and journal remain on the execution host. Each arm summary is about3.26GB.
The exact allocation responsible for the swap event is still being reviewed;
post-journal report fingerprinting and serialization duplicate the large result.

Reported best scores are both0.496723; fixed final control0.492768 versus
fusion0.468709 (delta−0.024059). At incident detection these were only reported
values. The later posthoc section below records the separate verified retained
aggregation, without rewriting this attempt or retraining.

## Verified retained fusion outcome

`EXP-FUSION-P512-POSTHOC-001` checks existing evidence only. The streaming
pass exited0 after696.62s with27040KiB maximumRSS observed before compact
publication, under512MiB address-space and3600s wall limits. It hashed and
syntax-checked both3.26GB arm summaries, retaining a1.11MB projection.
Per-target prediction arrays account for99.896% of each original summary.
The arrays are preserved in original evidence; they are excluded from this
compact projection, together with dense confusions. No original report or
manifest is recreated, and original aborted records are not relabelled.

| Applied updates | Control primary macro-F1 | Fusion primary macro-F1 | Fusion minus control |
| ---: | ---: | ---: | ---: |
| 0 | 0.496723 | 0.496723 | 0.000000 |
| 256 | 0.495822 | 0.470151 | −0.025671 |
| 512 | 0.492768 | 0.468709 | −0.024059 |

Both earliest-best choices are update0, so neither endpoint improves its
starting score. The final fusion model is worse on13of18heads, including
7of8primary heads. Only bass improves among primary heads (+0.002872).
Largest primary regressions are tonicized key−0.059210, local key−0.043091
and primary degree−0.032040. Other small improvements include cadence and
metrical strength; this is not uniform degradation of every head.

Final harmonic-event joint accuracy is0.297326 control versus0.294470 fusion;
unseen-tuple joint accuracy is0.051376 versus0.037615. Root-in-predicted-
pitch-set consistency is0.972492 versus0.967334. That narrow consistency
constraint is not music quality and supplies no positive fusion result here.

Local verification recomputed108head means and six primary means from3744
saved per-class rows, checked supports, all six162-record/147-component
populations,10104hash-chained journal events and identical1024TRAIN draws
per arm. Four model-only exports match the streamed summary byte pins, have
finite tensors and valid contract/state fingerprints; both selected best base
states match the sourceC2 fingerprint. Local tensor checking took0.159s,
with337.11MiB peakRSS. No model forward or training was repeated.

The initial0.496723 differs from the historical C2 best0.496669 by0.000054393.
Its cause is unresolved; it must not be silently described as exact historical
metric reproduction. The two compared arms have identical initial scores and
base weights. The short fresh-AdamW continuation is not a full120k comparison.

This posthoc check validates retained artifact consistency and aggregation; it
does not recompute per-class F1 from omitted prediction rows/confusions, rerun
raw inference, inspect full optimizer/RNG/checkpoint tensors, or demonstrate
resume. Summary hashes are newly observed because the original success manifest
was never published. These limits remain explicit despite successful recovery.

A first local import-check helper rejected an archived planned-record filename
before publishing any verification output. A versioned correction compares its
bytes to the correctly named, canonical root record; the filename rule remains
enforced. Neither streaming computation nor tensor verification was repeated.

## Remaining engineering limitation

The optional fusion dispatcher still embeds per-target prediction lists into
validation histories, repeated full checkpoints and whole-object JSON reports.
The bounded posthoc helper resolves evidence access for this completed pilot;
it does not modify that dispatcher. Before any further real-data fusion run,
a separate tested logging change must stream prediction rows once and store
compact metrics plus immutable file references in checkpoints and reports.
No such new run or source redesign is launched in this completion task.

## Final closure

All three authorized pilot outcomes have been reviewed. The independent fusion
review agrees with the coordinator on108head means,6primary means, fixed best
selection, common target supports/populations and model/provenance boundaries.
Its660arithmetic/compact-evidence assertions pass; these are not independent
scientific observations or implementation unit tests. Review artifacts are in
ignored `outputs/research_resume_20260913/EXP-FUSION-P512-POSTHOC-001/`.

The posthoc record is terminal `completed`; original P512 arm records remain
`aborted`. All66registry records validate. The helper's36independent unit tests
passed, and final ledger/diff checks pass. The original C2 remains the retained
model candidate; no new baseline is selected and no next experiment is launched.
The next phase is a user discussion of these completed results. The known fusion
logging limitation and small historical-initial-metric discrepancy remain
explicit; neither blocks this bounded descriptive comparison.
