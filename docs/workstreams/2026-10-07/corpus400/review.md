# T08 implementation review

Scope: new corpus400 files against HEAD
`287bf856643f9e555352e1661a7bf0563772178a`, dirty main. Existing unrelated WIP,
legacy, model training and musical-label accuracy are excluded. In-scope files
are untracked additions, read directly; no Git index mutation was performed.

## Catalog

Independent Astra standards/spec review found two evidence-loss cases: a
same-family heldout entry from a supplemental/match lane was not propagated,
and duplicate catalog identities could overwrite split evidence. Both were
reproduced and fixed; 14 focused tests passed, independently repeated.
Source SHA256 after review:
`5598abd954ec6da772428d2a81d80da16c1a9af5739f57662bf771b92c879108`.
The first metadata output remains as an earlier attempt; selection002 replaces
its ranking, without deleting prior evidence. No music payload was read.

## Materialization

Independent standards/spec review found invalid default MIDI IDs, unhandled
adapter exceptions, missing native-theory/canonical hash comparison, an
unvalidated experiment record, lack of a pitched-event gate and a direct CLI
import failure. All were corrected. Per-source failures remain in the ledger;
the active record and source admission are retained before reading music.
17 tests passed, including CLI invocation; one existing Torch deprecation.

## Completeness and object semantics

Parent reviewed the seven-layer accounting, exact per-view coverage, masks,
fingerprints and independent review gate. Review requested explicit typed
preparation/resolution pointers and rejection of placeholder semantic booleans.
The amendment checks edge existence, view, direction, note/context participants,
chronology and contradictions with reviewed absence. 43 new tests and 25 T07
regression tests passed; one existing Torch deprecation.
Source SHA256:
`cb6b9a445d61d9040304f521dd93443e2417778552e7585cb261e1195d4775cc`.

Standards and spec: no remaining actionable findings in these reviewed
mechanics. Provenance attestations still require trustworthy workflow receipts;
passing these checks does not establish full-work identity or musical accuracy.
Packet generation and authoring utilities require their own subsequent review.

## Packets, writer and selected archive extraction

Parent reviewed the packetizer's unique core-note ownership, exact temporal
partition, context omission receipts, source field preservation, authenticated
materialization index and separation of computational from musical boundaries.
15 synthetic tests pass. Source SHA256:
`dcc4c100a21c7d7df49f2e5b2da61dc044763ce0269b5db5d54e4e4b80d08dff`.

The authored-view writer received independent standards/spec review and 27
independently repeated tests. It constructs new harmony/context views, retains
unknown roles and derives membership/degrees only from explicit authored pitch
maps. It never manufactures independent reviewer acceptance. Source SHA256:
`6e44963851b74b2a82a785483821ccfe66fa839e106ef19bdeb6476fa873ed39`.

Selected-member archive extraction passed independent review and 7 tests.
Admission precedes archive access; selected regular files alone are copied,
with archive/member hashes, safe-path checks and explicit missing members.
The latest materializer additionally binds its admission manifest and retains
uninterpreted PDMX analytical extras separately from raw canonical inputs.

Final combined in-scope test run: **149 passed, 1 existing Torch warning,
3.26 seconds**. No remaining actionable findings in the reviewed engineering
mechanics. Real-source import failures are retained and need case-specific
diagnosis; these unit tests cannot establish musical correctness.

## Opt-in scope accounting v2

Independent Astra Standards/Spec review of the separate v2 module found one
blocker: overlapping known harmony/context claims in one view could pass strict
completion. The author added exclusive half-open harmony and note-context
geometry, including duplicate points and left/right boundary cases. Higher
musical objects and alternative views may still overlap. Independent delta
review accepted the correction and confirmed v1/writer bytes unchanged.

Final source SHA256:
`94575ac4efd09744a82670c25387856145ac554cc32b4faf0c7665d44103e424`.
Test SHA256:
`75019215118ce914d70076be0cdc0b676a5094528fb47d2344a60bf82728328e`.

Author verification: **115 passed, 1 existing Torch warning, 0.76s**.
Independent verification: **115 passed, 1 existing Torch warning, 0.74s**;
1,000 additional synthetic interval/point multisets matched an independent
pairwise overlap oracle. Tests cover the new module plus unchanged v1 and
writer. No real musical payload was used for these engineering checks.

V2 separates source extent, reviewed uncertainty and strict target availability.
External evidence authentication remains the caller's responsibility. Default
v2 does not itself add task-specific tension fields: the corpus profile must
explicitly require them on every layer. A supported-field floor and accepted
review are not a claim of dense, correct annotations or corpus-quota completion.
