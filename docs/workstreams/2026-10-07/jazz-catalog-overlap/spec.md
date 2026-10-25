# Jazz catalog overlap pass

Authorized by the user's request on 2026-10-07. Experiment
`EXP-JAZZ-CORPUS-OVERLAP-001` is planned before source collection.

Compare Jazz Harmony Treebank, WJD, JAAH and Chord-Jazzification identity
catalogs against PDMX, HookTheory, original POP909 and Dilemmadata metadata.
Retain source IDs, title/composer/performer distinctions, source version/URL,
all candidates, ambiguous/unmatched records, unavailable source fields,
local musical-file existence (stat only) and native/current split evidence.

Metadata from every split is admitted. Mixed JSON source bytes may be fetched,
hashed and lexically skipped; notes/chords/trees/annotations must not be
materialized or inspected. No local MIDI, score or musical-target reads.
No model calls, training, GPU, annotation transfer, dependencies or legacy use.
Do not execute downloaded code. Store full evidence under ignored
`outputs/jazz_corpus_overlap_20261007_001/`; terminal earlier outputs immutable.

Catalog JSONL rows share:
`dataset`, `id`, `titles` (list of `{field,value}`), `creators` (same, with
field identifying composer/performer/artist), and `source_metadata`.
Optional `clip_ids`, `source_clips`, `membership`, `availability` retain
provenance. Missing values stay missing; IDs remain source-specific.

Retrieval lanes: exact normalized full title; separate deterministic variants
(article movement/leading article, conservative bracketed arrangement markers,
accent folding) with operation evidence; explicit documented source aliases.
No substring/fuzzy hit is confirmed identity. Composer and performer agreement
are separate fields; absent/disagreeing creator does not discard title hits.
Treat duplicate arrangements and different performances as distinct records.
Avoid consolidating source records into unproven work identities.

Deliver all 16 corpus-pair summaries, complete JSONL/CSV candidate exports,
unmatched source records, coverage/limitations, and a small human-readable
shortlist with musical-version verification still pending. Count unique left
records, unique right records and pairs separately. No zero-overlap claim for
unavailable/unsearchable catalogs. Do not sum dependent WJD/JAAH sources as
independent works. Record metadata-only access observations explicitly.

Implementation tests use synthetic catalogs for Unicode/missing names,
creator roles, variants and false-positive boundaries, duplicate IDs,
all-pair accounting, missing splits, safe file existence and CSV formula cells.
Independent Standards and Spec reviews precede terminal completion.
Final report, STATUS and generated experiment ledger are required. No change
to model architecture or accepted training decision is made by this audit.

Acquisition attempt 001 was finalized as invalid after an incidental JAAH web
snippet exposed beat values and a commit-API response materialized an unread
annotation patch string. The record retains those observations and all source
evidence. The actual sixteen-pair comparison is planned separately as
`EXP-JAZZ-CORPUS-OVERLAP-002`, reading only eight fingerprinted identity
projections from sealed attempt 001, with no further source search or annotation
read. Its output root is `outputs/jazz_corpus_overlap_20261007_002/`.
The primary JHT population is the 150 entries with a root `trees` field;
the 1170-entry metadata catalog is retained separately, without claiming all
1170 have analysis trees. Local candidate availability is an explicit stat-only
enrichment stage: PDMX JSON/MIDI/MusicXML separately, HT/POP canonical cache,
and Dilemmadata raw TSV. These checks do not establish musical alignment.
