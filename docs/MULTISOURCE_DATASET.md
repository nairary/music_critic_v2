# Multi-source corpus Dataset contract

Phase 5B.2 defines the reproducible boundary between production corpus
adapters and the Phase 5B.1 collator. It does not define a training split,
mixture weights, a model, or a loss.

## Versioned artifacts

The public versions are:

- `MULTISOURCE_CORPUS_INDEX_VERSION = "1.0.0"`;
- `MULTISOURCE_CACHE_VERSION = "1.0.0"`;
- `MIXTURE_SAMPLER_VERSION = "1.0.0"`;
- `SPLIT_MANIFEST_VERSION = "1.0.0"`.

`CorpusIndexHeader` binds the dataset and adapter configuration to the
canonical, graph, feature, ontology, and encoding contracts. It also records
the source identity/fingerprint, creation policy, accepted record count, and
the deterministic index fingerprint. `IndexedCorpusRecord` contains portable
source evidence, source and lineage groups, a suggested source split,
target-availability counts, and the relative cache artifact path/SHA-256.
Absolute paths, path traversal, duplicate `(dataset_id, piece_id)` identities,
non-canonical ordering, and fingerprint/version mismatches are rejected.

Accepted records and `CorpusQuarantineRecord` values are separate. Quarantine
never becomes a Dataset item. `CorpusBuildReport` provides CPU-side accepted,
quarantined/failure-category, cache hit/miss, group, suggested-split conflict,
raw-only, and target availability/mask counts.

## Offline canonical cache

The HookTheory builder streams `4_merged.json` once through the production
adapter. It retains only index/report metadata while each accepted
`CanonicalPiece` is serialized and released. Structure `ori_uid` evidence is
preserved as its authoritative source grouping. Conversion failures are
structured report rows. The POP909-CL builder uses
`discover_pop909_cl_corpus` and the Phase 4B adapter; song 172 remains
quarantined, while the adapter's accepted masked-target behavior is unchanged.

The cache key is the SHA-256 of canonical JSON containing:

```text
cache_version
source_identity
source_sha256
adapter_name
adapter_version
adapter_config_fingerprint
canonical_schema_version
target_ontology_version
target_ontology_fingerprint
```

The artifact filename is another safe SHA-256 identity over that cache key and
the canonical artifact SHA-256. Consequently a target-only canonical change
can create a different artifact while the Phase 3A raw graph fingerprint stays
unchanged. One deterministic canonical JSON artifact is written per accepted
piece using a same-directory temporary file, `fsync`, and atomic rename.
Partial files are never cache hits. Existing content is verified; stale
namespaces are retained rather than deleted. PyG graphs, batches, tensors,
pickle, and `torch.save` are not cache formats in this phase.

Generated caches, real indices, and corpus reports are local artifacts and
must not be committed.

## Lazy Dataset and split boundary

`IndexedMultiSourceDataset` is a map-style PyTorch Dataset. Construction reads
index metadata only and opens no persistent handles. `__getitem__` resolves one
relative artifact under the configured namespace, reads that artifact only,
checks its SHA-256 and canonical identity/validation, and calls
`prepare_multisource_sample`. Negative and out-of-range indices are explicit
structured errors. The Dataset and the private raw-graph binding survive
spawn/pickle serialization without weakening graph fingerprint validation.

`suggested_split` is diagnostic evidence only. `DatasetView` requires a
validated external `SplitManifest`; it never filters directly on the source
suggestion. A manifest contains exact piece assignments, source/lineage group
evidence, transitive atomic-component fingerprints, seed, policy/config
fingerprint, constituent index fingerprints, and its own fingerprint. Missing,
duplicate, stale, or cross-split assignments are rejected through the existing
group validation. No production ratios or seed are selected here.

The optional `plan_group_hash_split` is target-blind and requires explicit
ratios and seed. It sorts and hashes complete source/lineage components, then
uses largest-remainder component quotas. It performs no fuzzy duplicate
matching and does not infer a scientific production split.

## Composition, sampling, and workers

`MultiCorpusDataset` accepts unique dataset IDs and views of one validated
split only. It sorts datasets by stable ID, exposes deterministic global
ranges and constituent index fingerprints, and maps global to local indices
without copying canonical pieces.

`DeterministicQuotaSampler` requires positive explicit dataset weights, seed,
epoch, and positive epoch size. Stable-ID largest-remainder allocation produces
exact quotas. A local `torch.Generator` shuffles the dataset schedule and
separate deterministic per-dataset cycles. A record is not repeated until its
local cycle is exhausted. `set_epoch` changes the schedule; the same
seed/epoch/contracts reproduce it. Epoch evidence records requested and
normalized weights, exact quotas, constituent fingerprints, repeats after
cycle exhaustion, and a schedule fingerprint. Mid-epoch resume is deferred;
epoch-level replay is the Phase 5B.2 guarantee.

`make_multisource_dataloader` accepts only a sampler bound to the exact
single-split composition. Batch size, worker count, seed, persistence,
prefetch, and multiprocessing context are explicit. Its top-level worker
initializer derives Python and torch seeds from the PyTorch worker seed.
NumPy is not a project dependency and is therefore not imported. The existing
`collate_multisource_samples` remains the only collator, and no target or
identity metadata is injected into PyG stores.

## Commands

Bounded cache build:

```bash
python scripts/build_multisource_cache.py \
  --cache-root /explicit/cache \
  --index-output /explicit/index.json \
  --report-output /explicit/report.json \
  --limit 4 hooktheory \
  --raw-path /explicit/4_merged.json \
  --structure-root /explicit/structure
```

Use the `pop909_cl --corpus-root ...` subcommand for a POP909-CL build.
Omitting `--limit` is an explicit full build and is never done by default CI.

Deterministic verification:

```bash
python scripts/audit_multisource_dataset.py \
  --index /explicit/index.json \
  --cache-root /explicit/cache \
  --split-manifest /explicit/split.json \
  --check
```

Bounded loader benchmark:

```bash
python scripts/benchmark_multisource_dataloader.py \
  --corpus /index.json:/cache:/split.json \
  --weight hooktheory=1 \
  --split train --epoch-size 16 --max-batches 4 --num-workers 0
```

Repeat with `--num-workers 2` for spawn evidence. Benchmark values are
diagnostic only and have no CI timing threshold.

## Deferred decisions

Before Phase 6, a production `SplitManifest`, scientific ratios/seed, and
training mixture weights require an explicit evidence-backed decision.
Mid-epoch sampler resume and any measured graph cache optimization are also
future work. Phase 5B.2 introduces no PDMX adapter, model head, loss, PU
objective, SSL/corruption pipeline, or target-dependent split/sampling rule.
