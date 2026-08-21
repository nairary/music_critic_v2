# Phase 9C-B: diagnostic onset-BiGRU decoder

## Hypothesis and claim boundary

Phase 9C-B tests whether the independent candidate MLP used by the completed
Dilemmadata pilot hides useful sequential information in an SSL encoder. It is
a seed-17 validation diagnostic, not a multi-seed scientific experiment. It
cannot establish statistical significance, scientific superiority, test
quality, or a final representation claim.

The fixed matrix is:

| Cell | Encoder initialization | Decoder |
|---|---|---|
| `scratch_mlp` | scratch | unchanged MLP |
| `ssl_mlp` | explicit encoder-only SSL transfer | unchanged MLP |
| `scratch_onset_bigru` | scratch | onset-BiGRU |
| `ssl_onset_bigru` | explicit encoder-only SSL transfer | onset-BiGRU |

All cells use the same class-balanced training artifact, seed domain,
Dilemmadata train/validation split, metadata-only batch schedule, optimizer,
learning rate, scheduler, AMP policy, attempted/applied update budget,
validation protocol, and final `last.pt` checkpoint policy. Test remains
locked.

## Architecture

`decoder.kind=mlp` is the accepted control. It creates no sequence-decoder
module, consumes no additional RNG draw, preserves all existing state names,
and retains the previous model-contract fingerprint and logits/loss path.
Existing MLP checkpoints omit a decoder field and continue to resolve to MLP.

`decoder.kind=onset_bigru` inserts one raw-only decoder between hierarchical
encoding and the unchanged `SourceNativeTaskHeads`:

```text
raw graph -> hierarchical encode -> encoded.fused
          -> onset-BiGRU decoder -> ordinary EncoderOutput
          -> unchanged four MLP heads -> predictions
          -> target-sidecar supervise -> source-entry loss
```

The GRU has one bidirectional layer, input width `hidden_dim`, and
`hidden_dim / 2` units per direction. Its internal dropout is zero. Odd or
invalid hidden widths fail with a structured configuration error. For each
onset:

```text
context = projection(bigru_output)
gate = sigmoid(gate_projection(concat(local_onset, context)))
onset_sequence = LayerNorm(local_onset + dropout(gate * context))
```

The GRU is called independently for every non-empty composition. This makes a
song independent of other sequence lengths in the batch and introduces no
padding-derived row. The results are written back to the original raw onset
row positions. Empty onset sequences produce no synthetic onset.

## Raw-only ordering and ownership

Sequence construction reads only `encoded.fused.embeddings["onset"]`, raw
node batch membership, and raw ownership edges. The graph contract already
groups batched node rows by sample and preserves the canonical graph builder's
exact onset order and deterministic raw identity tie-break; the decoder never
sorts float features or target spans.

Onset context is mean-pooled through exact raw `onset -> beat` and
`onset -> bar` ownership. Beat and bar each use a separate learned two-state
availability embedding, projection, gate, residual, and LayerNorm. Owners
without onset retain their local embedding through the residual and receive
the explicit unavailable state. Song, track, and note fused tensors are not
modified. Candidate enumeration remains onset then beat then bar for all four
tasks, so candidate identities and the source-entry loss denominator do not
change.

Target values, masks, labels, provenance, confidence, and sidecars are absent
from sequence construction. One prediction object may be supervised against
original or mutated targets without changing decoder outputs, candidate
identities, logits, or raw prediction fingerprints.

## Transfer, initialization, and checkpoints

The explicit SSL artifact is supplied by immutable absolute path, SHA-256, and
source kind. Only these existing prefixes transfer failure-atomically:

- `local_baseline.encoder.`;
- `context_encoder.pooling.`;
- `context_encoder.transformer.`;
- `context_encoder.fusion.`.

The GRU, onset/beat/bar fusion, four supervised heads, optimizer, scheduler,
and scaler stay fresh. Training evidence fingerprints all non-encoder tensors
before transfer and requires the same fingerprint after transfer. The verifier
requires exact paired scratch/SSL fresh fingerprints for each decoder.
Cross-decoder resume is rejected by the run-manifest/checkpoint contract.

## Metrics and interpretation

Every task and aggregate report includes normalized NLL, macro-F1, balanced
accuracy, accuracy, per-class precision/recall/F1, confusion matrix, true
support, predicted distribution, prediction entropy, majority baseline, and
available/masked source-entry counts. The primary metric remains mean
`NLL / log(class_count)` over the four tasks; lower is better.

Aggregation reports four deltas:

- decoder effect under scratch;
- decoder effect under SSL;
- SSL effect with MLP;
- SSL effect with onset-BiGRU.

For NLL it stores both raw `right - left` and improvement-signed `left - right`
values. The generated interpretation is descriptive only: probable old-decoder
bottleneck; sequence information visible only with BiGRU; SSL
objective/distribution as the next question; or target/noise/alignment/subset
as the next question.

## Production configuration and RTX 3090 command

The JSON configuration must name existing artifacts; the runner never chooses
an SSL variant automatically and never rebuilds caches:

```json
{
  "ssl_checkpoint": "/absolute/evidence/best-validation-only-ssl.pt",
  "ssl_checkpoint_sha256": "64-lowercase-hex-characters",
  "ssl_encoder_export": "/absolute/evidence/best-validation-only-encoder.pt",
  "ssl_encoder_export_sha256": "64-lowercase-hex-characters",
  "ssl_source_kind": "phase8b_multilevel_ssl",
  "raw_index": "/absolute/cache/dilemmadata.index.json",
  "raw_cache_root": "/absolute/cache/dilemmadata",
  "target_index": "/absolute/cache/dilemmadata-target.index.json",
  "target_cache_root": "/absolute/cache/dilemmadata-target",
  "split_manifest": "/absolute/cache/dilemmadata.split.json",
  "class_weight_artifact": "/absolute/evidence/class_weights.json",
  "train_priors": "/absolute/evidence/train_priors.json",
  "epochs": 1,
  "steps_per_epoch": 3000,
  "batch_size": 2,
  "learning_rate": 0.0003
}
```

If the selected checkpoint itself is already an accepted encoder export, omit
the two `ssl_encoder_export` fields and the runner uses the checkpoint path and
SHA for both bindings.

First run the independent bounded profile on a fresh output root:

```bash
EXPECTED_SHA=$(git rev-parse HEAD)
scripts/run_phase9cb_rtx3090_matrix.sh profile "$EXPECTED_SHA" \
  /absolute/phase9cb.json /absolute/evidence/phase9cb-profile
```

After review, launch the four production cells explicitly on another root:

```bash
EXPECTED_SHA=$(git rev-parse HEAD)
scripts/run_phase9cb_rtx3090_matrix.sh run "$EXPECTED_SHA" \
  /absolute/phase9cb.json /absolute/evidence/phase9cb-seed17
```

Resume uses the same exact SHA, config, and root with action `resume`. The shell
runner requires a clean exact HEAD and an NVIDIA GeForce RTX 3090 at `cuda:0`.
It invokes the independent verifier and creates a regular-file tar plus SHA-256
sidecar only after a successful `run` or `resume`. `profile` never starts
production automatically.

## Non-goals

No raw/target cache rebuild, corpus audit, split or accepted-subset change,
target ontology/crosswalk change, class-weight formula change, new SSL
objective, attention pooling, onset Transformer, multi-seed run, test
evaluation, PDMX, Phase 10, or long CPU training is part of Phase 9C-B.
