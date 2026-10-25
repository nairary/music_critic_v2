"""Explicit byte-bound weight migration from HCM-1 into a new whole model.

This operation is not a training resume. Shared encoder and event/field decoder
parameters migrate; old context memory, optimizer, RNG and progress do not.
"""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import torch

from music_critic.data.serialization import canonical_json_sha256
from .contracts import SCHEMA, CodecProfile, ContextProfile, ModelConfig
from .runtime import CHECKPOINT_SCHEMA
from .whole_model import WholePieceHCM


_NEW_DECODER_PREFIXES = ('decoder.global_attention.', 'decoder.motif_attention.',
                         'decoder.branch_norms.', 'decoder.branch_gates.')


def warm_start_hcm1(model, path, *, expected_sha256, expected_size_bytes, expected_update):
    """Validate all shared tensors before applying any source weight.

The source model/codec and conditional target contracts must match exactly.
Context length may differ intentionally: the destination observes whole songs.
Only the new model should exist at this point; construct its optimizer after
this function so the trainer's binding includes the migration provenance.
"""
    if not isinstance(model, WholePieceHCM):
        raise TypeError('pll.warmstart_whole_model_required')
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or any(c not in '0123456789abcdef' for c in expected_sha256)
            or type(expected_size_bytes) is not int or expected_size_bytes < 1
            or type(expected_update) is not int or expected_update < 0):
        raise ValueError('pll.warmstart_byte_binding_required')
    if model.warm_start_binding is not None:
        raise ValueError('pll.warmstart_already_applied')
    raw = Path(path).read_bytes()
    if len(raw) != expected_size_bytes or sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('pll.warmstart_checkpoint_bytes_mismatch')
    payload = torch.load(BytesIO(raw), map_location='cpu', weights_only=True)
    if (not isinstance(payload, dict) or payload.get('schema') != CHECKPOINT_SCHEMA
            or not isinstance(payload.get('binding'), dict)
            or type(payload.get('updates')) is not int or payload['updates'] != expected_update):
        raise ValueError('pll.warmstart_checkpoint_schema_or_update')
    source_contract = payload['binding'].get('model')
    expected_contract = model.contract()
    contract_keys = {'schema', 'model', 'context', 'codec', 'freeze_policy', 'initialization',
                     'event_projection', 'mask_ownership', 'likelihood'}
    if (not isinstance(source_contract, dict) or set(source_contract) != contract_keys
            or source_contract.get('schema') != SCHEMA
            or source_contract.get('model') != asdict(model.config)
            or source_contract.get('codec') != asdict(model.codec_profile)
            or source_contract.get('freeze_policy') not in ('frozen', 'upper', 'full')
            or not isinstance(source_contract.get('initialization'), str)
            or not source_contract['initialization']
            or any(source_contract.get(key) != expected_contract[key]
                   for key in ('event_projection', 'mask_ownership', 'likelihood'))):
        raise ValueError('pll.warmstart_source_model_contract_mismatch')
    try:
        ModelConfig(**source_contract['model']).validate_codec(CodecProfile(**source_contract['codec']))
        context = ContextProfile(**source_contract['context'])
    except (TypeError, ValueError) as exc:
        raise ValueError('pll.warmstart_source_model_contract_mismatch') from exc
    if context.max_tracks > model.codec_profile.max_tracks:
        raise ValueError('pll.warmstart_source_context_codec_mismatch')
    current = model.state_dict()
    shared = {key for key in current if key.startswith(('encoder.', 'decoder.'))
              and not key.startswith(_NEW_DECODER_PREFIXES)}
    source = payload.get('model')
    if (not isinstance(source, dict) or any(not isinstance(key, str) for key in source)
            or any(not key.startswith(('encoder.', 'decoder.', 'memory.')) for key in source)
            or {key for key in source if key.startswith(('encoder.', 'decoder.'))} != shared):
        raise ValueError('pll.warmstart_shared_key_coverage_mismatch')
    for key in sorted(shared):
        value, expected = source[key], current[key]
        if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
                or value.shape != expected.shape or value.dtype != expected.dtype
                or not bool(torch.isfinite(value).all())):
            raise ValueError('pll.warmstart_shared_tensor_invalid:' + key)
    receipt = dict(schema='HCM1WholeWarmStart@1', source_sha256=expected_sha256,
        source_size_bytes=expected_size_bytes, source_update=expected_update,
        source_model_contract_sha256=canonical_json_sha256(source_contract),
        copied_tensors=len(shared), copied_elements=sum(current[k].numel() for k in shared),
        copied_key_fingerprint=canonical_json_sha256(sorted(shared)),
        copied_scopes=['encoder', 'compatible_event_field_decoder'],
        retained_random_scopes=['whole_memory', 'motifs', 'decoder_distant_branches'],
        discarded_source_scopes=['memory', 'optimizer', 'rng', 'sampler_state', 'updates', 'draws'],
        optimizer_restored=False, rng_restored=False, progress_restored=False)
    # Prevalidation above is atomic. Retain a rollback snapshot as protection
    # against an exceptional device copy; no module loading hooks are invoked.
    before = {key: current[key].clone() for key in shared}
    with torch.no_grad():
        try:
            for key in sorted(shared):
                current[key].copy_(source[key])
        except BaseException:
            for key, value in before.items():
                current[key].copy_(value)
            raise
    model.warm_start_binding = receipt
    model.encoder.weights_origin = 'hcm1-warm-start:sha256:' + expected_sha256
    return receipt
