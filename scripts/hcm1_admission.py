"""Bind native HCM prerequisites to the exact parent and current scripts."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

from music_critic.experiments.registry import load_record
from scripts.evaluate_hcm1_handoff import binding


IDS = {'gate': 'EXP-HCM1-3090-GATE-001', 'evaluate': 'EXP-HCM1-3090-EVAL-001'}


def verify_prerequisites(mode, records, *, parent_sha256, cases, seed=20260928):
    """Return immutable receipts only for exactly matching completed attempts."""
    if mode not in ('gate', 'evaluate', 'train'):
        raise ValueError('unknown handoff mode')
    required = () if mode == 'gate' else ('gate',) if mode == 'evaluate' else ('gate', 'evaluate')
    receipts = []
    for prerequisite in required:
        path = Path(records) / (IDS[prerequisite] + '.json')
        record = load_record(path)
        expected = binding(SimpleNamespace(mode=prerequisite, parent_sha256=parent_sha256,
            cases=str(cases), seed=seed, wall_seconds=3600))
        if record['status'] != 'completed' or record.get('hcm_diagnostic') != expected:
            raise ValueError(f'{prerequisite} prerequisite binding mismatch')
        if prerequisite == 'gate' and any(record['results'].get(flag) is not True
                for flag in ('native_admitted', 'full_state_exact', 'resume_verified')):
            raise ValueError('native prerequisite did not admit complete state and replay')
        if prerequisite == 'evaluate' and record['results'].get('parent_update') != 10000:
            raise ValueError('evaluation prerequisite has wrong parent progress')
        receipts.append(dict(experiment_id=record['experiment_id'],
            record_fingerprint=record['record_fingerprint'], sha256=sha256(path.read_bytes()).hexdigest()))
    return receipts
