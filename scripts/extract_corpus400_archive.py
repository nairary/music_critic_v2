"""Extract only explicitly admitted PDMX members from an opaque source archive.

Other musical entries are never decoded. Archive headers are discovery metadata;
no imported code, links or paths are executed. Preserve failed output attempts.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

from scripts.materialize_corpus400 import validate_admission, save, save_json


def safe_member(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name:
        raise ValueError('unsafe archive member path')
    return str(path).removeprefix('./')


def extract_admitted(archive_binding, items, output, *, max_member_bytes=50_000_000):
    """Caller registers acquisition; receipts precede opening archive bytes."""
    by_member = {}
    for item in items:
        validate_admission(item)
        member = safe_member(item['source']['archive_member'])
        if member in by_member:
            raise ValueError('duplicate admitted archive member')
        by_member[member] = item
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    save_json(output / 'admission-receipt.json', items)
    path = Path(archive_binding['path'])
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    if path.stat().st_size != archive_binding['size_bytes'] or digest.hexdigest() != archive_binding['sha256']:
        raise ValueError('archive hash/size mismatch')
    found, header_count = {}, 0
    with tarfile.open(path, mode='r|gz') as archive:
        for member in archive:
            header_count += 1
            name = safe_member(member.name)
            if name not in by_member:
                continue
            if name in found or not member.isfile() or member.size > max_member_bytes:
                raise ValueError('duplicate, nonregular or oversized admitted member')
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError('admitted member unreadable')
            raw = stream.read(max_member_bytes + 1)
            if len(raw) != member.size or len(raw) > max_member_bytes:
                raise ValueError('admitted member size mismatch')
            destination = output / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            found[name] = save(destination, raw)
    receipt = dict(schema='Corpus400ArchiveExtraction@1', archive=archive_binding,
                   header_count=header_count, selected_count=len(by_member), extracted=found,
                   missing=sorted(set(by_member) - set(found)),
                   musical_decode_scope='none; copied explicitly admitted selected member bytes only')
    save_json(output / 'extraction-receipt.json', receipt)
    return receipt
