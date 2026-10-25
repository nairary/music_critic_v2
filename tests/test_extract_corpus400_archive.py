from hashlib import sha256
from io import BytesIO
import tarfile

import pytest

from scripts.extract_corpus400_archive import extract_admitted


def fixture(tmp_path, names=('data/a.json', 'data/b.json')):
    archive = tmp_path / 'raw.tar.gz'
    with tarfile.open(archive, 'w:gz') as stream:
        for name in names:
            member = tarfile.TarInfo(name); member.size = 2
            stream.addfile(member, BytesIO(b'{}'))
    raw = archive.read_bytes()
    binding = dict(path=str(archive), sha256=sha256(raw).hexdigest(), size_bytes=len(raw))
    item = dict(candidate_id='fixture:a', work_family_id='work:a', primary_genre='jazz', genre_evidence=['synthetic'],
        source=dict(kind='pdmx', binding=dict(path='pending.json'), archive_member='data/a.json'),
        admission=dict(status='admitted', current_split='train', known_heldout_family_conflict=False, evidence=['synthetic']))
    return binding, [item]


def test_extract_only_admitted_member_and_preserve_opaque_other(tmp_path):
    binding, items = fixture(tmp_path)
    result = extract_admitted(binding, items, tmp_path / 'out')
    assert set(result['extracted']) == {'data/a.json'}
    assert not (tmp_path / 'out/data/b.json').exists()
    assert result['missing'] == []


@pytest.mark.parametrize('member', ['../bad.json', '/absolute.json', 'data\\bad.json'])
def test_unsafe_archive_header_is_rejected(tmp_path, member):
    binding, items = fixture(tmp_path, (member,))
    with pytest.raises(ValueError, match='unsafe'):
        extract_admitted(binding, items, tmp_path / 'out')


def test_duplicate_selected_member_rejected(tmp_path):
    binding, items = fixture(tmp_path, ('data/a.json', 'data/a.json'))
    with pytest.raises(ValueError, match='duplicate'):
        extract_admitted(binding, items, tmp_path / 'out')


def test_closed_source_rejected_before_archive_read(tmp_path):
    binding, items = fixture(tmp_path); items[0]['admission']['current_split'] = 'test'
    with pytest.raises(ValueError, match='TRAIN'):
        extract_admitted(binding, items, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_missing_and_hash_failures_are_explicit(tmp_path):
    binding, items = fixture(tmp_path, ('data/b.json',))
    result = extract_admitted(binding, items, tmp_path / 'out')
    assert result['missing'] == ['data/a.json']
    binding['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='hash'):
        extract_admitted(binding, items, tmp_path / 'failed')
    assert (tmp_path / 'failed/admission-receipt.json').exists()
