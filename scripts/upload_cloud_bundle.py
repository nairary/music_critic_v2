"""Upload explicit artifact files to private Yandex Object Storage using an IAM token.

No credentials enter logs, command arguments, manifests, or the bucket. Existing
keys are reusable only with matching size, SHA256 metadata and single-PUT MD5.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import http.client
import json
from pathlib import Path
import re
import time
from urllib.parse import quote


def hashes(path):
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)
            md5.update(chunk)
    return sha.hexdigest(), md5.hexdigest(), base64.b64encode(md5.digest()).decode()


def object_path(bucket, key):
    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", bucket):
        raise ValueError("invalid bucket")
    if not key or key.startswith("/") or ".." in key.split("/") or "\x00" in key:
        raise ValueError("invalid object key")
    return "/" + bucket + "/" + quote(key, safe="/")


def upload(path, *, bucket, key, token_file):
    path, token_file = Path(path), Path(token_file)
    if path.resolve() == token_file.resolve() or path.is_symlink() or not path.is_file():
        raise ValueError("explicit regular artifact required")
    size = path.stat().st_size
    if size > 4 * 1024**3:
        raise ValueError("split artifacts larger than 4 GiB before upload")
    sha, md5, md5_b64 = hashes(path)
    target = object_path(bucket, key)
    token = token_file.read_text().strip()
    if not token or any(c.isspace() for c in token):
        raise ValueError("invalid IAM token file")
    auth = {"Authorization": "Bearer " + token}

    def head():
        conn = http.client.HTTPSConnection("storage.yandexcloud.net", timeout=120)
        try:
            conn.request("HEAD", target, headers=auth)
            response = conn.getresponse()
            status, headers = response.status, dict((k.lower(), v) for k, v in response.getheaders())
            response.read()
            return status, headers
        finally:
            conn.close()

    def matches(headers):
        return (headers.get("content-length") == str(size) and
                headers.get("x-amz-meta-sha256") == sha and
                headers.get("etag", "").strip('"') == md5)

    status, headers = head()
    if status == 200:
        if not matches(headers):
            raise ValueError("existing object differs; immutable key will not be overwritten")
        action = "already_verified"
    elif status == 404:
        conn = http.client.HTTPSConnection("storage.yandexcloud.net", timeout=120)
        try:
            conn.putrequest("PUT", target)
            for name, value in {**auth, "Content-Length": str(size), "Content-MD5": md5_b64,
                                "x-amz-meta-sha256": sha, "Content-Type": "application/octet-stream"}.items():
                conn.putheader(name, value)
            conn.endheaders()
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    conn.send(chunk)
            response = conn.getresponse()
            status = response.status
            response.read()
            if status not in (200, 201):
                raise RuntimeError(f"Object Storage PUT returned HTTP {status}")
        finally:
            conn.close()
        status, headers = head()
        if status != 200 or not matches(headers):
            raise RuntimeError("uploaded object integrity verification failed")
        action = "uploaded_verified"
    else:
        raise RuntimeError(f"Object Storage HEAD returned HTTP {status}")
    return {"key": key, "sha256": sha, "size_bytes": size, "md5": md5, "action": action}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-file", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    for filename in args.files:
        for attempt in range(3):
            try:
                row = upload(filename, bucket=args.bucket,
                             key=args.prefix.strip("/") + "/" + Path(filename).name,
                             token_file=args.token_file)
                break
            except (OSError, http.client.HTTPException):
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        with Path(args.receipt).open("a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)
