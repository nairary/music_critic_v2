import hashlib
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("upload_cloud_bundle", Path(__file__).parents[1] / "scripts/upload_cloud_bundle.py")
cloud = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cloud)


def test_upload_verifies_and_reuses_exact_object_without_replacing_it(tmp_path, monkeypatch):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"test artifact")
    token = tmp_path / "token"
    token.write_text("fixture-token")
    stored, requests = {}, []

    class Response:
        def __init__(self, status, headers):
            self.status, self.headers = status, headers
        def read(self):
            return b""
        def getheaders(self):
            return self.headers.items()

    class Connection:
        def __init__(self, host, timeout):
            assert host == "storage.yandexcloud.net"
            self.data, self.headers = b"", {}
        def request(self, method, path, headers):
            self.method, self.path = method, path
        def putrequest(self, method, path):
            self.method, self.path = method, path
        def putheader(self, name, value):
            self.headers[name.lower()] = value
        def endheaders(self):
            pass
        def send(self, data):
            self.data += data
        def getresponse(self):
            requests.append(self.method)
            if self.method == "HEAD":
                return Response(200 if self.path in stored else 404, stored.get(self.path, {}))
            assert self.headers["authorization"] == "Bearer fixture-token"
            stored[self.path] = {"content-length": str(len(self.data)),
                                 "x-amz-meta-sha256": self.headers["x-amz-meta-sha256"],
                                 "etag": hashlib.md5(self.data).hexdigest()}
            return Response(200, {})
        def close(self):
            pass

    monkeypatch.setattr(cloud.http.client, "HTTPSConnection", Connection)
    args = {"bucket": "fixture-bucket", "key": "code/artifact.bin", "token_file": token}
    first = cloud.upload(artifact, **args)
    assert first["action"] == "uploaded_verified"
    assert "fixture-token" not in str(first)
    assert cloud.upload(artifact, **args)["action"] == "already_verified"
    assert requests.count("PUT") == 1
    artifact.write_bytes(b"changed artifact")
    with pytest.raises(ValueError, match="will not be overwritten"):
        cloud.upload(artifact, **args)
    assert requests.count("PUT") == 1


def test_token_is_never_an_uploadable_artifact(tmp_path):
    token = tmp_path / "token"
    token.write_text("secret")
    with pytest.raises(ValueError, match="artifact"):
        cloud.upload(token, bucket="fixture-bucket", key="token", token_file=token)


@pytest.mark.parametrize("key", ["/absolute", "../escape", "prefix/../escape", "x\x00y"])
def test_unsafe_key_rejected(key):
    with pytest.raises(ValueError, match="key"):
        cloud.object_path("fixture-bucket", key)
