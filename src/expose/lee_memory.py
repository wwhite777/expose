"""In-memory public-MAT adapter for the unchanged Lee offline epoch loader."""

import hashlib
import io
from pathlib import Path
import urllib.request

from .lee import EpochConfig, load_offline_epochs


class NamedBytesIO(io.BytesIO):
    """BytesIO accepted by scipy.loadmat and Path metadata handling."""

    def __init__(self, value, display_path):
        super().__init__(value)
        self._display_path = str(display_path)

    def __fspath__(self):
        return self._display_path


def download_verified(url, expected_bytes, expected_md5, timeout=1800):
    """Stream one public object into RAM while enforcing frozen size and MD5."""
    if (not isinstance(url, str) or not url.startswith("https://")
            or not isinstance(expected_bytes, int) or expected_bytes <= 0
            or not isinstance(expected_md5, str) or len(expected_md5) != 32):
        raise ValueError("invalid frozen public-object identity")
    buffer = io.BytesIO(); md5 = hashlib.md5(); sha = hashlib.sha256(); count = 0
    request = urllib.request.Request(url, headers={"User-Agent": "EXPOSE-confirmation/1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        declared = response.headers.get("Content-Length")
        if declared is not None and int(declared) != expected_bytes:
            raise IOError("public-object Content-Length differs from frozen inventory")
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            count += len(block)
            if count > expected_bytes:
                raise IOError("public object exceeds frozen byte count")
            buffer.write(block); md5.update(block); sha.update(block)
    if count != expected_bytes or md5.hexdigest().lower() != expected_md5.lower():
        raise IOError("public-object size or published MD5 differs")
    return buffer.getvalue(), {"bytes": count, "md5": md5.hexdigest(), "sha256": sha.hexdigest()}


def load_offline_epochs_from_bytes(value, public_uri, subject, session,
                                   config=EpochConfig()):
    """Run the unchanged numeric loader and replace its local display-path metadata."""
    display = Path("memory_public_mat") / f"s{int(subject):02d}" / f"session{int(session)}.mat"
    handle = NamedBytesIO(value, display)
    try:
        epochs = load_offline_epochs(handle, config=config)
    finally:
        handle.close()
    epochs.metadata["path"] = f"memory://openbmi/s{int(subject):02d}/session{int(session)}"
    epochs.metadata["public_uri"] = str(public_uri)
    epochs.metadata["raw_persisted_to_disk"] = False
    return epochs
