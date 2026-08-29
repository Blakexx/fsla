# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""Dockerfile build-time helper (K37): download and sha256-verify the exact
sccache release `tools/sccache_pin.json` pins, then install it to
`/usr/local/bin/sccache`.

A CHECKED-IN FILE, not an inline `RUN <<'PY' ... PY` heredoc: the default
Dockerfile frontend (no `# syntax=docker/dockerfile:1.4` heredoc support
enabled) parses each line of an un-marked heredoc as a candidate Dockerfile
instruction -- `import hashlib, ...` was read as the unknown instruction
`IMPORT`, failing the build before a single layer ran. A real file sidesteps
the frontend-version question entirely and is what the Dockerfile `COPY`s and
`RUN python3`s, exactly like any other build input.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tarfile
import urllib.request

PIN_PATH = "/tmp/sccache_pin.json"


def main() -> int:
    with open(PIN_PATH) as f:
        pin = json.load(f)
    url = pin["release_url"]
    tar_path = "/tmp/sccache.tar.gz"
    urllib.request.urlretrieve(url, tar_path)
    with open(tar_path, "rb") as f:
        got = hashlib.sha256(f.read()).hexdigest()
    want = pin["tarball_sha256"]
    if got != want:
        sys.exit(f"sccache tarball sha256 mismatch: pinned {want}, got {got}")
    with tarfile.open(tar_path) as tf:
        member = next(m for m in tf.getmembers() if m.name.endswith("/sccache"))
        tf.extract(member, "/tmp/sccache_extract")
        src = f"/tmp/sccache_extract/{member.name}"
    with open(src, "rb") as f:
        body = f.read()
    bin_got = hashlib.sha256(body).hexdigest()
    bin_want = pin["binary_sha256"]
    if bin_got != bin_want:
        sys.exit(f"sccache binary sha256 mismatch: pinned {bin_want}, got {bin_got}")
    shutil.copy(src, "/usr/local/bin/sccache")
    os.chmod("/usr/local/bin/sccache", 0o755)
    print(f"sccache OK: {pin['version_string']}  sha256 {bin_got[:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
