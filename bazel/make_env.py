#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE ENVIRONMENT'S STORE: the pinned Python, requirements.lock's packages and rola-devtools at tools/devtools.txt's
commit, built once a key into the store every worktree's graph shares, under a file lock -- a second builder of the same key waits and reuses it.

The key hashes the lock, the pinned rola-devtools commit, the Python version, uv's version and the extra index, so a
change to any is a new environment and an unchanged one is never rebuilt. uv's cache in the store hardlinks package files, so environments share their
bytes. Run by the graph's environment repository (bazel/environment.bzl) under the machine's python3; prints the
environment's paths as JSON on its last line. Docs: docs/build-graph.md#environment.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--uv", required=True)
    ap.add_argument("--python", required=True, help="the interpreter version uv installs")
    ap.add_argument("--lock", type=Path, required=True)
    ap.add_argument("--index", required=True, help="the toolchain's torch index")
    ap.add_argument("--devtools", type=Path, required=True,
                    help="tools/devtools.txt: the rola-devtools commit the tree is tested with")
    a = ap.parse_args()

    uv_version = subprocess.run([a.uv, "--version"], capture_output=True, text=True, check=True).stdout.strip()
    key = hashlib.sha256(b"\0".join([a.lock.read_bytes(), a.devtools.read_bytes(), a.python.encode(),
                                     uv_version.encode(), a.index.encode()])).hexdigest()[:16]
    a.store.mkdir(parents=True, exist_ok=True)
    env = a.store / "venvs" / key
    python = env / "bin" / "python3"
    #: uv's own state lives in the store: its interpreters and its cache, never the user's
    uv_env = {**os.environ, "UV_PYTHON_INSTALL_DIR": str(a.store / "python"), "UV_CACHE_DIR": str(a.store / "uv-cache"),
              "UV_PYTHON_PREFERENCE": "only-managed", "UV_NO_CONFIG": "1"}
    with open(a.store / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (env / ".complete").is_file():
            #: an environment without its completion mark is an interrupted build's
            shutil.rmtree(env, ignore_errors=True)
            subprocess.run([a.uv, "python", "install", a.python], env=uv_env, check=True)
            subprocess.run([a.uv, "venv", "--python", a.python, str(env)], env=uv_env, check=True)
            subprocess.run([a.uv, "pip", "sync", "--python", str(python), "--require-hashes", str(a.lock),
                            "--extra-index-url", a.index, "--index-strategy", "unsafe-best-match"], env=uv_env,
                           check=True)
            #: rola-devtools at its pinned commit (a git requirement: the commit is its pin, no hash to require)
            subprocess.run([a.uv, "pip", "install", "--python", str(python), "--no-deps", "-r", str(a.devtools)],
                           env=uv_env, check=True)
            (env / ".complete").write_text(f"{uv_version}\n")
    probe = ("import json, sys, sysconfig; print(json.dumps({'site': sysconfig.get_paths()['purelib'], "
             "'include': sysconfig.get_paths()['include'], 'base': sys.base_prefix}))")
    paths = json.loads(subprocess.run([str(python), "-c", probe], capture_output=True, text=True, check=True).stdout)
    print(json.dumps({"venv": str(env), "python": str(python), **paths}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
