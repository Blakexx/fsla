"""THE SCCACHE WRAPPER PIN -- closed-world rule 1, extended to the compiler cache.

**WHY A CACHE NEEDS A RULE-1 EXTENSION AT ALL.** Rule 1 (`setup.py:_gate_toolchain`,
`tools/ratify.py:check_toolchain`) refuses a build or a ratification whose `ptxas`
does not match the string the manifest was measured under, because a spill/register
profile is a property of the assembler as much as of the source. A compiler *cache*
sits in front of that assembler and can, if unpinned, serve an object nobody
measured: install a different `sccache` build with different argument handling, or
point it at a different `nvcc`, and every gate downstream of the object file would
still see a plausible-looking `.o` with no way to tell it was never compiled by the
gated toolchain at all. `tools/sccache_pin.json` pins the wrapper the way
`csrc/third_party/cutlass/PIN.json` pins a vendored codegen input, and `gate()`
below is what refuses a mismatch before the wrapper is ever invoked.

**WHY THE WRAPPER ITSELF CANNOT BE THE THING THAT GOES WRONG.** `tools/sccache_nvcc.py`
never resolves `nvcc`; it execs whatever `ROLA_REAL_NVCC` says, and this module is
the only thing that sets that variable, always to the same `cuda_bin("nvcc")` path
`setup.py`'s and `tools/ratify.py`'s own rule-1 gates already resolved and checked
`ptxas` against. So "the cache key must carry the assembler identity the manifest
pins" is true by construction, not by a runtime comparison this file would otherwise
have to perform and could get wrong: sccache's own cache key is derived from the
full `nvcc --dryrun` decomposition of the invoked compiler
(https://github.com/mozilla/sccache/blob/main/src/compiler/nvcc.rs), and the
invoked compiler is always the one `ptxas`-gated path.

**THE DEPFILE QUESTION, MEASURED ON THIS TREE (2026-08-06).** Reading the sources
alone could not determine whether torch's forced
`--generate-dependencies-with-compile --dependency-output $out.d` (always added by
`torch.utils.cpp_extension._write_ninja_file` to every nvcc invocation) breaks
sccache's cache key, and recommended treating a 0% `.cu` hit rate as the expected
starting state. Measured here with sccache 0.17.0 pinned below: it does NOT. A cold
compile of a real shard TU WITH the depfile flags, followed by the same compile
WITHOUT them (an argv difference of exactly `-MP --generate-dependencies-with-
compile --dependency-output PATH`), produced a 100% cache hit on the second
invocation and a byte-identical `.o` (`cmp` exit 0). So the flag is NOT stripped
before this file's use of the wrapper, and `tools/ratify.py`'s post-build recompile
-- which never passed those flags to begin with, because it does no incremental
compilation of its own -- lands in the SAME cache namespace `setup.py`'s ninja
build populated. That is the mechanism behind the goal this card names: the second
compile of the same ten translation units becomes a cache hit, not a second
7-minute compile.

**DEFAULT BEHAVIOUR.** `ROLA_SCCACHE` defaults to `"auto"`: use the pinned wrapper
if it resolves and matches, otherwise build uncached and say so. `"0"` disables it
outright. `"1"` REQUIRES it -- a missing or mismatched wrapper is a hard failure,
for measurement runs and CI where a silent 0%-hit "pass" would be vacuous
(`per-phase-gate-protocol`: a gate that cannot fail is not a gate).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIN_PATH = HERE / "sccache_pin.json"
WRAPPER_PATH = HERE / "sccache_nvcc.py"


def pin() -> dict:
    return json.loads(PIN_PATH.read_text())


def resolve_sccache() -> str | None:
    """`ROLA_SCCACHE_BIN` overrides; otherwise `PATH` lookup, exactly like any other
    tool this project shells out to (`ptxas`, `nvcc`) -- no hidden search path."""
    override = os.environ.get("ROLA_SCCACHE_BIN")
    if override:
        return override
    return shutil.which("sccache")


def sccache_version(path: str) -> str:
    proc = subprocess.run([path, "--version"], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"`{path} --version` failed (rc={proc.returncode}); a "
                           f"closed-world build cannot gate a cache it cannot ask")
    return proc.stdout.strip()


def binary_sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gate() -> dict | None:
    """Verify the resolved `sccache` against the pin and return
    `{"version", "sha256", "path"}` if it may be used, else `None`.

    Runs BEFORE the wrapper is ever put in `PYTORCH_NVCC` or `ROLA_REAL_NVCC`'s
    caller path, for the same reason rules 1/3/5 run before the compile: refusing a
    mismatch after objects have already been produced under it helps nobody.
    """
    mode = os.environ.get("ROLA_SCCACHE", "auto")
    if mode == "0":
        print("sccache: disabled (ROLA_SCCACHE=0)")
        return None

    p = pin()
    path = resolve_sccache()
    if path is None:
        msg = (f"sccache: not found on PATH (or ROLA_SCCACHE_BIN). Pinned build is "
               f"{p['version_string']} ({p['release_asset']}, {p['release_url']}).")
        if mode == "1":
            raise RuntimeError(
                f"CLOSED-WORLD RULE 1 (sccache wrapper pin): ROLA_SCCACHE=1 requires "
                f"the pinned cache and none was found.\n{msg}")
        print(f"{msg} Building without a compiler cache (set ROLA_SCCACHE=1 to "
              f"require it once installed).")
        return None

    version = sccache_version(path)
    digest = binary_sha256(path)
    if version != p["version_string"] or digest != p["binary_sha256"]:
        msg = (
            "CLOSED-WORLD RULE 1 (sccache wrapper pin): the resolved `sccache` does "
            f"not match tools/sccache_pin.json.\n"
            f"  pinned  : {p['version_string']}  sha256 {p['binary_sha256'][:16]}...\n"
            f"  resolved: {version}  sha256 {digest[:16]}...  ({path})\n"
            "A cache under an unpinned wrapper can serve an object nobody measured "
            "this ratification against. Install the pinned build, or move the pin "
            "DELIBERATELY (tools/sccache_pin.json) after re-verifying the bit-"
            "identity gate (docs/build.md#sccache) under the new one."
        )
        if mode == "1":
            raise RuntimeError(msg)
        print(f"sccache PIN MISMATCH -- building without a compiler cache.\n{msg}")
        return None

    print(f"sccache OK  {version}  sha256 {digest[:16]}...  ({path})")
    return {"version": version, "sha256": digest, "path": path}


def wrap(real_nvcc: str) -> tuple[str, dict]:
    """Point env at the wrapper for `real_nvcc`, returning `(wrapper_path, env_updates)`.

    Callers `os.environ.update(env_updates)` (or pass them to `subprocess.run`'s
    `env=`) BEFORE invoking anything with the returned path as the compiler. The
    wrapper itself performs no resolution (see module docstring); `real_nvcc` must
    already be the rule-1-gated path.
    """
    return str(WRAPPER_PATH), {"ROLA_REAL_NVCC": real_nvcc}
