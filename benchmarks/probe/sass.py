"""The `sass` subcommand: codegen identity and the register/spill census, unforked.

Neither the disassembly nor the compile flags are re-implemented here.
`tools/ratify.py` owns the flag list the build uses and `tools/sass_bodies.py` owns
what "the same code" means; this subcommand is the front door to both, so that the
hand-transcribed `nvcc` command lines that campaign scripts kept beside them --
admitted in their own comments to be "ratify.py's EXACT flags" copied by hand -- have
nothing left to copy.

**The map is path-dependent and this refuses to pretend otherwise.** `decode.cu` and
`entmax.cu` place their kernels in an anonymous namespace, so the mangled name embeds
a discriminator derived from the SOURCE PATH. Two maps are comparable only if both
binaries were built at the same path; `--diff-against` therefore takes a map produced
by this same tool and states the requirement rather than checking a condition it
cannot see.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))


def _built_extension() -> Path:
    so = next((ROOT / "rola").glob("_C*.so"), None)
    if so is None:
        raise SystemExit("no built extension under rola/; build it first: "
                         "python setup.py build_ext --inplace")
    return so


def run(*, out_json: str | None, diff_against: str | None, census: bool) -> int:
    import sass_bodies

    if census:
        #: The census is ratify's, run through ratify, at ratify's flags. Shelling
        #: out rather than importing keeps its argument surface the one documented
        #: place a reader has to learn.
        return subprocess.run(
            [sys.executable, str(ROOT / "tools" / "ratify.py"), "--census"],
            cwd=str(ROOT)).returncode

    so = _built_extension()
    got = sass_bodies.bodies(so)
    print(f"{len(got)} entry points hashed in {so.name}")
    if out_json:
        Path(out_json).write_text(json.dumps(got, indent=1, sort_keys=True) + "\n")
        print(f"written to {out_json}")
    if diff_against:
        before = json.loads(Path(diff_against).read_text())
        print("\nBOTH SIDES MUST HAVE BEEN BUILT AT THE SAME PATH -- the anonymous-"
              "namespace discriminator in decode/entmax names is derived from it.")
        return 0 if sass_bodies.compare(before, got, diff_against, so.name) else 1
    return 0
