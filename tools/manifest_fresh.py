"""THE COMMIT-TIME MANIFEST FRESHNESS GATE.

WHY THIS EXISTS, AND WHY THE BUILD'S OWN CHECK WAS NOT ENOUGH.  `setup.py`
already refuses to finish a build whose sources no longer match the committed
manifest, and that check is sound -- but it only fires WHEN YOU BUILD.  A
final source edit that does not change behaviour (a comment sweep, a renamed
marker, a doc block) is exactly the edit an author takes straight to a commit
without rebuilding: the binary in hand is still behaviourally right, the
device stamp still matches, the test battery is still green, and `git status`
is clean.  Nothing in that sequence looks wrong, and the manifest ships one
edit stale -- describing a source tree that no longer exists.

That is a reproducibility defect, not a cosmetic one.  The manifest is the
record of WHICH SOURCES the ratified register and spill numbers were measured
against; a reader who cannot reproduce the digest cannot trust the numbers.
Observed once for real (K35-E, commit `07c4409`, corrected at `7ee9278`).

TWO RULES, and this gate is the second one made structural:

  1. Ratify AFTER the last source edit -- `edit; build; ratify --write; commit`.
  2. A clean `git status` is NOT evidence the manifest matches the source.
     Only recomputing the digest is.

THE COST.  Hashing the source tree, and nothing else.  No compile, no device,
no toolchain -- which is what lets it run on every commit and in the CPU lane
of CI.  It answers a strictly weaker question than `ratify.py --verify` (which
recompiles every translation unit to check the BOUNDS): this asks only whether
the manifest is describing the sources it ships beside.  That weaker question
is the one that was being missed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ratify  # noqa: E402


def stale_arches(root: Path, arches=ratify.DEFAULT_ARCHES) -> list[str]:
    """The arches whose manifest does not describe `root`'s sources.

    Returns a list of human-readable complaints, empty when the tree is clean.
    A MISSING manifest is a complaint too: an unratified arch is not a fresh
    one, and silence there would be the same hole in a different shape.
    """
    src = root / "csrc" / "rola" / "src"
    if not src.is_dir():
        return [f"no kernel sources at {src}"]
    digest = ratify.csrc_digest(src)
    out = []
    for arch in arches:
        path = root / "tools" / "manifests" / f"sm_{arch}.json"
        if not path.is_file():
            out.append(f"sm_{arch}: no manifest at {path}")
            continue
        try:
            recorded = json.loads(path.read_text())["source"]["csrc_sha256"]
        except (ValueError, KeyError) as exc:
            out.append(f"sm_{arch}: unreadable source record ({exc})")
            continue
        if recorded != digest:
            out.append(
                f"sm_{arch}: manifest was ratified against csrc {recorded[:16]}..., "
                f"but the tree's sources digest to {digest[:16]}..."
            )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                    help="repository root to check (default: this checkout)")
    ap.add_argument("--arch", action="append",
                    help=f"arch to check, repeatable (default: {' '.join(ratify.DEFAULT_ARCHES)})")
    ap.add_argument("filenames", nargs="*", help=argparse.SUPPRESS)
    a = ap.parse_args()
    arches = tuple(a.arch) if a.arch else ratify.DEFAULT_ARCHES
    problems = stale_arches(a.root.resolve(), arches)
    if not problems:
        print(f"manifest freshness OK for {' '.join('sm_' + x for x in arches)}")
        return 0
    print("FAIL  STALE MANIFEST. The manifests do not describe the sources beside them.\n")
    for p in problems:
        print(f"  {p}")
    print(
        "\nThe ratified register and spill numbers were measured against DIFFERENT\n"
        "sources than the ones being committed, so they cannot be reproduced.\n"
        "The order is: edit; build; `python tools/ratify.py --write`; commit --\n"
        "ratify runs after the LAST source edit, including a comment-only one."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
