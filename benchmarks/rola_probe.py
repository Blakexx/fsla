"""`rola-probe`'s entry point, beside the harness it drives.

    python benchmarks/rola_probe.py --help

Not a console script in `pyproject.toml`: the installed wheel ships `rola/`, not
`benchmarks/`, and a `rola-probe` on PATH that cannot import `bench` would be a
command that exists and does not work. The invocation is the one every other driver
in this directory already uses.

The GPU lock is taken by the harness, so this line carries no `flock`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
#: AND the repo root, AHEAD of it: `rola` is installed EDITABLE into the shared
#: venv, so without this a worktree run imports the MAIN tree's extension while
#: stamping this tree's sha (P11's discarded A/B). `probe.discipline` REFUSES if
#: the two still disagree.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from probe.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
