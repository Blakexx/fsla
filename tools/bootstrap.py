# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""WHAT A TOOL KNOWS BEFORE THE DEV CONFIG'S READER IS IMPORTABLE: where the sibling checkouts are, and the
rola-devtools checkout the reader comes from. Run, it is the machine's one step outside the graph:

    python3 tools/bootstrap.py      # the pinned bazelisk installed and `bazel` linked to it; then `bazel run //:init`


The reader (`rola_devtools.config`) and the host budget (`rola_devtools.locks.host`) are standard-library Python in the
rola-devtools checkout. The graph's front door (`bazel/front_door.py`) runs before any venv exists, under the
machine's own `python3`, so it reads one value here first -- `workspace.devtools` -- and imports the reader from that
checkout. Standard library only. Docs: docs/internals/tools/dev_config.md#bootstrap.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: THE DIRECTORY THE CHECKOUTS SHARE: a worktree's checkout sits in the worktrees folder, beside the others' parent
SIBLINGS = (ROOT.parent if ROOT.parent.name == "worktrees" else ROOT).parent
#: `workspace.devtools`' default
DEVTOOLS = SIBLINGS / "rola-devtools"
#: `rola_devtools.config`'s directory rule, read here because the reader is not importable yet
POINTER = "ROLA_DEV_CONFIG"


def devtools() -> Path:
    """The rola-devtools checkout: `workspace.devtools` from the config's workspace section, else its default."""
    pointer = os.environ.get(POINTER)
    section = (Path(pointer).expanduser() if pointer else Path.home() / ".config" / "rola") / "workspace.json"
    value = json.loads(section.read_text()).get("devtools") if section.is_file() else None
    return Path(value).expanduser() if value else DEVTOOLS


def main() -> int:
    """The pinned bazelisk and its `bazel` link (`tools/dev.py`'s `install_bazelisk`), by the machine's own python3 with
    the reader imported from the devtools checkout, as the front door imports it."""
    sys.path[:0] = [str(devtools()), str(ROOT / "tools")]
    import dev

    dev.install_bazelisk()
    print("bootstrap: `bazel` is the pinned bazelisk; next, bazel run //:init")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
