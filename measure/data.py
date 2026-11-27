# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE PINNED DATA: the data repository (fsla-results) at the ONE commit this tree pins (MODULE.bazel's
`environment.data`, the repository `@fsla_data`), where every record a doc cites (`data:lab/...`) and a tool reads lives.
The graph hands its root to what reads it as `FSLA_DATA`: the lab's run targets (bazel/defs.bzl's `lab_run`), and the
checks and tiers that read a record (`repos`). Stdlib only. docs/measurement.md#the-data.
"""
from __future__ import annotations

import os
from pathlib import Path

#: the handoff that names the pinned data's root (tools/dev_config.py's HANDOFFS)
VARIABLE = "FSLA_DATA"
#: how a doc and a tool's argument name a file of the pinned data: `data:lab/<experiment>/<arch>.json`
PREFIX = "data:"


def root() -> Path:
    """The pinned data's root, as the graph handed it."""
    value = os.environ.get(VARIABLE)
    if not value:
        raise SystemExit(f"{VARIABLE} is unset: run this through its Bazel target, which hands it the pinned data "
                         "(@fsla_data, MODULE.bazel's `environment.data`)")
    return Path(value)


def lab(name: str) -> Path:
    """An experiment's records in the pinned data: `lab/<name>/`."""
    return root() / "lab" / name


def resolve(spelled: str | Path) -> Path:
    """A path a tool is given: `data:<path>` is that file of the pinned data, anything else a path as it is."""
    text = str(spelled)
    return root() / text[len(PREFIX):] if text.startswith(PREFIX) else Path(text)
