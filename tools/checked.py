# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE CHECKER'S SEAM: warpline's checker (`warpline.checker`) as this tree calls it, in one place -- the lab's check
(`tools/lab.py`) and the production SASS gate (`tools/sass_gate.py`) both read compiled code through it, so the model and the
rules are loaded once and one way.

    checker()                 the checker's modules, imported from the directory the graph hands in (`ROLA_WARPLINE_PYTHON`)
    kernels_of(cubin)         the checker's model of every kernel of one cubin, each by its mangled name (the graph's toolkit
                              holds no demangler)

The disassemblers are `tools/sass.py`'s (the pinned toolkit's, or the graph's). -- see docs/internals/tools/sass_gate.md#checker"""
from __future__ import annotations

import functools
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).absolute().parent))
import sass  # noqa: E402 -- path insert must precede this import


@functools.cache
def checker() -> SimpleNamespace:
    """warpline's checker modules, imported where first needed: the package's directory is handed in (`bazel run
    //tools:lab`, `bazel run //tools:sass_gate`, the graph's gate action, a test tier)."""
    root = os.environ.get("ROLA_WARPLINE_PYTHON")
    if not root:
        import dev_config  # the graph's gate action has no dev config: it names the directory itself

        root = dev_config.warpline_python()
    if root is None:
        raise SystemExit("the checker is warpline's, which the graph hands the tool: bazel run //tools:<tool> -- ...")
    if root not in sys.path:
        sys.path.insert(0, root)
    from warpline.checker import declare, model, readers, rules
    from warpline.checker import notices as ledgers
    from warpline.checker.rules import _steps as steps
    return SimpleNamespace(declare=declare, model=model, readers=readers, rules=rules, ledgers=ledgers, steps=steps,
                           root=Path(root))


def kernels_of(cubin: Path) -> list:
    """The checker's model of every kernel of `cubin`: `cuobjdump`'s listing and resource report and `nvdisasm`'s graph of it."""
    def run(*argv: str) -> str:
        return subprocess.run(argv, capture_output=True, text=True, check=True).stdout

    c = checker()
    listing = run(sass.tool("cuobjdump"), "-sass", str(cubin))
    usage = run(sass.tool("cuobjdump"), "-res-usage", str(cubin))
    graphs = run(sass.tool("nvdisasm"), "-bbcfg", "-poff", str(cubin))
    return list(c.model.kernels_from_text(listing, graphs, usage).values())
