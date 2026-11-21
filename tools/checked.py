# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE CHECKER'S SEAM: warpline's checker (`warpline.checker`) as this tree calls it, in one place -- the lab's check
(`tools/lab.py`) and the production SASS gate (`tools/sass_gate.py`) both read compiled code through it, so the model and the
rules are loaded once and one way.

    checker()                       the checker's modules, imported from the directory `dev_config.warpline_python()` names; in the
                                    graph's gate action, which has no dev config, the one it hands in (`ROLA_WARPLINE_PYTHON`)
    kernels_of(binary, demangled)   the checker's model of every kernel of a cubin or of a binary's images, each by its mangled name,
                                    or by the toolkit's `cu++filt` name where `demangled` (the lab's patterns are of that form)
    check(kernels, rules, ...)      the named rules over the kernels, under loaded declarations and rola's notes: the refusals
    profile_refusal(dump, ...)      the profile rule `shared_wavefronts_excess` on one kernel's SourceCounters dump: its refusal or None
    demangle(names)                 `cu++filt` of mangled names, each mapped to its own (the graph's toolkit holds no demangler)

The disassemblers are `tools/sass.py`'s (the pinned toolkit's, or the graph's). -- see docs/internals/tools/sass_gate.md#checker"""
from __future__ import annotations

import functools
import os
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).absolute().parent))
import sass  # noqa: E402 -- path insert must precede this import


def _dev_config():
    """The dev config, loaded where a lookup needs it: the graph's gate action has none (no devtools), and names the directory itself."""
    import dev_config

    return dev_config


@functools.cache
def checker() -> SimpleNamespace:
    """warpline's checker modules, imported where first needed: the package's directory is handed in (`bazel run
    //tools:lab`, `bazel run //tools:sass_gate`, the graph's gate action, a test tier)."""
    root = os.environ["ROLA_WARPLINE_PYTHON"] if sass.in_graph() else _dev_config().warpline_python()
    if root is None:
        raise SystemExit("the checker is warpline's, which the graph hands the tool: bazel run //tools:<tool> -- ...")
    if root not in sys.path:
        sys.path.insert(0, root)
    from warpline.checker import declare, model, ncu, readers, rules
    from warpline.checker import notices as ledgers
    from warpline.checker.rules import _steps as steps
    return SimpleNamespace(declare=declare, model=model, ncu=ncu, readers=readers, rules=rules, ledgers=ledgers, steps=steps,
                           root=Path(root))


def demangle(names: list[str]) -> dict[str, str]:
    """Each mangled name of `names` with the toolkit's own `cu++filt` name."""
    out = subprocess.run([_dev_config().cuda_bin("cu++filt")], input="\n".join(names), capture_output=True, text=True,
                         check=True).stdout.splitlines()
    return dict(zip(names, out, strict=True))


def kernels_of(binary: Path, demangled: bool = False) -> list:
    """The checker's model of every kernel of `binary` (a cubin, or a binary whose images `sass.cubins` extracts): `cuobjdump`'s
    listing and resource report and `nvdisasm`'s graph of each image; named by the mangled name, or demangled where `demangled`."""
    def run(*argv: str) -> str:
        return subprocess.run(argv, capture_output=True, text=True, check=True).stdout

    c = checker()
    listing = run(sass.tool("cuobjdump"), "-sass", str(binary))
    usage = run(sass.tool("cuobjdump"), "-res-usage", str(binary))
    graphs = "\n".join(run(sass.tool("nvdisasm"), "-bbcfg", "-poff", str(cubin)) for cubin in sass.cubins(binary))
    if not demangled:
        return list(c.model.kernels_from_text(listing, graphs, usage).values())
    mangled = list(c.readers.sass_functions(listing))
    pretty = demangle(mangled)
    return list(c.model.kernels_from_text(listing, graphs, usage, "".join(f"{pretty[m]}\n" for m in mangled)).values())


def check(kernels: list, rules: Iterable[str], declarations: list, notes: dict[str, str]) -> list:
    """The checker's refusals of `kernels` under the rules named (the model rules of its registry), the loaded `declarations` and
    rola's `notes` (a line a rule, printed after the checker's fix)."""
    registry = checker().rules.registry()
    return checker().declare.check(kernels, {n: registry[n] for n in sorted(rules)}, declarations, notes)


def profile_refusal(dump: str, kernel: str, declarations: list, notes: dict[str, str]):
    """The checker's profile rule `shared_wavefronts_excess` on `kernel`'s SourceCounters `dump` (the text of its header and
    instruction rows), under the loaded `declarations` and rola's `notes`: its refusal, or None where within what is declared."""
    c = checker()
    profile, = c.ncu.profiles_of([dump])
    rule = c.rules.profile_registry()["shared_wavefronts_excess"]
    return c.declare.judge(kernel, rule.name, rule.check(profile), rule.refusal, rule.fix, declarations, notes)
