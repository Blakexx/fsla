# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE CHECKER'S SEAM: warpline's checker (`warpline.checker`) as this tree calls it, in one place -- the lab's check
(`tools/lab.py`) and the production SASS gate (`tools/sass_gate.py`) both read compiled code through it, so the model and the
rules are loaded once and one way.

    checker()                       the checker's modules, imported from the directory `dev_config.warpline_python()` names; in the
                                    graph's gate action, which has no dev config, the one it hands in (`ROLA_WARPLINE_PYTHON`)
    kernels_of(binary)              the checker's model of every kernel of a cubin or of a binary's images, each by its mangled name
    kernels_in(listing, usage, ...) the same model of the disassembly texts alone, named by the mangled name or by the demangled names given
                                    (the lab's built experiments carry the texts and the names: no tool runs)
    project_ledger()                the notice ledger every build is read against: the checker's base rows with the project's (`tools/notice_ledger.json`)
    declarations(entries)           the checker's declarations of `{"kernel": regex, "rule", "allow", "reason"}` entries (the production gate's, from the
                                    graph's declarations beside the arms, and the lab's, from `b.expect`, are this one form)
    compare(kernels, rules, ...)    THE ONE JUDGMENT, the lab's and the production gate's: the facts of the kernels under the named rules and of the build's
                                    notices (counted under their rows' rules), compared with the declarations under rola's notes: (comparison, facts)
    check(kernels, rules, ...)      its refusals
    profile_refusal(dump, ...)      the profile rule `shared_wavefronts_excess` on one kernel's SourceCounters dump, its declaration a CTA-window's: its refusal or None

The disassemblers are `tools/sass.py`'s (the pinned toolkit's, or the graph's). -- see docs/internals/tools/sass_gate.md#checker"""
from __future__ import annotations

import dataclasses
import functools
import json
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
    root = os.environ.get("ROLA_WARPLINE_PYTHON") or _dev_config().warpline_python()
    if root is None:
        raise SystemExit("the checker is warpline's, which the graph hands the tool: bazel run //tools:<tool> -- ...")
    if root not in sys.path:
        sys.path.insert(0, root)
    from warpline.checker import declare, facts, model, ncu, readers, rules
    from warpline.checker import notices as ledgers
    from warpline.checker.rules import _steps as steps
    return SimpleNamespace(declare=declare, facts=facts, model=model, ncu=ncu, readers=readers, rules=rules, ledgers=ledgers, steps=steps,
                           root=Path(root))


def kernels_of(binary: Path) -> list:
    """The checker's model of every kernel of `binary` (a cubin, or a binary whose images `sass.cubins` extracts): `cuobjdump`'s
    listing and resource report and `nvdisasm`'s graph of each image; named by the mangled name, each also by its demangled name
    (`cu++filt`'s, which the declarations' patterns are over)."""
    def run(*argv: str, text: str | None = None) -> str:
        return subprocess.run(argv, input=text, capture_output=True, text=True, check=True).stdout

    listing = run(sass.tool("cuobjdump"), "-sass", str(binary))
    usage = run(sass.tool("cuobjdump"), "-res-usage", str(binary))
    graphs = "\n".join(run(sass.tool("nvdisasm"), "-bbcfg", "-poff", str(cubin)) for cubin in sass.cubins(binary))
    mangled = list(checker().readers.sass_functions(listing))
    pretty = run(sass.tool("cu++filt"), text="".join(f"{name}\n" for name in mangled)).splitlines() if mangled else []
    return kernels_in(listing, usage, graphs, dict(zip(mangled, pretty, strict=True)))


def kernels_in(listing: str, usage: str, graphs: str, names: dict[str, str] | None = None) -> list:
    """The checker's model of the kernels a binary's disassembly texts describe (`listing`: `cuobjdump -sass`, `usage`: its
    `-res-usage`, `graphs`: `nvdisasm -bbcfg -poff` of each image), named by the mangled name, or by `names`' (mangled to demangled,
    one for each function of the listing) where given. The lab reads its built experiments' texts through this, in the graph's
    actions and out of them alike."""
    c = checker()
    if names is None:
        return list(c.model.kernels_from_text(listing, graphs, usage).values())
    mangled = list(c.readers.sass_functions(listing))
    return list(c.model.kernels_from_text(listing, graphs, usage, "".join(f"{names[m]}\n" for m in mangled)).values())


#: what a declaration of a PERMIT allows: the signature a probe or an experiment means to keep, whatever its count (no bound): the lab's presence
#: expectations (`STEP_XU`, `BSSY`, a `CALL` opcode) and a production probe's kept signatures are declarations of this allowance
PERMIT = 10**9

#: the project's ledger of the toolkit's notices: its rows override or add to the checker's base ledger (the rows that only repeat the base are not
#: here), the default every build's notices are read against -- the lab's and the production gate's
PROJECT_LEDGER = Path(__file__).absolute().with_name("notice_ledger.json")


@functools.cache
def project_ledger() -> list:
    """The ledger the checker reads a build's notices against: its base ledger with the project's rows (`PROJECT_LEDGER`)."""
    ledgers = checker().ledgers
    return ledgers.merge(ledgers.base_ledger(), ledgers.load_ledger(PROJECT_LEDGER.read_text()))


def declarations(entries: list[dict]) -> list:
    """The checker's loaded declarations of `entries` (`kernel`: a regex over the demangled name, `rule`, `allow`, `reason`): the rules of
    the model, the profiles and the project's ledger are the rules a declaration may name."""
    c = checker()
    return c.declare.load_declarations(json.dumps({"declarations": entries}), c.rules.registry(), c.declare.rule_names(project_ledger()))


def compare(kernels: list, rules: Iterable[str], declarations: list, notes: dict[str, str], notices: list | None = None):
    """THE ONE JUDGMENT: the facts of `kernels` under the rules named (the model rules of the checker's registry) and of the build's `notices`
    (read, counted under their rows' rules of `project_ledger()` at the function they name; none: a build with no messages), compared with
    the loaded `declarations` and rola's `notes` (a line a rule, printed after the checker's fix). The lab's check and the production gate both
    judge through it; returns the comparison (the refusals, the figures of bound rules with no bound declared, the stale declarations) and the facts
    it compared (every rule's count and sites at every place: the production gate's report reads its figures from them)."""
    c = checker()
    registry = c.rules.registry()
    facts = c.facts.collect(kernels, {n: registry[n] for n in sorted(rules)}, notices=notices, ledger=project_ledger())
    return c.declare.compare(facts, declarations, notes), facts


def check(kernels: list, rules: Iterable[str], declarations: list, notes: dict[str, str], notices: list | None = None) -> list:
    """The refusals of `compare`."""
    return compare(kernels, rules, declarations, notes, notices)[0].refusals


def profile_refusal(dump: str, kernel: str, declarations: list, notes: dict[str, str], windows: float | None):
    """The checker's profile rule `shared_wavefronts_excess` on `kernel`'s SourceCounters `dump` (the text of its header and
    instruction rows), under the loaded `declarations` and rola's `notes`: its refusal, or None where within what is declared.
    A declaration counts the excess wavefronts of a CTA-WINDOW of the profiled launch, never the launch's total (a launch's
    total moves with the grid and the windows a row times): `windows`, the launch's CTA-windows, scales it to the launch's
    allowance, and the refusal's `count` and `allowed` are the launch's, to be shown beside a window's. Where `windows` is
    None (the profile's CTA-windows are not stated) no declaration can be read, and none allows."""
    c = checker()
    profile, = c.ncu.profiles_of([dump])
    rule = c.rules.profile_registry()["shared_wavefronts_excess"]
    scaled = [dataclasses.replace(d, allow=int(d.allow * windows + 1e-6)) if d.rule == rule.name else d
              for d in declarations if windows is not None or d.rule != rule.name]
    return c.declare.judge(kernel, rule.name, rule.check(profile), rule.refusal, rule.fix, scaled, notes)
