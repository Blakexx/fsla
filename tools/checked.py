# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE LAB'S SEAM TO THE CHECKER: warpline's checker (`warpline.checker`) as the lab (`tools/lab.py`) calls it, in one place, so the
model and the rules are loaded once and one way. The production SASS gate is warpline's own check in the graph (bazel/defs.bzl's
`_sass_gate`); the judgment here is the one it makes, the manifest of the kernels compared with the declarations
(`warpline.declare.compare`), on the same project ledger and the same notes (the `checker` section of `docs/rules.json`).

    checker()                       the checker's modules, imported from the directory `dev_config.warpline_python()` names; in the
                                    lab's graph actions, which have no dev config, the one they hand in (`FSLA_WARPLINE_PYTHON`)
    kernels_in(listing, usage, ...) the checker's model of the disassembly texts alone, named by the mangled name or by the demangled names given
                                    (the lab's built experiments carry the texts and the names: no tool runs)
    project_ledger()                the notice ledger every build is read against: the checker's base rows with the project's (`tools/notice_ledger.json`)
    declarations(entries)           the checker's declarations of `{"kernel": regex, "rule", "allow", "reason"}` entries (the lab's, from `b.expect`:
                                    the production gate's form)
    check(kernels, rules, ...)      THE ONE JUDGMENT: the manifest of the kernels under the named rules and of the build's notices (counted under their
                                    rows' rules), compared with the declarations under fsla's notes: its refusals
    profile_refusal(dump, ...)      the profile rule `shared_wavefronts_excess` on one kernel's SourceCounters dump, its declaration a CTA-window's: its refusal or None

-- see docs/internals/tools/lab.md#hazards"""
from __future__ import annotations

import dataclasses
import functools
import json
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace


def _dev_config():
    """The dev config, loaded where a lookup needs it: the lab's graph actions have none (no devtools), and name the directory themselves."""
    import dev_config

    return dev_config


@functools.cache
def checker() -> SimpleNamespace:
    """warpline's checker modules, imported where first needed: the package's directory is handed in (`bazel run //tools:lab`, the
    lab's graph actions, a test tier)."""
    root = os.environ.get("FSLA_WARPLINE_PYTHON") or _dev_config().warpline_python()
    if root is None:
        raise SystemExit("the checker is warpline's, which the graph hands the tool: bazel run //tools:lab -- ...")
    if root not in sys.path:
        sys.path.insert(0, root)
    from warpline import declare
    from warpline.checker import manifest, model, ncu, readers, rules
    from warpline.checker import notices as ledgers
    from warpline.checker.rules import _steps as steps
    return SimpleNamespace(declare=declare, manifest=manifest, model=model, ncu=ncu, readers=readers, rules=rules, ledgers=ledgers, steps=steps,
                           root=Path(root))


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


#: what a declaration of a PERMIT allows: the signature an experiment means to keep, whatever its count (no bound): the lab's presence
#: expectations (`STEP_XU`, `BSSY`, a `CALL` opcode) are declarations of this allowance
PERMIT = 10**9

#: the project's ledger of the toolkit's notices: its rows override or add to the checker's base ledger (the rows that only repeat the base are not
#: here), the default every build's notices are read against -- the lab's and the production gate's (bazel/defs.bzl's `_SASS_GATE_ATTRS`)
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
    return c.declare.load_declarations(json.dumps({"declarations": entries}), c.rules.registry(), c.manifest.rule_names(project_ledger()))


def check(kernels: list, rules: Iterable[str], declarations: list, notes: dict[str, str], notices: list | None = None) -> list:
    """THE ONE JUDGMENT, the production gate's on a manifest: the manifest of `kernels` under the rules named (the model rules of the
    checker's registry) and of the build's `notices` (read, counted under their rows' rules of `project_ledger()` at the function they
    name; none: a build with no messages), compared with the loaded `declarations` and fsla's `notes` (a line a rule, printed after the
    checker's fix): the refusals."""
    c = checker()
    registry = c.rules.registry()
    manifest = c.manifest.collect(kernels, {n: registry[n] for n in sorted(rules)}, notices=notices, ledger=project_ledger())
    return c.declare.compare(manifest, declarations, notes).refusals


def profile_refusal(dump: str, kernel: str, declarations: list, notes: dict[str, str], windows: float | None):
    """The checker's profile rule `shared_wavefronts_excess` on `kernel`'s SourceCounters `dump` (the text of its header and
    instruction rows), under the loaded `declarations` and fsla's `notes`: its refusal, or None where within what is declared.
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
