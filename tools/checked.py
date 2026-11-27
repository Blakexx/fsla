# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE SEAM TO WARPLINE: warpline's checker (`warpline.checker`) as this tree's tools call it, in one place, so the model, the
rules and the readers are loaded once and one way. warpline reads and checks the SASS and the profiler's exports; what is fsla's is
which code is which phase (`tools/region_ledger.py`'s attribution) and the budgets. The production SASS gate is warpline's own check
in the graph (bazel/defs.bzl's `_sass_gate`); the judgment here is the one it makes, the manifest of the kernels compared with the
declarations (`warpline.declare.compare`), on the same project ledger and the same notes (the `checker` section of `docs/rules.json`).

    checker()                       the checker's modules, imported from the directory `dev_config.warpline_python()` names; in the
                                    lab's graph actions, which have no dev config, the one they hand in (`FSLA_WARPLINE_PYTHON`)
    cubin(so, member, arch)         the one image of a built extension the device ran, from warpline's cubin cache in the dev config's
                                    scratch, on the graph's toolkit (`runner.cubin`)
    kernel_of(cubin, kernel)        the checker's model of the one function `kernel` names, its registers by the life ranges
    frames(listing, kernel, where)  SASS offset -> the inline frame chain of the one function `kernel` names (`readers.inline_frames`)
    source_page(path)               a SourceCounters dump's source page (`ncu.source_rows`), past the profiler's own lines
    device_arch()                   the running device's architecture as a cubin names it
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
import re
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
    from warpline import manifest as formats
    from warpline.checker import flow, manifest, model, ncu, readers, rules, runner
    from warpline.checker import notices as ledgers
    from warpline.checker.rules import _steps as steps
    return SimpleNamespace(declare=declare, formats=formats, flow=flow, manifest=manifest, model=model, ncu=ncu, readers=readers, rules=rules,
                           runner=runner, ledgers=ledgers, steps=steps, root=Path(root))


def cubin(so: Path, member: str, arch: str) -> Path:
    """The one cubin of the built extension `so` whose name holds `member` and `arch` (`runner.cubin`: a fat binary holds an image an
    architecture, and a launch is attributed to the image the device ran), extracted once a build into the dev config's scratch, by the
    graph's toolkit."""
    return checker().runner.cubin(Path(so), _dev_config().scratch("cubins"), member, arch, _dev_config().cuda_home())


def one_kernel(names: list[str], kernel: str | None, where: str) -> str:
    """The one name of `names` matching the regex `kernel` -- or the only name, with none given; refuses otherwise, so a reader never
    merges two functions' offsets (they restart in every function)."""
    found = [n for n in names if kernel is None or re.search(kernel, n)]
    if len(found) != 1:
        raise SystemExit(f"{where}: {len(found)} functions match {kernel!r} among {names}; name one (`--kernel`)")
    return found[0]


def kernel_of(cubin_path: Path, kernel: str | None):
    """The checker's model of the one function of `cubin_path` the regex `kernel` names (over the mangled names), its registers the
    life-range listing's (`runner.kernels_of(..., life=True)`), on the graph's toolkit."""
    found = checker().runner.kernels_of(str(cubin_path), toolkit=_dev_config().cuda_home(), life=True)
    return found[one_kernel(list(found), kernel, Path(cubin_path).name)]


def inline_listing(cubin_path: Path) -> str:
    """`nvdisasm --print-line-info-inline` of a cubin, by the graph's toolkit (`runner.inline_listing`)."""
    return checker().runner.inline_listing(Path(cubin_path), _dev_config().cuda_home())


def frames(listing: str, kernel: str | None, where: str = "the listing") -> dict[int, list[tuple[str, int]]]:
    """SASS offset -> the inline frame chain at that instruction, innermost first, as (file name, line), of the one function of a
    listing with inline frames the regex `kernel` names (`readers.inline_frames`; `one_kernel`): the path cut to the file's name,
    which is how an attribution names a source."""
    by_function = checker().readers.inline_frames(listing)
    name = one_kernel(list(by_function), kernel, where)
    return {at: [(f.rsplit("/", 1)[-1], line) for f, line in chain] for at, chain in by_function[name].items()}


def source_page(path: Path):
    """The source page of a SourceCounters dump (`ncu --section SourceCounters --page source --print-source sass --csv`, of ONE launch)
    as the checker reads it (`ncu.source_rows`): from its `Kernel Name` line, its header and its instruction rows, so the profiler's own
    lines printed into the dump (`==PROF==`) are not the page."""
    lines = Path(path).read_text().splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith('"Kernel Name"')), None)
    if start is None:
        raise SystemExit(f"{path} is no SourceCounters dump: no `Kernel Name` line")
    return checker().ncu.source_rows("\n".join(line for i, line in enumerate(lines[start:]) if i < 2 or line.startswith('"0x')))


def device_arch() -> str:
    """The running device's architecture as a cubin names it (`sm_90a`): what a launch profiled on this card ran."""
    import torch

    sys.path.insert(1, str(Path(__file__).resolve().parents[1]))
    from hardware import registry

    return registry.device_arch(*torch.cuda.get_device_capability())


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
