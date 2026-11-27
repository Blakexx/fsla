"""Generate the support tables: the README's summary and the contract's page (docs/support.md). **A REPORT, NOT A CLAIM.**

A hand-maintained support table says what someone hoped was true when
they last edited it; this one is derived from the CONTRACT, the registry of run-time contexts FSLA accepts (``hardware/``,
docs/internals/hardware/registry.md), and the toolchain that builds each context's architecture. The build fails if any context of the contract
reaches no arm (``arms_gen``), so the published support and the guaranteed support are one file. The commit gate fails on drift
(`//:readme`, which holds the committed README to what the graph writes, docs/build-graph.md#committed-files):

* THE SUMMARY (the README's block): a row a kernel and architecture, in a reader's terms -- the card, the depth, a level's
  width, the value width, the activations, the toolchain, built or not -- read from the call facts each column names
  (`SUMMARY`); a kernel whose descriptors are all owed is a row "not built".
* THE CONTRACT (docs/support.md's block): each kernel's contexts (``hardware/kernels/<kernel>/descriptors``, ``hardware.registry.contexts``), one table a kernel that has a built contract: each descriptor's architecture, the GPUs it covers and, for every fact the kernel's
  call supplies, the values the contract admits.
* the BUILT architectures (``hardware.registry.built``: the architectures a device part states) and the toolchain that builds them
  (``tools/toolchains.py``). A contract architecture the build does not build is refused here: the contract is what is built.

    bazel run //tools:supported              # print the tables
    bazel run //:readme_update               # write README.md's block (the graph's action runs
                                             # `supported.py --readme README.md --out <file>`)
    bazel run //:support_update              # write docs/support.md's block, the same way
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

#: unresolved: in the graph's action the script is a link, and its siblings are the action's declared inputs beside it
ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(1, str(ROOT))
import toolchains  # noqa: E402

from hardware import registry  # noqa: E402 -- the contract

#: The generated regions: `marker -> producer`. A file holds the markers of the blocks it shows (the README the summary,
#: docs/support.md the contract), and no hand edit may touch one.
BLOCKS = {
    "tools/supported.py": lambda: summary(),
    "tools/supported.py --contract": lambda: table(),
}

#: A `levels` call fact is the hierarchy's depth when its unit says so; a launch's levels (the routing solve's) are not.
DEPTH_UNITS = ("hierarchy levels", "levels of the density")
LAUNCH_UNITS = ("levels of one launch",)


def markers(marker: str) -> tuple[str, str]:
    """A generated block's two markers in the README."""
    return f"<!-- BEGIN GENERATED: {marker} -->", f"<!-- END GENERATED: {marker} -->"


def block(text: str, marker: str) -> str | None:
    """The README's generated block for `marker`, its text between the markers; None when it has none. The one reader
    of the format: `render` reads it here."""
    begin, end = markers(marker)
    if begin not in text or end not in text:
        return None
    return text.split(begin, 1)[1].split(end, 1)[0].strip("\n")


def built() -> dict[str, list[dict]]:
    """`sm_XX -> [{toolchain, ptxas}]`: each built architecture, with the toolchain that builds it."""
    return {arch: [{"toolchain": t.name, "ptxas": t.ptxas} for t in toolchains.records().values()]
            for arch in registry.built()}


def _ptxas_short(version: str) -> str:
    m = re.search(r"release ([\d.]+), V([\d.]+)", version)
    return f"ptxas {m.group(2)}" if m else version


def _spell(values: set) -> str:
    """A fact's admitted values: points as a sorted list, a range as `lo..hi`."""
    points = sorted(v[0] for v in values if v[0] == v[1])
    ranges = sorted(f"{v[0]}..{v[1]}" for v in values if v[0] != v[1])
    return ", ".join([str(p) for p in points] + ranges)


def contract(kernel: str) -> dict[tuple[str, str], dict[str, set]]:
    """`(arch, gpus) -> {call fact: the values the contract admits}`, over every context of the kernel's registry."""
    calls = list(registry.kernel_schema(kernel))
    out: dict[tuple[str, str], dict[str, set]] = {}
    for c in registry.contexts(kernel):
        seen = out.setdefault((c.arch, c.gpus), {n: set() for n in calls})
        for n in calls:
            seen[n].add(c.facts[n])
    return out


def kernel_table(kernel: str) -> str:
    """ONE KERNEL'S TABLE: a row an architecture and GPU label its contract holds, the columns the kernel's own call facts."""
    built_rows = built()
    calls = list(registry.kernel_schema(kernel))
    rows = [f"| Compute capability | GPUs | {' | '.join(calls)} | Toolchain | Assembler | Status |",
            "|---" * (len(calls) + 5) + "|"]
    for (arch, gpus), admitted in sorted(contract(kernel).items()):
        if arch not in built_rows:
            raise SystemExit(f"the {kernel} contract holds {arch}, which the build does not build: the contract is what is built")
        spelled = " | ".join(_spell(admitted[n]) for n in calls)
        for r in built_rows[arch]:
            rows.append(f"| `{arch}` | {gpus} | {spelled} | `{r['toolchain']}` | {_ptxas_short(r['ptxas'])} | built |")
    return "\n".join(rows)


def _span(values: set) -> str:
    """A fact's admitted values for a reader: `lo to hi` where they make one unbroken run, else as `_spell` lists them."""
    lo, hi = min(v[0] for v in values), max(v[1] for v in values)
    covered = set()
    for a, b in values:
        covered.update(range(a, b + 1))
    return f"{lo} to {hi}" if covered == set(range(lo, hi + 1)) and lo != hi else _spell(values)


def _cuda(ptxas: str) -> str:
    m = re.search(r"V(\d+)\.(\d+)", ptxas)
    return f"CUDA {m.group(1)}.{m.group(2)}" if m else ptxas


def _widths(admitted: dict[str, set]) -> str:
    """A level's widths: the `width` fact as it is, or the `width_bits_<l>` facts as powers of two (0, no such level, left out)."""
    if "width" in admitted:
        return _span(admitted["width"])
    bits = [v for name, values in admitted.items() if name.startswith("width_bits_") for v in values]
    lo = min(max(v[0], 1) for v in bits if v[1] > 0)
    hi = max(v[1] for v in bits)
    return f"{2 ** lo} to {2 ** hi}"


def _depth(kernel: str, admitted: dict[str, set]) -> str:
    if "levels" not in admitted:
        return "--"
    unit = registry.kernel_schema(kernel)["levels"].unit
    if unit in DEPTH_UNITS:
        return _span(admitted["levels"])
    if unit in LAUNCH_UNITS:
        return "--"
    raise SystemExit(f"{kernel}'s `levels` fact has the unit {unit!r}, which the summary does not read: name it in "
                     "DEPTH_UNITS or LAUNCH_UNITS")


def _activations(admitted: dict[str, set]) -> str:
    if "alpha_tenths" not in admitted:
        return "--"
    return ", ".join(f"entmax {v[0] / 10:g}" for v in sorted(admitted["alpha_tenths"]))


def summary() -> str:
    """THE README'S TABLE: a row a kernel and architecture, its columns the facts a reader asks for; a kernel with no built
    contract a row "not built"."""
    built_rows = built()
    rows = ["| Kernel | Card | Depth | Level width | Value width (d_v) | Activations | Toolchain | Status |",
            "|---" * 8 + "|"]
    for kernel in registry.kernels():
        held = contract(kernel)
        if not held:
            rows.append(f"| `{kernel}` | -- | -- | -- | -- | -- | -- | not built |")
            continue
        for (arch, gpus), admitted in sorted(held.items()):
            if arch not in built_rows:
                raise SystemExit(f"the {kernel} contract holds {arch}, which the build does not build: the contract is what is built")
            dv = _span(admitted["d_v"]) if "d_v" in admitted else "--"
            for r in built_rows[arch]:
                rows.append(f"| `{kernel}` | {gpus} (`{arch}`) | {_depth(kernel, admitted)} | {_widths(admitted)} | {dv} | "
                            f"{_activations(admitted)} | {_cuda(r['ptxas'])} | built |")
    return "\n".join(rows)


def table() -> str:
    """The support table: a table for each kernel that has a built contract (a kernel whose descriptors are all owed has none), under its name."""
    return "\n\n".join(f"**`{kernel}`**\n\n{kernel_table(kernel)}" for kernel in registry.kernels() if registry.contexts(kernel))


def render(readme: str) -> str:
    """A file's text with each generated block it holds written: the one writer of the blocks. A file holding none is
    refused."""
    new = readme
    held = [marker for marker in BLOCKS if block(new, marker) is not None]
    if not held:
        raise SystemExit(f"the file holds no generated block (the markers: {sorted(BLOCKS)})")
    for marker in held:
        produce = BLOCKS[marker]
        begin, end = markers(marker)
        head, rest = new.split(begin, 1)
        _, tail = rest.split(end, 1)
        new = f"{head}{begin}\n{produce()}\n{end}{tail}"
    return new


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--readme", type=Path, help="the file (the README, docs/support.md) to write its generated blocks into")
    ap.add_argument("--out", type=Path, help="where to write the file's text with its generated blocks")
    a = ap.parse_args()

    if bool(a.readme) != bool(a.out):
        raise SystemExit("--readme and --out go together")
    if not a.readme:
        for marker, produce in BLOCKS.items():
            print(f"--- {marker}")
            print(produce())
        return 0
    a.out.write_text(render(a.readme.read_text()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
