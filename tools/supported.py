"""Generate the README's generated blocks: the "Supported configurations" table. **A REPORT, NOT A CLAIM.**

A hand-maintained support table says what someone hoped was true when
they last edited it; this one is derived from the CONTRACT, the registry of run-time contexts FSLA accepts (``hardware/``,
docs/internals/hardware/registry.md), and the toolchain that builds each context's architecture. The build fails if any context of the contract
reaches no arm (``arms_gen``), so the published support and the guaranteed support are one file. The commit gate fails on drift
(`//:readme`, which holds the committed README to what the graph writes, docs/build-graph.md#committed-files):

* ``hardware/descriptors``' contexts (``hardware.registry.contexts``): each descriptor's architecture, the GPUs it covers and, for every fact the
  CALL supplies, the values the contract admits.
* the BUILT architectures (``measure.hardware.built``: the rows whose build chooses a window) and the toolchain that builds them
  (``tools/toolchains.py``). A contract architecture the build does not build is refused here: the contract is what is built.

    bazel run //tools:supported              # print the tables
    bazel run //:readme_update               # write README.md's generated blocks (the graph's action runs
                                             # `supported.py --readme README.md --out <file>`)
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
from measure import hardware  # noqa: E402 -- which architectures the build builds

#: The README's generated regions: `marker -> producer`. One today: the support table, which no hand edit may touch.
BLOCKS = {
    "tools/supported.py": lambda: table(),
}


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
            for arch in hardware.built()}


def _ptxas_short(version: str) -> str:
    m = re.search(r"release ([\d.]+), V([\d.]+)", version)
    return f"ptxas {m.group(2)}" if m else version


def _spell(values: set) -> str:
    """A fact's admitted values: points as a sorted list, a range as `lo..hi`."""
    points = sorted(v[0] for v in values if v[0] == v[1])
    ranges = sorted(f"{v[0]}..{v[1]}" for v in values if v[0] != v[1])
    return ", ".join([str(p) for p in points] + ranges)


def contract() -> dict[tuple[str, str], dict[str, set]]:
    """`(arch, gpus) -> {call fact: the values the contract admits}`, over every context of the registry."""
    facts = registry.schema()
    calls = [n for n, f in facts.items() if f.read == "CALL"]
    out: dict[tuple[str, str], dict[str, set]] = {}
    for c in registry.contexts():
        seen = out.setdefault((c.arch, c.gpus), {n: set() for n in calls})
        for n in calls:
            seen[n].add(c.facts[n])
    return out


def table() -> str:
    built_rows = built()
    calls = [n for n, f in registry.schema().items() if f.read == "CALL"]
    rows = [f"| Compute capability | GPUs | {' | '.join(calls)} | Toolchain | Assembler | Status |",
            "|---" * (len(calls) + 5) + "|"]
    for (arch, gpus), admitted in sorted(contract().items()):
        if arch not in built_rows:
            raise SystemExit(f"the contract holds {arch}, which the build does not build: the contract is what is built")
        spelled = " | ".join(_spell(admitted[n]) for n in calls)
        for r in built_rows[arch]:
            rows.append(f"| `{arch}` | {gpus} | {spelled} | `{r['toolchain']}` | {_ptxas_short(r['ptxas'])} | built |")
    return "\n".join(rows)


def render(readme: str) -> str:
    """The README's text with each generated block written: the one writer of the blocks."""
    new = readme
    for marker, produce in BLOCKS.items():
        if block(new, marker) is None:
            raise SystemExit(f"README.md has no generated block for {marker}")
        begin, end = markers(marker)
        head, rest = new.split(begin, 1)
        _, tail = rest.split(end, 1)
        new = f"{head}{begin}\n{produce()}\n{end}{tail}"
    return new


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--readme", type=Path, help="the README to write the generated blocks into")
    ap.add_argument("--out", type=Path, help="where to write the README's text with its generated blocks")
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
