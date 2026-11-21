"""Generate the README's "Supported configurations" table. **A REPORT, NOT A CLAIM.**

A hand-maintained support table says what someone hoped was true when
they last edited it; this one is derived from the two artifacts that decide the
answer, and the commit gate fails on drift (the standards lint's `check_supported_table`; the capability table's
test is the CPU tier's, `tests/unit/test_capability_table_docs.py`):

* ``csrc/rola/src/common/hardware.json``'s rows (``measure.hardware.tabulated``, which also writes
  ``common/hardware.cuh``'s ``tabulated``) -- the architectures the KERNEL knows how to size itself for: a
  ``static_assert`` at compile time and the card's admission a refusal at run time, so an arch missing from it
  cannot execute whatever the table says.
* the same file's BUILT architectures (``measure.hardware.built``: the rows whose build chooses a window) and the
  toolchain that builds them (``tools/toolchains.py``).

The two are not the same set, and **the difference is the product statement**. An
arch that is tabulated but not built is listed with an empty toolchain column
and the status *not built -- will refuse to load*. A support claim is worth
nothing if the table quietly implies coverage nobody brought up.

    bazel run //tools:supported              # print the tables
    bazel run //tools:supported -- --write   # rewrite README.md's generated blocks

GPU names are the only hand-written column, in ``GPUS`` below, because a compute
capability does not know its own marketing name. They are labels; the hardware table
is what the table is actually reporting.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import toolchains

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(1, str(ROOT))
from measure import hardware  # noqa: E402 -- the architecture rows' own names

README = ROOT / "README.md"

#: The README's generated regions: `marker -> producer`. TWO of them, because two
#: different artifacts decide two different tables and neither may be hand-edited.
#: `capability_table()` is the activation registry's own generator -- this
#: script only places it, so there is still exactly one table in the codebase.
BLOCKS = {
    "tools/supported.py": lambda: table(),
    "rola.routing.activations.capability_table": lambda: _capability_table(),
}


def markers(marker: str) -> tuple[str, str]:
    """A generated block's two markers in the README."""
    return f"<!-- BEGIN GENERATED: {marker} -->", f"<!-- END GENERATED: {marker} -->"


def block(text: str, marker: str) -> str | None:
    """The README's generated block for `marker`, its text between the markers; None when it has none. The one reader
    of the format: `--write` and the checks read it here."""
    begin, end = markers(marker)
    if begin not in text or end not in text:
        return None
    return text.split(begin, 1)[1].split(end, 1)[0].strip("\n")


def _capability_table() -> str:
    from rola.routing.activations import capability_table

    return capability_table().rstrip()


#: Marketing names per compute capability. The ONLY hand-written column.
GPUS = {
    "sm_90a": "H100",
    "sm_100a": "B200",
}


def tabulated_archs() -> list[str]:
    """The architectures the kernel can size itself for: the hardware table's rows, by compute capability
    (`measure.hardware.tabulated`, which writes `common/hardware.cuh`'s `tabulated`; the hardware lint holds the header
    to it)."""
    return hardware.tabulated()


def built() -> dict[str, list[dict]]:
    """`sm_XX -> [{toolchain, ptxas}]`: each built architecture, with the toolchain that builds it."""
    return {arch: [{"toolchain": t.name, "ptxas": t.ptxas} for t in toolchains.records().values()]
            for arch in hardware.built()}


def _ptxas_short(version: str) -> str:
    m = re.search(r"release ([\d.]+), V([\d.]+)", version)
    return f"ptxas {m.group(2)}" if m else version


def table() -> str:
    built_rows = built()
    rows = ["| Compute capability | GPUs | Toolchain | Assembler | Status |",
            "|---|---|---|---|---|"]
    for arch in tabulated_archs():
        gpus = GPUS.get(arch, "—")
        for r in built_rows.get(arch, []):
            rows.append(f"| `{arch}` | {gpus} | `{r['toolchain']}` | {_ptxas_short(r['ptxas'])} | built |")
        if arch not in built_rows:
            rows.append(f"| `{arch}` | {gpus} | — | — | **not built — will refuse to load** |")
    return "\n".join(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="rewrite README.md's generated blocks")
    a = ap.parse_args()

    if not a.write:
        for marker, produce in BLOCKS.items():
            print(f"--- {marker}")
            print(produce())
        return 0

    new = README.read_text()
    for marker, produce in BLOCKS.items():
        if block(new, marker) is None:
            raise SystemExit(f"README.md has no generated block for {marker}")
        begin, end = markers(marker)
        head, rest = new.split(begin, 1)
        _, tail = rest.split(end, 1)
        new = f"{head}{begin}\n{produce()}\n{end}{tail}"

    README.write_text(new)
    print(f"README.md: {len(BLOCKS)} generated blocks rewritten")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
