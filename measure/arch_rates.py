# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE MACHINE'S RATES, READ FROM THE ONE TABLE: `csrc/rola/src/common/arch_rates.cuh`'s `rates_of` parsed (never
restated), the check that each value's source holds -- a measured value matches the lab record it names, a documented
one cites its document, an announced one says why -- and the cost of a form's load, the same integer arithmetic as the
header's `cost`. Files only (no `rola`, no GPU), so the lint runs it. docs/internals/common/arch_rates.md.

    python -m measure.arch_rates            # the table
    python -m measure.arch_rates --check    # the sources
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "csrc" / "rola" / "src" / "common" / "arch_rates.cuh"
RESULTS = ROOT / "measure" / "lab" / "results"
MANIFESTS = ROOT / "tools" / "manifests"
SCALARS = ("schedulers", "issue", "wavefront", "atomic", "atomic_wavefront")
#: a measured value may differ from its record's median by its own rounding (three decimals) and no more
TOLERANCE = 0.002


@dataclass(frozen=True)
class Rate:
    cycles: float
    kind: str
    where: str
    row: str


def _enum(text: str, name: str) -> list[str]:
    body = re.search(rf"enum {name} : int \{{([^}}]*)\}}", text)
    if body is None:
        raise SystemExit(f"cannot read enum {name} from {HEADER.relative_to(ROOT)}")
    return [m.strip() for m in body.group(1).split(",") if m.strip()][:-1]


def _args(text: str) -> list[float | str]:
    out: list[float | str] = []
    for token in re.finditer(r'((?:"(?:[^"\\]|\\.)*"\s*)+)|([-+]?\d+(?:\.\d+)?)', text):
        if token.group(1):
            out.append("".join(re.findall(r'"((?:[^"\\]|\\.)*)"', token.group(1))))
        else:
            out.append(float(token.group(2)))
    return out


def table(text: str | None = None) -> tuple[dict[int, dict[str, Rate]], list[str], list[str]]:
    """`{cc: {field: Rate}}` for every rated architecture (fields `issue`, `mma[kHmmaF16F32]`, `tax[k][c]`, ...), and
    the MMA kinds and taxed classes in their enum order; `text` the header's (the file's by default)."""
    text = HEADER.read_text() if text is None else text
    mmas, ops = _enum(text, "Mma"), _enum(text, "Op")
    body = text.split("constexpr Rates rates_of(int cc) {", 1)[1].split("\n  return r;", 1)[0]
    rated: dict[int, dict[str, Rate]] = {}
    for block in re.finditer(r"if \(cc == (\d+)\) \{(.*?)\n  \}", body, re.S):
        cc, rates = int(block.group(1)), {}
        for statement in " ".join(block.group(2).split()).split(";"):
            statement = statement.strip()
            if not statement or statement == "r.known = true":
                continue
            copied = re.fullmatch(r"taxes_announced_from\(r, (\d+)\)", statement)
            if copied:
                for field, rate in rated[int(copied.group(1))].items():
                    if field.startswith("tax["):
                        rates[field] = Rate(rate.cycles, "announced",
                                            "sm_86's measured tax, unmeasured on this architecture", "")
                continue
            assign = re.fullmatch(r"r\.(\w+(?:\[\w+\])*) = (measured|documented|announced)\((.*)\)", statement)
            if assign is None:
                raise SystemExit(f"cannot read `{statement}` in {HEADER.relative_to(ROOT)}'s rates_of")
            field, kind, args = assign.group(1), assign.group(2), _args(assign.group(3))
            rates[field] = Rate(float(args[0]), kind, str(args[1]), str(args[2]) if len(args) > 2 else "")
        rated[cc] = rates
    return rated, mmas, ops


def fields(mmas: list[str], ops: list[str]) -> list[str]:
    return [*SCALARS, *(f"mma[{k}]" for k in mmas), *(f"tax[{k}][{c}]" for k in mmas for c in ops)]


def check(text: str | None = None) -> list[str]:
    """Each built architecture rated; each rate present and its source holding (`text` the header's)."""
    findings = []
    rated, mmas, ops = table(text)
    built = sorted({int(p.stem[3:]) * 10 for p in MANIFESTS.glob("*/sm_*.json")})
    for cc in built:
        if cc not in rated:
            findings.append(f"sm_{cc // 10} is built (tools/manifests) but {HEADER.name} has no rates row for it")
    for cc, rates in rated.items():
        for field in fields(mmas, ops):
            if field not in rates:
                findings.append(f"sm_{cc // 10}: {field} has no rate in {HEADER.name}")
        for field, rate in rates.items():
            where = f"sm_{cc // 10} {field}"
            if rate.kind != "measured":
                if not rate.where:
                    findings.append(f"{where}: a {rate.kind} rate names no source")
                continue
            record = RESULTS / rate.where / f"sm_{cc // 10}.json"
            if not record.is_file():
                findings.append(f"{where}: its lab record {record.relative_to(ROOT)} does not exist")
                continue
            medians = {r["name"]: r["median"] for r in json.loads(record.read_text())["results"]}
            if rate.row not in medians:
                findings.append(f"{where}: {record.relative_to(ROOT)} has no result `{rate.row}`")
            elif abs(rate.cycles - medians[rate.row]) > TOLERANCE * max(1.0, medians[rate.row]):
                findings.append(f"{where}: {rate.cycles} in {HEADER.name}, {medians[rate.row]:.4f} in "
                                f"{record.relative_to(ROOT)}")
    return findings


def quantum(cycles: float) -> int:
    return int(cycles * 256.0 + 0.5)


@dataclass(frozen=True)
class Quanta:
    issue: int
    wavefront: int
    atomic: int
    atomic_wavefront: int
    mma: tuple[int, ...]
    tax: tuple[tuple[int, ...], ...]


def quanta(cc: int) -> Quanta:
    rated, mmas, ops = table()
    r = rated[cc]
    return Quanta(quantum(r["issue"].cycles), quantum(r["wavefront"].cycles), quantum(r["atomic"].cycles),
                  quantum(r["atomic_wavefront"].cycles), tuple(quantum(r[f"mma[{k}]"].cycles) for k in mmas),
                  tuple(tuple(quantum(r[f"tax[{k}][{c}]"].cycles) for c in ops) for k in mmas))


def cost(q: Quanta, mma: list[int], op: list[int], other: int, wavefronts: int, atomics: int,
         atomic_wavefronts: int) -> int:
    """The header's `cost`, in 1/256 cycles: the max of tensor (MMAs plus the taxes beside them at the kind with the
    most MMA cycles), issue and shared memory."""
    cycles = [n * c for n, c in zip(mma, q.mma)]
    kind = max(range(len(cycles)), key=lambda k: (cycles[k], -k))
    tensor = sum(cycles) + sum(n * t for n, t in zip(op, q.tax[kind]))
    issue = (other + sum(mma) + sum(op)) * q.issue
    smem = wavefronts * q.wavefront + max(atomics * q.atomic, atomic_wavefronts * q.atomic_wavefront)
    return max(tensor, issue, smem)


def main() -> int:
    if "--check" in sys.argv[1:]:
        findings = check()
        for f in findings:
            print(f)
        print(f"arch_rates: {len(findings)} finding(s)")
        return 1 if findings else 0
    rated, mmas, ops = table()
    for cc, rates in rated.items():
        print(f"sm_{cc // 10}")
        for field in fields(mmas, ops):
            r = rates[field]
            print(f"  {field:28s} {r.cycles:8.3f}  {r.kind:10s} {r.where}{' / ' + r.row if r.row else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
