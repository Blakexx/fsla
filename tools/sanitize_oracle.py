#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""K39 -- the compute-sanitizer NAMED gate on the carry/intra/decode oracle cells.

`docs/testing.md`'s "sanitizer lane" section records the shape (a SCHEDULED lane
over a named subset, never a PR gate -- the tools cost one to two orders of
magnitude -- `--error-exitcode=1`, a committed skip-list whose entries carry a
reason) and its one measurement so far: memcheck/initcheck/synccheck/racecheck all
clean on the TILED consumer's two registered probes. It also states the gap this
script closes: "no such measurement exists for the chunk arm yet."

This is that measurement, promoted from an ad-hoc invocation (recorded once, by
hand, in FINDINGS `## PROBE-K35` under "memcheck on paging") to a script anyone
can run and CI can schedule: `racecheck`/`synccheck`/`initcheck` (memcheck already
ran, ad hoc, on paging -- K39 card) against ONE named oracle cell per family
(carry, intra via the combined prefill operator, decode), each run as its own
`compute-sanitizer`-wrapped pytest process so a hang or an error in one tool
cannot block another, under the GPU lock every campaign shares.

Usage:
    python tools/sanitize_oracle.py [--tool TOOL ...] [--family FAMILY ...]
        [--timeout SECONDS]

Invoked BARE (K46): this script takes `gpu_lock()` itself, once, for the whole
run, exactly the way `tools/probe_cells.py` already did before K46 and every
other GPU entry point does now -- never wrap it in an external `flock` on the
same lock path, which self-deadlocks (see `tools/gpu_lock.py`'s module
docstring for the K38 incident this rule exists to prevent).

Exit code is 0 iff every (family, tool) pair is CLEAN or DECLARED. DECLARED is
K47's one adjudicated exception, not a skip: the pair still runs and its whole
report is still read, and it passes only if EVERY record in it is the one
declared idiom (see `ASSERTED`). A per-pair timeout (default
600s) turns a hang into a NAMED failure rather than a stuck lane holding the GPU
lock forever (KERNEL_STANDARDS §14: "a hang here blocks the whole lock queue for
every other agent").
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import host_budget
from gpu_lock import gpu_lock

ROOT = Path(__file__).resolve().parents[1]

#: one cell per family, chosen for being ALREADY a named, gated test -- this script
#: adds no new cell, it reruns an existing one under the sanitizer. `-k` narrows to
#: exactly one parametrization so each (family, tool) pair is one launch, not a
#: whole file's worth serialized under the heaviest instrumentation this repo has.
#:
#: THE CARRY SHIPS TWO BODIES AND THE LANE WAS COVERING ONE (K35 F1).  Every cell in
#: the carry's oracle file declared `level_modes = 0` until F1, which dispatches the
#: CHANNEL-SPLIT body -- so "carry racecheck CLEAN" was a statement about the body with
#: no exchange, no panel ring, no shared write list and no carved barriers.  The carved
#: body is where every shared-memory publish/consume protocol in this family lives, so
#: it gets its own family row rather than replacing the split one: both bodies run.
#:
#: THE FACTS PASS' TWO BITMAP KERNELS GET THEIR OWN ROWS (K48).  Their shared liveness
#: flag is written by one lane and read by all of them, which is a shared-memory
#: publish/consume protocol like any other; it reached the lane only incidentally, as a
#: record inside the carry cell's report, because the carry op launches the facts pass.
#: A kernel whose hazards are only ever seen through another family's cell has no gate.
_FAMILY_TESTS = {
    "carry": "tests/oracle/test_carry_vs_oracle.py::test_m3_two_sided_alt_small",
    "carry_carved": "tests/oracle/test_carry_vs_oracle.py::test_f1_carved_alt_small",
    "facts_atom": "tests/oracle/test_atom_bitmap.py::test_atom_bitmap_is_exact[d2b16-mixed]",
    "facts_block":
        "tests/oracle/test_block_bitmap.py::test_read_and_write_bits_are_exact[d2b16bc64-mixed]",
    "intra": "tests/oracle/test_prefill_vs_oracle.py::test_the_intra_term_is_actually_present",
    "decode": "tests/oracle/test_decode_vs_oracle.py::test_g1_is_byte_reproducible_against_itself",
}

#: memcheck is NOT re-run here: it already ran (ad hoc) on paging and is recorded
#: in FINDINGS; the K39 card's own text is "memcheck already runs on paging" --
#: this script's job is the three tools that never had ANY committed invocation.
_TOOLS = ("racecheck", "synccheck", "initcheck")

#: entries here are (family, tool) pairs with a recorded reason, mirroring
#: `docs/testing.md`'s "committed suppressions file, and an explicit skip-list
#: whose entries carry their reason" -- empty by design (docs/testing.md: "An
#: empty skip-list is the standard the successor lane inherits").
SKIP: dict[tuple[str, str], str] = {}


@dataclass(frozen=True)
class AssertedRaces:
    """ONE IDIOM'S REPORTS, DECLARED AND PINNED TO THE CODE THAT PRODUCES THEM.

    This is NOT a skip and NOT a suppression: the tool still runs, its whole report
    is still read, and the pair still FAILS on anything the declaration does not
    name. What the declaration buys is the ability to say "these exact reports, from
    these exact expressions, in this exact kernel, are the tool modelling less than
    the program does" -- with the proof of that claim attached and re-checkable.
    KERNEL_STANDARDS.md §18: a declared exception carrying its exit condition, never
    a blanket skip.

    The declaration LOCATES ITSELF IN THE SOURCE rather than pinning line numbers:
    `writes` and `reads` are the expressions whose lines racecheck may name, found in
    `source` at gate time. An unrelated edit above the combine shifts the lines and
    changes nothing; an edit TO THE COMBINE moves the expressions and re-opens the
    proof, which is the property that matters.
    """

    kernel: str
    source: str
    writes: tuple[str, ...]
    reads: tuple[str, ...]
    hazard_types: frozenset[str]
    proof: str
    exit_condition: str

    def lines(self, exprs: tuple[str, ...]) -> set[int]:
        text = (ROOT / self.source).read_text().splitlines()
        found: set[int] = set()
        for expr in exprs:
            hits = [i + 1 for i, line in enumerate(text) if expr in line]
            if len(hits) != 1:
                raise LookupError(
                    f"{self.source}: the declared expression {expr!r} was found "
                    f"{len(hits)} times, expected exactly 1 -- the combine moved, so "
                    f"the K47 proof no longer describes this code. Re-derive it "
                    f"(FINDINGS ## K47) before re-declaring the exception.")
            found.add(hits[0])
        return found


#: THE ONE DECLARED EXCEPTION (K47, 2026-08-29). The decode step's CTA combine ends
#: the kernel's barriers with its prologue (P79): each warp deposits its partial into
#: its own shared row, fences, and bumps an arrival count with one atomic; the LAST
#: ARRIVER fences and sums the rows in fixed warp order. racecheck reports 48 RAW
#: races between the deposit and that sum, because the synchronization it models is
#: enumerated by its own documentation -- barrier, syncwarp, cuda::barrier, cluster
#: barrier, async-copy commit/wait -- and a fence plus an atomic RMW chain is not in
#: it. See FINDINGS `## K47` for the whole proof; the summary is in `proof` below.
ASSERTED: dict[tuple[str, str], AssertedRaces] = {
    ("decode", "racecheck"): AssertedRaces(
        kernel="decode_step_kernel",
        source="csrc/rola/src/decode/decode.cu",
        writes=("my_row[lane + 32 * q] = acc[q]", "my_row[DV] = acc_m"),
        reads=("part[q] += row[lane + 32 * q]", "part_m += row[DV]"),
        hazard_types=frozenset({"RAW"}),
        proof=(
            "PTX ISA memory model, on the emitted PTX (membar.cta / bar.warp.sync / "
            "atom.shared.add.u32 / membar.cta): the writer's `membar.cta; "
            "atom.shared.add [s_arrive]` is a RELEASE PATTERN on the arrival counter "
            "(PTX 8.8, third bullet, verbatim `fence.release; atom.relaxed [M];`), and "
            "the last arriver's `atom.shared.add [s_arrive]; membar.cta` is an ACQUIRE "
            "PATTERN on it (PTX 8.8, third acquire bullet). Every writer's counter "
            "write precedes the last arriver's counter read in OBSERVATION ORDER via "
            "the atomic RMW chain (PTX 8.9.2: `for some atomic operation Z, W precedes "
            "Z and Z precedes R`), so release synchronizes-with acquire (PTX 8.9.4); "
            "both patterns' bounding fences are morally strong (PTX 8.7 -- cta scope, "
            "same CTA); causality order is transitive (PTX 8.9.5), so the deposits "
            "precede the sum's loads and PTX 8.10.6 forbids the stale read. The "
            "non-lane-0 deposits reach the release through bar.warp.sync's own memory "
            "ordering (PTX 9.7.14.2). membar.cta is fence.sc.cta on sm_70+. "
            "MEASURED: 80,000 launches of a probe build with randomized __nanosleep "
            "skew at all three rendezvous points produced y and state BIT-IDENTICAL to "
            "the unperturbed kernel across four cells; a probe build replacing the "
            "arrival count with __syncthreads produced BIT-IDENTICAL y and state over "
            "20,000 more launches AND reported ZERO hazards, which attributes all 48 "
            "to the fence-for-barrier substitution alone."),
        exit_condition=(
            "DELETE THIS DECLARATION when compute-sanitizer models fence+atomic "
            "happens-before (the run goes clean on its own and this script says so), "
            "or when the combine stops using the arrival-count rendezvous."),
    ),
}

#: the kernel's demangled name contains spaces, so the name group is GREEDY and the
#: file:line is anchored on the LAST " in <path>:<line>" of the record line.
_RACE_WRITE = re.compile(r"Race reported between (\w+) access at (.+) in (\S+):(\d+)")
_RACE_OTHER = re.compile(r"and (\w+) access at (.+) in (\S+):(\d+)")
_HAZARD = re.compile(r"Potential (\w+) hazard")
_SUMMARY = re.compile(r"RACECHECK SUMMARY: (\d+) hazards displayed \((\d+) errors, (\d+) warnings\)")


def _conforms(m: re.Match, kind: str, want: set[int], decl: AssertedRaces) -> bool:
    """One access line IS the declared idiom -- direction, kernel, file and line.

    The DIRECTION is declared too: the proof covers a deposit READ BY the last
    arriver, so a WAW or WAR record on the very same two lines is not it.
    """
    return (m.group(1) == kind
            and decl.kernel in m.group(2)
            and Path(m.group(3)).name == Path(decl.source).name
            and int(m.group(4)) in want)


def adjudicate(decl: AssertedRaces, out: str) -> tuple[bool, str]:
    """Read the WHOLE report and pass only if every record is the declared idiom."""
    m = _SUMMARY.search(out)
    if m and m.group(1) == "0":
        return True, ("DECLARED EXCEPTION IS NOW DEAD -- racecheck reported no hazards. "
                      f"Remove ASSERTED[('decode', 'racecheck')]. {decl.exit_condition}")
    want_w, want_r = decl.lines(decl.writes), decl.lines(decl.reads)
    types = set(_HAZARD.findall(out))
    if types - decl.hazard_types:
        return False, f"hazard types {sorted(types - decl.hazard_types)} are not declared (declared: {sorted(decl.hazard_types)})"

    records, offenders = 0, []
    pending = False
    for line in out.splitlines():
        mw = _RACE_WRITE.search(line)
        if mw:
            pending = True
            if not _conforms(mw, "Write", want_w, decl):
                offenders.append(line.strip())
            continue
        mo = _RACE_OTHER.search(line)
        if mo and pending:
            records += 1
            if not _conforms(mo, "Read", want_r, decl):
                offenders.append(line.strip())
    if offenders:
        head = "\n".join(f"    {o}" for o in offenders[:10])
        return False, (f"{len(offenders)} racecheck record(s) fall OUTSIDE the declared "
                       f"K47 combine idiom -- these are NOT covered by any proof:\n{head}")
    if records == 0:
        return False, "racecheck failed but produced no parsable race records -- read the raw output"
    return True, (f"{records} race record(s), every one the DECLARED K47 combine idiom "
                  f"(writes at {sorted(want_w)} -> reads at {sorted(want_r)} in "
                  f"{decl.source}, {sorted(decl.hazard_types)} only).")


def run_one(python: str, family: str, tool: str, timeout: int) -> tuple[bool, str]:
    test_id = _FAMILY_TESTS[family]
    cmd = [
        "compute-sanitizer", f"--tool={tool}", "--error-exitcode=1",
        python, "-m", "pytest", test_id, "-x", "-q",
    ]
    try:
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT after {timeout}s -- a hang under {tool}, not a correctness verdict"
    out = proc.stdout + proc.stderr
    ok = proc.returncode == 0
    #: A SKIPPED CELL EXITS ZERO AND READS AS CLEAN (K48: the carved family reported
    #: CLEAN on a binary that did not carry its TEST arm, so the kernel never ran).
    #: The tool refuses the vacuous pass rather than reporting coverage it does not have.
    if ok and not re.search(r"\b[1-9]\d* passed", out):
        return False, ("VACUOUS: the wrapped pytest exited 0 without passing a test (a skip "
                       "or an empty selection). The kernel never ran, so this pair measures "
                       "nothing -- build the cell's arm (ROLA_CARRY_ARMS=all) and re-run.\n"
                       + "\n".join(out.splitlines()[-20:]))
    #: THE DECLARED EXCEPTION IS ADJUDICATED, NOT ASSUMED. The tool ran, the whole
    #: report is read, and a pair with a declaration passes ONLY if every record in it
    #: is the declared idiom -- so a real new hazard in the same kernel still fails.
    decl = ASSERTED.get((family, tool))
    if decl is not None:
        passed, note = adjudicate(decl, out)
        return passed, ("DECLARED EXCEPTION (K47) -- " + note + "\n  proof: " + decl.proof
                        + "\n  exit:  " + decl.exit_condition) if passed else (
                        "DECLARED EXCEPTION (K47) DOES NOT COVER THIS RUN -- " + note)
    tail = "\n".join(proc.stdout.splitlines()[-30:] + proc.stderr.splitlines()[-30:])
    return ok, tail


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tool", action="append", choices=_TOOLS, default=None)
    ap.add_argument("--family", action="append", choices=tuple(_FAMILY_TESTS), default=None)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    tools = args.tool or list(_TOOLS)
    families = args.family or list(_FAMILY_TESTS)

    failures = []
    #: LOCKS brief, 2026-08-29: a sanitizer run is CORRECTNESS work on the GPU
    #: (`mode="shared"`, item 2) AND draws K=2 of the host-compute budget
    #: (item 1) -- `compute-sanitizer` itself is CPU-heavy (instrumenting every
    #: memory access), on top of whatever the wrapped pytest process spends.
    with host_budget.acquire(2, label="sanitize_oracle"), gpu_lock(mode="shared"):
        for family in families:
            for tool in tools:
                reason = SKIP.get((family, tool))
                if reason is not None:
                    print(f"SKIP {family}/{tool}: {reason}")
                    continue
                print(f"RUN  {family}/{tool} ({_FAMILY_TESTS[family]}) ...", flush=True)
                ok, tail = run_one(args.python, family, tool, args.timeout)
                declared = (family, tool) in ASSERTED
                status = ("DECLARED" if declared else "CLEAN") if ok else "FAIL"
                print(f"{status} {family}/{tool}")
                if not ok or declared:
                    print(tail)
                if not ok:
                    failures.append((family, tool))

    if failures:
        print(f"\n{len(failures)} sanitizer failure(s): {failures}")
        return 1
    print("\nall requested (family, tool) pairs CLEAN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
