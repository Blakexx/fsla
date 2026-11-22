"""THE CENSUS: every shipped instantiation's registers, spills and local accesses, read from the graph's compiles.

`bazel run //:ratify -- --census` reports, per kernel and architecture, what `ptxas` allocated for each instantiation
the build compiled -- registers, spill stores and loads, the register-side CTAs an SM, the local accesses inside a loop
-- and exits 1 on a launch bound the assembler discarded (`Value of threads per SM ... is out of range`: that entry
compiled with no bound at all, so `ptxas` spent registers as it liked). It reads the units the graph's CUDA rule
compiled (`--graph`, `--runfiles`: the target passes both): each unit's `ptxas -v` report and its object, with the
graph's disassemblers. The scope is the build's parameters (`--//:archs`, one `--//:<family>_arms` per family of `ARM_FAMILIES`), never
a flag of its own, and the tool compiles nothing: a tool that can force a launch bound is a tool that can report a
number nobody ships.

    bazel run //:ratify -- --census                 # the per-kernel table of the build's instantiations
    bazel run //:ratify -- --census --json PATH     # the raw per-instantiation record, for arm-versus-arm diffing

Docs: docs/internals/tools/ratify.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dev_config  # noqa: E402
import kernels  # noqa: E402
import sass  # noqa: E402
import toolchains  # noqa: E402
from csrc_stamp import digest as _csrc_digest  # noqa: E402
from rola_devtools.locks import host  # noqa: E402

#: ONE toolchain location for nvcc and ptxas, the build's own (`dev_config.cuda_bin`): the version this gate reads is
#: from the assembler that produced the numbers it parses.
cuda_bin = dev_config.cuda_bin

#: THE GRAPH'S BUILD (`--graph`, bazel/defs.bzl's `ratify_census`): the units its compile rule compiled (each one's ptxas
#: report and object), its assembler and disassemblers and the build's parameters. See
#: docs/internals/tools/ratify.md#graph
_GRAPH: dict = {}


def _graph_bin(name: str) -> str:
    """The graph's toolkit binary `name`."""
    return str(_GRAPH["tools"][name])


def use_graph(census: Path, runfiles: Path) -> None:
    """Read the graph's census record, whose paths are relative to `runfiles`: this process's assembler and
    disassemblers become the graph's, and its census reads the graph's compiles."""
    global cuda_bin
    blob = json.loads(census.read_text())

    def where(short: str) -> Path:
        return Path(os.path.normpath(runfiles / short))

    _GRAPH.update(tools={name: where(path) for name, path in blob["tools"].items()},
                  units=[{"source": Path(u["source"]), "report": where(u["report"]), "object": where(u["object"])}
                         for u in blob["units"]],
                  parameters=blob["parameters"])
    cuda_bin = _graph_bin
    sass.use_graph(cuobjdump=_graph_bin("cuobjdump"), nvdisasm=_graph_bin("nvdisasm"))


SRC = Path(__file__).resolve().parents[1] / "csrc" / "rola" / "src"


ENTRY = re.compile(r"Compiling entry function '([^']+)' for 'sm_(\d+a?)'")
SPILL = re.compile(r"(\d+) bytes spill stores, (\d+) bytes spill loads")
REGS = re.compile(r"Used (\d+) registers")
OUT_OF_RANGE = re.compile(r"Value of threads per SM for entry (\S+) is out of range")


def shard_field(entries: dict) -> dict[str, str]:
    """`{entry key -> the kernel it instantiates}`, the census's per-row kernel column.

    The kernel is read off the symbol the assembler saw (`tools/kernels.py`'s demangling), so it cannot disagree
    with the binary and every family answers in the same schema.
    """
    keys = sorted(entries)
    return dict(zip(keys, kernels.demangled_names([k.split(":", 1)[1] for k in keys])))


def csrc_digest() -> str:
    """A digest of the kernel sources a census was read from (`tools/csrc_stamp.py`'s)."""
    return _csrc_digest(SRC)


def ptxas_version() -> str:
    """`ptxas --version`, whitespace-normalized to one line.

    The WHOLE output, not a parsed release number: the build string
    (`cuda_12.4.r12.4/compiler.33961263_0`) distinguishes two shipments of the same
    nominal release, and it is exactly those that can move an allocation while the
    release number says nothing changed. Normalizing whitespace makes the comparison
    insensitive to line endings and nothing else.
    """
    ptxas = cuda_bin("ptxas")
    proc = subprocess.run([ptxas, "--version"], capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"`{ptxas} --version` FAILED (rc={proc.returncode}); the gate "
                         f"cannot read a census against an assembler it cannot ask")
    return " ".join(proc.stdout.split())


def _parse_report(stderr: str) -> tuple[dict, list[str]]:
    entries: dict[str, dict] = {}
    unhonored = sorted(set(OUT_OF_RANGE.findall(stderr)))
    key = None
    for line in stderr.splitlines():
        m = ENTRY.search(line)
        if m:
            #: The MANGLED name is the key. It is exact, stable, and encodes every
            #: template argument, so an entry cannot be silently re-pointed at a
            #: different instantiation by a rename or a reordering.
            key = f"sm_{m.group(2)}:{m.group(1)}"
            entries[key] = dict(regs=0, spill_stores=0, spill_loads=0)
            continue
        if key is None:
            continue
        m = SPILL.search(line)
        if m:
            entries[key]["spill_stores"] = int(m.group(1))
            entries[key]["spill_loads"] = int(m.group(2))
        m = REGS.search(line)
        if m:
            entries[key]["regs"] = int(m.group(1))
    return entries, unhonored


def _scan_object(obj: str, wanted: set) -> dict:
    with sass.sass_lines(obj) as lines:
        return sass_scan(lines, wanted)


def compiled_entries() -> list[str]:
    """The mangled name of every entry function the graph's units compiled, from each unit's ptxas report."""
    return [key.split(":", 1)[1] for unit in _GRAPH["units"]
            for key in _parse_report(unit["report"].read_text())[0]]


def measure_built() -> tuple[dict, list[str], dict]:
    """THE GRAPH'S CENSUS: each unit its build compiled, read from that compile -- ptxas's report and the object it
    wrote -- with no compile of its own."""
    units = _GRAPH["units"]
    t0 = time.perf_counter()

    def one(unit: dict):
        got, bad = _parse_report(unit["report"].read_text())
        return unit["source"], {"entries": got, "unhonored": bad, "scan": _scan_object(str(unit["object"]), set(got))}

    #: the disassemblers' reads hold the host budget too, a job a slot (KERNEL_STANDARDS §14 addendum 2): the
    #: machine's under `bazel run //:ratify`; under a sandboxed test, whose /tmp and dev config are its own,
    #: a pool of the sandbox's, the default slot count
    with host.acquire(host.budget_slots(), label="ratify's census") as jobs, ThreadPoolExecutor(
            max_workers=min(jobs, len(units))) as pool:
        measured = list(pool.map(one, units))
    print(f"{len(units)} translation unit(s) read from the graph's build in {time.perf_counter() - t0:.1f} s at "
          f"{min(jobs, len(units))} jobs")
    return _merge_units(measured)


def _merge_units(measured: list[tuple[Path, dict]]) -> tuple[dict, list[str], dict]:
    """One census from each unit's record: its entries, its unhonored bounds, its SASS scan."""
    entries: dict[str, dict] = {}
    unhonored: set[str] = set()
    scan: dict[str, dict] = {}
    owner: dict[str, str] = {}
    for source, rec in measured:
        for key, value in rec["entries"].items():
            if key in entries:
                #: TWO TRANSLATION UNITS EMITTING ONE INSTANTIATION is the exact state the split exists to prevent, and
                #: merging their reports would hide it behind numbers that look fine.
                raise SystemExit(
                    f"{key} was compiled by more than one translation unit "
                    f"({source.name} is the second, after {owner[key]}): an arm declared twice, or a hand unit that "
                    f"instantiates what an arm's unit does (bazel/arms.bzl)")
            entries[key] = value
            owner[key] = source.name
        unhonored |= set(rec["unhonored"])
        scan.update(rec["scan"])
    scan = {name: scan.get(name, {"body_sha256": None, "hot_locals": [], "local_ops": 0})
            for name in entries}
    return entries, sorted(unhonored), scan


#: THE SASS SCAN. One pass over `cuobjdump -sass` answers every question asked of the assembled code: a body's
#: identity, the local accesses in a loop and the local accesses at all.
#: the section headers and the body lines, off the shared grammar (`tools/sass.py`, the one reader of the disassemblers)
_SASS_FUNC = sass.FUNCTION
_SASS_ARCH = sass.ARCH
_SASS_INST = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(.*?);")
_SASS_LABEL_DEF = re.compile(r"^\s*(\.L_[\w.]+):")
_SASS_LOCAL = re.compile(r"\b(LDL|STL)\b")
_SASS_BRA = re.compile(r"\bBRA\b.*?(?:`?\((\.L_[\w.]+)\)|0x([0-9a-f]+))")

#: THE BODY-HASH LINE RULES, and they are deliberately WIDER than `_SASS_INST`'s.
#: The hot-loop parser wants an instruction it can reason about, so it requires a
#: terminating `;`; the body hash wants EVERY line that carries codegen, which
#: includes the continuation line holding an instruction's second encoding word. Two
#: instructions can print identically and encode differently -- a predicate, a
#: scheduling control field, a reuse flag -- so the encoding words are in the hash
#: and must be. Not in it: `Function :` headers, `.headerflags`, lineinfo, and the
#: padding that aligns the encoding column, which is a property of the LISTING and
#: moves with the longest instruction in it.
#: THE SCANNER'S OWN VERSION. What reaches the digest is a property of
#: :func:`sass_scan`, not of the binary: widening the capture once moved every digest
#: in an unchanged binary, which reads exactly like a codegen change and is not one.
#: Every artifact carrying one of these hashes is stamped with it -- the maps
#: `tools/sass_bodies.py` writes and the census's `sass_sha256` -- and each refuses a comparison across two versions. BUMP IT in the same
#: commit that changes which lines `sass_scan` feeds the digest.
#: 1 = instruction text only (the pre-2026-08-05 capture, which dropped the encoding
#: words); 2 = every offset-marked and encoding line, whitespace-normalised.
SASS_HASH_VERSION = 2

_SASS_BODY_INST = re.compile(r"/\*[0-9a-f]{4,}\*/")
_SASS_BODY_ENC = sass.ENCODING


def _hot_locals_in_function(items: list) -> list:
    """`items` = [(pos, kind, payload)] for one function, in order; kinds are
    'label' (payload: name), 'inst' (payload: (addr, text)). Returns the hot
    local accesses: [(addr, opcode_text)] that sit inside a backward-BRA interval."""
    labels: dict[str, int] = {}
    addr_pos: dict[int, int] = {}
    for pos, kind, payload in items:
        if kind == "label":
            labels[payload] = pos
        else:
            addr_pos.setdefault(payload[0], pos)
    intervals = []
    for pos, kind, payload in items:
        if kind != "inst":
            continue
        addr, text = payload
        m = _SASS_BRA.search(text)
        if not m:
            continue
        target = labels.get(m.group(1)) if m.group(1) else \
            addr_pos.get(int(m.group(2), 16))
        if target is not None and target < pos:
            intervals.append((target, pos))
    if not intervals:
        return []
    hot = []
    for pos, kind, payload in items:
        if kind != "inst":
            continue
        addr, text = payload
        if _SASS_LOCAL.search(text) and any(a <= pos <= b for a, b in intervals):
            hot.append((addr, text.strip()))
    return hot


def sass_scan(sass_lines, wanted: set | None = None) -> dict:
    """ONE pass over `cuobjdump -sass`, answering every question asked of the SASS.

    `{key -> {"body_sha256", "hot_locals": [(addr, inst)], "local_ops": int}}`:

    * `body_sha256` is the assembled body's identity, and is what "the same code"
      means in this repository -- names are the weaker check twice over, since a
      mangling can move while the code is identical (a translation-unit restructure
      does exactly that to anonymous-namespace kernels) and the code can move while
      every name is identical.
    * `hot_locals` is the in-loop local traffic.
    * `local_ops` is EVERY `LDL`/`STL`, in a loop or not -- a seconds-cheap presence
      count that catches local traffic the byte-level spill report does not (a
      dynamically indexed local array is not a spill) and that the in-loop count does
      not see outside a backward-branch interval.

    `wanted` filters; `None` scans every entry point, which is what a whole-binary
    comparison needs. A name appearing twice within one arch with DIFFERENT bodies
    is refused rather than resolved last-writer-wins: it means the kernel is
    duplicated across objects -- the one-instantiation-per-TU rule broken -- and
    hashing one of the two copies would hide it behind a green comparison.
    """
    report: dict[str, dict] = {}
    arch, key, items, pos, digest, locals_seen = None, None, [], 0, None, 0

    def flush():
        if key is None or (wanted is not None and key not in wanted):
            return
        entry = {"body_sha256": digest.hexdigest(),
                 "hot_locals": _hot_locals_in_function(items),
                 "local_ops": locals_seen}
        if key in report and report[key]["body_sha256"] != entry["body_sha256"]:
            raise SystemExit(f"{key} is disassembled twice with DIFFERENT bodies; "
                             f"the same kernel is emitted by two objects")
        report[key] = entry

    for line in sass_lines:
        m = _SASS_ARCH.match(line)
        if m:
            flush()
            arch, key, items = f"sm_{m.group(1)}", None, []
            continue
        m = _SASS_FUNC.match(line)
        if m:
            flush()
            key, items, pos = f"{arch}:{m.group(1)}", [], 0
            digest, locals_seen = hashlib.sha256(), 0
            continue
        if key is None:
            continue
        if _SASS_BODY_INST.search(line) or _SASS_BODY_ENC.match(line):
            digest.update(" ".join(line.split()).encode())
            digest.update(b"\n")
        m = _SASS_LABEL_DEF.match(line)
        if m:
            items.append((pos, "label", m.group(1)))
            pos += 1
            continue
        m = _SASS_INST.search(line)
        if m:
            items.append((pos, "inst", (int(m.group(1), 16), m.group(2))))
            pos += 1
            if _SASS_LOCAL.search(m.group(2)):
                locals_seen += 1
    flush()
    return report


def _regs_ctas(regs: int) -> int:
    """Register-file-side CTAs/SM at 256 threads after the 8-register
    allocation granularity. The REGISTER term only -- the driver's own occupancy
    query stays the sole residency authority."""
    alloc = 256 * ((regs + 7) // 8) * 8
    return min(6, 65536 // alloc) if alloc else 6


def census_report(entries: dict, unhonored: list[str],
                  hot: dict[str, int] | None = None) -> None:
    hot = hot or {}
    shard = shard_field(entries)
    by_key: dict[tuple, list] = {}
    for name, e in sorted(entries.items()):
        by_key.setdefault((name.split(":", 1)[0], shard[name]), []).append((e, hot.get(name, 0)))
    print(f"\n{'arch':6} {'shard':24} {'n':>4} {'regs':>9} {'ctas(reg)':>9} "
          f"{'st':>6} {'ld':>6} {'#spill':>6} {'worst st':>8} {'hot':>5}")
    for k, es in sorted(by_key.items()):
        regs = sorted(e["regs"] for e, _ in es)
        ctas = sorted(_regs_ctas(r) for r in regs)
        st = sum(e["spill_stores"] for e, _ in es)
        ld = sum(e["spill_loads"] for e, _ in es)
        nsp = sum(1 for e, _ in es if e["spill_stores"] or e["spill_loads"])
        worst = max(e["spill_stores"] for e, _ in es)
        nhot = sum(h for _, h in es)
        rg = f"{regs[0]}-{regs[-1]}" if regs[0] != regs[-1] else f"{regs[0]}"
        cg = f"{ctas[0]}-{ctas[-1]}" if ctas[0] != ctas[-1] else f"{ctas[0]}"
        print(f"{k[0]:6} {k[1]:24} {len(es):>4} {rg:>9} {cg:>9} {st:>6} "
              f"{ld:>6} {nsp:>6} {worst:>8} {nhot:>5}")
    total_st = sum(e["spill_stores"] for e in entries.values())
    total_ld = sum(e["spill_loads"] for e in entries.values())
    zero = sum(1 for e in entries.values()
               if not e["spill_stores"] and not e["spill_loads"])
    print(f"\nTOTAL {len(entries)} entries: {total_st} spill st / {total_ld} "
          f"spill ld, {zero} zero-spill, register sum "
          f"{sum(e['regs'] for e in entries.values())}, "
          f"{sum(hot.values())} in-loop local accesses, "
          f"{len(unhonored)} unhonored bound(s)")


def _self_test() -> int:
    """Prove the readers can fail: the loop parser counts an in-loop `LDL` and clears a cold one, the body hash separates
    changed code and ignores listing padding (a gate that cannot fail is not a gate)."""
    #: THE LOOP PARSER, against synthetic SASS: one LDL
    #: inside a backward-branch interval MUST be flagged, and the same LDL
    #: outside every loop MUST NOT be.
    hot_sass = [
        "arch = sm_90a",
        "                Function : k_hot",
        "        /*0000*/ MOV R1, c[0x0][0x28] ;",
        ".L_x_0:",
        "        /*0010*/ LDL R2, [R1] ;",
        "        /*0020*/ @P0 BRA `(.L_x_0) ;",
        "                Function : k_cold",
        "        /*0000*/ LDL R2, [R1] ;",
        ".L_x_1:",
        "        /*0010*/ IADD3 R3, R3, 1, RZ ;",
        "        /*0020*/ @P0 BRA `(.L_x_1) ;",
    ]
    rep = sass_scan(iter(hot_sass), {"sm_90a:k_hot", "sm_90a:k_cold"})
    print("\n--- SELF-TEST: the hot-loop parser on synthetic SASS ---")
    counts = {k: len(v["hot_locals"]) for k, v in rep.items()}
    if counts != {"sm_90a:k_hot": 1, "sm_90a:k_cold": 0}:
        print(f"\nSELF-TEST FAILED: hot-loop parser reported {counts}; it "
              f"must count exactly the in-loop LDL (k_hot=1, k_cold=0)")
        return 1
    print("SELF-TEST PASSED: the parser counts the in-loop LDL and clears the "
          "cold one")

    #: NON-VACUITY OF THE BODY HASH, both directions. Two functions whose
    #: instruction text differs must hash differently, or the comparison is
    #: decorative; the SAME text laid out with different column padding must hash
    #: identically, or every rebuild reads as a codegen change and the gate gets
    #: turned off.
    print("\n--- SELF-TEST: the SASS body hash on synthetic SASS ---")
    moved_sass = [
        "arch = sm_90a",
        "                Function : k_a",
        "        /*0000*/ IADD3 R3, R3, 1, RZ ;",
        "                                     /* 0x000fe200078e0203 */",
        "                Function : k_b",
        "        /*0000*/ IADD3 R3, R3, 2, RZ ;",
        "                                     /* 0x000fe200078e0203 */",
        "                Function : k_a_repadded",
        "   /*0000*/     IADD3   R3,  R3, 1, RZ ;",
        "     /* 0x000fe200078e0203 */",
    ]
    got = sass_scan(iter(moved_sass))
    a, b = got["sm_90a:k_a"]["body_sha256"], got["sm_90a:k_b"]["body_sha256"]
    repad = got["sm_90a:k_a_repadded"]["body_sha256"]
    if a == b:
        print("\nSELF-TEST FAILED: two different instruction texts hashed the "
              "same; the body hash cannot detect a codegen change")
        return 1
    if a != repad:
        print("\nSELF-TEST FAILED: re-padding the listing moved the body hash; "
              "the hash is over the LISTING's column alignment, not the code")
        return 1
    print("SELF-TEST PASSED: the body hash separates changed code and ignores "
          "listing padding")
    return 0


def _census(a: argparse.Namespace) -> int:
    """The census: each instantiation's report, the SASS read of its object, and the launch bounds the assembler
    discarded."""
    #: every SASS body below is read with the record's disassemblers, or it is not the toolchain's: checked before the
    #: first read
    toolchains.check_readers(toolchains.for_ptxas(ptxas_version()),
                             {tool: toolchains.version_text(sass.tool(tool)) for tool in toolchains.READERS})
    entries, unhonored, scan = measure_built()
    hot = {name: len(s["hot_locals"]) for name, s in scan.items()}
    for name, s in scan.items():
        entries[name]["hot_locals"] = len(s["hot_locals"])
        entries[name]["local_ops"] = s["local_ops"]
        entries[name]["sass_sha256"] = s["body_sha256"]
    census_report(entries, unhonored, hot)
    if a.json:
        Path(a.json).write_text(json.dumps(
            dict(ptxas=ptxas_version(), csrc_sha256=csrc_digest(), entries=entries, shards=shard_field(entries)),
            indent=1, sort_keys=True) + "\n")
        print(f"census written to {a.json}")
    if unhonored:
        print(f"\nFAIL  {len(unhonored)} instantiation(s) compiled with the launch bound DISCARDED ('out of range'); "
              f"`ptxas` was unconstrained there:")
        for name in unhonored:
            print(f"        {name}")
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--census", action="store_true",
                      help="print the build's per-kernel register and spill table; exit 1 on a discarded launch bound")
    mode.add_argument("--self-test", action="store_true",
                      help="prove the loop parser and the body hash can fail")
    ap.add_argument("--json", metavar="PATH", help="census only: dump the raw per-instantiation record")
    ap.add_argument("--graph", type=Path, metavar="CENSUS", required=True,
                    help="THE GRAPH'S BUILD: its census record (bazel/defs.bzl's ratify_census) names each unit's ptxas "
                         "report and object, the graph's toolkit and the build's parameters")
    ap.add_argument("--runfiles", type=Path, metavar="DIR", required=True,
                    help="the directory the census record's paths are relative to")
    a = ap.parse_args()
    if a.json and not a.census:
        raise SystemExit("--json is the census's: pass --census --json PATH")

    use_graph(a.graph, a.runfiles)
    if a.census:
        return _census(a)
    return _self_test()


if __name__ == "__main__":
    raise SystemExit(main())
