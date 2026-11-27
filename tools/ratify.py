"""THE CENSUS: every shipped instantiation's registers, spills and local accesses, read from warpline's static manifests of the
graph's library (docs/internals/tools/ratify.md).

`bazel run //:ratify -- --census` reports, per kernel and architecture, what the static manifest of each built architecture
(`//:static_manifest`, the SASS gate's own) says of each instantiation the library carries -- registers, the bytes `ptxas`
spilled, the register-side CTAs an SM, the local accesses inside a loop -- and exits 1 on a launch bound the assembler
discarded (the checker's `launch_bound_discarded`: that entry compiled with no bound at all, so `ptxas` spent registers as it
liked; the SASS gate refuses it too, and the census names it). It reads the graph's manifests (`--graph`, `--runfiles`: the
target passes both), with warpline's reader of the manifest (`checked.checker().formats`). The scope is the build's parameters
(`--//:archs`, `--//:contract`), never a flag of its own, and the tool compiles and disassembles nothing: a tool that can force a
launch bound is a tool that can report a number nobody ships.

    bazel run //:ratify -- --census                 # the per-kernel table of the build's instantiations
    bazel run //:ratify -- --census --json PATH     # the raw per-instantiation record, for arm-versus-arm diffing

Docs: docs/internals/tools/ratify.md.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import checked  # noqa: E402
import dev_config  # noqa: E402
import kernels  # noqa: E402
import toolchains  # noqa: E402
from csrc_stamp import digest as _csrc_digest  # noqa: E402

#: ONE toolchain location for ptxas and the readers, the build's own (`dev_config.cuda_bin`): the version this census
#: states is the assembler's that produced the numbers it reads.
cuda_bin = dev_config.cuda_bin

#: THE GRAPH'S BUILD (`--graph`, bazel/defs.bzl's `ratify_census`): the static manifest of each built architecture, the graph's
#: assembler and readers and the build's parameters. See docs/internals/tools/ratify.md#graph
_GRAPH: dict = {}

#: the checker's rule a discarded launch bound is counted under (warpline's base ledger)
DISCARDED = "launch_bound_discarded"


def _graph_bin(name: str) -> str:
    """The graph's toolkit binary `name`."""
    return str(_GRAPH["tools"][name])


def use_graph(census: Path, runfiles: Path) -> None:
    """Read the graph's census record, whose paths are relative to `runfiles`: this process's assembler and readers become the
    graph's, its census reads the graph's manifests, and warpline is the record's (`FSLA_WARPLINE_PYTHON`, which `checked` reads)."""
    global cuda_bin
    blob = json.loads(census.read_text())

    def where(short: str) -> Path:
        return Path(os.path.normpath(runfiles / short))

    _GRAPH.update(tools={name: where(path) for name, path in blob["tools"].items()},
                  manifests={arch: where(path) for arch, path in blob["manifests"].items()},
                  parameters=blob["parameters"])
    os.environ["FSLA_WARPLINE_PYTHON"] = str(where(blob["warpline"]))
    cuda_bin = _graph_bin


SRC = Path(__file__).resolve().parents[1] / "csrc" / "fsla" / "src"


def manifests() -> dict[str, list]:
    """Each built architecture's kernels as its static manifest holds them (warpline's `manifest.loads`, which validates the
    form): `{arch: [Place]}`, the places that are kernels."""
    formats = checked.checker().formats
    return {arch: [p for p in formats.loads(path.read_text()).places if p.kernel]
            for arch, path in sorted(_GRAPH["manifests"].items())}


def compiled_entries() -> list[str]:
    """The mangled name of every kernel the library carries, over the built architectures' manifests."""
    return [place.name for found in manifests().values() for place in found]


def measure_built() -> tuple[dict, list[str]]:
    """THE GRAPH'S CENSUS: each instantiation's figures, `{"sm_XX:mangled": {...}}`, from its architecture's static manifest --
    the registers, the spill bytes `ptxas` reported (`spill_stores`, `spill_loads`: the SASS gate reads the build's notices, so a
    kernel lacking them is refused), the local accesses inside a loop (`hot_locals`) and in all (`local_ops`) -- and the kernels
    whose launch bound the assembler discarded."""
    entries: dict[str, dict] = {}
    unhonored = []
    for arch, found in manifests().items():
        for place in found:
            figures = place.figures["kernel"]
            key = f"{arch}:{place.name}"
            if "spill_stores" not in figures:
                raise SystemExit(f"{key}: its manifest has no spill bytes -- the gate read no `ptxas -v` report naming it")
            entries[key] = {"regs": figures["registers"], "spill_stores": figures["spill_stores"],
                            "spill_loads": figures["spill_loads"], "hot_locals": figures["local_in_loops"],
                            "local_ops": figures["local_loads"] + figures["local_stores"]}
            if place.rules.get(DISCARDED) is not None and place.rules[DISCARDED].count:
                unhonored.append(key)
    return entries, sorted(unhonored)


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
    release number says nothing changed.
    """
    ptxas = cuda_bin("ptxas")
    proc = subprocess.run([ptxas, "--version"], capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"`{ptxas} --version` FAILED (rc={proc.returncode}); the census cannot be read against an "
                         f"assembler it cannot ask")
    return " ".join(proc.stdout.split())


def _regs_ctas(regs: int) -> int:
    """Register-file-side CTAs/SM at 256 threads after the 8-register
    allocation granularity. The REGISTER term only -- the driver's own occupancy
    query stays the sole residency authority."""
    alloc = 256 * ((regs + 7) // 8) * 8
    return min(6, 65536 // alloc) if alloc else 6


def census_report(entries: dict, unhonored: list[str]) -> None:
    hot = {name: e["hot_locals"] for name, e in entries.items()}
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


def _census(a: argparse.Namespace) -> int:
    """The census: each instantiation's figures off its manifest, and the launch bounds the assembler discarded."""
    #: the manifests were written by the record's readers, or they are not the toolchain's: checked before the first read
    toolchains.check_readers(toolchains.for_ptxas(ptxas_version()),
                             {tool: toolchains.version_text(cuda_bin(tool)) for tool in toolchains.READERS})
    entries, unhonored = measure_built()
    census_report(entries, unhonored)
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
    ap.add_argument("--census", action="store_true", required=True,
                    help="print the build's per-kernel register and spill table; exit 1 on a discarded launch bound")
    ap.add_argument("--json", metavar="PATH", help="also dump the raw per-instantiation record")
    ap.add_argument("--graph", type=Path, metavar="CENSUS", required=True,
                    help="THE GRAPH'S BUILD: its census record (bazel/defs.bzl's ratify_census) names each built architecture's "
                         "static manifest, the graph's toolkit and the build's parameters")
    ap.add_argument("--runfiles", type=Path, metavar="DIR", required=True,
                    help="the directory the census record's paths are relative to")
    a = ap.parse_args()
    use_graph(a.graph, a.runfiles)
    return _census(a)


if __name__ == "__main__":
    raise SystemExit(main())
