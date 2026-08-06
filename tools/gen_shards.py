"""THE ONE ENUMERATION of the consumer's instantiation matrix.

**WHAT THIS OWNS.** Before this file the matrix
`consumer_forward_kernel<BT, BC, DV, D, FOLD, READ, DECAY, GLOBAL>` was
written out THREE times -- the dispatch ladder in `consumer.cu`, the occupancy and
footprint ladders beside it, and a probe translation unit under `tools/`. Three
independent enumerations of one set is the drift shape this project has already been
bitten by once (the shared-router bug agreed with a sibling and disagreed with the
definition). Here the matrix is DEFINED ONCE, and everything that enumerates it is
GENERATED from that definition:

* `csrc/rola/src/instantiations/consumer_*.cu` -- the eight shard translation units,
  each carrying `TARGET_MEMBERS_PER_SHARD` explicit instantiations of the host launcher
  (and with them, exactly once in the whole fatbin, that many device kernels), plus one
  `ShardRow` record each;
* `csrc/rola/src/instantiations/dispatch_table.inc` -- the shard row blocks, the
  mixed-radix key over the matrix axes, and the index that turns a key into a row;
  the launch and both device-backed queries all resolve through it;
* `csrc/rola/src/instantiations/README.md` -- the layout, regenerated with them.

The files are CHECKED IN, not built: what is reviewed must be what compiles
(closed-world codegen policy), and a source file that exists only inside a build
directory cannot be reviewed. `--check` regenerates into memory and diffs, which is
what CI and `tests/unit/test_shard_partition.py` run.

**MEMBERSHIP IS A STRUCTURAL FUNCTION.** `shard_of` maps a template tuple to a shard
name using an ordered AXIS PRIORITY list and a target member count, and nothing else
-- no measured compile time, no hand-written file list, no size threshold. "Rebalance
the shards by how long they took to compile last week" is therefore not a thing that
can be expressed here; changing the partition (P17 adds a read-path axis and takes
the matrix to 1,024) is a change to `SHARD_AXIS_PRIORITY` or
`TARGET_MEMBERS_PER_SHARD` and nothing else. `tests/unit/test_shard_partition.py`
asserts every member of every generated file maps back to that file.

**WHY THESE THREE AXES.** PREC, GLOBAL and BT, in that priority. Every recorded
ratchet co-movement episode clustered perfectly on ONE of PREC, GLOBAL or BT and never
on BC/DV/D, so a shard-per-axis-value makes a ratchet refusal a one-shard-slice event;
the fourth axis is RESERVED rather than spent, because the next queued template axis
(P17's dense read path) doubles the matrix and will need it. The evidence:
`rola-scratch/workflows/refactor/p02_build_measure.md`; the adjudication:
`p02_shard_review.md` answer 1.

Usage:

    python tools/gen_shards.py --write     # regenerate the tree
    python tools/gen_shards.py --check     # regenerate into memory and diff (CI)
"""
from __future__ import annotations

import argparse
import hashlib
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "csrc" / "rola" / "src" / "instantiations"

#: The formatter the pre-commit hook and CI run, pinned to one version in
#: `.pre-commit-config.yaml`. Generated C++ is piped through it rather than
#: hand-aligned: unformatted generated files would fail either the hook or the
#: regenerate-and-diff gate, and "generated output happens to be formatted" is not a
#: property anybody can maintain by eye.
CLANG_FORMAT = "clang-format"


# ---------------------------------------------------------------------------
# THE MATRIX
# ---------------------------------------------------------------------------
class Axis:
    """One structural template axis: its legal values, and how each value is
    spelled in C++, in a file name, and in the runtime dispatch test."""

    def __init__(self, name, values, cxx, token, selector):
        self.name = name
        self.values = values
        self.cxx = cxx            # value -> the C++ template argument
        self.token = token        # value -> the file-name fragment
        self.selector = selector  # value -> the host-side predicate/case label


def _b(v):
    return "true" if v else "false"


#: DECLARATION ORDER of `consumer_forward_kernel`'s template parameters, minus the
#: duplicated precision parameter. The kernel takes FOLD and READ separately; the
#: dispatch has always passed them EQUAL (one `mma_precision` plan field selects both
#: GEMM sites together), so the matrix carries one PREC axis and the generator writes
#: it twice. That funnel is the reason the space is 256 and not 512, and it is
#: recorded here rather than rediscovered from a macro.
AXES = (
    Axis("BT", (16, 32), str, lambda v: f"bt{v}", str),
    Axis("BC", (16, 64), str, lambda v: f"bc{v}", str),
    Axis("DV", (32, 64), str, lambda v: f"dv{v}", str),
    Axis("D", (1, 2, 3, 4), str, lambda v: f"d{v}", str),
    Axis("PREC", ("bf16", "tf32x3"),
         lambda v: f"MmaPrec::{'kBf16' if v == 'bf16' else 'kTf32x3'}",
         lambda v: v, None),
    Axis("DECAY", (False, True), _b, lambda v: "decay" if v else "nodecay", None),
    Axis("GLOBAL", (False, True), _b, lambda v: "global" if v else "raw", None),
)
AXIS = {a.name: a for a in AXES}

#: THE PARTITION, as two constants and nothing else. The prefix of this list that is
#: long enough to reach `TARGET_MEMBERS_PER_SHARD` is the set of sharded axes; the
#: rest of the matrix lives inside a shard.
SHARD_AXIS_PRIORITY = ("PREC", "GLOBAL", "BT", "BC", "DV", "DECAY", "D")
TARGET_MEMBERS_PER_SHARD = 32


def matrix() -> list[dict]:
    """Every tuple of the matrix, in axis-declaration order (the canonical order)."""
    rows: list[dict] = [{}]
    for axis in AXES:
        rows = [dict(row, **{axis.name: v}) for row in rows for v in axis.values]
    return rows


def shard_axes() -> tuple[str, ...]:
    """The shortest priority prefix that partitions the matrix finely enough.

    Structural by construction: the answer is a function of the axis cardinalities
    and the target, so a new template axis re-derives it instead of inheriting a
    stale hand-written file list.
    """
    total = math.prod(len(a.values) for a in AXES)
    want = math.ceil(total / TARGET_MEMBERS_PER_SHARD)
    used: list[str] = []
    have = 1
    for name in SHARD_AXIS_PRIORITY:
        if have >= want:
            break
        used.append(name)
        have *= len(AXIS[name].values)
    if have < want:
        raise SystemExit(f"no prefix of SHARD_AXIS_PRIORITY reaches {want} shards")
    return tuple(used)


def shard_of(row: dict) -> str:
    """The shard file (stem) a template tuple belongs to. THE membership function."""
    return "consumer_" + "_".join(AXIS[n].token(row[n]) for n in shard_axes())


def shards() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for row in matrix():
        out.setdefault(shard_of(row), []).append(row)
    return out


def canonical(row: dict) -> str:
    """One tuple, as the line that participates in the membership hash."""
    return " ".join(f"{a.name}={row[a.name]}" for a in AXES)


def members_sha256(rows: list[dict]) -> str:
    """The shard's IDENTITY: a hash of WHICH instantiations it carries.

    Not of the file's bytes -- a comment edit must not read as a membership change,
    and a membership change must not be able to hide behind an unchanged file size.
    The manifest records this hash, so "only shards X and Y were re-measured" is a
    provable statement rather than a claim.
    """
    blob = "\n".join(sorted(canonical(r) for r in rows)).encode()
    return hashlib.sha256(blob).hexdigest()


def shard_manifest_block() -> dict:
    """`{shard: {members_sha256, count}}` -- what `tools/ratify.py` files in the
    per-arch manifests. Membership only: no paths and no ordering, so the digest
    stays a property of the ratification and not of its filing."""
    return {name: {"members_sha256": members_sha256(rows), "count": len(rows)}
            for name, rows in sorted(shards().items())}


def targs(row: dict, prec_twice: bool = False) -> str:
    """The C++ template argument list for one tuple."""
    prec = AXIS["PREC"].cxx(row["PREC"])
    args = [AXIS["BT"].cxx(row["BT"]), AXIS["BC"].cxx(row["BC"]),
            AXIS["DV"].cxx(row["DV"]), AXIS["D"].cxx(row["D"]), prec]
    if prec_twice:
        args.append(prec)
    args += [AXIS["DECAY"].cxx(row["DECAY"]), AXIS["GLOBAL"].cxx(row["GLOBAL"])]
    return ", ".join(args)


# ---------------------------------------------------------------------------
# THE EMITTERS
# ---------------------------------------------------------------------------
BANNER = "// GENERATED by tools/gen_shards.py -- do not edit, regenerate.\n"


def shard_source(name: str, rows: list[dict]) -> str:
    """One shard translation unit.

    It is TORCH-FREE on purpose (`consumer_launch.cuh` takes the stream as an
    argument): eight translation units that each parsed the torch host headers would
    spend more on header parse than the split saves, and `tools/ratify.py` must be
    able to compile this exact file under the exact shipped flags.
    """
    axes = ", ".join(f"{n}={AXIS[n].token(rows[0][n])}" for n in shard_axes())
    body = [
        BANNER,
        f"// members-sha256: {members_sha256(rows)}\n",
        "//\n",
        f"// SHARD {name} -- {len(rows)} of the consumer's "
        f"{len(matrix())} instantiations:\n"
        f"// the ones with {axes}.\n"
        "// The partition, what the membership hash means, and what breaks if this\n"
        "// file is hand-edited: csrc/rola/src/instantiations/README.md\n",
        "\n",
        '#include "consumer_launch.cuh"\n',
        "\n",
        "namespace rola {\n",
        "\n",
    ]
    for row in rows:
        body.append(f"template void launch_consumer_arm<{targs(row)}>("
                    "const ConsumerParams&, cudaStream_t);\n")
    body.append("\n")
    for row in rows:
        body.append(f"template int query_consumer_occupancy_arm<{targs(row)}>();\n")
    body.append("\n")
    #: THE ROW TABLE. Not a placeholder for P06's dispatch table -- it IS that table's
    #: payload, and it is live already: `rola_consumer_occupancy` and
    #: `rola_consumer_smem_bytes` resolve through it, which is what let the two
    #: hand-written query ladders (and with them the last reason for `consumer.cu` to
    #: see the kernel header at all) be deleted. The `extern` is what gives the
    #: definition external linkage -- a namespace-scope `const` array is internal by
    #: default, and an internal one would link but resolve to nothing.
    key = members_sha256(rows)
    body.append(f"extern const ShardRow kShardRows_{name}[{len(rows)}];\n")
    body.append(f"const ShardRow kShardRows_{name}[{len(rows)}] = {{\n")
    for row in rows:
        body.append(
            f"    {{{row['BT']}, {row['BC']}, {row['DV']}, {row['D']}, "
            f"{AXIS['PREC'].cxx(row['PREC'])}, {_b(row['DECAY'])}, "
            f"{_b(row['GLOBAL'])}, "
            f"&launch_consumer_arm<{targs(row)}>, "
            f"&query_consumer_occupancy_arm<{targs(row)}>, "
            f"consumer::consumer_smem_bytes<{targs(row, prec_twice=True)}>(), "
            f"consumer::consumer_amp_prefetch<{targs(row, prec_twice=True)}>(), "
            f'"{name}", "{key}"}},\n')
    body.append("};\n")
    body.append("\n")
    body.append("}  // namespace rola\n")
    return "".join(body)


#: THE KEY'S RADIX, in axis-declaration order. The key is a mixed-radix index over
#: the axis VALUE POSITIONS, so it is dense exactly when the matrix is a full cross
#: product -- which it is, and which `--check` proves by construction because both the
#: key function and the row index come out of this one definition.
def axis_index_fn(axis) -> str:
    """`dispatch_axis_<NAME>` -- one axis value -> its position, or -1.

    A value the matrix does not carry returns -1 and the key returns -1 with it, so an
    unreachable choice is REFUSED at the lookup rather than resolved to a neighbour.
    That is the whole difference between a table and a threshold.
    """
    lines = [f"inline int dispatch_axis_{axis.name}({axis_cxx_type(axis)} v) {{\n"]
    if axis.values == (False, True):
        lines.append("  return v ? 1 : 0;\n")
    else:
        lines.append("  switch (v) {\n")
        for i, v in enumerate(axis.values):
            lines.append(f"    case {axis.cxx(v)}: return {i};\n")
        lines.append("  }\n  return -1;\n")
    lines.append("}\n")
    return "".join(lines)


def axis_cxx_type(axis) -> str:
    """The C++ type a choice carries this axis in."""
    if axis.values == (False, True):
        return "bool"
    if axis.name == "PREC":
        return "MmaPrec"
    return "int"


def dispatch_table_inc() -> str:
    """The whole dispatch table: the shard row blocks, the key, and the row index.

    THE LAUNCH RESOLVES A ROW, exactly as the residency and footprint queries already
    do, so there is ONE lookup in the translation unit and no arm-shaped code anywhere
    in it. The row index is a mixed-radix array over the axes' value positions:
    `kRowByKey[dispatch_key(choice)]`, a load, against the 1,054-line nested switch it
    replaces (2,462 lines before the COMPACT axis left and the matrix halved). Both
    the key function and the index are emitted from the matrix here, so they cannot
    disagree about the axis order -- which is the failure a hand-written index formula
    would reintroduce.

    ABSENT KEYS ARE `nullptr`, not a neighbouring row. The matrix is a full cross
    product today, so every key is populated; the null arm is what makes a
    non-rectangular matrix (an axis whose values differ per shard) a REFUSAL instead
    of a silent mis-dispatch.
    """
    sh = sorted(shards().items())
    lines = [BANNER,
             "// The dispatch table: every arm's row, the mixed-radix key over the\n"
             "// matrix axes, and the index that turns a key into a row. Included by\n"
             "// `csrc/rola/src/consumer.cu`.\n"]
    for name, rows in sh:
        lines.append(f"extern const ShardRow kShardRows_{name}[{len(rows)}];\n")
    lines.append("\nconstexpr ShardRowBlock kShardRowBlocks[] = {\n")
    for name, rows in sh:
        lines.append(f"    {{kShardRows_{name}, {len(rows)}}},\n")
    lines.append("};\n\n")
    for axis in AXES:
        lines.append(axis_index_fn(axis))
        lines.append("\n")
    lines.append(f"constexpr int kDispatchKeys = {len(matrix())};\n\n")
    lines.append("//: The choice's mixed-radix key, or -1 if any axis carries a value\n"
                 "//: this binary has no arm for.\n"
                 "inline int dispatch_key(const DispatchChoice& c) {\n")
    lines.append("  int key = 0;\n")
    for axis in AXES:
        field = {"PREC": "prec", "DECAY": "decay", "GLOBAL": "global_norm"}.get(
            axis.name, axis.name)
        lines.append(f"  const int i_{axis.name} = dispatch_axis_{axis.name}(c.{field});\n")
        lines.append(f"  if (i_{axis.name} < 0) return -1;\n")
        lines.append(f"  key = key * {len(axis.values)} + i_{axis.name};\n")
    lines.append("  return key;\n}\n\n")
    #: THE INDEX. Each entry is the address of a row inside the shard array that owns
    #: it -- an address constant, so the array is statically initialized and no code
    #: runs before `main`.
    where = {}
    for name, rows in sh:
        for i, row in enumerate(rows):
            where[canonical(row)] = (name, i)
    lines.append("const ShardRow* const kRowByKey[kDispatchKeys] = {\n")
    for row in matrix():
        name, i = where[canonical(row)]
        lines.append(f"    &kShardRows_{name}[{i}],\n")
    lines.append("};\n")
    return "".join(lines)


README = """\
# `csrc/rola/src/instantiations` — the consumer's explicit-instantiation shards

Every file here is GENERATED by `tools/gen_shards.py` from the single matrix
definition in that file. Hand-editing one is not a small liberty: the membership
hash in its header, the `shards` block of every ratification manifest and the
regenerate-and-diff gate all disagree with it immediately, and CI fails.

    python tools/gen_shards.py --write     # regenerate
    python tools/gen_shards.py --check     # what CI runs

## What is here

`consumer_forward_kernel<BT, BC, DV, D, FOLD, READ, DECAY, GLOBAL>` has
{total} reachable instantiations — the exact set `launch_consumer` can select, no
more and no fewer. Compiling them in ONE translation unit put {total} template
expansions through one `cicc`; they are split across **{nshards} shards of
{per} instantiations** on **{axes}**, whole-program (no `-rdc`), so the compile
parallelizes and a ratchet refusal scopes to one shard's slice of the manifest.

Each shard carries, for its {per} tuples: an explicit instantiation of the host
launcher `launch_consumer_arm` (which is what pulls in the device kernel, exactly
once in the whole fatbin), one of `query_consumer_occupancy_arm`, and a `ShardRow`
record holding both plus the arm's `constexpr` shared-memory footprint. The
launcher's DEFINITION lives in `../consumer_launch.cuh`, which `consumer.cu` must
not include — that is what stops the dispatch translation unit from satisfying an
arm by re-instantiating it locally, which would leave the build green and nothing
sharded.

It does NOT make a missing member a link error. This is a Python extension module,
so `ld -shared` leaves the reference undefined for the loader and a dropped arm
would reach a shipped artifact as an `ImportError`. Two gates close it instead:
`python tools/gen_shards.py --check` refuses a shard that disagrees with the matrix
before anything compiles, and `python tools/fatbin_entries.py --check` refuses a
built binary carrying an unresolved `launch_consumer_arm` after the link.

`dispatch_table.inc` carries the shard row blocks, a mixed-radix `dispatch_key`
over the matrix axes, and `kRowByKey`, which turns a key into a row. The launch and
both device-backed queries resolve through it, so there is ONE lookup in the dispatch
translation unit. It comes out of the same matrix definition as the shards, so the
dispatch and the shards cannot enumerate different sets — which is the whole reason
this generator exists.

## The partition

The axes are chosen by `SHARD_AXIS_PRIORITY` and `TARGET_MEMBERS_PER_SHARD`, and by
nothing else — membership is a structural function of the template tuple, never a
measured compile time. Repartitioning (P17's read-path axis takes the matrix to
1,024) is a change to those two constants.

## The membership hash

Each file's `// members-sha256:` line hashes the SORTED LIST OF ITS TUPLES — not the
file's bytes. A comment edit does not move it; adding, dropping or moving one
instantiation does, and the same hash is recorded in `tools/manifests/sm_*.json`'s
`shards` block, which enters the manifest digest. That is what makes "only shards X
and Y were re-measured" a provable claim.

## The files

| shard | members |
|---|---|
{table}
"""


def readme() -> str:
    sh = shards()
    table = "\n".join(f"| `{name}.cu` | {len(rows)} |" for name, rows in sorted(sh.items()))
    return README.format(total=len(matrix()), nshards=len(sh),
                         per=TARGET_MEMBERS_PER_SHARD,
                         axes=" x ".join(shard_axes()), table=table)


def generated() -> dict[str, str]:
    """Every file this generator owns: relative path -> content."""
    out = {f"{name}.cu": shard_source(name, rows) for name, rows in sorted(shards().items())}
    out["dispatch_table.inc"] = dispatch_table_inc()
    for name in list(out):
        out[name] = clang_format(out[name], name)
    out["README.md"] = readme()
    return out


def clang_format(text: str, filename: str) -> str:
    """Pipe generated C++ through the pinned formatter.

    `.inc` is not a language `clang-format` recognizes, so it is formatted as the
    `.cu` fragment it is; the pre-commit hook globs `.cu`/`.cuh` and never sees it,
    but a fragment formatted differently from the file that includes it would be a
    gratuitous inconsistency.
    """
    assume = filename if filename.endswith(".cu") else filename[:-4] + ".cu"
    proc = subprocess.run(
        [CLANG_FORMAT, "--style=file", f"--assume-filename={OUT_DIR / assume}"],
        input=text, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"{CLANG_FORMAT} FAILED on generated {filename}; the "
                         f"generator may not emit files the lint gate would reject")
    return proc.stdout


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="regenerate the tree")
    ap.add_argument("--check", action="store_true",
                    help="regenerate into memory and fail on any difference")
    a = ap.parse_args()
    if a.write == a.check:
        raise SystemExit("pass exactly one of --write / --check")

    want = generated()
    if a.write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        for name in sorted({p.name for p in OUT_DIR.iterdir() if p.is_file()}):
            if name not in want and name != ".gitkeep":
                (OUT_DIR / name).unlink()
                print(f"removed stale {name}")
        for name, text in sorted(want.items()):
            (OUT_DIR / name).write_text(text)
        print(f"wrote {len(want)} files to {OUT_DIR.relative_to(ROOT)} "
              f"({len(shards())} shards x {TARGET_MEMBERS_PER_SHARD} instantiations "
              f"on {' x '.join(shard_axes())})")
        return 0

    bad = []
    on_disk = {p.name for p in OUT_DIR.iterdir() if p.is_file() and p.name != ".gitkeep"}
    for name in sorted(on_disk - set(want)):
        bad.append(f"    {name}: on disk, not generated")
    for name, text in sorted(want.items()):
        path = OUT_DIR / name
        if not path.exists():
            bad.append(f"    {name}: missing")
        elif path.read_text() != text:
            bad.append(f"    {name}: differs from the generated content")
    if bad:
        print("FAIL  the checked-in instantiation shards do not match the matrix "
              "definition in tools/gen_shards.py:")
        print("\n".join(bad))
        print("\n    python tools/gen_shards.py --write")
        return 1
    print(f"shards OK  {len(want)} generated files match the matrix "
          f"({len(matrix())} instantiations over {len(shards())} shards)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
