"""Read one .ncu-rep and print the P57 measurement table.

Two pages are used, both from the SAME capture:
  --page raw     : the scalar counters (issue-active, stalls, occupancy, ...)
  --page source  : per-SASS-line `Instructions Executed`, which is summed by
                   opcode to give the opcode-class partition. The partition is
                   EXHAUSTIVE by construction -- anything the table does not
                   name lands in `residual` and is printed by name, so the
                   classes always sum to the measured total.

    python ncu_extract.py report.ncu-rep
"""
from __future__ import annotations

import csv
import io
import re
import subprocess
import sys

CLASSES = [
    ("tensor", ("HMMA", "IMMA", "BMMA", "OMMA", "HGMMA")),
    ("data movement", ("LDG", "STG", "LDS", "STS", "LDSM", "LDGSTS", "LDL", "STL",
                       "LD", "ST", "MOV", "SHFL", "LDC", "ULDC", "R2P", "P2R",
                       "R2UR", "S2R", "S2UR", "SEL", "PRMT", "REDUX", "MATCH", "CS2R")),
    ("integer/address", ("IADD3", "IMAD", "IABS", "IMNMX", "ISCADD", "LEA", "LOP3",
                         "LOP", "SHF", "SHL", "SHR", "POPC", "FLO", "BREV", "VIADD",
                         "UIADD3", "UIMAD", "ULEA", "ULOP3", "USHF", "UMOV", "VOTEU",
                         "IADD", "XMAD", "BMSK")),
    ("conversion", ("F2F", "F2I", "I2F", "I2I", "FRND", "CVT", "F2FP", "I2IP")),
    ("fp math", ("FADD", "FMUL", "FFMA", "FSEL", "FMNMX", "MUFU", "FCHK", "DADD",
                 "DMUL", "DFMA", "HADD2", "HMUL2", "HFMA2", "FSET")),
    ("predicate/control", ("ISETP", "FSETP", "DSETP", "HSETP2", "PLOP3", "PSETP",
                           "UISETP", "UPLOP3", "BRA", "BRX", "JMP", "JMX", "BSSY",
                           "BSYNC", "SSY", "SYNC", "CAL", "CALL", "RET", "EXIT",
                           "BREAK", "PBK", "NOP", "YIELD", "KILL", "VOTE")),
    ("sync", ("BAR", "MEMBAR", "DEPBAR", "WARPSYNC", "ERRBAR", "ATOMS", "ATOMG",
              "ATOM", "RED", "LDGDEPBAR", "MEMBAR")),
]


def classify(op: str) -> str:
    root = op.split(".")[0]
    for name, prefixes in CLASSES:
        if root in prefixes:
            return name
    return "residual:" + root


def page(rep: str, pg: str) -> str:
    return subprocess.run(["ncu", "-i", rep, "--page", pg, "--csv"],
                          capture_output=True, text=True).stdout


def scalars(rep: str) -> dict:
    rows = list(csv.DictReader(io.StringIO(page(rep, "raw"))))
    return rows[-1]


def opcodes(rep: str) -> dict:
    text = page(rep, "source")
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith('"Address"'))
    rdr = csv.DictReader(io.StringIO("\n".join(lines[start:])))
    hist: dict[str, int] = {}
    for row in rdr:
        src = (row.get("Source") or "").strip()
        if not src:
            continue
        m = re.match(r"^(?:@!?U?P\d+\s+)?([A-Z][A-Z0-9_.]*)", src)
        if not m:
            continue
        n = row.get("Instructions Executed") or "0"
        try:
            n = int(n.replace(",", ""))
        except ValueError:
            continue
        hist[m.group(1)] = hist.get(m.group(1), 0) + n
    return hist


def num(row, key, default=0.0):
    v = row.get(key)
    if v in (None, ""):
        return default
    try:
        return float(str(v).replace(",", ""))
    except ValueError:
        return v


def report(rep: str):
    r = scalars(rep)
    hist = opcodes(rep)
    total_src = sum(hist.values())
    mma = num(r, "smsp__inst_executed_pipe_tensor_op_hmma.sum") or \
        num(r, "sm__inst_executed_pipe_tensor_op_hmma.sum") or \
        sum(n for op, n in hist.items() if op.split(".")[0] in
            ("HMMA", "IMMA", "BMMA", "OMMA"))
    inst = num(r, "smsp__inst_executed.sum") or num(r, "sm__inst_executed.sum")

    print(f"kernel   {r.get('Kernel Name','?')[:110]}")
    print(f"grid {r.get('Grid Size')}  block {r.get('Block Size')}")
    print(f"{'duration (raw metric)':34s} {num(r,'gpu__time_duration.sum'):12,.1f}")
    print(f"{'inst executed (warp-level)':34s} {inst:12,.0f}")
    print(f"{'mma.sync':34s} {mma:12,.0f}")
    print(f"{'INSTRUCTIONS PER mma.sync':34s} {inst/mma:12.2f}" if mma else "no mma")
    print(f"{'  (SASS source-page total)':34s} {total_src:12,.0f}"
          f"   ratio to raw counter {total_src/inst if inst else 0:.4f}")

    print("\n-- opcode-class partition (per mma.sync, and % of stream)")
    agg: dict[str, int] = {}
    for op, n in hist.items():
        agg[classify(op)] = agg.get(classify(op), 0) + n
    resid = {k: v for k, v in agg.items() if k.startswith("residual:")}
    named = {k: v for k, v in agg.items() if not k.startswith("residual:")}
    named["residual"] = sum(resid.values())
    for cls, n in sorted(named.items(), key=lambda kv: -kv[1]):
        print(f"  {cls:22s} {n:12,.0f}  {100*n/total_src:6.2f}%   "
              f"{n/mma if mma else 0:8.3f} /mma")
    print(f"  {'TOTAL':22s} {total_src:12,.0f}  100.00%   "
          f"{total_src/mma if mma else 0:8.3f} /mma")
    if resid:
        print("  residual opcodes:", ", ".join(
            f"{k[9:]}={v:,}" for k, v in sorted(resid.items(), key=lambda kv: -kv[1])))

    print("\n-- top opcodes")
    for op, n in sorted(hist.items(), key=lambda kv: -kv[1])[:16]:
        print(f"  {op:26s} {n:12,.0f}  {100*n/total_src:6.2f}%")

    print("\n-- issue / occupancy")
    for label, key in [
        ("issue_active %peak", "smsp__issue_active.avg.pct_of_peak_sustained_active"),
        ("eligible warps/sched", "smsp__warps_eligible.avg.per_cycle_active"),
        ("active warps/sched", "smsp__warps_active.avg.per_cycle_active"),
        ("achieved occupancy %", "sm__warps_active.avg.pct_of_peak_sustained_active"),
        ("registers/thread", "launch__registers_per_thread"),
        ("dyn smem/CTA (B)", "launch__shared_mem_per_block_dynamic"),
        ("static smem/CTA (B)", "launch__shared_mem_per_block_static"),
        ("CTAs/SM limit: smem", "launch__occupancy_limit_shared_mem"),
        ("CTAs/SM limit: regs", "launch__occupancy_limit_registers"),
        ("CTAs/SM limit: warps", "launch__occupancy_limit_warps"),
        ("CTAs/SM limit: blocks", "launch__occupancy_limit_blocks"),
        ("waves per SM", "launch__waves_per_multiprocessor"),
        ("tensor pipe %peak",
         "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active"),
        ("DRAM %peak", "dram__throughput.avg.pct_of_peak_sustained_elapsed"),
        ("L2 hit %", "lts__t_sector_hit_rate.pct"),
        ("ldmatrix", "smsp__inst_executed_op_ldsm.sum"),
        ("shared LD", "smsp__inst_executed_op_shared_ld.sum"),
        ("shared ST", "smsp__inst_executed_op_shared_st.sum"),
    ]:
        print(f"  {label:24s} {r.get(key, '--')}")

    print("\n-- stall reasons (pcsamp warp-cycle shares)")
    st, tot = {}, 0.0
    for key, val in r.items():
        if key.startswith("smsp__pcsamp_warps_issue_stalled_") and not key.endswith("_not_issued"):
            try:
                x = float(str(val).replace(",", ""))
            except (TypeError, ValueError):
                continue
            st[key[len("smsp__pcsamp_warps_issue_stalled_"):]] = x
            tot += x
    for n, x in sorted(st.items(), key=lambda kv: -kv[1]):
        if x:
            print(f"  {n:26s} {x:12,.0f}  {100*x/tot:6.2f}%")
    print(f"  {'TOTAL':26s} {tot:12,.0f}")


if __name__ == "__main__":
    report(sys.argv[1])
