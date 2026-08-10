"""Print the RUNG 1 delta table from the ncu reports of base and rung arms.

Uses the base's extractor verbatim for the opcode-class partition (the SAME
instrument as p57), and adds the traffic rows the ladder needs in order to see
the kernel move off the bandwidth wall as features accumulate.

The headline uses the EXHAUSTIVE SASS source-page total, which the opcode-class
partition sums to by construction. Under this metric set `smsp__inst_executed.sum`
comes back a multiplexed partial, so it is reported as a consistency ratio rather
than used; the offset is common to both arms and cancels in the delta.

    python ladder_table.py <evidence_dir> <cell suffix, e.g. L4096B2H8> <arms...>
    arms are `C_S_SWZ` triples, e.g. 16_4_0 32_4_1
"""
from __future__ import annotations

import sys

ROOT = __file__.rsplit("/", 2)[0]
sys.path.insert(0, ROOT)
from denseref.ncu_extract import classify, num, opcodes, scalars  # noqa: E402

#: masked share of all mma, from p57b's CORRECTED (element-wise) normalisation.
#: The rung does not change the mask fraction: its mma count is identical.
MASKED = {16: 0.047, 32: 0.081, 64: 0.123}


def row(rep):
    r = scalars(rep)
    hist = opcodes(rep)
    total = sum(hist.values())
    mma = num(r, "smsp__inst_executed_pipe_tensor_op_hmma.sum") or \
        num(r, "sm__inst_executed_pipe_tensor_op_hmma.sum") or \
        sum(n for op, n in hist.items()
            if op.split(".")[0] in ("HMMA", "IMMA", "BMMA", "OMMA"))
    agg = {}
    for op, n in hist.items():
        agg[classify(op)] = agg.get(classify(op), 0) + n
    cls = {k: v for k, v in agg.items() if not k.startswith("residual:")}
    cls["residual"] = sum(v for k, v in agg.items() if k.startswith("residual:"))
    d = dict(inst=total, mma=mma, scalars=r)
    d.update({k: v / mma for k, v in cls.items()})
    return d


def g(d, k, default=0.0):
    return d.get(k, default)


COLS = ("integer/address", "tensor", "data movement", "conversion", "fp math",
        "predicate/control", "sync", "residual")
HDR = ("INT/mma", "tensor", "data mv", "conv", "fp", "pred/ctl", "sync", "resid")


def main(ev, cell, arms):
    b, u, pk = {}, {}, {}
    for a in arms:
        b[a] = row(f"{ev}/base{a}_{cell}.ncu-rep")
        u[a] = row(f"{ev}/rung{a}_0_{cell}.ncu-rep")
        pk[a] = row(f"{ev}/rung{a}_1_{cell}.ncu-rep")

    print(f"== headline, per mma.sync -- cell {cell} ==")
    print(f"{'arm':22s}{'inst/mma':>10s}{'/USEFUL':>9s}" + "".join(f"{h:>10s}" for h in HDR))
    for tag, tbl in (("base", b), ("rung1 WIDEN", u), ("rung1 PACKED", pk)):
        for a in arms:
            d = tbl[a]
            c = int(a.split("_")[0])
            ipm = d['inst'] / d['mma']
            print(f"{tag + ' ' + a.replace('_', '/'):22s}{ipm:10.3f}"
                  f"{ipm / (1 - MASKED[c]):9.2f}"
                  + "".join(f"{g(d, k):10.3f}" for k in COLS))
    print(f"{'DELTA (WIDEN - base)':22s}")
    for a in arms:
        d, e = b[a], u[a]
        c = int(a.split("_")[0])
        dd = e['inst'] / e['mma'] - d['inst'] / d['mma']
        print(f"{'  ' + a.replace('_', '/'):22s}{dd:+10.3f}{dd / (1 - MASKED[c]):+9.2f}"
              + "".join(f"{g(e, k) - g(d, k):+10.3f}" for k in COLS))

    print(f"{'PACKED - WIDEN':22s}")
    for a in arms:
        d, e = u[a], pk[a]
        c = int(a.split("_")[0])
        dd = e['inst'] / e['mma'] - d['inst'] / d['mma']
        print(f"{'  ' + a.replace('_', '/'):22s}{dd:+10.3f}{dd / (1 - MASKED[c]):+9.2f}"
              + "".join(f"{g(e, k) - g(d, k):+10.3f}" for k in COLS))

    print("\n== mma count (independent check that the algorithm is unchanged) ==")
    for a in arms:
        same = b[a]['mma'] == u[a]['mma'] == pk[a]['mma']
        print(f"  {a:10s} base {b[a]['mma']:12,.0f}   rung1 {u[a]['mma']:12,.0f}"
              f"   {'IDENTICAL' if same else 'DIFFERENT'}")

    keys = [
        ("issue_active %peak", "smsp__issue_active.avg.pct_of_peak_sustained_active"),
        ("eligible warps/sched", "smsp__warps_eligible.avg.per_cycle_active"),
        ("active warps/sched", "smsp__warps_active.avg.per_cycle_active"),
        ("achieved occupancy %", "sm__warps_active.avg.pct_of_peak_sustained_active"),
        ("regs/thread", "launch__registers_per_thread"),
        ("dyn smem/CTA (B)", "launch__shared_mem_per_block_dynamic"),
        ("CTAs/SM limit: smem", "launch__occupancy_limit_shared_mem"),
        ("CTAs/SM limit: regs", "launch__occupancy_limit_registers"),
        ("tensor pipe %peak",
         "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active"),
        ("DRAM %peak", "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"),
        ("DRAM %peak (alt)", "dram__throughput.avg.pct_of_peak_sustained_elapsed"),
        ("DRAM read (B)", "dram__bytes_read.sum"),
        ("DRAM write (B)", "dram__bytes_write.sum"),
        ("L2 hit %", "lts__t_sector_hit_rate.pct"),
        ("L1 hit %", "l1tex__t_sector_hit_rate.pct"),
        ("duration", "gpu__time_duration.sum"),
    ]
    print("\n== occupancy / traffic ==")
    print(f"{'':24s}" + "".join(f"{('b ' + a):>15s}{('r ' + a):>15s}" for a in arms))
    for label, k in keys:
        cells = "".join(f"{str(tbl[a]['scalars'].get(k, '--')):>15s}"
                        for a in arms for tbl in (b, u))
        print(f"{label:24s}{cells}")

    print("\n== stall reasons (pcsamp share) ==")
    names = set()
    for tbl in (b, u):
        for a in arms:
            for k in tbl[a]['scalars']:
                if k.startswith("smsp__pcsamp_warps_issue_stalled_") and \
                        not k.endswith("_not_issued"):
                    names.add(k)

    def share(d, k):
        tot = sum(num(d['scalars'], x) for x in names)
        return 100 * num(d['scalars'], k) / tot if tot else 0.0
    order = sorted(names, key=lambda k: -share(u[arms[0]], k))
    print(f"{'':24s}" + "".join(f"{('b ' + a):>12s}{('r ' + a):>12s}" for a in arms))
    for k in order:
        if all(share(tbl[a], k) < 0.05 for a in arms for tbl in (b, u)):
            continue
        cells = "".join(f"{share(tbl[a], k):12.2f}" for a in arms for tbl in (b, u))
        print(f"{k[len('smsp__pcsamp_warps_issue_stalled_'):]:24s}{cells}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
