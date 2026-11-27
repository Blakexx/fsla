# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The measured section of `docs/lab/clock_time.md`, generated from the newest samples of `clock_time`'s five records.

`write(root, data, instance)` reads, in the pinned data `data`, the full grid's and one CTA's records (the clock locked: the lab's sessions lock it)
and the variant `unlocked`'s two (the clock as the card left it) -- the four of five bodies, one commit -- and, read apart,
the variant `reads` at one CTA (the clock locked, all seven bodies: the integer body and the two with a read of a clock a
step, a later commit) and rewrites
`docs/lab/clock_time.md` under `root` from its `MEASURED (` line (the first run: from its `NOT YET MEASURED` line)
through its `Record:` line, every figure the records' own and every comparative word chosen by the figures it
compares. `instance` is the rented box's label of the locked run (`nb...` Nebius, `hs...` Hyperstack, digits vast.ai),
which only words the card's line. It returns the files it wrote (`lab page clock_time --instance LABEL`,
docs/internals/tools/lab.md).
"""

from __future__ import annotations

import pathlib

from _record import Record, read

#: two figures within this share of each other (of the lesser) are said to agree
SAME = 0.005
#: the four records of five bodies, each the selection of its record: the full grid and one CTA, locked and unlocked
RECORDS = {"locked": {}, "locked, one CTA": {"blocks": 1}, "unlocked": {"variant": "unlocked"},
           "unlocked, one CTA": {"blocks": 1, "variant": "unlocked"}}
#: the record of all seven bodies, the clock locked, one CTA: the third table's, read apart from the four
READS = {"blocks": 1, "variant": "reads"}


def write(root: pathlib.Path, data: pathlib.Path, instance: str) -> list[pathlib.Path]:
    inst = instance
    samples = {f: read(data, "clock_time", **selected) for f, selected in RECORDS.items()}
    recs = {f: sample.body() for f, sample in samples.items()}
    rows = {f: Record({r["name"]: r["median"] for r in d["results"]}) for f, d in recs.items()}
    reads_sample = read(data, "clock_time", **READS)
    reads = reads_sample.body()
    read_rows = Record({r["name"]: r["median"] for r in reads["results"]})
    assert all(c["ok"] for c in reads["checks"]), "lab page: the reads record holds a check that failed"
    d = recs["locked"]
    assert all(c["ok"] for r in recs.values() for c in r["checks"]), "lab page: a record holds a check that failed"
    assert len({r["git"]["sha"] for r in recs.values()}) == 1, "lab page: the four records are of different commits"
    assert len({r["reps"] for r in recs.values()}) == 1, "lab page: the four records differ in repetitions"
    assert len({r["driver"] for r in recs.values()}) == 1, "lab page: the four records differ in driver"

    BODIES = (
        "eight independent integer operations a step",
        "a chain of four shared loads a step",
        "a chain of four scattered loads of a 64 MB table a step",
        "a store to global memory a step",
        "one wgmma a step, its commit and wait",
    )
    INT, SHARED, GLOBAL, STORE, WGMMA = BODIES
    #: the four columns: the record, its head
    COLS = (
        ("unlocked, one CTA", "unlocked, one CTA"),
        ("unlocked", "unlocked, full grid"),
        ("locked, one CTA", "locked, one CTA"),
        ("locked", "locked, full grid"),
    )

    def rate(f, body):
        return rows[f][f"{body}: the cycle counter's rate"]

    def step_ns(f, body):
        return rows[f][f"{body}: a step by the nanosecond timer"]

    def table(figure):
        out = ["| a body |" + "".join(f" {h} |" for _, h in COLS), "|---|" + "---|" * len(COLS)]
        for b in BODIES:
            out.append(f"| {b} |" + "".join(f" {figure(f, b):.1f} |" for f, _ in COLS))
        return "\n".join(out)

    def pct(x):
        return f"{100.0 * x:.1f}%"

    def against(a, b):
        """a against b: its share over (positive) or under (negative) b, and the word the share chooses"""
        share = (a - b) / min(a, b)
        word = "the same as" if abs(share) <= SAME else ("over" if share > 0 else "under")
        return share, word

    def long_or_short(a, b):
        share, word = against(a, b)
        return word if word == "the same as" else ("longer than" if share > 0 else "shorter than")

    CARD = (
        f"{d['device']}, 132 SMs, Nebius preemptible VM {inst[2:]}"
        if inst.startswith("nb")
        else f"{d['device']}, Hyperstack spot VM {inst[2:]}"
        if inst.startswith("hs")
        else f"{d['device']}, vast.ai instance {inst}"
    )
    U, UF, L, LF = "unlocked, one CTA", "unlocked", "locked, one CTA", "locked"
    others = (INT, SHARED, GLOBAL, STORE)

    # the counter through waits on memory: the global-load chain's rate against the integer body's, unlocked
    wait = []
    for f, head in ((U, "one CTA"), (UF, "the full grid")):
        share, word = against(rate(f, GLOBAL), rate(f, INT))
        wait.append((head, rate(f, GLOBAL), rate(f, INT), share, word))
    keeps = all(w[4] == "the same as" for w in wait)
    wait_detail = "; ".join(
        f"at {h} the global-load chain's {g:.1f} cycles a microsecond is {word} the integer body's {i:.1f}"
        + ("" if word == "the same as" else f" ({pct(abs(s))})")
        for h, g, i, s, word in wait
    )
    wait_line = (
        f"- THE COUNTER {'KEEPS ONE RATE THROUGH WAITS ON MEMORY' if keeps else 'DOES NOT KEEP ONE RATE THROUGH WAITS ON MEMORY'} "
        f"({wait_detail}; two rates are called the same within {pct(SAME)})."
    )

    # the unlocked wgmma body against the others' mean
    wg = []
    for f, head in ((U, "one CTA"), (UF, "the full grid")):
        mean = sum(rate(f, b) for b in others) / len(others)
        share, word = against(rate(f, WGMMA), mean)
        wg.append((head, rate(f, WGMMA), mean, share, word))
    wg_detail = "; ".join(
        f"at {h} the wgmma body's {w:.1f} is "
        + ("the same as" if word == "the same as" else f"{pct(abs(s))} {word}")
        + f" the other four bodies' mean {m:.1f}"
        for h, w, m, s, word in wg
    )

    wg_head = (
        "UNLOCKED, THE WGMMA BODY'S RATE IS THE OTHERS'"
        if all(w[4] == "the same as" for w in wg)
        else "UNLOCKED, THE WGMMA BODY'S RATE IS OFF THE OTHERS'"
    )

    # the locked rates: every body within M% of one rate
    lock = []
    for f, head in ((L, "one CTA"), (LF, "the full grid")):
        rs = [rate(f, b) for b in BODIES]
        spread = (max(rs) - min(rs)) / min(rs)
        lock.append((head, spread, sum(rs) / len(rs), min(rs), max(rs)))
    lock_detail = "; ".join(
        f"at {h} the five bodies' rates run {lo:.1f} to {hi:.1f} cycles a microsecond, within {pct(s)} of one another"
        f" (mean {m:.1f})"
        for h, s, m, lo, hi in lock
    )
    lock_head = (
        "LOCKED, EVERY BODY IS ONE RATE" if all(x[1] <= SAME for x in lock) else "LOCKED, THE BODIES' RATES STILL DIFFER"
    )
    locked_rate = sum(rate(LF, b) for b in BODIES) / len(BODIES)

    # the steps, locked against unlocked
    def step_line(body, label):
        parts = []
        for (fu, head), (fl, _) in zip((COLS[0], COLS[1]), (COLS[2], COLS[3])):
            u, lk = step_ns(fu, body), step_ns(fl, body)
            share, _ = against(lk, u)
            word = long_or_short(lk, u)
            parts.append(
                f"{head.split(', ')[1]}: {lk:.1f} ns locked is {word} {u:.1f} ns unlocked"
                + ("" if word == "the same as" else f" by {pct(abs(share))}")
            )
        return f"{label} ({'; '.join(parts)})"

    # THE READS (the third table): the integer body and the two with a read of a clock a step, from the reads record
    READ_BODIES = (
        (INT, "eight independent integer operations, no read"),
        ("eight independent integer operations and a read of the cycle counter a step", "the same and a read of the cycle counter"),
        ("eight independent integer operations and a read of the nanosecond timer a step", "the same and a read of the nanosecond timer"),
    )
    r_int, r_clock, r_timer = (b for b, _ in READ_BODIES)

    def r_cycles(b):
        return read_rows[f"{b}: a step by the cycle counter"]

    def r_ns(b):
        return read_rows[f"{b}: a step by the nanosecond timer"]

    def r_rate(b):
        return read_rows[f"{b}: the cycle counter's rate"]

    reads_table = "\n".join(
        ["| a body | cycles a step | nanoseconds a step | the cycle counter's rate (cycles a microsecond) |", "|---|---|---|---|"]
        + [f"| {label} | {r_cycles(b):.1f} | {r_ns(b):.1f} | {r_rate(b):.1f} |" for b, label in READ_BODIES]
    )

    def read_cost(b, name):
        dc, dn = r_cycles(b) - r_cycles(r_int), r_ns(b) - r_ns(r_int)
        return (
            f"a read of the {name} costs {dc:.1f} cycles a step ({r_cycles(b):.1f} against the integer body's {r_cycles(r_int):.1f},"
            f" {r_cycles(b) / r_cycles(r_int):.1f} times) and {dn:.1f} ns ({r_ns(b):.1f} against {r_ns(r_int):.1f})"
        )

    share_c, word_c = against(r_cycles(r_timer), r_cycles(r_clock))
    share_n, word_n = against(r_ns(r_timer), r_ns(r_clock))
    if word_c == word_n == "the same as":
        same_reads = (
            f"THE TWO READS COST THE SAME: the timer's step is {r_cycles(r_timer):.1f} cycles and {r_ns(r_timer):.1f} ns, the counter's"
            f" {r_cycles(r_clock):.1f} cycles and {r_ns(r_clock):.1f} ns (called the same within {pct(SAME)})"
        )
    else:
        same_reads = (
            f"THE TWO READS DO NOT COST THE SAME: the timer's step is {pct(abs(share_c))} {word_c} the counter's in cycles"
            f" ({r_cycles(r_timer):.1f} against {r_cycles(r_clock):.1f}) and {pct(abs(share_n))} {word_n} it in nanoseconds"
            f" ({r_ns(r_timer):.1f} against {r_ns(r_clock):.1f})"
        )
    rate_spread = (max(r_rate(b) for b, _ in READ_BODIES) - min(r_rate(b) for b, _ in READ_BODIES)) / min(
        r_rate(b) for b, _ in READ_BODIES
    )
    reads_sha = reads["git"]["sha"][:8] + (" (a dirty tree)" if reads["git"].get("dirty") else "")
    reads_text = f"""A READ'S COST, one CTA, the clock locked (`{reads_sample.cited}`: the experiment at {reads_sha}, {reads["reps"]} repetitions, driver {reads["driver"]},
{reads["date"][:10]}; every check passed): the integer body beside the same body with one 32-bit read a step of the cycle counter
and of the nanosecond timer's low word, the read's value xor-ed into a value the step keeps:

{reads_table}

WHAT IT DECIDES, the reads: {read_cost(r_clock, "cycle counter")}; {read_cost(r_timer, "nanosecond timer")}. {same_reads}.
THE RATE IS {"ONE" if rate_spread <= SAME else "NOT ONE"} FOR ALL THREE BODIES: they run {min(r_rate(b) for b, _ in READ_BODIES):.1f} to {max(r_rate(b) for b, _ in READ_BODIES):.1f} cycles a microsecond, within {pct(rate_spread)} of one another.
"""

    unlocked_date = recs[UF]["date"][:10]
    text = f"""MEASURED ({CARD}, driver {d["driver"]}; the locked records {d["date"][:10]}, the unlocked {unlocked_date}, the experiment at
{d["git"]["sha"][:8]}; {d["reps"]} repetitions; every check passed in all four records: every body's clocks advanced in
every warp and the launch's rate was a GPU clock's; the median): THE CYCLE COUNTER'S RATE, in cycles a microsecond, a body's
cycles over its nanoseconds, one CTA and the full grid, the clock as the card left it (unlocked) and locked:

{table(rate)}

A STEP'S TIME, in nanoseconds by the nanosecond timer, the same four columns:

{table(step_ns)}

WHAT IT DECIDES.
{wait_line}
- {wg_head}: {wg_detail}.
- {lock_head}: {lock_detail}. What the lock asked for and the clock the card then reported are not in the
  record (the box's own account of the lock is beside its output, not in it); the record gives only the measured rate,
  {locked_rate:.1f} cycles a microsecond at the full grid across the five bodies.
- A STEP IN REAL TIME, locked against unlocked: {step_line(WGMMA, "a wgmma step")}; {step_line(INT, "an integer step")}.

{reads_text}
Record: `{samples[LF].cited}` and `{samples[L].cited}` (the clock locked; the full grid, one CTA) and
`{samples[UF].cited}` and `{samples[U].cited}` (unlocked), these four of five bodies; `{reads_sample.cited}` (locked, one CTA,
all seven bodies).
"""
    p = root / "docs/lab/clock_time.md"
    s = p.read_text()
    i = s.index("MEASURED (") if "MEASURED (" in s else s.index("NOT YET MEASURED")
    # through the end of the `Record:` paragraph (the next blank line, or the file's end)
    r = s.index("\nRecord: `", i)
    j = s.find("\n\n", r + 1)
    j = len(s) if j < 0 else j + 1
    p.write_text(s[:i] + text + s[j:])
    return [p]
