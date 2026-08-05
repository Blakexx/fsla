"""HOW it is measured: **the interleaved paired-arm engine. THIS IS THE METHODOLOGY.**

Every performance number this repository publishes comes through here, so the rules
are written down once, in the place that implements them, rather than repeated in
each driver's docstring where they can drift apart.

**1. THE ARMS ALTERNATE REP BY REP.** Never arm-A-block then arm-B-block. A block
layout lets a drift step -- a clock change, a thermal step, another process arriving
-- land BETWEEN the arms, where it is indistinguishable from the effect being
measured. Interleaving puts each pair's two measurements adjacent in time, so a
drift affects both nearly equally and mostly divides out of their ratio. **Measured
on this box, the bias was real**: sequential blocks read x3.27-3.30 where
interleaving read x3.33-3.35, and interleaving cut the run-to-run spread by ~7x.

**2. THE REPORTABLE STATISTIC IS THE MEDIAN OF THE PER-PAIR RATIOS**, each pair
being two measurements adjacent in time. Not the ratio of the medians -- that
discards the pairing, which is the entire point of having paid for it.

**3. ABSOLUTE TIMES AND IQRs ARE REPORTED ALONGSIDE, NEVER INSTEAD.** A ratio
without its spread is a claim without an error bar.

**4. A CELL WHOSE IQR CANNOT DISCRIMINATE THE EFFECT BEING CLAIMED IS REPORTED AS A
RANGE OR DROPPED, AND WHICH WAS DONE IS STATED.** This is the rule that keeps the
harness from manufacturing precision it does not have.

**5. ONE STOPWATCH, AND ITS NAME TRAVELS WITH EVERY NUMBER.** The timer is chosen
from :data:`INSTRUMENTS` by name, the committed choice is
:data:`CANONICAL_INSTRUMENT`, and the name is recorded beside every measurement.
Two numbers taken with different instruments are two different quantities:
:func:`same_instrument` refuses to compare them, so a baseline recorded under one
stopwatch cannot be gated against a run taken with another. ``ncu`` durations are a
third instrument (``source: "ncu"``) and the same refusal covers them.

**6. ONE ESCAPE HATCH, AND IT IS NAMED.** :func:`arm_time` measures ONE arm's
absolute times, because the paired doctrine cannot express the only comparison it
cannot hold in one process: two builds differing by a COMPILE-TIME kernel constant.
The caller alternates PROCESSES and pairs across them. It is the exception, it is
documented as one, and nothing else in this package times an arm alone.

No standard benchmarking library provides this -- interleaved pairing, ratio IQRs,
counter corroboration, and a committed statistics manifest are the harness's own.
What IS standardized is the plumbing around it (§6.9): pytest-benchmark stores and
compares the per-``(cell, arm)`` latencies, at thresholds DERIVED from each cell's
measured IQR.
"""
from __future__ import annotations

import statistics
import time

import torch

from bench.cells import CELLS
from rola.ops.consumer import _consumer_forward
from rola.routing.types import LeafMassDecay

# ---------------------------------------------------------------------------
# INTERLEAVED PAIRED-RATIO A/B
# ---------------------------------------------------------------------------

def _iqr(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[int(0.75 * n)] - xs[int(0.25 * n)]




# ---------------------------------------------------------------------------
# THE STOPWATCH
# ---------------------------------------------------------------------------

def _elapsed_cuda_events(call):
    """Device-side elapsed time between two events straddling the launch.

    The events are recorded on the current stream, so what is measured is the
    device's own view of the work: submission latency and the host's return from
    the launch are outside the interval. Both syncs are still taken -- the first so
    the start event is not queued behind unrelated work, the last because
    `elapsed_time` is only defined once both events have completed.
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    out = call()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end), out


def _elapsed_perf_counter(call):
    """Host wall time around a full device sync, on both sides.

    A strictly larger quantity than the event interval: it contains the launch's
    host cost and the sync's return latency as well as the kernel. Kept as a named
    instrument rather than deleted because the launch-side cost is a real part of
    this kernel's step (the un-fused glue), and a driver that wants to measure it
    should say so by name instead of by choosing a different timing idiom.
    """
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = call()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3, out


#: THE INSTRUMENT REGISTRY. A timer is selected BY NAME, and the name is what gets
#: recorded beside the number; adding a timing idiom means adding an entry here, so
#: an instrument cannot enter the record without a name to record.
INSTRUMENTS = {
    "cuda_events": _elapsed_cuda_events,
    "perf_counter": _elapsed_perf_counter,
}

#: THE COMMITTED CHOICE. Device events, because the quantity every published claim
#: is about is the kernel's own duration, and because it is the instrument the
#: campaign's banked numbers were actually taken with -- that recipe lived only in a
#: workflow document, so the tree and the record measured different quantities. It
#: is one name in one place: changing it is a re-measurement of every committed
#: baseline, which is what it should be.
CANONICAL_INSTRUMENT = "cuda_events"

#: Launches discarded before any is recorded. The kernel's first call builds its
#: schedule and warms the instruction cache; ten is the campaign recipe's count.
WARMUP_LAUNCHES = 10


def same_instrument(a: str, b: str, what: str) -> None:
    """Refuse to compare two numbers taken with different stopwatches.

    Not a warning. The gap between the instruments is not noise -- it is the launch
    and sync cost one of them includes -- so a comparison across them reports that
    difference as though it were a property of the kernel. The same refusal covers
    ``ncu`` durations, which are taken under profiling conditions (replay, flushed
    caches, base clocks) that no wall-clock instrument reproduces.
    """
    if a != b:
        raise ValueError(
            f"{what}: refusing to compare a number taken with instrument {a!r} "
            f"against one taken with {b!r}. These are different quantities, not the "
            f"same quantity measured twice. Re-measure the older side with the "
            f"current instrument ({CANONICAL_INSTRUMENT!r}) and record it as a new "
            f"measurement.")


def _timed(cell, fx, plan, decay, g_read, schedule, *,
           instrument: str = CANONICAL_INSTRUMENT):
    def call():
        _, _, sch = _consumer_forward(
            fx["v_f32"], fx["read_f32"], fx["write_f32"],
            None if g_read is None else fx["g_read_f32"], fx["g_write_f32"],
            plan, decay=decay, schedule=schedule)
        return sch

    return INSTRUMENTS[instrument](call)




# ---------------------------------------------------------------------------
# PER-FEATURE PRICE -- the new kernel against ITSELF
#
# A production A/B cannot price a FEATURE, only a kernel: production implements
# decay with a different algorithm (the token-sequential walk) and `global` with a
# different state layout, so its decay-vs-none delta moves two things at once.
# Holding the kernel fixed and flipping one axis is what isolates the feature, and
# residency (3 CTAs/SM on every arm) is held fixed with it.
# ---------------------------------------------------------------------------

def _paired_one_kernel(cell, fx, arm_a, arm_b, reps, label, extra="",
                       instrument: str = CANONICAL_INSTRUMENT):
    """Interleaved paired ratio between two ARMS OF THE NEW KERNEL on one fixture.

    `arm_*` are `(norm, decay_temp)`. Interleaved rep by rep, per the standing
    rule's interleaving clause; the reportable statistic is the median of the
    per-pair ratios. Both arms are timed with ONE instrument by construction --
    they share this argument -- and its name is returned with the result.
    """
    def mk(arm):
        norm, temp = arm
        d = None if temp is None else LeafMassDecay(dials=fx["dials"][temp])
        gr = None if norm == "global" else fx["g_read"]
        pl, sch = cell.plan(norm=norm, decay_on=d is not None, prec="bf16",
                            fx=fx)
        for _ in range(WARMUP_LAUNCHES):
            _, sch = _timed(cell, fx, pl, d, gr, sch, instrument=instrument)
        return pl, d, gr, sch

    pa, da, ga, sa = mk(arm_a)
    pb, db, gb, sb = mk(arm_b)
    ta, tb, pair = [], [], []
    for _ in range(reps):
        x, _ = _timed(cell, fx, pa, da, ga, sa, instrument=instrument)
        y, _ = _timed(cell, fx, pb, db, gb, sb, instrument=instrument)
        ta.append(x)
        tb.append(y)
        pair.append(y / x)
    ma, mb, mr = statistics.median(ta), statistics.median(tb), statistics.median(pair)
    print(f"{cell.name} {label}{extra}  [{instrument}]")
    print(f"  a={arm_a} {ma:.3f} ms (IQR {_iqr(ta):.3f})   "
          f"b={arm_b} {mb:.3f} ms (IQR {_iqr(tb):.3f})")
    print(f"  PAIRED b/a {mr:.4f} (IQR {_iqr(pair):.4f})   = {100 * (mr - 1):+.1f} %")
    return dict(paired=mr, paired_iqr=_iqr(pair), a_ms=ma, b_ms=mb,
                a_times=ta, b_times=tb, source=instrument)




def arm_time(name, *, reps=31, norm="global", decay_temp=None, bt=None,
             instrument: str = CANONICAL_INSTRUMENT):
    """ONE arm of the new consumer, absolute times -- the CROSS-BUILD instrument.

    **Every other timing mode in this file is a same-process paired A/B, and that is
    the standing doctrine. This mode exists for the one comparison the doctrine
    cannot express: two arms that differ by a COMPILE-TIME constant of the kernel
    itself** (the launch bound's register budget, `consumer_reg_budget`), which cannot
    coexist in one binary and therefore cannot alternate rep by rep in one process.

    The pairing is recovered one level up: the caller alternates PROCESSES (build A,
    build B, build A, ...), each process reporting the median of its own reps, and
    the per-pair ratio is taken across adjacent process pairs. That is weaker than
    rep-level interleaving -- a drift step can land between two processes rather
    than between two reps -- so this mode prints median, IQR and min, the caller
    reports the IQR alongside, and the verdict is corroborated with ncu counters,
    which are contention-independent.
    """
    cell = CELLS[name]
    fx = cell.build()
    decay = None if decay_temp is None else LeafMassDecay(dials=fx["dials"][decay_temp])
    gr = None if norm == "global" else fx["g_read"]
    plan, sch = cell.plan(norm=norm, decay_on=decay is not None, prec="bf16",
                          fx=fx, bt=bt)
    for _ in range(WARMUP_LAUNCHES):                     # warm + schedule
        _, sch = _timed(cell, fx, plan, decay, gr, sch, instrument=instrument)
    ts = []
    for _ in range(reps):
        t, _ = _timed(cell, fx, plan, decay, gr, sch, instrument=instrument)
        ts.append(t)
    med = statistics.median(ts)
    print(f"ARMTIME {cell.name} norm={norm} decay={decay_temp} reps={reps} "
          f"[{instrument}]")
    print(f"  median {med:.4f} ms   IQR {_iqr(ts):.4f}   min {min(ts):.4f}")
    return dict(median_ms=med, iqr=_iqr(ts), min_ms=min(ts), times=ts,
                source=instrument)

