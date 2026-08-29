# benchmarks/ — the chunk arm's harness

> **K31 (2026-08-20).** The consumer subjects and the `bench_chunk.py` front door are
> DELETED (baseline = tag `baseline/pre-k31`). The surviving subjects — `chunk_facts`,
> `chunk_block_bits`, `decode_step`, `entmax_solve` — run through their unit drivers
> under `benchmarks/unit/`.

The methodology, the instruments and the record are `docs/measurement.md`; the lock rule
is `docs/testing.md`. This file is the map.

**`benchmarks/unit/` — one driver per custom kernel, and the roster is the oracle tests'.**
The chunk forward, the facts pass, the block bits, the divide, the backward seeds, the
entmax solves and the decode step each get one driver, because a bench roster that does
not match the correctness roster is a roster with kernels nobody measures. Each driver
prices exactly one launch: everything the launch does not pay for — the packed sides,
the state plane, the routing — is built outside the timed callable, so a row is
attributable to the kernel it names. The facts pass is one launch that emits two facts,
so it is one row: the union table is priced there and nowhere else.

**`benchmarks/integration/` — the compositions the unit rows cannot be summed into.**
The whole layer step (projections, producer, kernel, readout, output projection) and the
prefill→decode continuation across one carried state. These are where the glue lives, and
the glue is more than half of a step; a suite of per-kernel rows that all improve while
the step does not is exactly what this tier exists to catch.

**`benchmarks/bench/` — the library both tiers share.** `chunk_cells.py` is the frozen
cell registry (a cell is its constructor and its seed) with its committed manifest;
`subjects.py` is what each roster entry times; `pairing.py` is the interleaved paired-ratio
engine; `discipline.py` is the preconditions, enforced; `stats.py` and `regression.py` are
whether a difference is real; `ledger.py` is where the number lives afterwards;
`chunk_baseline.py` is the committed per-cell latency baseline and the read-only gate — its
rows are medians and IQRs over SEVEN SESSION MEDIANS, because the gate's comparison crosses
two processes and a within-session block IQR does not bound that gap (`docs/measurement.md`).

**THE LOCK: this harness takes it itself (K46), and every driver is invoked BARE.**
`bench.discipline.disciplined` holds `tools/gpu_lock.py`'s `gpu_lock()` for its whole
body; never wrap a driver in an external `flock` on the same path (self-deadlocks).
(`bench_chunk.py`, the front door these example lines used to name, was deleted in K31
per the callout box above — the roster now runs through the per-kernel drivers below.)

```bash
python benchmarks/unit/bench_chunk_facts.py --tier landing
python benchmarks/unit/bench_block_bits.py --tier landing
python benchmarks/unit/bench_entmax.py --tier landing
python benchmarks/unit/bench_decode_step.py --tier landing
```

`benchmarks/bench_layer.py`, `bench_scaling.py` and `bench_paging.py` predate this harness
and are curve and attribution studies rather than gated cells; they state their own usage
lines and use the same statistics.

`benchmarks/bench_prefill.py`, `bench_intra.py` and `bench_prefill_backward.py` are the
K31 family's own tables. They are NOT gates — the acceptance bars live on the queue card
— and they take every row in ONE process on one quiet card, including the
flash-attention row, because a number from another session carries that session's clock
and card temperature as alternative explanations. `bench_prefill_backward.py` reports the
reverse pass beside the forward's own wall and splits the two carry scans from the intra
backward, since those are the two halves a ratio can move for different reasons.
