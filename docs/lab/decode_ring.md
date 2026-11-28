# `decode_ring`

Mirrors `measure/lab/experiments/decode_ring.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

What form of decode's walk holds the card's HBM rate on the state's pages? A decode step's bytes are the touched pages'
alone, scattered over a plane far larger than the cache, so the walk's form is decided by the bytes it holds in flight
and what it costs a byte. Four forms walk five patterns of a step's touched leaves:

- REG: a warp loads its position's pieces into registers and computes from them: one position in flight a warp (the
  walk as the library builds it today).
- BULK: a ring of `K` slots a warp in shared memory, each position's pieces landed by bulk copies (`cp.async.bulk`,
  issued by the warp's first lane, a slot's mbarrier counting its bytes) `K - 1` positions ahead of its use.
- PIECES: the same ring landed by 16-byte `cp.async` copies, every lane its chunks, a position a copy group.
- PRODUCER: warp 0 fills fifteen consumer warps' rings by bulk copies, a full and an empty barrier a slot.

The patterns: WRITE_ALL (an atom written whole: its page read, every leaf deposited, re-rounded to its group exponent
and stored, then read by every head); READ_ALL (an atom's sixteen leaves read only: its hi plane, exponents and masses);
WRITE_ONE and READ_ONE (one leaf an atom, scattered, written or read only; a position is one atom a lane group); and
RECORD (READ_ONE on a page laid out with a leaf's exponents and mass beside its hi line, a 160-byte record a leaf at
`d_v` 64 and 96 at 32: the layout a scattered read would want, measured and not adopted: the state's page is the owner's format). Each
at G = 1 and 8 read heads (`G x 9` accumulators a lane) and `d_v` 64, and at G = 8 and `d_v` 32; the `d_v` 64, G = 8
rows again with every warp's timeline stamped.

THE FORM. A CTA an SM of 512 threads; the plane 2^21 pages at `d_v` 64 and 2^22 at 32 (6.6 and 6.8 GB). Position `q` of
CTA `b` is its items `(b Q + q) L + k` (`L` the atoms a position), each item's atom an odd multiple of its index over
the plane, so a launch touches an atom at most once and every form walks the same atoms. The ring's depth is the slots
a warp's share of the 227 KB holds, at most 8: at `d_v` 64 WRITE_ALL 4, READ_ALL 6 (7 under the producer), the ONE
patterns 8; at `d_v` 32 every pattern 8. The plane holds small integers (an s16 hi at exponent -8, a zero lo plane, a
mass of 1 to 8) and every weight is 1 or 2, so every sum is exact whatever its order: the check sums every thread's
accumulators by head and channel against the host's, and compares every leaf CTA 0 wrote, byte for byte, against the
host's codec. A row is GB/s the device: the bytes the pattern moves (its 32-byte sectors: a WRITE_ALL page in and out, a
READ_ALL page's hi plane and tail, a scattered leaf's hi line, lo line where written, and one sector each of exponents
and masses) over the launch's time by events.

## The floors and the prices they rest on

| price | the record | figure |
|---|---|---|
| P1 HBM read | warpline's H100 rates, its floor fixture of the H100 80GB HBM3 (the row `hbm.read`) | 12.415 bytes a cycle an SM at 1.963 GHz: 3,218 GB/s |
| P2 HBM copy | the same, `hbm` (the bytes read and written) | 11.097 at 1.979 GHz: 2,899 GB/s |
| P3 a scattered row | [`gather_rate`](gather_rate.md), a 2 GB table, 512 rows in flight a CTA | 9.71 cycles a 128-byte row a CTA: a scattered line at the stream's rate when 64 KB a CTA are in flight |
| P4 the copy engine's requests | `tma_box`'s record in the pinned data, `data:records/fsla/measure/lab/tma_box.run/20261128T034956Z-802475-98afad64/output.json` (a 48 KB stage by tensor copies, its rows `a tensor copy, 16-byte lines` and `a tensor copy, 128-byte lines swizzled`) | a 16-byte-line box 5.55x the 128-byte box (5,360 against 966 cycles a stage): about 0.57 requests a cycle an SM |
| P5 a dependent DRAM load | [`clock_time`](clock_time.md), four chained scattered loads of a 64 MB table | 2,069 ns: about 517 ns, 1,015 cycles at 1.963 GHz, unloaded |
| P6 issue | warpline's H100 rates, `issue` | 3.18 warp instructions a cycle an SM |

A row's FLOOR is its bytes at P1 (a read-only pattern) or P2 (a written one). HYPOTHESIS H-L, no record: under a
full card's load a scattered DRAM load's latency `L` is 1,000 to 2,000 cycles (P5 unloaded, up to twice it queued), so
a CTA holds its floor only with `12.4 L` = 12 to 25 KB in flight. REG's rows measure `L`: a REG warp holds one
position, so its rate is `16 x` the position's bytes over `L` plus its compute.

A position's instructions (counted from the source, G = 8, `d_v` 64; each class's share approximate): a leaf a lane
group read by every head, about 117 warp instructions a trip (8 conversions, a scale and 8 FMAs and a mass FMA a head, 8
table reads); written first, about 190 (the deposit's 90 and the reads'). Per SM: READ_ALL 4 trips a page, 2,144 bytes,
about 2.9 instructions a cycle at P1 (0.91 of P6); READ_ONE about 150 a position of 768 bytes, about 2.4 (0.76); BULK's
READ_ONE adds its first lane's 12 copies a position, about 100 instructions, 4.0 a cycle: over P6. WRITE_ALL about 1.4,
WRITE_ONE about 1.2. The producer's one warp issues a position's copies alone: about 100 instructions a READ_ONE position,
130 a WRITE_ONE one, 30 a WRITE_ALL page.

PREDICTION (written 2026-11-28, before the experiment's first run). A row's band is a fraction of its floor; the
prices each band rests on are named, so a miss stops only the rows that rest on the missed price.

| pattern | form | G 1, `d_v` 64 | G 8, `d_v` 64 | G 8, `d_v` 32 | rests on |
|---|---|---|---|---|---|
| WRITE_ALL | REG | 0.85-1.00 | not run | not run | P2, P5, H-L (16 x 3,168 bytes in flight is 50 KB) |
| WRITE_ALL | BULK, PIECES | 0.85-1.00 | 0.85-1.00 | 0.85-1.00 | P2, P3 (3 or 7 pages in flight a warp) |
| WRITE_ALL | PRODUCER | 0.85-1.00 | 0.85-1.00 | 0.85-1.00 | P2, P3, P6 (one copy a page) |
| READ_ALL | REG | 0.85-1.00 | not run | not run | P1, P5, H-L |
| READ_ALL | BULK, PIECES, PRODUCER | 0.85-1.00 | 0.70-0.95 | 0.70-0.95 | P1, P3, P6 (0.91 of the issue) |
| WRITE_ONE | REG | 0.75-1.00 | 0.75-1.00 | 0.75-1.00 | P2, P5, H-L (16 KB in flight, half the floor's bytes reads) |
| WRITE_ONE | BULK, PIECES | 0.80-1.00 | 0.80-1.00 | 0.80-1.00 | P2, P3, P4 |
| WRITE_ONE | PRODUCER | 0.35-0.80 | 0.35-0.80 | 0.35-0.80 | P2, P4, P6 (one warp's 16 copies a position) |
| READ_ONE | REG | 0.45-0.90 | 0.45-0.90 | 0.55-1.00 | P1, P5, H-L (12 KB in flight; 16 at `d_v` 32) |
| READ_ONE | BULK | 0.55-0.80 | 0.55-0.80 | 0.55-0.80 | P1, P4, P6 (its issue over P6) |
| READ_ONE | PIECES | 0.75-1.00 | 0.75-1.00 | 0.75-1.00 | P1, P3, P6 (0.84 of the issue) |
| READ_ONE | PRODUCER | 0.15-0.45 | 0.15-0.45 | 0.15-0.45 | P1, P4, P6 (one warp's 12 copies a position) |
| RECORD | REG | 0.45-0.90 | 0.45-0.90 | 0.55-1.00 | as READ_ONE's REG: its leaves a second 0.90-1.10 of READ_ONE's (latency, not bytes, binds) |
| RECORD | PIECES | 0.75-1.00 | 0.75-1.00 | 0.75-1.00 | as READ_ONE's PIECES: its leaves a second 1.10-1.30 of READ_ONE's (192 bytes against 160, at the sector's grain) |

NOT RUN, A FACT OF THE BUILD: REG holds a whole atom's pieces in registers, and beside G = 8 heads' 72 accumulators a
thread's 128 registers (512 threads on the 64K-register SM) do not hold them: ptxas spilled REG's WRITE_ALL and READ_ALL
at G = 8 at both widths (20 to 208 bytes of spill stores a thread, the experiment's build at its first commit), which the
lab refuses. The rings' trips are not unrolled, so no trip's shared loads are hoisted into another's registers, and they
hold G = 8 with no spill. Today's register-loaded walk cannot be widened to G = 8 heads a lane as it is.

THE SECTORS (`-- ncu --metric dram__bytes_read.sum --metric dram__bytes_write.sum`, a kernel's first launch): every
row's DRAM bytes within 10% of the bytes its pattern declares (its 32-byte sectors), so a scattered leaf moves 192 bytes
on the ruled page and 160 as a record at `d_v` 64 (128 and 96 at 32). Rests on: the L2 fetching a miss at the sector's
32 bytes. A row over by about a third would say the L2 fetches 64 bytes a miss (256 and 224), and every scattered
pattern's floor would then be priced on 64-byte grains.

Across the bands: the ring a warp by PIECES holds 0.75 of its floor or more in every pattern; REG falls under 0.9 on a
scattered read only; the producer falls on the scattered patterns, its one warp's issue the bound. G = 8 against G = 1
costs nothing but where the issue binds (READ_ALL, READ_ONE). The stamped rows' windows lie within 5% of their twins'.

WHAT IT WILL DECIDE: the walk's form (a free choice, made by measurement); whether a page laid out by leaf records
would pay on a scattered read (the page's format is the owner's: the row is his evidence); and `L`, the price every
decode prediction after it rests on.

## The second run: the separating rows, predicted

PREDICTION (written 2026-11-28, before the second run). The first run's timed rows were not stored: lab refused its
stamped PRODUCER rows, whose producer stamped its first `K` waits on a fresh empty barrier, a phase no arrival releases
(warpline's `wait_before_release`). The producer now stamps only the waits after a consumer's arrival (`n >= K`). The
second run's first try was refused too: the producer's one warp stamps every position, and READ_ONE's 3,072 positions
a CTA overflow its timeline row, so its launch's end is lost (`launch_edge_missing`). PRODUCER is no longer stamped: the
form is dropped by its unstamped rows, and the checker finds nothing on the stamped rows of the other forms. Every other row runs as it ran, with the bands above; the bands of the
ONE patterns and RECORD rest on the 32-byte sector, which the sectors' record below refutes, and are re-based on the
bytes measured there (a leaf's read bytes from the counters). Three questions the first run left, each a row:

| row | the question | prediction | if it misses |
|---|---|---|---|
| the profiler's counters on every row: `smsp__inst_executed.sum`, `smsp__issue_active.sum`, `sm__cycles_elapsed.avg` and the issue stalls (`smsp__average_warps_issue_stalled_{long_scoreboard,short_scoreboard,mio_throttle,wait,math_pipe_throttle,not_selected}_per_issue_active.ratio`) | does the issue (P6) bind the rings at G = 8 (READ_ALL 0.70x and WRITE_ALL 0.81x of G = 1)? | the executed instructions a position are the compiled loop's (the static ledger, `-- ledger --static`) within 10%: READ_ALL by PIECES 229 at G 1 (the steady trip's 85 and four leaf trips of 36) and 564 at G 8 (84 and four of 120); WRITE_ALL by PIECES about 773 and 1,115 (117 and four of 164; 127 and four of 247). At G 8 the SM issues about 2.0 instructions a cycle on READ_ALL and 1.4 on WRITE_ALL, at most 0.70 of P6: the issue does not bind, and the leading stall is a leaf trip's shared loads (`short_scoreboard`) | over 0.90 of P6, or instructions 1.5x the compiled count: P6 binds, and my count of a trip is the wrong price |
| WRITE_SECTORS, REG, `d_v` 64, G 1 and 8 | does a partial sector's store (a leaf's 2 exponent bytes and 4 mass bytes) cost a DRAM read at its write-back? | as WRITE_ONE by REG within 5% in bytes read (366 a leaf) and in rate: the tail's sectors are read by the leaf's own loads before its stores, so their merge reads nothing | its reads fall by 32 bytes a leaf or more and its rate rises 10% or more: a partial sector's write-back reads its sector, and the scattered write is priced by whole sectors |
| the scattered patterns with the L2's fetch granularity held to 32 bytes, REG and PIECES, `d_v` 64 G 1 and `d_v` 32 G 8 | is the 64-byte grain the device's fetch granularity (`cudaLimitMaxL2FetchGranularity`, the run prints the device's own)? | the reads fall to the sectors: READ_ONE 192 to 216 bytes a leaf at `d_v` 64 (from 272; the odd atoms' hi lines start 32 bytes into a block, a 3,168-byte page), RECORD 160 (from 192); at `d_v` 32 READ_ONE 128 to 160, RECORD 96. PIECES' READ_ONE rises 1.15-1.30x in leaves a second, REG's less (one position in flight, latency binds) | the reads stand at the 64-byte grain: the grain is the L2's own, not a limit a step sets |

MEASURED, THE SECTORS (sm_90a, Nebius preemptible VM bench-20261128-084912; 2026-11-28, the experiment at 3db2b445;
`-- ncu --metric dram__bytes_read.sum --metric dram__bytes_write.sum`, each kernel's median over its launches in one
repetition, a launch 132 CTAs). A row's DRAM bytes against the bytes its pattern declares (its 32-byte sectors); a
read-only row's written bytes, which it declares none of, in MB (the sums, and a stamped row's timeline):

| pattern | form | `d_v`, G | read, of declared | written, of declared | read a leaf, bytes (declared) |
|---|---|---|---:|---:|---:|
| WRITE_ALL | REG | 64, 1 | 1.01 | 0.95 | 200 (198) |
| WRITE_ALL | BULK | 64, 1 | 1.01 | 0.95 | 200 (198) |
| WRITE_ALL | PIECES | 64, 1 | 1.01 | 0.95 | 200 (198) |
| WRITE_ALL | PRODUCER | 64, 1 | 1.01 | 0.95 | 200 (198) |
| READ_ALL | REG | 64, 1 | 1.03 | (0 MB) | 138 (134) |
| READ_ALL | BULK | 64, 1 | 1.03 | (0 MB) | 138 (134) |
| READ_ALL | PIECES | 64, 1 | 1.03 | (0 MB) | 138 (134) |
| READ_ALL | PRODUCER | 64, 1 | 1.03 | (0 MB) | 138 (134) |
| WRITE_ONE | REG | 64, 1 | 1.43 | 0.96 | 366 (256) |
| WRITE_ONE | BULK | 64, 1 | 1.67 | 0.96 | 427 (256) |
| WRITE_ONE | PIECES | 64, 1 | 1.67 | 0.96 | 427 (256) |
| WRITE_ONE | PRODUCER | 64, 1 | 1.43 | 0.96 | 366 (256) |
| READ_ONE | REG | 64, 1 | 1.42 | (0 MB) | 272 (192) |
| READ_ONE | BULK | 64, 1 | 1.42 | (0 MB) | 272 (192) |
| READ_ONE | PIECES | 64, 1 | 1.42 | (0 MB) | 272 (192) |
| READ_ONE | PRODUCER | 64, 1 | 1.42 | (0 MB) | 272 (192) |
| RECORD | REG | 64, 1 | 1.20 | (0 MB) | 192 (160) |
| RECORD | PIECES | 64, 1 | 1.20 | (0 MB) | 192 (160) |
| WRITE_ALL | BULK | 64, 8 | 1.01 | 0.99 | 200 (198) |
| WRITE_ALL | PIECES | 64, 8 | 1.01 | 0.99 | 200 (198) |
| WRITE_ALL | PRODUCER | 64, 8 | 1.01 | 0.99 | 200 (198) |
| READ_ALL | BULK | 64, 8 | 1.03 | (1 MB) | 138 (134) |
| READ_ALL | PIECES | 64, 8 | 1.03 | (1 MB) | 138 (134) |
| READ_ALL | PRODUCER | 64, 8 | 1.04 | (5 MB) | 139 (134) |
| WRITE_ONE | REG | 64, 8 | 1.43 | 1.01 | 367 (256) |
| WRITE_ONE | BULK | 64, 8 | 1.68 | 1.00 | 429 (256) |
| WRITE_ONE | PIECES | 64, 8 | 1.67 | 1.01 | 428 (256) |
| WRITE_ONE | PRODUCER | 64, 8 | 1.44 | 1.01 | 369 (256) |
| READ_ONE | REG | 64, 8 | 1.42 | (3 MB) | 273 (192) |
| READ_ONE | BULK | 64, 8 | 1.42 | (1 MB) | 273 (192) |
| READ_ONE | PIECES | 64, 8 | 1.42 | (2 MB) | 273 (192) |
| READ_ONE | PRODUCER | 64, 8 | 1.43 | (5 MB) | 275 (192) |
| RECORD | REG | 64, 8 | 1.21 | (2 MB) | 193 (160) |
| RECORD | PIECES | 64, 8 | 1.20 | (1 MB) | 193 (160) |
| WRITE_ALL | BULK | 32, 8 | 1.05 | 1.00 | 106 (101) |
| WRITE_ALL | PIECES | 32, 8 | 1.05 | 1.00 | 106 (101) |
| WRITE_ALL | PRODUCER | 32, 8 | 1.04 | 1.01 | 105 (101) |
| READ_ALL | BULK | 32, 8 | 1.09 | (1 MB) | 75 (69) |
| READ_ALL | PIECES | 32, 8 | 1.09 | (1 MB) | 75 (69) |
| READ_ALL | PRODUCER | 32, 8 | 1.10 | (5 MB) | 76 (69) |
| WRITE_ONE | REG | 32, 8 | 1.86 | 1.18 | 298 (160) |
| WRITE_ONE | BULK | 32, 8 | 2.59 | 1.18 | 415 (160) |
| WRITE_ONE | PIECES | 32, 8 | 2.59 | 1.18 | 414 (160) |
| WRITE_ONE | PRODUCER | 32, 8 | 1.83 | 1.18 | 293 (160) |
| READ_ONE | REG | 32, 8 | 1.69 | (3 MB) | 216 (128) |
| READ_ONE | BULK | 32, 8 | 1.69 | (1 MB) | 216 (128) |
| READ_ONE | PIECES | 32, 8 | 1.69 | (2 MB) | 216 (128) |
| READ_ONE | PRODUCER | 32, 8 | 1.70 | (4 MB) | 217 (128) |
| RECORD | REG | 32, 8 | 1.34 | (2 MB) | 129 (96) |
| RECORD | PIECES | 32, 8 | 1.34 | (2 MB) | 128 (96) |
| WRITE_ALL | BULK | 64, 8, stamped | 1.01 | 1.01 | 201 (198) |
| WRITE_ALL | PIECES | 64, 8, stamped | 1.01 | 1.01 | 201 (198) |
| WRITE_ALL | PRODUCER | 64, 8, stamped | 1.01 | 1.02 | 201 (198) |
| READ_ALL | BULK | 64, 8, stamped | 1.03 | (11 MB) | 138 (134) |
| READ_ALL | PIECES | 64, 8, stamped | 1.03 | (12 MB) | 138 (134) |
| READ_ALL | PRODUCER | 64, 8, stamped | 1.03 | (19 MB) | 138 (134) |
| WRITE_ONE | REG | 64, 8, stamped | 1.43 | 1.04 | 367 (256) |
| WRITE_ONE | BULK | 64, 8, stamped | 1.68 | 1.05 | 431 (256) |
| WRITE_ONE | PIECES | 64, 8, stamped | 1.68 | 1.05 | 430 (256) |
| WRITE_ONE | PRODUCER | 64, 8, stamped | 1.44 | 1.06 | 367 (256) |
| READ_ONE | REG | 64, 8, stamped | 1.42 | (15 MB) | 273 (192) |
| READ_ONE | BULK | 64, 8, stamped | 1.42 | (17 MB) | 273 (192) |
| READ_ONE | PIECES | 64, 8, stamped | 1.42 | (18 MB) | 272 (192) |
| READ_ONE | PRODUCER | 64, 8, stamped | 1.42 | (26 MB) | 273 (192) |
| RECORD | REG | 64, 8, stamped | 1.20 | (14 MB) | 193 (160) |
| RECORD | PIECES | 64, 8, stamped | 1.20 | (17 MB) | 192 (160) |

WHAT IT SHOWS.
- THE WHOLE ATOMS (WRITE_ALL, READ_ALL), `d_v` 64: every row within 4.9% of its declared bytes, read and written (the widest WRITE_ALL by BULK, G 1): HELD.
- THE WHOLE ATOMS (WRITE_ALL, READ_ALL), `d_v` 32: every row within 10.4% of its declared bytes, read and written (the widest READ_ALL by PRODUCER, G 8): MISSED.
- READ_ONE, `d_v` 64: 272 bytes read a leaf against the 192 declared, 1.42x to 1.43x: MISSED. At the 64-byte block's grain a leaf reads 256, the measured 16 to 19 bytes over it.
- RECORD, `d_v` 64: 192 bytes read a leaf against the 160 declared, 1.20x to 1.21x: MISSED. At the 64-byte block's grain a leaf reads 192, the measured within 1 of it.
- READ_ONE, `d_v` 32: 216 bytes read a leaf against the 128 declared, 1.69x to 1.70x: MISSED. At the 64-byte block's grain a leaf reads 192, the measured 24 to 25 bytes over it.
- RECORD, `d_v` 32: 128 bytes read a leaf against the 96 declared, 1.34x: MISSED. At the 64-byte block's grain a leaf reads 128, the measured within 1 of it.
- WRITE_ONE reads more than it declares in every form, and the rings more than REG: 427 bytes a leaf by
  PIECES against REG's 366 at `d_v` 64, G 1 (414 against 298 at `d_v` 32, G 8); its written
  bytes are its declared ones within 6% at `d_v` 64.
- A stamped row reads within 1.0% of its twin's bytes.

THE PREDICTION'S PRICE NAMED. The L2 does not fetch a scattered miss at the sector's 32 bytes: the record's leaves read
the 64-byte blocks their records span, within 1.3 bytes, and the ruled page's scattered leaf reads its blocks and a residual
beside them. Every scattered pattern's floor is priced on 64-byte blocks from here; the timed rows' bands that rest on
the 32-byte grain (every ONE pattern and RECORD) are re-based on it with the next recorded run.

Record: `data:records/fsla/measure/lab/decode_ring.run/20261128T085609Z-468416-ebc1cbff/output.json`.
