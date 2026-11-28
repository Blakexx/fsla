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
`d_v` 64 and 96 at 32: the layout a scattered read would want, the design's decision 6, measured and not adopted). Each
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
| P4 the copy engine's requests | `tma_box`, the producer's experiment, its run 20261128T034956Z-802475-98afad64 (recorded, not yet in this tree's pinned data) | a 16-byte-line box 5.55x the 128-byte box: about 0.57 requests a cycle an SM |
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

WHAT IT WILL DECIDE: the walk's form (the design's decision 5, a free choice by measurement); whether a page laid out by
leaf records would pay on a scattered read (decision 6, the owner's: the row is its evidence); and `L`, the price every
decode prediction after it rests on.
