# `gather_group`

Mirrors `measure/lab/experiments/gather_group.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

The gather of a sparse block's window in the grouped digit stream's plainest form -- every list a run of its own,
values only -- what does a block's window cost a CTA, and a token a side? [`gather_rate`](gather_rate.md) priced the
rows' access pattern alone; this is the gather's whole trip, its lists and barriers included.

THE FORM. A CTA an SM of 16 warps, a warp a digit of the block, a trip one (block, window) of 1,024 tokens, each token
live on 4 of the 256 digits, the block's 16 kept, the mask digit-major.
- THE LISTS: a lane loads 32 tokens' bits of its warp's digit, counts them, and the warp sums its lanes' counts
  before each and its digit's whole; behind the CTA's barrier a warp knows the lists before its own, and a lane writes
  each of its live tokens' indices at its place in the packet.
- THE VALUES: behind the barrier thread `e` takes entry `e`: its token's eight value lines from the token-major
  table (8 MB) to the packet's eight planes -- line `c` of entry `e` at `c (R + 1) + e`, the operand a fold step reads
  in place -- and its sparse byte on its list's digit; an entry past the lists' end is the zero token's.
- THE STORE: the packet (a head: R, the entries, a list's count; the entries' tokens; their sparse bytes; the
  planes) to the CTA's ring by one bulk copy, a packet's copy waited two trips on, three packets of shared memory.
- THE PIPELINE: a trip issues its packet's 8 loads and a byte, and under them plans the NEXT packet (the mask's word
  loaded a trip ahead, 5 shuffles, the barrier, a lane's live tokens' stores, the barrier), then stores this packet,
  fences, barriers, and thread 0 copies, commits and waits on the copy before.
Every packet is checked byte for byte against the host's.
Rows: a thread an entry's eight loads; the loads and stores with 8 lanes a row; that form with each thread 0's phases
clocked (the lists and their barrier, the entries' tokens and their barrier, the values landing, the values' stores
and their barrier, the packet's store); and two PROBES of the 8-lanes form, unchecked: no packet stored, and no entry's
token written. A form's window is
also stated a token a side (SM-cycles a token) and an entry (cycles an entry a CTA).
- LIVE: 8 lines a thread through the plan.

WHAT IT WILL DECIDE: the plain form's cost a token a side against the carry's, so whether the gather is a small enough
share of the carry that its plain form is good enough to compose; and, where it is not, which phase of the trip holds
the cycles before anything else is changed.

MEASURED (NVIDIA H100 80GB HBM3, Nebius preemptible VM rgfp2qac, driver 580.196.07; 2026-11-20, the experiment at
7d0d6925; 7 repetitions; the 3 checked rows pass, every packet exact byte for byte; the median, the widest
spread of a row 0.4%). Cycles a block's window a CTA:

| form | cycles a block's window a CTA | SM-cycles a token a side | cycles an entry a CTA |
|---|---:|---:|---:|
| a thread an entry's eight loads | 5,532 | 86.4 | 21.6 |
| loads and stores, 8 lanes a row | 3,045 | 47.6 | 11.9 |
| the same, its phases clocked | 3,355 | 52.4 | 13.1 |

WHAT IT SHOWS.
- THE 8-LANES FORM AGAINST A THREAD AN ENTRY'S EIGHT LOADS: 3,045 cycles a block's window a CTA, 2,488 cycles below the 5,532 of the thread-an-entry form.
- PROBE, NO PACKET STORED: 2,305 cycles a block's window a CTA, 739 cycles below the 8-lanes form's 3,045.
- PROBE, NO ENTRY'S TOKEN WRITTEN: 2,923 cycles a block's window a CTA, 122 cycles below the 8-lanes form's 3,045.

THREAD 0'S TRIP BY PHASE (the clocked form; thread 0 alone stores the packet):
- THREAD 0, THE LISTS AND THEIR BARRIER: 1,706 cycles a trip, 51% of thread 0's 3,340.
- THREAD 0, THE ENTRIES' TOKENS AND THEIR BARRIER: 651 cycles a trip, 19% of thread 0's 3,340.
- THREAD 0, THE VALUES LANDING: 104 cycles a trip, 3% of thread 0's 3,340.
- THREAD 0, THE VALUES' STORES AND THEIR BARRIER: 551 cycles a trip, 16% of thread 0's 3,340.
- THREAD 0, THE PACKET'S STORE: 329 cycles a trip, 10% of thread 0's 3,340.

Record: `data:records/fsla/measure/lab/gather_group.run/20261120T235125Z-9e8e6e00/output.json`.
