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
and their barrier, the packet's store); and four PROBES of the 8-lanes form, unchecked: every entry the zero line (no
table line read), no packet stored, the zero line and no packet stored, no entry's token written. A form's window is
also stated a token a side (SM-cycles a token) and an entry (cycles an entry a CTA).
- LIVE: 8 lines a thread through the plan.

WHAT IT WILL DECIDE: the plain form's cost a token a side against the carry's, so whether the gather is a small enough
share of the carry that its plain form is good enough to compose; and, where it is not, which phase of the trip holds
the cycles before anything else is changed.

MEASURED: no record is taken at this commit (the experiment's results file for sm_90a, which this commit does not have does not exist). This
section is generated from one by `bazel run //tools:lab -- page gather_group --instance LABEL`.
