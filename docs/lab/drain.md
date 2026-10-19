# `drain`

Mirrors `measure/lab/experiments/drain.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

What does the readout's drain -- a pass's panel words added into the head's `num` by `red.global.add` at each slot's
token, the panel cleared for the next pass's adds -- cost a CTA-window, in today's form and in forms that pair
conflict-free slots, clear by exchange or by a 16-byte sweep, skip the requests past the union, and beside a stream's
copies?

80 CTAs, box b of one head (the carry's order); the inputs (`drain.inputs.py`) are `read_stream`'s three cells, each
box-window's union in rank order (the facts pass's rank slots). Each pass every thread fills its slot's 16 panel words
(four 16-byte stores) and its token, a CTA barrier, the drain, a barrier. A warp drains its 32 slots two a
`red.global` (lanes 0-15 one slot's 16 words, 16-31 another's), each a token's half line of `num` (d_v 64: a token's
128-byte row, two channel tiles), so the 8 slice groups' boxes and every read group holding the token add into the
same words. Every draining form's `num` is checked word for word.

The forms: today's (slots (s, s + 1) at the panel's 20-word pitch: 4 banks shared, 2 wavefronts an access; a load and
a 4-byte clearing store); slots (s, s + 4) (16 banks apart: 1 wavefront); an `atom.shared.exch` reading and clearing
in one; no clear in the drain and a 16-byte sweep after its barrier; the sweep with a warp's groups of 4 requests past
the union not issued (a warp-uniform count); copies of 512 bytes a warp beside it every 4 adds or every add; bounds:
no clear, the fill alone.

sm_86 (RTX 3080 Ti), SM clock locked at 1.665 GHz, the drain's cycles a CTA-window (the form's less the fill's 587 /
955 / 947), every CTA draining at once:

| form | alt-k4 | alt-k16 | cohort-k16 |
|---|---|---|---|
| today's: slots (s, s + 1), load and 4-byte clear | 1,641 | 3,309 | 3,402 |
| slots (s, s + 4), load and 4-byte clear | 1,635 | 3,297 | 3,400 |
| slots (s, s + 4), exchange | 1,489 | 3,017 | 3,200 |
| slots (s, s + 4), load; 16-byte clear sweep | 1,371 | 2,816 | 3,150 |
| bound: no clear | 1,432 | 2,968 | 3,162 |
| the sweep, groups past the union skipped | 1,099 | 2,542 | 2,460 |
| (s, s + 4) beside copies, 512 bytes every 4 adds | 2,176 | 4,375 | 4,424 |
| (s, s + 4) beside copies, 512 bytes every add | 4,247 | 8,372 | 8,189 |

Requests a CTA-window (ncu, `lts__t_requests_srcunit_tex_op_red`): 256 / 512 / 509 issuing every slot of each pass;
215 / 459 / 458 skipping the groups past the union (alt-k4's union 212 slots). L2's atomic unit runs ~36-41% of the
launch's elapsed cycles (about half the drain's); global-queue stalls ~0.

What it settles:

- THE DRAIN'S FLOOR IS THE SM'S GLOBAL REQUEST PATH: ~5 SM cycles a line request with every SM draining (the skipping
  form at alt-k4: 215 requests in ~1,100 cycles) -- the per-SM rate on record (2026-09-27), L2's atomic unit about
  half busy. The requests are the union's slots: one half line a slot at d_v 64. The drain is no shared-pipe or issue
  question on sm_86: today's slot pairing and the conflict-free one run alike.
- SKIP WHAT THE UNION DOES NOT HOLD: a pass's slots past the union issued as zero adds cost 16% more requests and 20%
  more time at alt-k4 (28% at cohort-k16, whose second pass is mostly empty). Never point dead lanes at one shared
  dummy row: the first form of this lab did, and every CTA's dead adds serialized on that one line -- the drain read
  11,300 cycles a window against 1,850 with a dummy row per CTA.
- A PREDICATED GLOBAL ADD IS A BRANCH: ptxas puts `@p red.global` in a convergence region (BSSY, BRA, BSYNC around it,
  ~4 instructions an add); the production drain's `red_nonzero` is that form. A dead lane's add is a select of a zero
  value and its own row, or the group is not issued.
- THE CLEAR: the exchange saves 9% of today's drain, the 16-byte sweep after the barrier 16% (below the no-clear bound:
  the sweep's stores leave the drain's loop shorter). On sm_80, where the shared pipe binds the readout, the
  conflict-free pairing also halves the drain's panel wavefronts.
- THE DRAIN AND THE STREAMS SHARE THE GLOBAL PATH: 512 bytes copied a warp every 4 adds cost the same form 33% more,
  every add 2.6x -- ~4-5 cycles a copied 128-byte line beside the adds. The streamed carry's copies and its drain are one
  per-SM budget (with L2's ~18 bytes a cycle an SM for copies alone, `smem_rates`), which `tools/pipes.py` does not
  price yet.

Records: `measure/lab/results/drain/sm_86.json`, `sm_86.counters.json`.
