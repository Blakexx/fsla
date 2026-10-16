# `carry/schedule.cuh` — the window's plan

Mirrors `csrc/rola/src/carry/schedule.cuh`: each window's PLAN, made by the plan pass before the carry runs from the
facts pass's counts -- a sequence of ITEMS, each a phase over a chunk, each buffer's place in the AREA (the shared memory
past the tenants, `plan.md#plan`) and each fill's host. The carry runs a plan item by item (`carry_kernel.md#kernel`):
shared memory holds what is active now and what lands next, sized by the window's actual counts, and a bigger card's
area takes larger chunks with no other change (ruled 2026-10-14). The plan's Python model is
`tests/oracle/carry_plan.py`, which a gate holds the plan pass to word for word.

Symbols: `U` a side's union slots in the window, `C` a chunk's slots (a multiple of 64 below the union, or the whole
union), `E` entries, `B` the area's bytes.

<a id="items"></a>
## The items

`ItemKind`: MAXIMA (each warp's digit maxima posted and joined), STORE (the next window's inputs staged; the image
stored whole in the whole mode), READ (a read chunk's tiles over every digit), DRAIN (a read chunk's panel to `num`),
FOLD (a fold chunk's steps into the state), BOUNDARY (the window's end; the units step where the window folds nothing).
A window's order: MAXIMA, STORE, each read chunk's READ followed by the DRAIN of the chunk before it -- PIPELINED, so
chunk `t + 1` reads while `t`'s panel waits -- or by its own DRAIN (SERIAL, one buffer reused), the last DRAIN, each
FOLD, the BOUNDARY; LATE puts the last DRAIN after the first FOLD.

An item (`Item`, 8 bytes): its kind; its flags (`ItemFlag`: `kFlagFilled` its buffer filled by a fill, `kFlagUnits`
the window's units step at it);
its chunk (its first slot, a multiple of 64, and its slots); its DEPENDENCE `dep` (READ on the whole mode's STORE,
DRAIN on its READ); its buffer (bytes / 16 into the area); and up to two FILLS it hosts, each a target item and the
item whose done it waits for, and their count (so an item hosting none skips the fills' loop). `kNoItem` names none; `kNextItem` names the next window's first read, its item
`kFirstRead`. A window's HEAD (`Head`): its items, its image mode, each side's chunk and union, its order.
`kPlanItems` = 27 (the maxima, the store, a read and a drain for each of 8 chunks, 8 folds, the boundary);
`kPlanBytes` = 240, a head and 28 item slots, written whole.

<a id="buffers"></a>
## The buffers

A chunk's buffer is laid out at its OWN slots (`parts.cuh`'s `read_pads`, `read_ents`, `fold_ents`): a read buffer's
rows (`C x row`), the dummy panel rows (`kPadBytes`), the panel (`C x 64`, 128-byte aligned: a slot pair shares a
128-byte line and a slot's row may sit in its line's upper half, so the whole line is zeroed), its entries (`E x 8`);
a fold buffer's rows (slices then values) and entries. `read_bytes` and `fold_bytes` are their sizes, so a chunk's
footprint is its actual count, never the window's largest chunk. The IMAGE (`ImageMode`) is whole (32 planes, 64 KB), a
ring of two 8-plane slots, or one 16-plane half (each 32 KB). An entry's row and panel offsets are the union's, so a
chunk from a multiple of 64 takes its pool and panel back by its first slot (`carry.md#lists`).

<a id="totals"></a>
## The segment sums

`side_counts`: a side's union slots and `S[j]`, its box lists' entries before union slot `64 (j + 1)` -- the facts
pass's prefix rows' totals row, one 16-byte load (`facts.md#lists-out`). Any chunk's entries are a difference of two of
them (`entries_of`); `chunk_max` is the most any chunk of `64 m` slots holds.

<a id="chunks"></a>
## The chunks

`largest_chunk`: the whole union where one buffer of it fits; else the largest `64 m` slots below the union two of
whose largest buffers fit (chunk `t + 1` lands while `t` computes); else the largest one of which fits. A read chunk
(`read_chunk_of`) is sized beside the image's 32 KB from the window's own counts, so the window before knows the first
read buffer it fills (`first_read_bytes`); a fold chunk is sized in the whole area.

<a id="planner"></a>
## The planner

THE RULE (`rule_form`): the image whole where it fits beside the read buffers in flight (64 KB and one, or with read
chunks two, of the largest read buffer), else in 16-digit halves; the fold's largest chunk; the read chunks pipelined,
and serial where their buffers do not place pipelined. `build` makes a window's plan in a form:

- the items in order (`#items`);
- the long-lived regions: the image to its last READ, the first read buffer at the area's top (filled by the window
  before) to its drain;
- each later buffer placed (`place`): of the area's ends and the regions' edges, the place whose overlapping regions'
  last user is earliest -- none in use by the buffer's consumer -- the lowest among equals; so chunk `t + 2` takes the
  place `t` frees when nothing frees earlier;
- each fill hosted (`host`) at the first item past its region's last user, never before the store (the maxima retire
  the window before), at most two an item, that user flagged done;
- THE BOUNDARY: the next window's first read buffer at the top, its fill hosted once its region's users here are done
  and past the store: a host's fills run before its work, and the store's work stages the next window's inputs the
  fill reads (its runs and union). A window that reads nothing has no first-read region holding it back.

Why a rule, not a search (measured 2026-10-14, nl64k-alt-k4 on sm_86, every named form timed warm): the rule's form is
the fastest (3.22-3.26 ms a call with the plan pass). Each extra fold chunk costs ~500 us there -- it splits every
list's run into two partial steps, and the fold's steps are latency-bound list by list -- and the 8-digit ring ~180 us
over the halves. A search over the named forms by a per-item cost model ranked them inverted on unmeasured prices (its
pick 5.2 ms) and cost 661 us a call.

THE NAMED FORMS (`named_form`): `kCandidates` = 60, an image mode by a fold chunk (the union, 256, 192, 128, 64 slots)
by an order (pipelined or serial, the last drain late or not). `CarryParams.prefer` names one: reserved, for the gates
and the measurements (`rola.ops.carry.CANDIDATES`); a window takes it where it builds and is distinct (a fold chunk
below the union), else the rule's.

Worked, on the even box (64-byte read rows, 128-byte write rows; the area 90,624 bytes on sm_86, 156,160 on sm_80 and
sm_87): a nl64k-alt-k4 window of ~216-slot unions and ~260 entries a side takes the halves, one read buffer (30.3 KB at
the top) and the whole union's fold buffer (29.8 KB), which misses fitting beside the two by ~2.6 KB: it goes over the
image, its fill hosted at the drain after the reads, and the next window's first read buffer is filled at the fold. On
sm_80 the image is whole and the fold buffer lands beside the readout.

<a id="pass"></a>
## The plan pass

`window_plan_kernel`, launched before the carry on its stream: a thread a (batch-head, box class, window), 64 a block
(`kPlanThreads`), its plan built in shared memory (`Sim`). Its counts (`window_counts`; a BOX CLASS is the pair of the
read and write list levels' groups, which fixes a CTA's counts, `box_classes`) and the next window's first read buffer
(`first_read_bytes`); the named form where it builds, else the rule's, its reads pipelined, else serial; the whole
record to `CarryParams.plans` (`[BH][windows][classes]`, `kPlanBytes` each). 9.8 us at nl64k-alt-k4 (8,192 plans).
Gated word for word against the model on every form cell under a spread of named forms
(`tests/oracle/test_carry_plan.py`).

<a id="plan"></a>
## The plan in the CTA

The CTA stages a window's record with its inputs, by parity (`carry_kernel.md#staging`), and reads its head and item
`k` at each use (`plan_head`, `plan_item`), so no field holds a register across items.

Per architecture: the same rule everywhere, its area the card's (`hardware.json`); a larger area takes whole images and
larger chunks; on sm_90 the fills become TMA copies on the same landed barriers. Scaling: the pass a thread a
(batch-head, box class, window), O(1) each; nothing O(L) or O(N) in the carry.
