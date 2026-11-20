# `fold_forms`

Mirrors `measure/lab/experiments/fold_forms.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

The fold's step takes its coefficients from a tile in shared memory, a step one sparse digit's 16 tokens: today's
form, its absent steps not issued and its short steps padded by the layout. The other place a step's coefficients can
be is the warpgroup's registers, a step 16 tokens by a group of four sparse digits, a warp a digit, the values the
step's one shared-memory operand. What does a window cost a CTA of two boxes in each?

THE FORM. A CTA an SM of two fold warpgroups, a box each of 16 lists of 12 to 24 entries (256 a window), 16
accumulator sets of 8 or 4 of 32 in registers. A box's two packets land once, before the clock, a bulk copy each
into its own buffer, and its windows take them in turn: no copy is in a window. Every accumulator is checked exactly. A warpgroup's trip is a window.
- THE TILE FORM: a thread expands its entries (a dense line and a sparse byte each, loaded from the packet's header)
  to two lines of the entry's step's tile; the fence, the warpgroup's barrier; a list's one or two steps in a rolled
  loop; the commit and the wait.
- THE TILE FORM, ITS DENSE LINES FETCHED: the same, the dense lines read from the level -- 65,536 tokens' rows as the
  bundle holds them, tile-major, 16 MB -- by the entry's token instead of from the packet.
- THE REGISTER FORM: a step's 16 positions are four CLASSES of four (a thread holds two dense rows of positions 2t,
  2t + 1, 8 + 2t, 9 + 2t), and the layout gives a digit's tokens classes of their own. A position's dense line and a
  step's cells (a thread's four positions' sparse bytes for its warp's digit, the classes its warp has) land with
  the values; a thread reads its cell and its positions' two dense bytes a step ahead, multiplies, converts and packs
  four words, and the step is issued. A commit group is four steps, each its own words, and the wait on it opens the
  next. EVERY CLASS COMPUTED, or ONLY THE CLASSES A WARP HAS.
- LIVE: 128 accumulators a thread; the tile form three entries' lines and bytes; the register form a stage's cell and
  four dense pairs.

MEASURED (NVIDIA H100 80GB HBM3, Nebius preemptible VM r1xrwyyv, driver 580.196.07; 2026-11-20, the experiment at
5979607b; 7 repetitions; the 4 checked rows pass, every accumulator exact; the median, the widest
spread of a row 1.1%). Cycles a window a CTA:

| form | cycles a window a CTA | against the tile form |
|---|---:|---:|
| the tile form: the coefficients a tile in shared memory, a step a list | 2,921 | 1.00 |
| the register form, every class computed: a step four digits | 4,100 | 1.40 |
| the register form, a warp's own classes computed | 5,253 | 1.80 |
| the tile form, its dense lines fetched from the level by token | 3,019 | 1.03 |

WHAT IT SHOWS.
- THE REGISTER FORM, EVERY CLASS COMPUTED: 4,100 cycles a window a CTA, 1.40 of the tile form's 2,921, 1,179 cycles above it.
- THE REGISTER FORM, A WARP'S OWN CLASSES COMPUTED: 5,253 cycles a window a CTA, 1.80 of the tile form's 2,921, 2,332 cycles above it.
- A WARP'S OWN CLASSES AGAINST EVERY CLASS: 1,153 cycles more a window a CTA,
  1.28 of the every-class form's 4,100.
- THE TILE FORM, ITS DENSE LINES FETCHED FROM THE LEVEL: 3,019 cycles a window a CTA, 1.03 of the tile form's 2,921, 98 cycles above it.
- THE CHEAPEST FORM is the tile form: 2,921 cycles a window a CTA, 1.00 of the tile form's.

Record: `measure/lab/results/fold_forms/sm_90a.json`.
