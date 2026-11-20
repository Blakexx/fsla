# `fold_forms`

Mirrors `measure/lab/experiments/fold_forms.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

The fold's step takes its coefficients from a tile in shared memory, a step one sparse digit's 16 tokens: today's
form, its absent steps not issued and its short steps padded by the layout. The other place a step's coefficients can
be is the warpgroup's registers, a step 16 tokens by a group of four sparse digits, a warp a digit, the values the
step's one shared-memory operand. What does a window cost a CTA of two boxes in each?

THE FORM. A CTA an SM of two fold warpgroups, a box each of 16 lists of 12 to 24 entries (256 a window), 16
accumulator sets of 8 or 4 of 32 in registers. A window's value lines land by one bulk copy a box into one of two
buffers, the copy for the window after next issued behind the wait on this one's steps; the packets land before the
clock, and every accumulator is checked exactly. A warpgroup's trip is a window.
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
