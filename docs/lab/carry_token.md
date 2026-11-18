# `carry_token`

Mirrors `measure/lab/experiments/carry_token.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

The carry's token-stream form -- a window's entries laid by the fold in one pass from one view in token order, its 32
steps one block, the readout mirrored beside it -- is one design among the 178 kernels of
[`carry_window`](carry_window.md), where it stands among forms since superseded behind 64 flag bits. What does a window
cost a CTA in that form alone, which role waits on which, and what do its own parts cost, in a file that holds that one
design and nothing else?

THE FORM. A CTA an SM of four warpgroups, 512 threads, registers traded by role: two fold warpgroups at 168, the
readout at 104, four drain warps at 72. It is `carry_window`'s row `kFold | kRead | kSuspend | kLean | kNoClock |
kWhole | kBlocks | kMirror | kChunk8 | kDrains4 | kToken | kThrough | kOne`, written as its own file with every
branch of a flag it does not use removed: the same instructions, held to it by the pack's registers and opcode counts
and by `cuobjdump -sass` of the plain, edges and coarse kernels of both files.
- THE FOLD: a window is a box's 256 entries in 16 lists of 8 to 24. Its table -- the entries' words, places and sparse
  bytes and the 32 steps' operands -- is one bulk copy issued a window ahead; each thread loads two entries' dense and
  value lines from global memory through its registers, expands the dense line to its two coefficient lines by its
  sparse byte, and stores the value lines to its list's run; then 32 steps, one block. The image is published a tile
  group at a time behind the readout.
- THE READOUT is mirrored: a (token, box) a thread, its 16 bytes expanded to two lines of its digit's coefficient tile
  in shared memory, a pass of 8 digits, 16 steps of 64 channels by 16 tokens one commit group, its sums stored to the
  panel as halves.
- THE DRAIN WARPS add each token's row as one f16 line to global memory.
The kernel is `window<TRACE, DETAIL>`. `TRACE` 0 is the plain kernel, 1 its launch's two edges stamped, 2 every warp's
phases stamped (the coarse row), 3 every stamp; `DETAIL` names a coarse row's one more group of waits and arrivals --
the tile groups read or published, the panels filled or drained. Eight rows: plain, edges, coarse, the four detail
rows and every stamp. The state and every output row are checked exactly in every row.

MEASURED: NOT YET MEASURED.
