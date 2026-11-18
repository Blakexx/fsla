# `carry_token`

Mirrors `measure/lab/experiments/carry_token.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

The carry's token-stream form: a window's value lines land once as the stream holds them and are read into the fold's
steps by `ldmatrix`; the fold expands only its coefficients; the readout is mirrored beside it. What does a window
cost a CTA, which role waits on which, and what do the fold's fill and its steps cost apart? And what is the order of
a window's entries worth -- placed so the 8 tokens a lane group reads for a step are on 8 bank groups, against token
order?

THE FORM. A CTA an SM of four warpgroups, 512 threads, registers traded by role: two fold warpgroups at 176, the
readout at 96, four drain warps at 64 (at 168, 104 and 72 the toolkit serializes the fold's steps where four sets of A
are in flight, and no split tried holds eight).
- THE FOLD: a window is a box's 256 entries in 16 lists of 8 to 24. It lands by two bulk copies behind one barrier,
  issued under the last steps of the window before: a table (an entry's word and sparse byte, a lane's row of the 32
  steps' token positions, the lists' counts) and the window's value lines, an entry's 8 at its position, a brick a
  run of 256 -- so a line's bank group is its entry's position mod 8. A thread loads its two entries' dense lines
  from global memory and expands each by its sparse byte to the two lines of its step's coefficient tile; no value
  line is moved. A step's A is read by `ldmatrix`, transposed, into registers, a lane naming its token's line of its
  channel octet; its B is its tile, addressed by the step's index, with no descriptor loaded. A step is a commit
  group and `SETS` sets of A are held in turn, a step waiting only on the one `SETS` before it: 2, and rows at 4. The image is published a tile group at a time behind the readout.
- THE READOUT is mirrored: a (token, box) a thread, its 16 bytes expanded to two lines of its digit's coefficient tile
  in shared memory, a pass of 8 digits, 16 steps of 64 channels by 16 tokens one commit group, its sums stored to the
  panel as halves.
- THE DRAIN WARPS add each token's row as one f16 line to global memory.
The kernel is `window<TRACE, DETAIL>`. `TRACE` 0 is the plain kernel, 1 its launch's two edges stamped, 2 every warp's
phases stamped (the coarse row: the fold's window, its expanding and its steps), 3 every stamp; `DETAIL` names a
coarse row's one more group of waits and arrivals -- the tile groups read or published, the panels filled or drained.
Twelve rows: plain, edges, coarse, the four detail rows, every stamp; the plain and coarse kernels on a window whose
entries are in token order; and the plain and coarse kernels with 4 sets of A in flight. The state and every output row are checked exactly in every row. THE PLACES ARE THE
HOST'S: an entry's position, its tile's lines and the steps' positions are a table the window lands, where the design's
CTA counts them from its mask; the rows price the fill and the steps, not that counting.

MEASURED: NOT YET MEASURED.
