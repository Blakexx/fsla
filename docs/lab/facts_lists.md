# `facts_lists`

Mirrors `measure/lab/experiments/facts_lists.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

A sparse level's liveness is RULED to be a mask the op is handed, a bit a digit a token, used blindly; the pass
this tree builds still tests the rows' values ([`ARCHITECTURE.md`](../ARCHITECTURE.md#62-the-facts-pass)). This
lab prices the ruled form beside the built one's access. The mask says, for a token, which digits are live; a
carry CTA needs the opposite each window -- for each of its digits, WHICH TOKENS are live on it, the data it pulls.
What does turning one into the other cost a token on hopper, and what does the value test it replaces cost beside
it?

THE FORM. A CTA a window of 1,024 tokens, a thread a token, one side's sparse level of 256 digits. The pass is a
counting sort of the window's live (token, digit) pairs by digit, in token order:
- a thread ORs its lane's bit into a word a (warp, digit) for each of its live digits, a shared-memory atomic each;
- a digit's thread sums the 32 warps' counts before each warp; the digits' offsets are a scan by the 8 warps of
  digits and their totals -- no thread works while the CTA waits;
- a thread stores each of its live digits' ENTRY at its digit's offset, the count before its warp and the lanes
  before it in its warp's word. AN ENTRY IS THE TOKEN, with no byte (KERNEL_STANDARDS 29, item 5): the pass emits
  the mask turned around and reads no row.
Two forms find a token's live digits: the mask's 32 bytes as handed, or a value test of the token's 256 bytes read
from the tile-major rows the bundle holds. A call is the window's COUNTS alone -- what sizes its lists before they
are written -- or the LISTS too. Each is timed with every table in cache (a call's 16 rounds over the same
windows) and ONCE-READ (a round a fresh window: 2.2 million tokens, 540 MB of rows), at a cell of 4 to 5 live
digits a token and at one of 32. Every count and entry of the first round is checked against the host's.

MEASURED: not yet run.
