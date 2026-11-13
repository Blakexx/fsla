# `facts_lists`

Mirrors `measure/lab/experiments/facts_lists.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

A sparse level's liveness is a MASK the op is handed, a bit a digit a token, used blindly
([`ARCHITECTURE.md`](../ARCHITECTURE.md#62-the-facts-pass)). The mask says, for a token, which digits are live; a
carry CTA needs the opposite each window -- for each of its digits, WHICH TOKENS are live on it, the data it pulls.
What does turning one into the other cost a token on hopper, and what does the value test it replaces cost beside
it?

THE FORM. A CTA a window of 1,024 tokens, a thread a token, one side's sparse level of 256 digits. The pass is a
counting sort of the window's live (token, digit) pairs by digit, in token order:
- a thread ORs its lane's bit into a word a (warp, digit) for each of its live digits, a shared-memory atomic each;
- a digit's thread sums the 32 warps' counts before each warp, and one thread the digits' offsets;
- a thread places each of its live digits at its digit's offset, the count before its warp and the lanes before it
  in its warp's word, as an ENTRY: the token and its byte on the digit, the byte read from the level's tile-major
  row.
Two forms find a token's live digits: the mask's 32 bytes as handed, or a value test of the token's 256-byte row.
A call is the window's COUNTS alone -- what sizes its lists before they are written -- or the LISTS too. The cell:
4 live digits a token and a fifth for a token in three. Every count and every entry is checked against the host's.

MEASURED: not yet run.
