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
- a digit's thread sums the 32 warps' counts before each warp, and one thread the digits' offsets;
- a thread places each of its live digits at its digit's offset, the count before its warp and the lanes before it
  in its warp's word, as an ENTRY: the token and its byte on the digit, the byte read from the level's tile-major
  row.
Two forms find a token's live digits: the mask's 32 bytes as handed, or a value test of the token's 256-byte row.
A call is the window's COUNTS alone -- what sizes its lists before they are written -- or the LISTS too. The cell:
4 live digits a token and a fifth for a token in three. Every count and every entry is checked against the host's.

MEASURED (H100 80GB HBM3, 132 SMs, vast.ai instance 60552186, driver 580.173.02; 2026-11-13, the experiment at
5d489776; 7 repetitions, 132 windows; every row's 33,792 counts and the lists' 585,728 entries exact; the median,
SM-cycles a token for one side's sparse level):

| a token's live digits from | a window's counts | a window's lists |
|---|---|---|
| the mask as handed | 4.8 | 23.2 |
| a value test of its row | 21.0 | 49.7 |

WHAT IT SHOWS.
- WITH EVERY TABLE IN CACHE -- the upside, not the price (R22.2): each timed call re-reads the same windows, and the
  pass over masks and rows the cache does not hold is unmeasured -- FROM A HANDED MASK THE PASS IS 23 SM-CYCLES A TOKEN a sparse level-side, lists and all, and 5 for the counts
  alone; a value test of the rows is 50 and 21. The mask halves the pass and takes the count to a quarter.
- THE LISTS COST 18 cycles a token past the counts: the digits' offsets (one thread's 256 sums while the CTA waits), a live digit's byte
  loaded from its row unstaged, and its entry's store. Which of them it is is NOT ISOLATED; an entry without its byte
  is not measured.
- BESIDE THE CARRY's window ([`carry_window`](carry_window.md)) the pass is a few hundredths a sparse level-side.
  ONE CELL: 4 to 5 live digits a token of 256, spread evenly; a denser or a clustered level is unmeasured. The value
  test reads token-major rows, which the hopper bundle does not hold.

Record: `measure/lab/results/facts_lists/sm_90a.json`.
