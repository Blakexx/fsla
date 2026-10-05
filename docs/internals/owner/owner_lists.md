# `csrc/rola/src/owner/owner_lists.cuh` — the window's words and lists

Mirror doc for the owner carry's first two components: one stages the box's rows of the facts pass's words a window
ahead, and the other turns them into what every later component reads — the unions, a live token's pool slot, each row's
live count, each warp's union words, and on the read side each warp's step form and its list entries. Vocabulary:
[`owner.md`](owner.md).

<a id="words"></a>

## The words

The box's `RD` write rows (level R's write digits) then its `RS` read rows (level S's read digits), a window's 16 words
a row (64 bytes), copied by `cp.async` a 16-byte chunk a thread into one of two buffers, so the next window's rows land
while this window is worked. A dense side's rows are not read: the facts pass's words for a dense level are its width
mask, and the lists never need them. At `(16, 64)` that is 80 rows, 5 KB a buffer.

<a id="lists"></a>

## The lists

Each warp first works alone on its own rows: its fold lists (side 0) and its read digits (side 1). A dense side's
counts are the window and its union every token, and it writes no entries (its slot is the token).

<a id="lanes"></a>

A side's rows sit in the warp's lanes, a lane `kWords` consecutive words of one row (`SideLanes`: 8 rows of 4 words
at `(16, 64)`'s fold side, 2 rows of 1 at its read side; 4 of 2 either side at `(32, 32)`). A few shuffles then give
every lane its row's live count, the row's live tokens before its words, the row's first entry (the rows before it), the
warp union of its word group, the union's tokens before them, the lists' total and their k-steps. From those the warp
picks its read side's form ([below](#step-cost)) — the fold side's is merged ([`owner_fold.md`](owner_fold.md#forms))
— writes its rows' counts and its union's words, and waits.

One warp then ORs the eight warps' unions into the CTA's write and read unions (and the write side's sector unions,
four warps' rows a sector at `(16, 64)`, all eight at `(32, 32)`) and takes their ranks, `base[k]` = the live tokens
before word `k`. A token's POOL SLOT is `base[k] + popc(union[k] & below(b))` for bit `b` of word `k`: its rank among
the window's live tokens of its side.

Last, every warp fills its share of the slot-to-token table (four union words, a lane a bit) and writes its READ
entries for its chosen form — pool slots in token order, per digit each row's tokens from its offset, merged the
union's once — into its 512-entry buffer, a lane walking the set bits of each of its words. A per-digit form whose
digits hold more than 512 entries between them takes the merged form. The fold writes no entries: it builds each pass's
rows from the slot-to-token table and its warp's union words ([`owner_fold.md`](owner_fold.md#rows)), which leaves 8 KB
of shared memory to the write slots.

<a id="write-lists"></a>

## The fold's write lists, a window ahead

The fold part keeps its own write-side lists (`WriteLists`), built a window ahead with no CTA barrier, so its passes run
across windows ([`owner_fold.md`](owner_fold.md#windows)); the read side's lists are the readout's. Every stage is an
mbarrier every thread arrives on, and every buffer is a window's (three for the words, two by parity for the rest):

1. `stage_write_words`: the box's 64 write rows of window W + 2, a 16-byte chunk a thread, at window W's start
   (`words_landed`).
2. `write_partial`: at window W's start each warp ORs its eight rows of window W + 1, a lane a word of four rows and one
   shuffle (`partials`).
3. `write_table`: in its first pass's middle (or at W's end when W has no pass), once every warp is past its last reads
   of the buffer, each warp forms the CTA union, its ranks and the sector unions in registers from the eight warps'
   words (a lane a word), warp 0 stores them and the live count, and each warp stores its two words' share of the
   slot-to-token table (`table`); a fill or a pass of W + 1 waits it.

A dense write side has none of it: its table is the identity and its live count the window, set once. The words' and
union words' reuse needs no wait of its own: a warp at window W has waited window W - 1's table, which every warp wrote
after its last reads of what W's staging overwrites. The table's does: it is written after the drain of the last pass
before, so no warp still reads the buffer. Shared memory: 15.9 KB at `(16, 64)` (12 KB of it the words).

<a id="measured"></a>

Measured (`flagship-alt-k4-L16384`, 4 heads): 5.4K warp-instructions a CTA-window at `(16, 64)`, 5.1K at `(32, 32)`,
0.17 ms a head at 64K; the first form (a half-warp a row, shuffles a row, a CTA pass for counts and unions) issued 6.8K.
Most of what is left is the entries (a lane walks its words' bits one at a time, and at alt-k4 a word holds about half a
live bit, so lanes idle) and the slot-to-token table. The card's ~0.76K line assumed eight lane-instructions an entry
at full lane use, which words this sparse do not give; one loop over a lane's four words together cost more (its
selection outweighed the iterations it saved).

<a id="step-cost"></a>

## The step cost

The read side's two forms are priced in issue slots by one model: a k-step of 32 tokens costs its coefficient operand
(`kCostOperandA`), its token operand (`kCostOperandB`) and its IMMAs (`kCostImmas`, eight at 16 cycles), and an entry
`kCostEntry`. Per list: `steps × (A + B + IMMAs) + entries`, `steps` = each list's ceil(count / 32) summed. Merged:
`ceil(union / 32) × (B + lists × (A + IMMAs)) + union entries` — more IMMAs (every list steps over the whole union) for
one token operand a step and one compaction. Per list therefore never issues more IMMAs; merged wins where the lists
overlap enough that their entries and token operands outweigh the extra IMMAs, a sparse-declared level that is dense in
fact. The weights are placeholders until the skeleton measures each forced form; they are the comparison the ruling
names (Blake, 2026-10-01: the step chosen per warp and window by a cost comparison over the exact counts), not a
threshold on a statistic.
