# `csrc/rola/src/owner/owner_lists.cuh` — the window's words and lists

Mirror doc for the owner carry's first two components: one stages the box's rows of the facts pass's words a window
ahead, and the other turns them into what every later component reads — the unions, a live token's pool slot, each row's
live count, each warp's step form and its list entries. Vocabulary: [`owner.md`](owner.md).

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

Last, every warp fills its share of the slot-to-token table (four union words, a lane a bit) and writes its entries for
its chosen form — pool slots in token order, per list each row's tokens from its offset, merged the union's once —
into its 512-entry buffer, a lane walking the set bits of each of its words. A per-list form whose lists hold more than
512 entries between them takes the merged form.

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
