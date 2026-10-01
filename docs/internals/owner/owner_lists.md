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

The whole CTA first: a thread a (word, row sixteenth) ORs its rows into a partial union and every half-warp counts one
row's live tokens (16 popcounts summed); one warp then ORs the partials into the write and read unions and takes their
ranks, `base[k]` = the live tokens before word `k`. A token's POOL SLOT is `base[k] + popc(union[k] & below(b))` for bit
`b` of word `k`: its rank among the window's live tokens of its side, so a side's pool holds each live token once.

Then each warp alone, for its fold lists (side 0) and its read digits (side 1): it ORs its rows into its own union,
compares the two step forms' costs over the exact counts, and writes the chosen form's entries — pool slots in token
order — into its 512-entry buffer: per list, each list's tokens one list after another (a list's offset is its
predecessors' counts), or merged, the warp union's tokens once. A dense side writes no entries: its form is merged and
its slot is the token itself. A per-list form whose lists hold more than 512 entries between them takes the merged form
(its union never exceeds the window).

The half-warps take rows and lists in pairs, and every loop that holds a shuffle has the same trip count on both
halves (the carve's row and list counts are even, asserted).

<a id="step-cost"></a>

## The step cost

Both forms are priced in issue slots by one model: a k-step of 32 tokens costs its coefficient operand
(`kCostOperandA`), its token operand (`kCostOperandB`) and its IMMAs (`kCostImmas`, eight at 16 cycles), and an entry
`kCostEntry`. Per list: `steps × (A + B + IMMAs) + entries`, `steps` = each list's ceil(count / 32) summed. Merged:
`ceil(union / 32) × (B + lists × (A + IMMAs)) + union entries` — more IMMAs (every list steps over the whole union) for
one token operand a step and one compaction. Per list therefore never issues more IMMAs; merged wins where the lists
overlap enough that their entries and token operands outweigh the extra IMMAs, a sparse-declared level that is dense in
fact. The weights are placeholders until the skeleton measures each forced form; they are the comparison the ruling
names (Blake, 2026-10-01: the step chosen per warp and window by a cost comparison over the exact counts), not a
threshold on a statistic.
