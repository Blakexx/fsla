---
name: rola-kernel-map
description: Where each carry-kernel mechanism lives in csrc/rola/src/carry — box.cuh's (k,m) law, ModeOrder/StreamSpans, carry_kernel.cuh sections, the dispatch and its refusal, the arm generator, terminology
---
Read the source files named below directly, not `docs/internals/carry/*.md`'s history sections —
those record HOW a mechanism arrived (superseded designs, ruled-out alternatives); the CURRENT
mechanism is only trustworthy read off the `.cuh`/`.cu` file itself. `docs/internals/carry/box.md`,
`carry_kernel.md`, `carry.md` etc. mirror `csrc/rola/src/carry/` file-for-file if you need prose
alongside code, but a doc and its source can drift — the source is the tiebreaker.

## `box.cuh` — the (k, m) box lattice, pure constexpr arithmetic
Read `BoxPlan<D, B, K, M>`'s own comments for the current parameter meanings and derived quantities
(owner span, owner-grid extent, leaves-per-owner, total leaf count) — don't re-derive them from
memory, the derivations are constexpr and self-documenting in the file. The load-bearing invariant
to know going in: a leaf's canonical index and its lattice index are related by a fixed BIT
PERMUTATION — nothing here divides or tabulates at runtime, every use site is a shift and a mask
(the file's own comments name the power-of-two and alignment constraints this relies on).
`ModeOrder<D, MO, SIDE>` orders levels by a fixed class order (read the enum/comment for the exact
order) — a pure function of its template args, no span consulted.
`StreamSpans<D, B, K, M, MO, SIDE, NS>` derives the per-level span vector one STREAM keeps by
carving `ModeOrder`'s level order until the stream grid's product matches the requested stream
count — the stream's box is an AFFINE RECTANGLE of digits (a prefix carve, contiguity never
required).
`CarryGeom::kWarps` (in `carry_kernel.cuh`) is the CTA's warp count; a carved arm's EXECUTOR count is
its STREAM count, not `kWarps` directly — see the terminology law below.

## `carry_kernel.cuh` — the carved body's sections, by banner comment
Find sections by the file's own top-of-file banner (state / schedule / fetch+expansion / staging /
fold / exchange / readout / ring) rather than by line number, which moves across stages. Landmarks
worth knowing directly (search for the function/section name, not a remembered line number):
`ring_scatter` (the fan-in ring's lock+CAS+accumulate retry loop), `tile_ready`/`place_group`/
`rank_live` (flow counters gating a pull), the exchange's row-map section (an affine bit map over a
publisher's slot). See `rola-ledger` for how each section's cost was measured and its (a)/(b)/(c)
classification.
`carry_split_kernel.cuh` is the uncarved (channel-split) body — no `MO`, no stream counts, dispatched
when neither side carves anything.

## `carry.cu` — dispatch only, and its refusal
This file instantiates NO kernel — the arm rows live in `tools/gen_shards.py`, which emits the arm
include file (rows + selection macros) and one per-arm `.cu` file (`rola-build` covers the
file-subset mechanics). `carry.cu` matches a call's shape against the generated dispatch rows and,
on a match, tries the level-mode word; grep the file for `"unbuilt carry arm"` to find the current
refusal point rather than trusting a remembered line number, which moves every time the file is
edited. This refusal overruled a removed "law 3" fallback that silently retried with a different
key and served a default body for an unbuilt declaration with no diagnostic — a defect that hid a
large, multi-stage performance loss before it was caught (see FINDINGS `## BISECT (corrected)` for
the incident). Never reintroduce a silent retry-with-a-different-key path here; an unbuilt arm is
refused by name, always.

## The generator — `tools/gen_shards.py`
`CARRY_ARMS` is THE ONE arm-row tuple list — read its own tuple-field comment for the current field
order (topology, value width, window, chunk, declaration scalar, mode word, warps-per-stream,
stream counts, park bit, etc.) rather than reconstructing it from memory. `CARRY_SHIPPED_ARMS` vs.
`CARRY_TEST_ARMS` is the shipped/conformance-only split — read both definitions directly, they move
as arms are added or retired. `carry_kernel_key(row)` decides whether two rows are the SAME kernel
(an uncarved row's body ignores `MO`, so declaration-only-different rows can share one TU); this is
what makes an arm-subset build a FILE subset rather than a macro trick. Re-derive the arm-row axes
and the current row/TU count from THIS file after any pruning — `len(gen_shards.CARRY_ARMS)` and
`CARRY_SHIPPED_ARMS`, read live — never from memory of an older count.

## The terminology law: STREAMS != EXECUTORS
A STREAM owns a box (an affine rectangle of digits, from `StreamSpans`) and walks its data into
tiles. An EXECUTOR runs streams — a carved arm's executor count is its STREAM count; the CTA's warp
count (`kWarps`) is a resourcing decision (how many warps cover how many streams, via the
warps-per-stream field), never the stream count itself. Getting this backwards is how a past ledger
entry misread a `kWarps` change as doubling work when it only doubled per-warp fixed costs (see
FINDINGS for the incident) — because the streams, and therefore the union list each executor
covers, hadn't doubled. When reading or writing a brief, say "N streams, M warps per stream,
therefore M×N executors" explicitly rather than conflating them.

## sm_80/90/100 mapping habit
Every structural decision in this family states its arch mapping (KERNEL_STANDARDS §8, card P76
§4): on sm_80 there is no register redistribution, so warp specialization takes the role-alternating
form the standards doc names, and pipelining uses an SMEM slot or registers, never the sm_90-only
register-reallocation instruction. When reading a new mechanism here, ask "what does this become on
sm_90 (mbarrier, wgmma/tcgen05 SMEM-sourced operands, TMA box loads) and sm_100 (clusters/DSM)" even
though only the Ampere arches are built today — a design that only makes sense as an Ampere register
trick is flagged, not shipped quietly.
