# `csrc/rola/src/segment_mask.cuh` — §6.4's per-segment tile-range mask

One `__device__` function: the arithmetic that decides which tiles a SEGMENT owns,
under sequence parallelism (`P > 1`).

Symbols: `P` is the sequence-parallel split factor; a SEGMENT is one contiguous tile
range assigned to one launch slot; `nseg` is the number of segments; `w` is a
candidate WORD index, so word `w` covers tiles `[32w, 32w+32)`; `[lo, hi)` is the
segment's tile range.

---

## <a id="own-header"></a>1. Where it came from, and why it is its own header

EXTRACTED from the REGISTER-RESIDENT consumer's kernel header when that consumer was
deleted (item 6b, 2026-08-01). That header was also named `consumer_kernel.cuh` and is
readable at `c3c2f845` in the source repository `docs/provenance.md` names (the
hash does not resolve in this repository); **it is a different kernel from today's file of that name.**

Why it is not simply inlined into the kernel: the mask is read by the consumer kernel
and, at `P > 1`, must agree BIT FOR BIT with the host-side segment partition the
schedule publishes. Keeping it at a named, single definition is what the
sequence-parallel gates pin; folding it into the kernel body would make the pin a
comment.

## <a id="state-pitch"></a>2. `state_pitch` is NOT here — a finding, not an omission

The other half of the brief's "split" was `detail::state_pitch<DV>() = DV + 1`, the
second symbol the old header shared.

Read back from the tree at the point of deletion: the consumer kernel declared
`using rola::detail::state_pitch;` and **NEVER CALLED IT**. Its own row pitch is
`SP = DV + 4`, derived from the bf16 fragment map and proven by `banks_distinct`.

So `state_pitch` dies with the kernel that used it, and the `state_pitch` CONDITIONAL
MANDATE — keep the divergence documented while two kernels disagree on the pitch —
**DISCHARGES**: there is one kernel and one pitch.

## <a id="word-mask"></a>3. `segment_word_mask(w, lo, hi)`

§6.4: the bits of candidate word `w` that lie in the tile range `[lo, hi)`.

A segment owns a CONTIGUOUS tile range, so restricting the candidate walk to it is one
more word-AND, of exactly the shape §4's `j+1` candidate ANDs already are.

The body clamps the two range endpoints into word-local `[0, 32]`, returns `0` when the
clamped range is empty — which is also the only way `a == 32` could reach the shift —
and otherwise returns `upto & ~((1u << a) - 1u)` with `upto` special-cased at `b == 32`
because `1u << 32` is undefined.

**At `nseg == 1` the range is the whole tile axis, every word comes back all-ones, and
the AND is the identity** — which is one half of why `P = 1` is bit-identical to the
pre-`P` kernel.
