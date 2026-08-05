# `csrc/rola/src/segment_mask.cuh` — §6.4's per-segment tile-range mask

One `__device__` function: the arithmetic that decides which tiles a SEGMENT owns,
under sequence parallelism (`P > 1`).

Symbols: `P` is the sequence-parallel split factor; a SEGMENT is one contiguous tile
range assigned to one launch slot; `nseg` is the number of segments; `w` is a
candidate WORD index, so word `w` covers tiles `[32w, 32w+32)`; `[lo, hi)` is the
segment's tile range.

---

## <a id="own-header"></a>1. Why the mask is its own header

The mask is read by the consumer kernel and, at `P > 1`, must agree BIT FOR BIT with
the host-side segment partition the schedule publishes. A named, single definition is
what the sequence-parallel gates can pin; folded into the kernel body, the pin would
be a comment.

## <a id="state-pitch"></a>2. `state_pitch` is NOT here, and there is no shared pitch symbol

The consumer kernel's state row pitch is `SP = DV + 4`, derived from the bf16 fragment
map and proven by `banks_distinct`. It is the ONLY state pitch in the engine: one
kernel, one pitch, and therefore no `state_pitch<DV>()` helper for two kernels to
disagree through. Nothing in this header is an expression in the state pitch.

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
unsplit kernel.
