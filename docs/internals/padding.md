# Host-side padding: any level width, any value width, no kernel change

`fsla/ops/padding.py` is the ONE translator between a caller's LOGICAL shape (any
level width `b_l >= 1`, any value width `d_v`) and an op's kernel-facing, PADDED shape
(a level width a power of two at or above 16, a shipped value width). The two
rulings ("PADDING AT THE PRODUCER", "d_v PADDING IS ALSO HOST-SIDE") name the
mechanism; this page states where it lives and why.

## Two contracts, kept separate (CONSTITUTION §R13)

**API CONTRACT.** The layer producer's bundle
(`RouteFactors`) and `fsla.layer.FSLA` stay at the caller's own, unpadded widths --
`tests/unit/test_api_contract.py` pins this: one tensor shape per declared level, and
every public tensor a VIEW of the solve's packed layout with zero copy. Padding those
would break both pins for no reason: nothing downstream of the producer needs a padded
`RouteFactors` to exist as an object.

**KERNEL CONTRACT.** The two padding seams -- the prefill's carry leg (`fsla.ops.prefill`)
and the decode step (`fsla.ops.decode._decode_step`) -- receive the
caller's logical tensors and widths and pad INTERNALLY, once, before any launch-facing
derivation. The kernel itself is never handed a width other than its own; `y` is sliced
back to the caller's own `d_v` before it leaves the op.

`fsla/ops/padding.py`'s primitives, used identically by both seams:

* `pad_routes(levels, padded_widths)` -- per level, `[..., b_l] -> [..., B_l]`, the
  trailing columns exact `0.0`.
* `pad_v(v, DV)` -- `[..., d_v] -> [..., DV]`, exact `0.0` tail.
* `slice_y(y, d_v)` -- `[..., DV] -> [..., d_v]`, a view.
* `PaddedShape.of(widths, d_v, shipped_dv=...)` -- the descriptor: `widths`,
  `padded_widths`, `d_v`, `DV`, and `is_padded`. The one place "logical vs padded" is
  computed; every op builds one of these before it touches a tensor.

## Why a zero-pad AFTER the solve is the ruling's mechanism, not an approximation of it

The ruling states the mechanism as padding the LOGITS with `-inf` before entmax/softmax. This
module pads the SOLVED level with `0.0` after it -- the identical tensor, because both
softmax and entmax normalize over, and threshold against, only the finite entries of a
row: an `exp(-inf) == 0.0` addend changes neither a softmax's normalizer nor an
entmax's support-selecting sort, so the `b_l` real columns take the same value either
way and the extra columns are `0.0` either way.
`tests/oracle/test_padding_seam.py::test_prefill_padding_is_the_ruled_mechanism_not_an_approximation_of_it`
proves the two tensors are `torch.equal`. The zero-pad-after form is also SAFER:
autograd never runs through a literal `-inf` logit, so there is no `0 * -inf` NaN risk
in a backward pass, and no parameter reads a pad column, so its gradient is exactly
`0.0` by construction rather than by cancellation.

## Per-op scope: not every op pads every axis

* **the prefill's carry leg** (`fsla.ops.prefill`'s `_padded_shape`/`_pack_operands`) pads
  BOTH axes: every level to `fsla.routing.topology.padded_level_width` (next power of two
  >= 16, the register-shape floor), and `v` to the smallest shipped value width that
  admits it.
* **decode** (`fsla.ops.decode._decode_step`) pads ONLY
  `d_v` -> the `DV` of the arm the planner's launch function
  (`fsla.ops.decode.decode_launch`, [`common/plan/decode_launch.md`](common/plan/decode_launch.md)) maps it to: the narrowest arm that holds it. Its
  per-level widths are NEVER padded: a level's width is a call fact of the descriptor (`width_bits_k`, its log2, [`docs/internals/hardware/registry.md`](hardware/registry.md#parts)), the contract's
  range runs from the state format's floor (`fsla._state.MIN_LEVEL_WIDTH`) to the decode path's capacity, and a width outside it is refused when the descriptor is made
  (the descriptor the decode step builds from its bundle, `fsla.ops.produce.decode_descriptor`'s rule), never padded to one inside it.
* **intra** (`fsla.ops.intra.intra_forward`) pads nothing: it reads the bundle's rows
  where the producer stores them, at their own widths, each level a whole number of
  16-digit granules up to 256 (`fsla.ops.intra.reads`) -- a bundle outside that is never
  padded at the seam. The values are the
  format's 64 channels.

## Where the shipped sets come from

Decode's value widths are the launch function's: it maps a call's `d_v` into the arm that holds it (`fsla.ops.decode.decode_launch`), and the
contract bounds it (`hardware/kernels/decode/parts/call.json`), so no literal tuple here can drift from what the binary carries. The intra reads one
value width, `fsla.ops.intra.VALUE_WIDTH`.

The prefill's carry leg pads nothing either: the carry (designed, not built) takes the value width as
runtime addressing over the carve's channel extent, which the prefill launch function derives
([`engine/rules.md#carve`](engine/rules.md#carve)); the prefill is refused by name until it is built. The arena's allocation
extent is the launch function's too ([`docs/open-work.md`](../open-work.md) A-c and A-f).

## What this does not do

Nothing here changes `RouteFactors`, `fsla.layer.FSLA`, or `fsla._state.StateFormat`'s
fields (`docs/internals/state.md#binding`): the public surface keeps the caller's own
widths, and padding exists only below the op seams.
