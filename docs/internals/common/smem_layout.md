# The shared-memory layout (`csrc/fsla/src/common/smem_layout.cuh`)

THE RULE (the owner, 2026-11-24): "all smem layout decisions are made in the kernel, using the template facts or runtime
facts that are able to be uniformly switched on"; the card's shared memory an SM is the fact everything is serviced
from ("it should be smem per sm, and we service everything from. That, including how many CTAs"), and the budget "can
just be the whole card". FACTS IN THE TEMPLATE, DECISIONS IN THE KERNEL: the card's facts reach a kernel as constants
of its build (`Bt::traits.caps`, generated from the registry's device parts), each branch of a kernel declares its
shared memory once through `SmemLayout`, and the host decides nothing about it: the launch reserves `bytes`, read off
the same type.

## <a id="layout"></a>A layout

`SmemLayout<Bt, CTAS, Tiles...>` is one CTA's shared memory on the build `Bt` where `CTAS` CTAs share an SM. Its
`budget` is the SM's shared memory over `CTAS` (`smem_per_sm`) less what the card keeps of each CTA's (`smem_per_sm`
less `smem_per_cta_optin`): with one CTA an SM that is what a CTA may reserve, and with more their budgets and what
the card keeps of each add to the SM. A tile is named by a type (`Tag`) and holds elements of a type; every tile starts on a 16-byte
boundary. The layout gives, as compile-time constants, each tile's `count<Tag>` (its elements ON THIS CARD), its
`bytes`, what the launch reserves (the tiles with a share), and `offset<Tag>()`, where such a tile starts in it; and at
run time `get<Tag>()`, the tile as its elements. `base()` declares what the launch reserves, and `own` a tile's own
object.

- `Fixed<Tag, T, COUNT>`: exactly `COUNT` elements. `One<Tag, T>`: one element.
- <a id="own"></a>A TILE WITH NO SHARE IS ITS OWN SHARED OBJECT (`own`), not a range of one buffer, so that the compiler
  knows no two such tiles overlap: decode's control fields carved out of one buffer cost every arm about 12 registers
  a thread and more convergence instructions than its hand-declared variables had, and as tiles of their own its arms
  are the same size or smaller. Only the tiles with a share are in the bytes the launch reserves (`bytes`); the budget
  covers both (`own_bytes` and `bytes`).
- <a id="rest"></a>`Rest<Tag, T>`: all the budget leaves, one part of it, with no least: a tile whose elements a call
  addresses at run time, the call's need checked against `count<Tag>` at the seam.
- <a id="share"></a>`Share<Tag, T, LEAST, WEIGHT>`: at least `LEAST` elements, and `WEIGHT` parts of what the budget
  leaves once every tile's least is placed (dealt in whole 16-byte steps). A share is how a layout ADAPTS: the same
  source gives a deeper ring on a card with more shared memory and a shallower one on a card with less. Code reads
  `count<Tag>` and never a number.

THE ONE COMPILE FAILURE is the `static_assert` that the tiles' least sizes fit the budget: it means the layout cannot
adapt to that card, and the compiler's instantiation trace names the build, the kernel and the case. A name no tile
has, or two tiles have, does not compile either: a tile's count, offset, type and pointer all go through `place`.

## The proof in the header

`smem_proof` instantiates one layout on two cards that differ in shared memory alone and asserts, at every build, that
it fits both, that a fixed tile is the same on both, that a share is larger on the larger card and above its least on
the smaller, that one CTA an SM may reserve what the card says, that two CTAs' budgets and what the card keeps of each
are the SM, and that a tile's offset is the bytes before it.

## Who uses it

`csrc/fsla/src/entmax/factor.cu`'s forward kernel (a slot a warp, a fixed tile) and `csrc/fsla/src/decode/decode.cu`'s step
(its control fields, scan storage, clock rows, value and fold rows as fixed tiles, and the call's staged rows and tables
in what the budget leaves). The lint that refuses a hand declaration is `tools/lint/facts_in_template.py`
([`../tools/lint.md`](../tools/lint.md#facts-in-the-template)).
