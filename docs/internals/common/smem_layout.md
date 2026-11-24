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
`budget` is the smaller of the SM's shared memory over `CTAS` (`smem_per_sm`) and the most one CTA may reserve
(`smem_per_cta_optin`). A tile is named by a type (`Tag`) and holds elements of a type; every tile starts on a 16-byte
boundary. The layout gives, as compile-time constants, each tile's `count<Tag>` (its elements ON THIS CARD), its
`offset<Tag>()`, and `bytes`, the whole; and at run time `get<Tag>()`, the tile as its elements. `base()` declares the
CTA's shared memory, the one place in the tree that does.

- `Fixed<Tag, T, COUNT>`: exactly `COUNT` elements.
- <a id="share"></a>`Share<Tag, T, LEAST, WEIGHT>`: at least `LEAST` elements, and `WEIGHT` parts of what the budget
  leaves once every tile's least is placed (dealt in whole 16-byte steps). A share is how a layout ADAPTS: the same
  source gives a deeper ring on a card with more shared memory and a shallower one on a card with less. Code reads
  `count<Tag>` and never a number.

THE ONE COMPILE FAILURE is the `static_assert` that the tiles' least sizes fit the budget: it means the layout cannot
adapt to that card, and the compiler's instantiation trace names the build, the kernel and the case. A name no tile
has, or two tiles have, does not compile either.

## The proof in the header

`smem_proof` instantiates one layout on two cards that differ in shared memory alone and asserts, at every build, that
it fits both, that a fixed tile is the same on both, that a share is larger on the larger card and above its least on
the smaller, that halving the budget by the CTAs an SM equals halving the card, and that a tile's offset is the bytes
before it.

## What is NOT built

No kernel uses the layout yet: `csrc/fsla/src/entmax/factor.cu` and `csrc/fsla/src/decode/decode.cu` declare their
shared memory by hand. No lint refuses a hand declaration or a literal size yet.
