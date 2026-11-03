# `common/window.cuh` -- the window W, a card's

Mirrors `csrc/rola/src/common/window.cuh`.

A call's token axis is cut into windows of W tokens, global to every box and every kernel: the carry reads each
window's state out to the window's readers and folds its writes, the intra takes the pairs inside one window, the facts
pass emits each window's lists, the scale pass each window's write masses. W is not a data size. A window's resident rows
follow its read and write unions (the plan's chunks, multiples of 64 union slots); W sets how many tokens share one
window's fixed work -- the hand-off of the state into the image, the staging of the next window's inputs, the barriers.
A card with more shared memory amortizes that work over more tokens (ruled 2026-10-31: W and the other structural
constants are per-card configuration).

<a id="window-for"></a>
## `window_for(cc)`

A card's W by its compute capability (x10), the one table every kernel and the host read. Device code reads the
compiling card's through `kWindow`, which is `window_for` at `arch_caps.cuh`'s `kArch`: the compiling card in the
device pass and, as for every cap, the pinned card in the host pass. So host code never reads `kWindow` -- on another
device it would be the pinned card's -- and takes its device's through `device_window(device)`
(`common/arch_runtime.cu`), the same table at the device's compute capability. A host shape or check built from the
pinned value would refuse the device's tensors at the call, and the carry's device stamp reports the device's own W, so
a binary and its host cannot disagree unseen. Sizes that both sides compute -- the carry's tenants and read tables
(`carry/plan.cuh`) -- take W as a template parameter, and the host instantiates them inside its
per-card switch.

Every card is 512 until the A100's W = 1,024 is built through every kernel and measured; the table is then the only
line that changes.

<a id="token-field"></a>
## `window_bits(w)`, `window_heads(w)`

A union slot is 16 bits: its token within the window in the low `window_bits(w)` bits (log2 W), its query head above
them, bit 15 a gap slot. A write head's query heads are at most `window_heads(w)`: 64 at W = 512, 32 at W = 1,024.
