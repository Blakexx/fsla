# `csrc/rola/src/intra/intra_api.cuh` / `intra.cu` — the intra's host seam

Mirror doc for the intra's host entries and the refusals they make before a launch exists.
The kernel is [`intra_kernel.md`](intra_kernel.md), the unit that defines these entries
[`intra.md`](intra.md); the tile unions it gates on are
[`facts/liveness_contract.md`](../facts/liveness_contract.md#tile-unions). `rola/ops/intra.py`
is the Python caller, `rola.ops.prefill.prefill` its one call site on the prefill path.

<a id="rola-intra-forward"></a>

## `rola_intra_forward`

    rola_intra_forward(read[], write[], scale_write[], g_write, v_hat, v_lo?, v_scale, unions,
                       num!, den!, widths, dense_read, dense_write, weight_digits, window) -> ()

Each side's rows a level (`[B, T, H, B_l]` u8), the write levels' scales, the write gain and
the value scale (`[B, T, H]` fp32, any strides), the value planes (`v_hat` s8, `v_lo` u8 or
absent at one value digit, `[B, T, H, 64]`), the facts pass's tile unions (contiguous int32
`[B·H, 2, groups, T / 64]`), and the two output planes the pairs are REDUCED into: `num`
contiguous fp32 `[B·H, T, 64]` and `den` `[B·H, T]`. The caller owns their contents; the
carry reduces into the same planes. `dense_*` is each side's dense-level bitmask, the same
statics the facts pass took (`rola.engine.facts.liveness.side_statics`).

The refusals (KERNEL_STANDARDS §18: a seam refuses rather than guesses):

* one to four levels, each side a row set a level and the write side a scale a level;
* `window` is the carry's 512 tokens — the two kernels run one window grid — and `T` a whole
  number of windows;
* the weights are one or two u8 digits;
* the rows are u8: the f16 rows' arms (depth above two) are not built, and the refusal says so;
* every level is one width, and an arm is built at it (`no intra arm at D levels of B digits`);
* a row set or value plane starts and strides on 16 bytes, digits contiguous — the kernel's
  copies are 16-byte `cp.async` chunks (the producer's planes are);
* every token stride fits 31 bits, and `v_lo` is laid out as `v_hat` is — a copy adds its
  tile's rows by one 32-bit stride, and one address serves both value planes;
* the unions and outputs are the shapes above, and the dense bitmasks name levels inside the
  depth.

The arm's footprint is past the 48 KB a kernel gets unasked, so the first launch of an arm
checks it against the device's opt-in and sets the attribute; the grid is `(windows × 6, B·H)`, a CTA a piece.

<a id="rola-intra-ledger-bind"></a>

## `rola_intra_ledger_bind`

Binds the phase ledger (`[8][7]` contiguous int64, [`intra_kernel.md`](intra_kernel.md#phase-ledger))
the next launches add into, or unbinds it with `None`.

<a id="intra-arms"></a>

## `intra_arms`

The built rows `(levels, level width, shared bytes)`, read off the same macro the dispatch
expands (`ROLA_INTRA_ARMS_X`), so the list and the dispatch cannot drift. `rola.ops.intra.level_width`
answers the one width an arm is built at for a depth.

<a id="intra-build-stamp"></a>

## `intra_build_stamp`

A fact only the loaded binary can produce: a one-thread kernel returns the built arm's
shared-memory footprint times 1000003 plus `sizeof(IntraParams)`, so the value moves when the
arm's shape or the operand block does. A path or hash check cannot catch a stale extension;
`test_the_loaded_binary_states_the_arm_it_carries` reads it against the arm table (the stamp
kernel's named namespace: [`intra.md`](intra.md#the-unit)).
