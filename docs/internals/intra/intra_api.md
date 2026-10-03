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

Each side's rows a level (`[B, T, H, w_l]` u8, the level's own width), the write levels' scales, the write gain and
the value scale (`[B, T, H]` fp32, any strides), the value planes (`v_hat` s8, `v_lo` u8 or
absent at one value digit, `[B, T, H, 64]`), the facts pass's tile unions (contiguous int32
`[B·H, 2, groups, T / 64]`), and the two output planes the pairs are REDUCED into: `num`
contiguous fp32 `[B·H, T, 64]` and `den` `[B·H, T]`. The caller owns their contents; the
carry reduces into the same planes. `widths` is the union table's layout, the facts pass's
(`rola.ops.facts.call_layout`, the padded widths), and `dense_*` each side's dense-level
bitmask, the same statics the facts pass took (`rola.engine.facts.liveness.side_statics`).

The refusals (KERNEL_STANDARDS §18: a seam refuses rather than guesses):

* each side a row set a level and the write side a scale a level, as many as the layout's
  levels;
* `window` is the carry's 512 tokens — the two kernels run one window grid — and `T` a whole
  number of windows;
* the weights are one or two u8 digits;
* the rows are u8;
* one or two levels (the built arms serve depth two), each a whole number of 16-digit copy granules
  from 16 to 256, and no wider than the layout's level, which its union groups come from;
* a row set or value plane starts and strides on 16 bytes, digits contiguous — the kernel's
  copies are 16-byte `cp.async` chunks (the producer's planes are);
* every token stride fits 31 bits, and `v_lo` is laid out as `v_hat` is — a copy adds its
  tile's rows by one 32-bit stride, and one address serves both value planes;
* the unions and outputs are the shapes above, and the dense bitmasks name levels inside the
  depth.

The footprint is the call's ([`intra_footprint`](#intra-footprint)); one past the 48 KB a kernel
gets unasked is checked against the device's opt-in and the depth's arm raised to it, once a
size; the grid is `(windows × 6, B·H)`, a CTA a piece.

<a id="rola-intra-ledger-bind"></a>

## `rola_intra_ledger_bind`

Binds the phase ledger (`[8][7]` contiguous int64, [`intra_kernel.md`](intra_kernel.md#phase-ledger))
the next launches add into, or unbinds it with `None`.

<a id="intra-footprint"></a>

## `intra_footprint`

    intra_footprint(widths) -> int

The shared-memory bytes a call at these level widths launches with, from the same geometry the
launch derives ([`intra.md`](intra.md#stages)), with the same refusals of widths it does not read.
`rola.ops.intra.admits` reads it against the device's opt-in; the prefill's envelope asks
`admits` whether the intra reads a call's widths.

<a id="intra-build-stamp"></a>

## `intra_build_stamp`

A fact only the loaded binary can produce: a one-thread kernel returns the fixed tenants'
size (`sizeof(Smem)`) times 1000003 plus `sizeof(IntraParams)`, so the value moves when the
tenants or the operand block do. A path or hash check cannot catch a stale extension;
`test_the_loaded_binary_states_the_footprint_it_carries` reads it against `intra_footprint`'s law
(the stamp kernel's named namespace: [`intra.md`](intra.md#the-unit)).
