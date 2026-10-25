# `csrc/rola/src/den/den_api.cuh` — the den pass's host seam

Mirror doc for the den pass's host entries and the refusals they make before a launch exists. The kernel is
[`den_kernel.md`](den_kernel.md), the unit that defines these entries [`den.md`](den.md); `rola/ops/den.py` is the
Python caller (`rola.ops.den.den`), run once a call before or beside the carry.

<a id="den-forward"></a>

## `den_forward`

    den_forward(read_rows[], write_rows[], read_lists[], write_lists[], list_exp, gains16, gains_exp,
                den!, num!, widths, level_s, read_list, write_list, heads) -> ()

Each side's rows a level (`[B, T, H, width]`, the read side's `H` the `heads` query heads of each write head), the
facts pass's lists for each side in their final form (five tensors a side, empty tensors where the form takes none),
the write lists' exponents, the window gains and their exponents, and the two output planes `den` and `num`,
contiguous fp32 `[B H_w heads, T]`, zeroed by the caller. `widths` is the layout's padded widths, `level_s` the
carry's level S (`rola.routing.factors.carry_level_s`), `read_list` and `write_list` each side's list level (-1 for
none).

<a id="refusals"></a>

The refusals (KERNEL_STANDARDS §18: a seam refuses rather than guesses):

* depth two, level S named, level S's width a multiple of 64 and level R's of 64;
* the read side listing level S with the write side listing level R, or both sides dense (every other pairing is
  owed: `rola.ops.den.serves` names it);
* the planes the form reads are f16 (a dense level's stored rows), each starting and striding on 16 bytes, digits
  contiguous -- the staging copies and the den's row quarters are 16-byte loads;
* `T` a whole number of 64-token chunks (a partial last window is a window of `live` tokens), 1 to 4 query heads a
  write head (the accumulator's rows), a head's rows within 31 bits, `B H_w` within the grid;
* the lists the facts pass's shapes; `list_exp` int16 `[BH, windows, width_R, 2]`; `gains16` f16 `[BH, L, 2]` and
  `gains_exp` int16 `[BH, windows x 2]`, contiguous;
* the outputs the shapes above.

<a id="den-ledger-bind"></a>

## `den_ledger_bind`

Binds the phase ledger the next launches add into -- a contiguous CUDA int64 tensor `[ctas][16][6]`, the kernel's
`DenPhase` order (`den_kernel.md#phases`) -- or clears it. Unbound, the clock issues nothing.

