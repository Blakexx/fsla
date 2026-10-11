# `csrc/rola/src/facts/facts_api.cuh` — the facts pass's host seam

Mirror doc for the pass's host entries and for the refusals they make before a launch
exists. The kernel is [`facts.md`](facts.md); the words' shape and meaning are
[`liveness_contract.md`](liveness_contract.md). `rola/ops/facts.py` is the Python
caller: `call_layout` and `call_statics` read a bundle's layout and its two sides'
static operands off its `sparse` flags and topology, and `facts` packs them into this argument list.

<a id="rola-facts-forward"></a>

## `rola_facts_forward`

    rola_facts_forward(read[], write[], scale_write[], g_write, v_scale, words!, unions!, list_exp!, fm!, lists![10],
                       widths, dense_read, mask_read, dense_write, mask_write, list_level, read_list_level,
                       window) -> ()

Each side's rows a level (`[B, T, H, b_l]`, u8 but a dense write level's f16), the
write levels' scales, the write gain and the value scale (`[B, T, H]` fp32, any
strides), and the three outputs the caller allocates: `words` the contiguous int32
`[B·H, 2, rows, ceil(T/32)]`, `unions` the contiguous int32 `[B·H, 2, groups,
ceil(T/64)]` ([`liveness_contract.md`](liveness_contract.md#tile-unions)) and `list_exp` the
contiguous int16 `[B·H, windows, B_list, 2]` ([`facts.md`](facts.md#lists)), `B_list` the list level's padded
width, and `fm` the contiguous fp32 `[B·H, T, 2]` factor products ([`facts.md`](facts.md#factor-products)) — both any
tensor when `list_level` is -1 (the write side has no one sparse level; `rola.ops.facts.list_level`). The pass
writes every element of `list_exp` and `fm` and, of `words` and `unions`, every sparse level's rows and groups; a dense
level's are left as allocated (its words are the width mask, which a reader takes from the statics,
[`liveness_contract.md`](liveness_contract.md#statics)). `widths` are the layout's PADDED widths; `dense_*`
is a bitmask over levels and `mask_*` one bit a row, packed by
`rola.engine.facts.liveness.side_statics` — the only place `b_l` appears.

The refusals (KERNEL_STANDARDS §18: a seam refuses rather than guesses):

* the depth is one to four levels, each side holds a row set a level and the write side
  a scale a level; a level is 1 to 256 digits wide;
* `window` is the carry's 512 tokens — the lists' grain is the carry's window, and a caller
  that means another has no fold to hand it to;
* `list_level` is a level of the depth, or -1;
* the rows are u8, a dense write level's f16, `[B, T, H, width]` with the first level's `B, T, H`, digits
  contiguous, under the row alignment law ([`liveness_contract.md`](liveness_contract.md#alignment));
* `lists` holds 10 tensors: the write side's records, starts, union tokens and counts, the read side's slots, starts, union
  tokens and counts, then the gains' f16 pairs `[B·H, T, 2]` and their int16 exponents `[B·H, windows, 2]`
  ([`facts.md`](facts.md#lists-out)); a side's four are touched only at its sparse listed level;
* each side's dense bitmask names only levels inside the depth, and its digit mask is
  exactly `mask_words(rows)` words;
* **a mask bit names a digit inside its level's rows.** The kernel reads a digit iff
  its bit is set (on a sparse level), so a mask wider than the rows would read the next
  level's digits, or past the tensor; it is caught here, not by a wrong answer;
* the columns are CUDA fp32 `[B, T, H]`, and the outputs are the shapes above.

<a id="rola-facts-ledger-bind"></a>

## `rola_facts_ledger_bind`

Binds the `[16][4]` int64 phase ledger the next launches book into, or unbinds it with
`None` ([`facts.md`](facts.md#phase-ledger)); it issues no launch.
