# `csrc/rola/src/entmax/entmax.{cuh,cu}` — the producer's union entmax solve

Hand CUDA over CUB warp primitives: the α-entmax solve that turns per-token read
and write LOGITS into route VALUES, a midpoint amplitude, and §4.1's packed
per-leaf support bitmap. `entmax.cuh` carries the host declaration pair
(`union_forward` / `union_backward`) AND [the warp-row vocabulary](#warp-row) the
whole entmax family speaks; `entmax.cu` is the union split's implementation — two
kernels and their host dispatch.

The family's other member is [`factor.md`](factor.md): the independent per-side
solves, whose forward is this file's phases B/C/D fed one logit tensor.

Design of record: this document.
Gates: `tests/oracle/test_entmax_cuda_gates.py`, against the **fp64 oracle only** —
never kernel-vs-kernel (trackE.md §4).

Symbols: a *stream* is one (batch, head)-like routing stream; a *token* is one row
of the solve; `width` is the level's branch width and `col` indexes it; `LW` is the
LOGICAL warp width (a power of two ≤ 32); `IPT` is items per lane, so
`W_PAD = LW * IPT` is the padded row width; `alpha` is 1.5 (α-entmax) or 2.0
(sparsemax); `tau` is the solve's threshold; *support* is the set of columns with
`midpoint > tau`.

---

## <a id="why-cuda"></a>1. Why this exists as CUDA at all

A Triton lowering of this solve measures **255 registers WITH SPILLS at 16.67%
occupancy** as written, and **115 reg / 33.33%** hand-restructured — where it still
carries an uncontrollable `tl.sort` scratch layout with **75%
bank-conflict-excessive wavefronts**. It is also a JIT against a closed-world AOT
policy that forbids exactly that.

Provenance: the producer bring-up measurements and
`producer-p3.md`.

## <a id="one-row-per-warp"></a>2. The structural idea: one row per (logical) warp

Every collective in the entmax solve — the sort, both prefix scans, every reduction
— runs ALONG ONE TOKEN'S ROW. Triton can only express them as block-scope
collectives over a 2D `(ROW_TILE, BLOCK_COLS)` tile, which is what

1. welds register demand to the CTA tile, so `num_warps` is not a lever, and
2. produces the bank-conflicted shared scratch.

Assigning one row to one warp makes every collective warp-scope and shuffle-based.
The consequences are all structural:

- **register demand becomes `O(items-per-lane)` and decouples from the warp count**
  — so the warp count is a free occupancy knob, which it provably is not under
  Triton (`producer-p3.md` §1);
- **the solve needs ZERO shared memory and ZERO block barriers.** `cub::WarpScan` /
  `cub::WarpReduce` `TempStorage` measures **1 byte** here — the pure-shfl
  specializations — so the 75%-bank-conflict `tl.sort` scratch is ELIMINATED, not
  tuned;
- **exactly ONE `__syncthreads` in the whole kernel** (§8, the epilogue).

CUB's warp primitives are the library half of the spec's "hand-roll semantics, never
primitives" rule.

The per-thread `TempStorage` declarations are only legal because CUB resolves all of
them to the pure-shfl specializations. That is MEASURED, not assumed
(`<scratchpad>/cub_sizeof.cu`: 1 byte for every `WarpScan` / `WarpReduce`
instantiation this kernel uses), and the `static_assert`s in both kernels turn a
future CUB that changes its mind into a compile error rather than a race.

## <a id="frozen-support-word"></a>3. The frozen support-word layout

`WORD_TOKENS = 32` is FROZEN (§4.1's per-leaf write bitmap, **nine downstream
consumers**). One CTA therefore owns exactly the 32 tokens of one support word, so
the `1 << (token % 32)` packing needs **no global atomics and no zero-init pass**.

`__launch_bounds__(128, 8)` — i.e. `MIN_BLOCKS_PER_SM = 8` — states the 64-register
target to ptxas, so a regression becomes a compile-time allocation decision instead
of a silent spill; the ratification manifest (`tools/ratify.py`) then fails the
BUILD on spill growth.

## <a id="strides"></a>4. `Strides` — one address expression, two route-value layouts

`read_values` / `write_values` accept BOTH route-value layouts through one stride
5-tuple `(stream, token, col, tile, lane)`:

| layout | `stream` | `token` | `col` | `tile` | `lane` |
|---|---|---|---|---|---|
| rank-3 dense level | `stride(0)` | `stride(1)` | `stride(2)` | 0 | 0 |
| rank-4 tile-major arena slice | `stride(0)` | 0 | `stride(2)` | `stride(1)` | `stride(3)` |

The single address expression

```
stream*s + tok*t + (tok>>5)*tile + col*c + (tok&31)*lane
```

is correct for both, because `five()` already zeroes the members the layout does not
use, deriving them from the tensor's own RANK. **Layout is therefore data, not a
codegen axis** — the Triton lowering this replaced needed a `tl.where(TILE_MAJOR,
...)` select and a specialization axis for it.

## <a id="warp-row"></a>4.1 The warp-row vocabulary lives in the header

`Strides`/`addr`, `store_value`/`load_value`, `Descending`, `Pair`/`PairAdd`,
`bf16_nonzero`, `logical_warp_mask`, `five` and the block constants are declared in
`entmax.cuh` rather than in either `.cu`, because `factor.cu`'s four kernels are the
same machine over a different phase set and a second copy would be a second thing to
keep in step.

They sit behind `#ifdef __CUDACC__`. That guard is load-bearing and not defensive:
`csrc/rola/rola_api.cpp` includes this header to reach the host declarations below,
and it is compiled by the HOST compiler, which knows nothing of `__device__` or
`__nv_bfloat16`.

The namespace is NAMED (`rola::entmax::detail`), so moving these out of `entmax.cu`
moved no mangled symbol and changed no generated code — `tools/ratify.py` reported
"the source digest moved but NO ratified body did" across all 380 bodies at the
move, which is the check that says so rather than the claim that it should.

## <a id="launch-shape"></a>5. Launch shape and the width class

The grid is `(words, streams, LEVELS)`: one CTA owns one 32-token support word of one
stream of one level, and `blockIdx.z` indexes a [`LevelTable`](factor.md#batched-levels)
that arrives in the kernel's parameter space. So one launch covers every union level
that selects the same template arm — same padded width, same alpha, same output dtype —
and `production.py`'s `_Plan` is what groups a walk into those launches.

**Why the class is in the grouping rule and the width is not.** `LW` and `IPT` are what
`w_pad = next_pow2(width)` selects, and they set how `WarpMergeSort` permutes the row and
how the `WarpScan` associates the prefix sums. Running a width-3 level in a width-64 arm
would therefore round `tau` differently in fp32 and could move a support boundary — it
would not be the loop it replaces. Widths INSIDE a class may differ freely; the kernel
reads each level's own `width` from the table. `build_table` refuses a mixed-class table
rather than padding to the largest.

That this is exactly the loop is gated, at `rtol = 0`:
`test_the_batched_solve_is_the_per_level_loop_bit_for_bit` runs four levels of one class
as one launch and as four, and asserts `torch.equal` on values, second output, midpoints
and support words.

## <a id="overflow-free-delta"></a>6. `overflow_free_delta` — the read-write difference

Verbatim from the shipped semantics: two opposite-sign finite extrema overflow a raw
subtraction while their MIDPOINT stays representable, so the MAGNITUDE is clamped
instead. Gate A1 holds this **bit-identical** against plain fp32 torch — it is pure
elementwise, so no reassociation exists.

The `saturated` flag it returns is not cosmetic: the backward zeroes `d_delta`
wherever it fires, because at saturation the difference is no longer a
differentiable function of the two logits.

`log_write_gate` is the companion: `log(sigmoid(-delta)) == -softplus(delta)`,
finite for every finite input, computed as `-(max(d,0) + log1p(exp(-|d|)))`.
**`log1pf` rather than `logf(1+x)`** because the governing baseline is the fp64
oracle, which uses `logsigmoid`, and `log1pf` is the closer composition to it.

## <a id="smem-slot-alias"></a>7. HAZARD — the support-word staging aliases the sort scratch

The `__shared__` slot is sized `max(SORT_BYTES, STAGE_BYTES)` per PHYSICAL warp and
serves both purposes in sequence. **No barrier is needed before the staging write**,
and that is what gets the whole kernel down to ONE `__syncthreads`: each physical
warp writes only into ITS OWN slot, and its own sorts are already retired
warp-synchronously by then, so there is no cross-warp hazard.

Introducing a shared slot, widening one warp's write past its slot, or letting a
warp read another's slot before the barrier all reintroduce a race the compiler will
not flag.

`SLOT` is `RAW_SLOT` rounded up to 16 B so warp `w`'s slot stays 16 B aligned:
`WarpMergeSort::TempStorage` is `4 + 4*W_PAD` bytes, i.e. NOT a multiple of its own
alignment.

## <a id="logical-warp-mask"></a>8. HAZARD — the logical-warp member mask

```cpp
const unsigned lmask = (LW == WARP_SIZE)
                           ? 0xffffffffu
                           : (((1u << LW) - 1u) << (((threadIdx.x % WARP_SIZE) / LW) * LW));
```

**A full `0xffffffff` mask here DEADLOCKS, and it is not theoretical.** With logical
warps (`LW < 32`) and a ragged final support word, only SOME logical warps of a
physical warp have a live token — the rest take the `continue` — so a broadcast
naming all 32 lanes waits on threads that are somewhere else entirely. The member
mask must name exactly this logical warp.

CUB's own collectives compute this internally; the hand `__shfl_sync` broadcasts of
their lane-0 results did not, which is why the constant is spelled here.

The same expression, and the same reason, appears in the backward kernel.

## <a id="forward-phases"></a>9. The forward kernel, phase by phase

A row past the end is skipped by `continue`, and that is exactly equivalent rather
than merely close: such a row yields `has_valid = 0` → `tau = 0` → `support = false`
→ every store masked off.

**PHASE A — load striped (coalesced), fold, RETIRE both logit tiles.** Both logit
reads happen once; `midpoint = (r + w)/2` and `delta = r - w` (§6) are all that
survive into the rest of the kernel.

**PHASE B — row max/min, sentinel-pad, sort descending.** `midpoint` is shifted by
`row_max` for stability. Padded lanes get a FINITE row-relative sentinel,
`(row_min - row_max) - 1.0f`, which keeps them strictly below every real value
without putting an infinity into the scan; `row_min` is pre-shift, so it is shifted
too.

A sort is a **PERMUTATION** — exact, no arithmetic — so the sorted sequence is
identical whatever algorithm produces it. Feeding striped data is therefore fine:
the output arrangement is what the scan reads, and CUB delivers it BLOCKED (rank
order across lanes), which is exactly what the prefix walk needs.

**PHASE C — `tau`.** `prefix`, `prefix_sq`, `tau_candidates`, `candidate` and
`chosen_col` **all collapse to scalars**; none is ever materialized full width. One
`WarpScan` over the `(sum, sum-of-squares)` pair supplies each lane's carry, and the
per-rank candidate test is

```
alpha = 1.5 :  disc = p^2 - k*(q - 4);  tau_c = (p - sqrt(max(disc,0))) / k
               candidate iff rank < width and disc >= 0 and key > tau_c
alpha = 2.0 :  tau_c = (p - 1) / k
               candidate iff rank < width and key > tau_c
```

with `k = rank + 1`.

**ONE reduction selects the greatest valid `k` AND carries its `tau`**: the column
goes in the high 32 bits of a `unsigned long long`, which makes an integer `Max`
pick it, and lane rank ranges are disjoint so a tie is impossible. `best_col = -1`
maps to key 0, so an all-empty row yields `tau = 0`.

**PHASE D — values, in place over `midpoint`.** `read` is emitted HERE, while
`delta` is still the logit difference and before phase E overwrites it. That
ordering is the reason the read side needs no saved tensor.

**PHASE E — log-weights IN PLACE over `delta`, then the write normalization.** A
max-subtracted `exp` / sum over the support, with `has_mass` and `has_weight` guards
so an empty support produces exact zeros rather than `0/0`.

## <a id="bf16-support-boundary"></a>10. HAZARD — support is defined at the native bf16 storage boundary

The support bit is packed from `bf16_nonzero(read) || bf16_nonzero(write)`, i.e.
from the values AS STORED, **not** from the pre-quantization solve. This makes the
scheduler and the route stream ONE authority.

Union has one scheduling stream for both roles, so it is the UNION of post-cast
nonzeros that is packed: either side may create harmless false-live work, but **a
numerically live stored factor is never skipped**. Testing the pre-cast float
instead flips that guarantee — it silently drops work the consumer will still read —
and nothing in the build complains.

## <a id="epilogue-butterfly"></a>11. The epilogue — packing the 32-token support word

Lanes holding the same column in one physical warp differ only in bits at or above
`LW`, so a `__shfl_xor_sync` butterfly over exactly those bits ORs across the logical
warps IN PLACE — no shared atomics, no zero-init pass. Logical warp 0 of each
physical warp then stages its words, one `__syncthreads` follows, and warp 0 ORs the
`BLOCK_WARPS` staged copies into the output.

## <a id="backward"></a>12. The backward — the α-entmax Jacobian action through the union split

**No sort and no scan: this is a THREE-REDUCTION elementwise pass.** It shares the
forward's block/warp decomposition, width classes, striped
access, stride expression and launch bounds, and it **RECOMPUTES `delta`** rather
than reloading a saved tensor (cheaper than a third logit-sized read).

**SUPPORT IS AN INPUT** here — unpacked from the same frozen `support_words` the
forward emitted — so the two directions cannot disagree about the support set by
construction.

On support the action is `diag(s) - s s^T / (1.s)` — the α-entmax Jacobian action,
with `s = 1` (α = 2) or `s = sqrt(p) = p^(2-alpha)` (α = 1.5) — plus the write side's
second rank-1 (the softmax form inside `d_log_weight`) and the diagonal gate terms.

### Dead directions are exactly zero, by multiplication

Off support: `s = 0`, `read_values = 0` and `write_values = 0` (bf16 of exact
zeros), hence

```
d_log_weight      = write_values * (...)                                   = 0
d_midpoint_logits = s * (...)                                              = 0
d_delta           = d_read*read_values*write_gate - d_log_weight*read_gate = 0
```

so both `0.5*d_ml + d_delta` and `0.5*d_ml - d_delta` are exact zeros. **The gate
asserts the FORM (`rtol=0`), not a magnitude.**

### The tie guard

The VJP divides by the amplitude; at a TIE the amplitude is exactly 0, so the guard
substitutes 1 and the support mask zeroes the term. See the Wolfram sheet.

### Two measured re-read decisions

Both are register-pressure trades. MEASURED under `__launch_bounds__(128, 8)` at
`IPT = 8`:

| staged instead of re-read | measured cost |
|---|---|
| `write_values` / `d_write` across the two passes | 2·`IPT` live registers, **32 B stack frame** (spill) |
| `s` as a third live `IPT`-array | **24 B of spill** |

Re-reading is the cheaper trade on a kernel whose DRAM throughput is only ~30%. The first pass therefore carries NOTHING into the next but two scalars
(`local_dot` and the support mask), and `s` is recomputed from a re-read
`midpoint_values` in the final pass; one extra load per element buys the spill back
to zero.
