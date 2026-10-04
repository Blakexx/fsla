# `csrc/rola/src/owner/owner_fold.cuh` — the fold, fp16

Mirror doc for the owner carry's fold: a warp's write lists over a pass of the write pool into the resident state and
mass, in the declared fp16 arithmetic of the rebuild plan card (step 3). Its reference is
`tests/oracle/fold_reference.py` (`fold_state`), which mirrors every rounding; `measure/harness/bench_owner_skeleton.py
--check fold` holds the part to it per slot on both arms. Vocabulary: [`owner.md`](owner.md); the
pool rows: [`owner_fill.md`](owner_fill.md); the forms and entries: [`owner_lists.md`](owner_lists.md).

<a id="arithmetic"></a>

## The arithmetic

A leaf `(d, s)` — `d` a write-dense digit of level S (16 a CTA, the atom's M), `s` a list, a write digit of level R (8
a warp) — gains, per channel `c`, `sum_t A(t, d, s) v(t, c)` over the window's tokens, `v` the f16 values (a token's
largest channel at 1):

    F(t) = ((g sw_0) sw_1) vs,   m(t) = (g sw_0) sw_1                       fp32, the facts pass's own products
    x(t, d) = max(b_S, 255 x 2^-13) / 256                                     f16, exact; the floor on a write-dense S
    c(t, s) = f16(fp32(b_R F) x 2^(4 - u_s))                                  b_R at the floor on a write-dense R
    A(t, d, s) = f16(x c)                                                     one HMUL2
    mass:  c_m(t, s) = f16(fp32(b_R m) x 2^(4 - v_s)),  M(d, s) += sum_t x c_m

`F` and `m` are the facts pass's products in its order, so `b_R F <= 2^K` holds exactly for the window's exponent `K`
([`facts.md`](../facts/facts.md#lists)) and every `c` is at most 16. The state holds `st` in its list's unit: the
absolute state is `st x 2^(u_s - 4) x 256 / 255^2` (the dense digit carries `b_S / 256`, `c` the raw `b_R`).

<a id="units"></a>

## The units

The atom adds straight into fp32, so a list's state lives in the unit its window's `c` was formed at. Each window, before
its passes, a list's four lanes move its units from the facts pass's `(K_m, K_v)`:

    u' = max(K, u + ceil(log2 bound) - 100, -123)      (a dead list, K = -126, keeps u)
    st  *= 2^(u - u')                                   exact; zero below the normal range
    bound' = bound x 2^(u - u') + 512 x 16

Following `K` keeps `c`'s top at 16, so a list-window keeps fp16's full range below its strongest writer. The bound is a
scalar a list (no reduction): every add of a window is at most 512 tokens x 16 in its units, so the state never exceeds
`bound`, and `u'` is held so that `bound` in the new unit stays under 2^100 — a list whose gain collapses (a window
2^100 below the last) cannot overflow its state. `-123` keeps the prescale `2^(4 - u)` a normal fp32. The mass's unit
`v` moves by the same rule from `K_m`.

<a id="state"></a>

## The accumulators

Per warp: each list's 16 leaves x the channel group in fp32 (the HMMA's own C fragments, 128 registers). The
ACCUMULATE ARM is a compile-time choice (`FoldArm`): fp32 straight into the state (datacenter), or fp16 partials
(consumer), one a list and channel tile, joined into the fp32 state by FADD every 256 tokens (two passes) — at most
256 x 16 = 4096 in a partial. On the fp16 arm the 192 accumulator registers leave ~60 for a step's operands.

THE MASS lives in shared memory (`FoldTenants::mass`, 16 leaves x the CTA's 64 lists, a warp owning its lists'
columns): each step's mass atom writes a fresh fp32 fragment, added by FADD into the pass's four sums, which join the
columns at the pass's end. Every mass term is positive, so any accumulation chain's rounding biases den: MEASURED
(`flagship-dense`, median error over the mass) a call-long HMMA chain 7.2e-6, fp16 partials 2.2e-4, fp32 HMMA partials
a 256 tokens 9e-7, fresh fragments 3.6e-8. Out of registers it also gives the fp16 arm's steps their room.

<a id="stage"></a>

## The restage

After a pass's fill lands, two threads a row restage it for the operands: the dense digits to f16 — the byte under a
0x64 high byte read as `1024 + b`, one HFMA2 to `b / 256`, one HMNMX2 to the floor — at a 48-byte pitch, the values
copied to an 80-byte pitch (both conflict-free for an 8-row `ldmatrix`; the pool's 160-byte rows are not), and each
row's `F` and `m`. Rows past the pass, to a whole 16-token step, are zeroed, so a dense side's last step reads only
zeros there.

<a id="operands"></a>

## The operands

A step is 16 tokens. `ldmatrix.x4.trans` gives A = the dense digits (leaves g and g + 8, tokens 2 tq.., 2 tq + 8..) and
two more give B = the values (four n-tiles of 8 channels). `StepRows` maps the step's tokens to pass rows: on a dense
side they are the step's own (`e0 - first + t`); on a listed side the entries, a token past its run reading row 0 with
`c = 0`.

<a id="coefficient"></a>

## The coefficient

Lane (g, tq) forms list g's `c` at its four tokens (2 tq, 2 tq + 1, 2 tq + 8, 2 tq + 9) — two fp32 products and one
F2FP a pair — and two shuffles a list carry it to the A layout. The mass's `c_m` is formed beside it, and the mass's
B is exactly lane (g, tq)'s own pair for list g, so it takes no shuffle.

<a id="forms"></a>

## The form

MERGED, the fold's one form: every list over the step's tokens — every row of a dense side, or the warp union's
entries — the operands once, `A_j = x c_j` and four atoms a list, one mass atom a step. A token outside a list carries
a zero byte and so a zero `c`. A per-list form (each list over its own entries) was built and measured slower on every
sparse cell — its steps a warp a CTA-window, fp16 arm, forced: `nl64k-alt-k4` 17.8K against merged 9.8K,
`flagship-cohort-k16` 40.5K against 18.0K — because a list holds a step or two a 128-row pass and its chain is
latency-bound, while interleaving the lists to overlap them needed the fp16 arm's last registers (it spilled). It was
removed (`docs/internals/DELETIONS.md`); one pool a window (the plan's step 8) is what would change the comparison.

<a id="measured"></a>

## Measured

Gate (`--check fold`, `flagship-dense`, `-alt-k4`, `-alt-k4-L4096`, `-cohort-k16`; the per-slot error over the slot's
envelope `sum |A| |v|`): fp32 arm median 2-6e-8, max <= 9e-7 (bound 2^-14); fp16 arm median 3-8e-5, max <= 1.2e-3
(bound 2^-8); the units exact; mass median <= 4e-8, max <= 3e-7 on both. The floor's exposures on the cells: no risky
leaf (strongest writer > 10 octaves below its list's top), mass inflation <= 3.7e-4.

Per head, sm_86 under ncu (1.36 GHz), 4 heads (cohort 16), the part's lists and fill included:

| cell | fp32 arm | fp16 arm | the steps a warp a CTA-window against their tensor floor (ledger) |
|---|---|---|---|
| `nl64k-dense` | 14.0 ms | 9.3 ms | 72.7-74.5K / 67.6K (1.08-1.10x); 44.0K / 34.8K (1.26x) |
| `nl64k-alt-k4` | 4.8 ms | 4.1 ms | 13.9K and 10.2K: ~2 steps a 128-row pass, 3 passes a window |
| `flagship-cohort-k16` | 0.22 ms | 0.18 ms | |

A dense step is 148 instructions for 33 HMMAs on the fp16 arm (32 HMUL2, 16 shuffles, `c`'s products and packs, the
mass's FADDs): the FP32-pipe work beside the tensor pipe is the fp16 arm's 1.26x, which the probe (precomputed `c`,
1.06x) did not carry. The sparse cases' steps are a pass's few dependent steps, and the fixed parts are 40-55% of the
part (lists ~22%, fill ~20%); one pool a window and the next pass's fill under this pass's steps are the plan's step 8.
