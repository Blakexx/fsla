# `csrc/rola/src/owner/owner_fold.cuh` — the fold, fp16

Mirror doc for the owner carry's fold: a warp's write lists over a pass of the write pool into the resident state and
mass, in the declared fp16 arithmetic of the rebuild plan card (step 3). Its reference is
`tests/oracle/fold_reference.py` (`fold_state`), which mirrors every rounding; `measure/harness/bench_owner_skeleton.py
--check fold` holds the part to it per slot. Vocabulary: [`owner.md`](owner.md); the
pool rows: [`owner_fill.md`](owner_fill.md); the forms and entries: [`owner_lists.md`](owner_lists.md).

<a id="arithmetic"></a>

## The arithmetic

A leaf `(d, s)` — `d` a write-dense digit of level S (16 a CTA, the atom's M), `s` a list, a write digit of level R (8
a warp) — gains, per channel `c`, `sum_t A(t, d, s) v(t, c)` over the window's tokens, `v` the f16 values (a token's
largest channel at 1):

    F(t) = ((g sw_0) sw_1) vs,   m(t) = (g sw_0) sw_1                       fp32, the facts pass's plane
    x(t, d) = level S's stored f16, or its byte / 256                         f16, exact
    c(t, s) = f16(fp32(b_R F) x 2^(4 - u_s))                                  b_R on the byte scale
    A(t, d, s) = f16(x c)                                                     one HMUL2
    mass:  c_m(t, s) = f16(fp32(b_R m) x 2^(4 - v_s)),  M(d, s) += sum_t x c_m

`F` and `m` are the facts pass's products ([`facts.md`](../facts/facts.md#factor-products)), and `b_R` is its byte
scale — a sparse level R's u8 digit, a dense one's f16 times 255, exact — so `b_R F <= 2^K` holds exactly for the
window's exponent `K` ([`facts.md`](../facts/facts.md#lists)) and every `c` is at most 16. A dense level's floor is in
its stored rows ([`liveness_contract.md`](../facts/liveness_contract.md#floor)); the fold takes none. The state holds
`st` in its list's unit: the absolute state is `st x 2^(u_s - 4) / 255`, times `256 / 255` when level S is sparse (its
`x` carries `b_S / 256`).

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

Per warp: each list's 16 leaves x the channel group in fp32 (the HMMA's own C fragments, 128 registers), on every card
(ruled 2026-10-04: the fp32 fold's tensor cost is paid so a sparse pass's few steps are not held to a partial's join;
the readout keeps its fp16 form).

THE MASS lives in shared memory (`FoldTenants::mass`, 16 leaves x the CTA's 64 lists, a warp owning its lists'
columns): each step's mass atom writes a fresh fp32 fragment, added by FADD into the pass's four sums, which join the
columns at the pass's end. Every mass term is positive, so any accumulation chain's rounding biases den: MEASURED
(`flagship-dense`, median error over the mass) a call-long HMMA chain 7.2e-6, fp16 partials 2.2e-4, fp32 HMMA partials
a 256 tokens 9e-7, fresh fragments 3.6e-8.

<a id="stage"></a>

## The restage

After a pass's fill lands, two threads a row restage it for the operands: level S's digits as f16 at a 48-byte pitch
— a dense level's copied, a sparse level's byte under a 0x64 high byte read as `1024 + b` and one HFMA2 to `b / 256` —
the values copied to an 80-byte pitch (both conflict-free for an 8-row `ldmatrix`; the pool's rows are not), and each
row's `(F, m)`. Rows past the pass, to a whole 16-token step, are zeroed, so a dense side's last step reads only zeros
there.

<a id="operands"></a>

## The operands

A step is 16 tokens. `ldmatrix.x4.trans` gives A = the dense digits (leaves g and g + 8, tokens 2 tq.., 2 tq + 8..) and
two more give B = the values (four n-tiles of 8 channels). `StepRows` maps the step's tokens to pass rows: on a dense
side they are the step's own (`e0 - first + t`), a row past the pass taking `c = 0` (its f16 coefficients are a
stale row's, or never written); on a listed side the entries, a token past its run reading row 0 with `c = 0`.

<a id="coefficient"></a>

## The coefficient

Lane (g, tq) forms list g's `c` at its four tokens (2 tq, 2 tq + 1, 2 tq + 8, 2 tq + 9) — on a dense level R one more
product for the byte scale, two fp32 products and one F2FP a pair — and two shuffles a list carry it to the A layout. The mass's `c_m` is formed beside it, and the mass's
B is exactly lane (g, tq)'s own pair for list g, so it takes no shuffle.

<a id="forms"></a>

## The form

MERGED, the fold's one form: every list over the step's tokens — every row of a dense level R (its f16 rows), or the
warp union's entries — the operands once, `A_j = x c_j` and four atoms a list, one mass atom a step. A token outside a list carries
a zero byte and so a zero `c`. A per-list form (each list over its own entries) was built and measured slower on every
sparse cell — its steps a warp a CTA-window, fp16 arm, forced: `nl64k-alt-k4` 17.8K against merged 9.8K,
`flagship-cohort-k16` 40.5K against 18.0K — because a list holds a step or two a 128-row pass and its chain is
latency-bound, while interleaving the lists to overlap them needed the fp16 arm's last registers (it spilled). It was
removed (`docs/internals/DELETIONS.md`); one pool a window (the plan's step 8) is what would change the comparison.

<a id="measured"></a>

## Measured

Gate (`--check fold`; the per-slot error over the slot's envelope `sum |A| |v|`, bound 2^-14), one write format a
cell — `WriteRows<C, kHalfS, kHalfR>` ([`owner_fill.md`](owner_fill.md#pools)) is a template of the fold, chosen by the
host from the write rows' types:

| cell | level S, level R write rows | state median, max | mass median, max |
|---|---|---|---|
| `flagship-dense` | f16, f16 | 5.4e-8, 6.6e-7 | 7.2e-8, 2.6e-7 |
| `flagship-alt-k4` | f16, u8 | 2.5e-8, 4.9e-7 | 3.0e-8, 2.0e-7 |
| `flagship-alt-k4-L4096` | f16, u8 | 5.5e-8, 7.9e-7 | 5.0e-8, 3.3e-7 |
| `flagship-cohort-k16` | f16, u8 | 1.8e-8, 3.9e-7 | 7.1e-8, 2.3e-7 |
| `flagship-both-k4` | u8, f16 | 3.0e-8, 6.8e-7 | 1.3e-8, 1.7e-7 |
| `flagship-both-k4-L4096` | u8, f16 | 6.8e-8, 1.0e-6 | 4.3e-8, 3.3e-7 |

The units exact on all. No cell declares both write levels sparse: the u8-u8 format is built and gated by no cell.

Per head, sm_86 under ncu (1.36 GHz), 4 heads (16 at L 1024), the part's lists and fill included:

| cell | time | the steps a warp a CTA-window against their tensor floor (ledger) |
|---|---|---|
| `nl64k-dense` | 14.4 ms | 73.7K / 67.6K (1.09x); fill 8.3K, restage 5.6K |
| `nl64k-alt-k4` | 4.8 ms | 13.7K: ~2 steps a 128-row pass, 3 passes a window |
| `flagship-cohort-k16` | 0.22 ms | |
| `flagship-both-k4` | 0.24 ms | |

Against the factor-column format (14.0, 4.8, 0.22 ms) the dense cell pays 3% for its f16 level R: twice the run's bytes
in the fill and the byte-scale product in `c`.

The sparse cases' steps are a pass's few dependent steps, and the fixed parts are 40-55% of the
part (lists ~22%, fill ~20%); one pool a window and the next pass's fill under this pass's steps are the plan's step 8.
