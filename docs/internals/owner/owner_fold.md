# `csrc/rola/src/owner/owner_fold.cuh` — the fold, fp16

Mirror doc for the owner carry's fold: a warp's write lists over the window's passes into the resident state and
mass, in the declared fp16 arithmetic of the rebuild plan card (step 3). Its reference is
`tests/oracle/fold_reference.py` (`fold_state`), which mirrors every rounding; `measure/harness/bench_owner_skeleton.py
--check fold` holds the part to it per slot. Vocabulary: [`owner.md`](owner.md); the
slot rows: [`owner_fill.md`](owner_fill.md); the unions and pool slots: [`owner_lists.md`](owner_lists.md).

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

<a id="pipeline"></a>

## The pass pipeline

A window's union is folded in passes of a write slot's rows ([`owner_fill.md`](owner_fill.md#pools)), through THREE
slots (`FoldPool`) under mbarriers, with no CTA barrier between passes: each slot has `landed` (every thread's fill
copies arrive as they land, `cp.async.mbarrier.arrive.noinc`) and `drained` (every thread arrives after its pass's LAST
LOADS, before those steps' MMAs). A warp's pass: wait its slot's landing; build its rows; then chunks of two steps --
the chunk's operands into registers, the slot signalled drained after the pass's last chunk's loads, the chunk's MMAs --
and after the MMAs of the chunk holding step `min(3, steps - 1)` its share of the fill of the pass two ahead, into the
slot the pass before drained (its drain waited first). Measured in the lab
(`sparse_pass`, KERNEL_STANDARDS 26 i-l): in-place operands need the third slot, and releasing a slot at its last load
frees the fill from the slowest warp's MMAs. Measured here: a chunk of two steps against three, dense 52.8 against 55.2
ms (4 heads), sparse the same; a one-step register stage 55.4. Placing the fill after the chunk's MMAs leaves no operand
live across it.

<a id="windows"></a>

## Across windows

The passes run across windows with no CTA barrier after the kernel's first: a FILL CURSOR (the next pass to fill, its
window, its index there and its global index), which every warp keeps alike from the windows' live counts, fills in
each pass's middle every pass up to two ahead whose window is this one or the next -- the next once its table is in
([`owner_lists.md`](owner_lists.md#write-lists)) -- so a window's first two passes are normally filled under the window
before; a window whose first passes the cursor did not reach (the one before had fewer than two passes) fills them at
its start. A fill waits its slot's drain by the pass three before. At a window's start each warp stages and ORs the
lists ahead, waits the window's table, and moves its own lists' units (`fold_window`, the list exponents loaded a window
ahead: warp-local, so no warp waits for another there). `--check fold` runs a sparse cell thinned too -- level R's rows
zeroed in every third window and all but every eighth token in the next -- so windows with no pass and with one pass
are gated.

<a id="rows"></a>

## The rows

A warp's rows for a pass are built at its start (`pass_rows`): each lane takes a slot of each 32 of the pass, reads its
token and the token's bit in the warp's union, and a ballot ranks the warp's tokens into its row buffer, in order -- a
dense level R's rows are the pass's own. `StepRows` maps a step's 16 entries to rows; an entry past the warp's count
reads row 0 (always filled) with `c = 0`, so no stale or never-written row is read. Building the rows word by word from
the union's ranks instead cost 3.65K cycles a warp a CTA-window at `nl64k-alt-k4` against 2.26K.

<a id="operands"></a>

## The operands

A step is 16 tokens, read in place from the slot. A dense level S's f16 digits give A by `ldmatrix.x4.trans` (leaves g
and g + 8, tokens 2 tq.., 2 tq + 8..); a sparse level S's bytes by `ldmatrix.x2.trans` over the tokens' 16-byte rows
-- lane (g, tq) then holds leaves 2g and 2g + 1 of tokens 2 tq and 2 tq + 1 -- and a PRMT and an HFMA2 a register to
`b / 256` (the byte under a 0x64 high byte read as `1024 + b`), so the accumulator's rows are leaves 2g and 2g + 1
(`acc_leaf`). Two more `ldmatrix.x4.trans` give B = the values (four n-tiles of 8 channels). The restage the first
fold ran (level S and the values copied to padded tiles behind a barrier) is gone.

<a id="coefficient"></a>

## The coefficient

Lane (g, tq) forms list g's `c` at its four tokens (2 tq, 2 tq + 1, 2 tq + 8, 2 tq + 9) from the slot's run and
`(F, m)` — on a dense level R one more product for the byte scale, two fp32 products and one F2FP a pair — and two
shuffles a list carry it to the A layout. The mass's `c_m` is formed beside it, and the mass's
B is exactly lane (g, tq)'s own pair for list g, so it takes no shuffle.

<a id="forms"></a>

## The form

MERGED, the fold's one form: every list over the step's tokens — every row of a dense level R (its f16 rows), or the
warp's rows of its union — the operands once, `A_j = x c_j` and four atoms a list, one mass atom a step. A token outside a list carries
a zero byte and so a zero `c`. A per-list form (each list over its own entries) was built and measured slower on every
sparse cell — its steps a warp a CTA-window, fp16 arm, forced: `nl64k-alt-k4` 17.8K against merged 9.8K,
`flagship-cohort-k16` 40.5K against 18.0K — because a list holds a step or two a 128-row pass and its chain is
latency-bound, while interleaving the lists to overlap them needed the fp16 arm's last registers (it spilled). It was
removed (`docs/internals/DELETIONS.md`).

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

Per head, sm_86 under ncu (1.36 GHz), 4 heads (16 at L 1024), the part's lists and fill included; the ledger in
cycles a warp a CTA-window against the steps' tensor floor (the busiest scheduler's paired steps x 1056, from the
cells' words):

| cell | time | ledger total / floor | passes in a window (8a) | first fold (barriers, restage) |
|---|---|---|---|---|
| `nl64k-dense` | 12.7 ms | 79.7K / 67.6K (1.18x): steps 59.0K, loads 9.1K, middle 8.7K | 13.2 ms, 81.3K | 14.4 ms, 93K |
| `nl64k-alt-k4` | 2.98 ms | 20.6K / 12.5K: steps 8.6K, middle 4.8K, rows 2.9K, loads 2.2K | 3.83 ms, 24.8K | 4.77 ms, 32.5K |
| `flagship-cohort-k16` | 0.120 ms | 46.9K / 45.9K: steps 22.3K, middle 9.3K, loads 5.6K | 0.138 ms, 53.1K | 0.22 ms, 83K |
| `flagship-both-k4` | 0.199 ms | | 0.217 ms | 0.24 ms |
| `flagship-alt-k4` | 0.0625 ms | | 0.080 ms | 0.107 ms |

The window boundary is gone (its lists 0.7-1.8K, from 5.4-12.4K); what it held moved in part into the middles' waits
-- a fill waits every warp's loads of the pass before, so a warp leads the slowest by at most a pass, and cohort's
imbalance (its busiest scheduler 1.7x the mean) is now its floor. Against step 8b's line (the rebuild plan card): dense
12.7 against ~12.0 ms, alt-k4 2.98 against ~2.8, cohort 0.120 against ~0.105.
