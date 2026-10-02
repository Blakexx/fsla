# `csrc/rola/src/owner/owner_fold.cuh` — the fold, first built form

Mirror doc for the owner carry's fold: a warp's write lists over a pass of the write pool, into the resident fp32
state, in the declared arithmetic of the rebuild plan card's fold line. This form is exact against its model
(`measure/harness/bench_owner_skeleton.py --check fold`, bit for bit, every step form) and MEASURED FAILING its budget;
it is the baseline the redesign on the same card ("THE FOLD REDESIGNED", 2026-10-02) is compared against.
Vocabulary: [`owner.md`](owner.md); the pool rows: [`owner_fill.md`](owner_fill.md).

<a id="arithmetic"></a>

## The arithmetic

A leaf (d, s) — d a write-dense digit (the fold's M), s a write-sparse list — gains, per channel c,
sum_t A[d, t] q[t, c], with q the token's 16-bit value (`v_hat` s8, `v_lo` u8). The coefficient is
A = min(RNE(u_wd · c · 2^(e − K + 16)), 65535), where c · 2^e = u_ws · F(t) is the token's list factor (F the token
factor) and K the leaf's scale: the largest bit length of the pass's live products u_wd · c · 2^e. A splits into a high
and a low u8 digit; the IMMAs sum hh = Σ hi·v_hat, mid = Σ (hi·v_lo + lo·v_hat), ll = Σ lo·v_lo; the state gains
ll · 2^(K − 16) + (256 hh + mid) · 2^(K − 8) in fp32.

The constants place every rounding on hardware: the token factor is stored as m · 2^(e − 50) and the leaf scale as
2^(−83 − K), so the product of a byte, the scale and the factor lands on fp32's denormal grid, where a value's bits are
its integer — one FMUL rounds the coefficient to nearest even, and one `I2IP` saturates and packs two.

<a id="factor"></a>

## The token factors

Once a pass, a thread a pool row: F = ((g · sw_S) · sw_R) · vs in fp32, as a 16-bit mantissa at an exponent in
[−79, 32] (below 2^−64 zero, above the top saturated), stored back over the row's first factor; and the dense digits'
bit lengths, biased, as halfwords in `WritePool::bits` — the scales' operand.

<a id="scale"></a>

## The scales

Each leaf's K is a maximum over the pass's live (token, leaf) products of bitlen(u_wd) + bitlen(c) + e, kept biased in
halfwords so the fp16 maximum (`HMNMX2`) is the integer one, two lanes an instruction. A lane loads its own token's
sixteen bit lengths by `ldmatrix.trans` and reduces leaves g and 8 + g over its token pairs; the quad and the two halves
reduce at the end, and each lane fetches the two leaves its A layout holds. Per list it is the list's own run; when the
lists share a run (merged, or a dense side) one pass over the run serves all eight, eight independent chains a step.

<a id="list"></a>

## A list's sums

One body for every list (eight inlined copies cost 3x the code and nothing in time): the leaves' scales, the first
step's operands whole, then the steps as bursts, each building the next step's coefficient digits beside its own IMMAs,
into hh, mid, ll; then the state gains them at the list's scales (`add_sums`, the state's list chosen at the add).

<a id="burst"></a>

## The burst

A step's sixteen IMMAs, the `v_hat` products first (hi into hh, lo into mid) then the `v_lo` ones (hi into mid, lo into
ll), so each value digit's last reader ends a group: the next step's `v_hat` digits reload into the same registers
after the first group, its raw `v_lo` matrices after the second (permuted under the next burst's first group), its two
coefficient halves between, the token after it loaded at the end; the coefficient digits alternate two sets. The token
slot is branch-free (past a run's end, the run's last row with a zero factor), so a step's loads schedule into the burst
before it.

<a id="operands"></a>

## The operands

A k-step is 32 tokens; `ldmatrix.x4.trans` gathers their pool rows: the value digits as B (channels on N, four n-tiles,
a byte permute a register for the even and odd channels) and the dense slice as A (leaves on M, one m-tile). Register
2h + l of A holds leaf 2g + l at tokens 16h + 8(b >> 1) + 2tq + (b & 1) (`a_token`) — the token order the B gather
gives, so any per-token operand placed by `a_token` lines up.

<a id="forms"></a>

## The forms

Per list: each list's entries in the pass. Merged (and a dense side): the warp union's entries, a token outside a list
carrying a zero byte and so a zero factor — the same sums. The cursors live in `FoldTenants`, a warp's own; a pass's
end is found by ballots over the sorted entries.

<a id="measured"></a>

## Measured

Per head over the part's own fill, 64K tokens x 4 heads, sm_86 at boost (2026-10-01/02): `nl64k-dense` +17.3 ms
against a 7.4 ms tensor floor; `nl64k-alt-k4` +5.5 ms. The phase ledger (`--ledger fold`), cycles a warp a window,
dense / alt-k4: bursts 49% / 17%, scales 17% / 18%, first steps 12% / 15%, state adds 10% / 17%, heads 5% / 13%, the
part's lists, fill, factors and sync 8% / 21%. Why: ~14 non-MMA instructions an IMMA (the per-element formation, the
per-pass scales and state add) and a list's fixed costs paid every 128-token pass. Probed the same days: a k32 IMMA
holds a scheduler's tensor pipe 16 cycles; a warp's ALU work overlaps its own IMMAs only under a uniform interleave;
FP32 and IADD3 instructions on the partner warp delay its IMMAs about a cycle each. The phase clock, compiled in and
unbound, costs ~6% here: timing builds must compile it out.
