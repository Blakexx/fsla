"""V3 consumer **algorithm** gate -- CPU-only, fp64, no GPU required.

This is a faithful transcription of ``cuda/src/v3_consumer.cu``'s *control
structure* into Python, run in float64 and compared against the canonical Stage 1
oracle ``naive_rola``. It exists to separate the two failure modes that the
CUDA gate (``test_rola_v3_consumer.py``) cannot tell apart:

* the **algorithm** is wrong -- the owner rectangle's ``(base, stride, modulus)``
  digit lowering, the ``j+1``-word bitmap AND candidate walk, the ordered epoch
  discipline, the causal ``jj <= t`` intra bound, or the split-K sum over owners;
* the **CUDA transcription** is wrong -- fragment layout, synchronisation, a race,
  a fp32 accumulation artefact.

Running the same algorithm in fp64 on the CPU pins the first class to ``~1e-16``.
A CUDA disagreement above the fp32 tolerance is then attributable to the second
class, and this file is the thing that makes that inference sound.

**It does NOT discharge the CUDA gate** and is not a substitute for it. It is also
NOT a second reference: the comparison target is the canonical oracle only (spec
§2.1's prohibition on validating against a freshly written reference), and the
emulator below is the *subject*, not the standard.

**Maintenance contract:** ``emulate()`` mirrors ``v3_consumer_forward_kernel``. A
change to the kernel's control structure that is not mirrored here makes this gate
test a stale algorithm -- which is still a real signal (it would then disagree with
the CUDA gate), but the intent is that they move together.

**P26 migration note (2026-08-06): the tolerance dropped from EXACT to
BF16_RTOL.** Before P26 this file forced the schedule's amplitude storage to
``amp_dtype=torch.float64`` specifically so precision and algorithm were
decoupled: any disagreement with the oracle at ``EXACT = 1e-10`` was
attributable to the *algorithm*, never to a storage rounding. ``Plan`` no
longer exposes an ``amp_dtype`` override -- ``level_amp_dtype`` now hardcodes
bf16 for every SPANNED level unconditionally (the family ruling, 2026-08-04) --
so a schedule built from this file's ``Plan`` always carries real bf16
quantization on its spanned amplitude stream, the same as the CUDA kernel's.
There is no longer a way to build a full-precision schedule to decouple
against, so the achievable tolerance is the bf16 family bound
``BF16_RTOL = 1e-2``, not ``EXACT``. This does not erase the file's
discriminating power for what it exists to catch: every property under test
here (owner-rectangle digit lowering, the causal ``jj <= t`` Gram, the ordered
epoch, the token-sequential decay walk) is a STRUCTURAL claim whose violation
is an O(1) disagreement (see e.g. ``test_decay_matches_oracle``'s docstring),
easily separated from a ~1e-2 rounding floor -- the file just can no longer
also certify the algorithm to fp64 precision on top of that.
"""

from __future__ import annotations

import pytest
import torch

from rola.ops.consumer import level_descriptors
from rola.ops.naive import naive_rola
from rola.planner.schedule import (
    TOKEN_WORD,
    Plan,
    build_clock_stream,
    build_schedule,
    unpack_bits,
)
from rola.routing.types import (
    IndependentRouting,
    LeafMassDecay,
    LevelConfig,
    SoftmaxActivation,
    Topology,
)

#: The derived bf16 family budget (see the module docstring's P26 note) -- the
#: one tolerance available now that the schedule's spanned-level amplitude
#: storage is unconditionally bf16.
BF16_RTOL = 1e-2

_ROUTING = IndependentRouting(
    tied=False, read_activation=SoftmaxActivation(), write_activation=SoftmaxActivation())


def emulate(v, read_levels, write_levels, g_read, g_write, plan, initial_state,
            decay=None, write_levels_stored=None):
    """Transcription of ``v3_consumer_forward_kernel`` (``v3_consumer.cu``), fp64.

    Deliberately written in the kernel's terms -- per-owner register state ``M``,
    a candidate walk over tiles, an ordered read-then-write epoch -- rather than in
    the shortest Python that would give the same answer, because the point is to
    exercise the kernel's *structure*.

    Under ``decay`` the epoch splits differently and that difference is the whole
    of increment 2: the state evolves *within* a mixed tile, so the "carry against
    a fixed tile-entry state, plus a causal Gram" decomposition stops holding.
    The exact repair used here (and in the kernel) is the **token-sequential
    walk** -- ``keep -> deposit -> read`` per token, in order -- which is
    ``O(BT * BC)`` and therefore not merely exact but *cheaper* than the ``O(BT^2 *
    BC)`` Gram it replaces. The two-panel factorization that §12 rule 1 guards
    exists only to batch read rows into an MMA against a state held fixed across
    them; a scalar readout has nothing to batch (findings §3.1).
    """
    topo = plan.topology
    B, T, H, d_v = v.shape
    BH, BT, BC = B * H, TOKEN_WORD, plan.BC
    N = topo.N
    glob = plan.normalization == "global"
    cols = d_v + (1 if glob else 0)

    sched = build_schedule(read_levels, write_levels, g_read, g_write, plan)
    # The clock stream is built by a SEPARATE function that never sees a bitmap
    # (P5.20.5 finding 2); tile liveness above stays keyed on the live deposit.
    clock_amps = (None if decay is None else build_clock_stream(
        write_levels if write_levels_stored is None else write_levels_stored,
        write_levels, plan))
    desc = level_descriptors(topo, plan)
    geom = plan.geometry
    j, c, ns = geom.first_spanned_level, geom.slice_digits, geom.n_slices

    n_tiles = sched.n_twords
    read_bits = unpack_bits(sched.read_tile_bits, n_tiles)      # [BH, sum(widths), n_tiles]
    write_bits = unpack_bits(sched.write_tile_bits, n_tiles)
    read_un = unpack_bits(sched.read_slice_union, n_tiles)      # [BH, n_slices, n_tiles]
    write_un = unpack_bits(sched.write_slice_union, n_tiles)

    v_f = v.permute(0, 2, 1, 3).reshape(BH, T, d_v).double()
    state = (torch.zeros(BH, N, cols, dtype=torch.float64) if initial_state is None
             else initial_state.reshape(BH, N, cols).double().clone())
    num = torch.zeros(BH, T, d_v, dtype=torch.float64)
    den = torch.zeros(BH, T, dtype=torch.float64)

    for row in range(sched.owners.shape[0]):
        bh, owner = int(sched.owners[row, 0]), int(sched.owners[row, 1])
        # --- CTA init: decompose the owner id into its per-level base digits ----
        sl, rest = owner % ns, owner // ns
        base = []
        for level in range(topo.D):
            if level < j:
                base.append((rest // desc["owner_stride"][level]) % desc["level_width"][level])
            elif level == j:
                base.append(sl * c)
            else:
                base.append(0)

        # --- CTA init: per-leaf survival, computed once per owner ---------------
        # `rate[i] = prod_l delta_l[d_l(i)]` is a plain product of DIALS (spec §2),
        # not V2's AND over per-branch retentions -- there is no per-branch
        # retention in V3 to log-union, so V2's stable pairwise composition does
        # not carry over (findings §1, rule 2). `log1p(-rate)` is the stable form;
        # in the kernel `rate` accumulates in fp32, never bf16, because `1 - rate`
        # cancels catastrophically near 1 and `log1p` cannot repair a cancellation
        # that already happened inside the product.
        log_keep = None
        if decay is not None:
            head = bh % H
            rate = torch.ones(BC, dtype=torch.float64)
            for level in range(topo.D):
                dial = decay.dials[level][head].double()
                for i in range(BC):
                    rate[i] *= dial[
                        base[level]
                        + (i // desc["level_stride"][level]) % desc["level_modulus"][level]]
            log_keep = torch.log1p(-rate)

        # --- CTA init: candidate tiles = slice union AND the j fixed levels -----
        cand_r = read_un[bh, sl].clone()
        cand_w = write_un[bh, sl].clone()
        for level in range(j):
            cand_r &= read_bits[bh, desc["level_row_offset"][level] + base[level]]
            cand_w &= write_bits[bh, desc["level_row_offset"][level] + base[level]]

        # --- owner-resident state -------------------------------------------------
        M = state[bh, owner * BC:(owner + 1) * BC, :d_v].clone()      # [BC, d_v]
        z = state[bh, owner * BC:(owner + 1) * BC, d_v].clone() if glob else None

        for tile in range(n_tiles):
            is_r, is_w = bool(cand_r[tile]), bool(cand_w[tile])
            if not (is_r or is_w):
                continue  # write-conditioned quiescence: skipping is bit-identical
            t0 = tile * BT

            def panel(amps, lo=j):
                """``A[t, i] = prod_{l >= lo} amp[l][tile, t, digit_l(i)]``.

                ``lo = j`` for the read/write panels, whose ``l < j`` factors hoist
                into the row scale. ``lo = 0`` for the clock, which is an exponent
                and therefore does not factor out multiplicatively.
                """
                out = torch.ones(BT, BC, dtype=torch.float64)
                for level in range(lo, topo.D):
                    digits = torch.tensor([
                        base[level]
                        + (i // desc["level_stride"][level]) % desc["level_modulus"][level]
                        for i in range(BC)])
                    out *= amps[level][bh, tile, :, :].double()[:, digits]
                return out

            def rowscale(amps, gains):
                """Gain times the ``l < j`` fixed-level factors -- constant over the
                owner's leafs, so it factors out of the panel as a row scale."""
                s = torch.zeros(BT, dtype=torch.float64)
                for t in range(BT):
                    token = t0 + t
                    if token >= T:
                        continue  # padded tail: structurally zero
                    s[t] = gains[bh, token].double() if gains is not None else 1.0
                    for level in range(j):
                        s[t] *= amps[level][bh, tile, t, base[level]].double()
                return s

            if is_r:
                Ar = panel(sched.read_amps)
                sr = rowscale(sched.read_amps, sched.g_read)
            if is_w:
                Aw = panel(sched.write_amps)
                sw = rowscale(sched.write_amps, sched.g_write)

            if decay is not None:
                # --- token-sequential walk: keep -> deposit -> read, in order -----
                # Exact under a state that evolves inside the tile, and O(BT * BC).
                # A read-only tile has `c = 0` on every leaf the owner holds (the
                # clock-live tile set is a subset of the write-live set), so
                # `keep = 1` exactly there and this reduces to the carry -- which is
                # why skipping a non-candidate tile stays bit-identical under decay.
                cpanel = panel(clock_amps, lo=0) if is_w else None
                for t in range(BT):
                    token = t0 + t
                    if token >= T:
                        break
                    if is_w:
                        keep = torch.exp(cpanel[t] * log_keep)
                        M *= keep[:, None]
                        if glob:
                            z *= keep
                        a = Aw[t] * sw[t]
                        M += a[:, None] * v_f[bh, token][None, :]
                        if glob:
                            z += a
                    if is_r:
                        # Inclusive read-after-write: the read at `t` sees the
                        # deposit at `t` with survival exactly 1 (spec §2).
                        num[bh, token] += sr[t] * (Ar[t] @ M)
                        if glob:
                            den[bh, token] += sr[t] * float(Ar[t] @ z)
                continue

            if is_r and is_w:
                G = Ar @ Aw.T                                             # leaf contraction
                G = G * torch.tril(torch.ones(BT, BT, dtype=torch.float64))  # causal jj <= t

            # --- epoch: reads first, against the tile-entry state ------------------
            if is_r:
                for t in range(BT):
                    token = t0 + t
                    if token >= T:
                        break
                    out = sr[t] * (Ar[t] @ M)
                    if is_w:
                        wgt = G[t, :t + 1] * sw[:t + 1]
                        out = out + sr[t] * (wgt @ v_f[bh, t0:t0 + t + 1])
                    num[bh, token] += out
                    if glob:
                        d = sr[t] * (Ar[t] @ z)
                        if is_w:
                            d = d + sr[t] * float((G[t, :t + 1] * sw[:t + 1]).sum())
                        den[bh, token] += d
            # --- epoch: writes second ----------------------------------------------
            if is_w:
                for t in range(BT):
                    token = t0 + t
                    if token >= T:
                        break
                    a = Aw[t] * sw[t]
                    M += a[:, None] * v_f[bh, token][None, :]
                    if glob:
                        z += a

        state[bh, owner * BC:(owner + 1) * BC, :d_v] = M
        if glob:
            state[bh, owner * BC:(owner + 1) * BC, d_v] = z

    y = num / (den[:, :, None] + 1e-5) if glob else num
    return y.view(B, H, T, d_v).permute(0, 2, 1, 3), state.view(B, H, N, cols)


def _simplex(shape, p_nonzero, generator):
    x = torch.rand(shape, dtype=torch.float64, generator=generator)
    if p_nonzero < 1.0:
        x = x * (torch.rand(shape, dtype=torch.float64, generator=generator) < p_nonzero)
    # An empty support at some level is unreachable from the producer (every
    # activation is a simplex); repair rather than test an impossible input.
    x = torch.where(x.sum(-1, keepdim=True) == 0, torch.ones_like(x), x)
    return x / x.sum(-1, keepdim=True)


def _dials(widths, H, generator, *, hot):
    """``delta in [0, 1)``, strict upper bound (spec §1).

    ``hot=True`` drives the per-level dials close to 1 so the leaf rate product
    approaches 1 and ``1 - rate`` cancels -- the regime the fp32 accumulation
    requirement in the §7.4 audit exists for. ``hot=False`` is the ordinary range.
    """
    if hot:
        return tuple(0.90 + 0.099 * torch.rand(H, w, dtype=torch.float64, generator=generator)
                     for w in widths)
    return tuple(0.05 + 0.85 * torch.rand(H, w, dtype=torch.float64, generator=generator)
                 for w in widths)


def _check(widths, BC, d_v, B, T, H, p_read, p_write, *, nonzero_m0, seed,
           decay_dials=None, quantize_stored=False):
    generator = torch.Generator().manual_seed(seed)
    topology = Topology(levels=tuple(LevelConfig(w, _ROUTING) for w in widths), d_v=d_v)
    read_levels = tuple(_simplex((B, T, H, w), p_read, generator) for w in widths)
    write_levels = tuple(_simplex((B, T, H, w), p_write, generator) for w in widths)
    g_write = torch.rand(B, T, H, dtype=torch.float64, generator=generator) + 0.5
    # `normalization` is always "global" now (Plan validates no other value), so
    # `g_read` is always None -- the kernel raises on a non-None `g_read` under
    # global normalization.
    g_read = None
    v = torch.randn(B, T, H, d_v, dtype=torch.float64, generator=generator)
    cols = d_v + 1
    initial_state = None
    if nonzero_m0:
        initial_state = 0.1 * torch.randn(
            B, H, topology.N, cols, dtype=torch.float64, generator=generator)
        # The mass column is a nonnegative accumulated write mass.
        initial_state[..., d_v] = initial_state[..., d_v].abs()

    decay = None if decay_dials is None else LeafMassDecay(dials=decay_dials)
    stored = None
    if quantize_stored:
        # A deliberately divergent stored representation: bf16 round-trip, which
        # can only flush a small live value TOWARD zero. P5.20.5 finding 2 -- the
        # deposit must survive a stored zero, so this is the fixture that would
        # catch a clock-keyed bitmap.
        stored = tuple(t.to(torch.bfloat16).to(torch.float64) for t in write_levels)

    y_ref, state_ref = naive_rola(
        v, read_levels, write_levels, g_read, g_write, topology, decay, "global",
        write_levels_stored=stored,
        initial_state=initial_state, output_final_state=True)
    plan = Plan(topology=topology, normalization="global", BC=BC,
                  warp_tile=(1, max(1, d_v // 32)),
                  decay=decay is not None)
    y, state = emulate(v, read_levels, write_levels, g_read, g_write, plan, initial_state,
                       decay=decay, write_levels_stored=stored)

    err_y = float((y - y_ref).abs().max()) / max(1e-30, float(y_ref.abs().max()))
    err_s = float((state - state_ref).abs().max()) / max(1e-30, float(state_ref.abs().max()))
    assert err_y < BF16_RTOL, f"algorithm disagrees with the canonical oracle: y relative {err_y:.3e}"
    assert err_s < BF16_RTOL, f"algorithm disagrees with the canonical oracle: state {err_s:.3e}"


# Each entry is a DIFFERENT owner rectangle -- `j` (first spanned level) and `c`
# (aligned sub-range width) are derived from `(widths, BC)` per §3, so this is the
# real structural axis, not a size sweep. The dead `BT=16` row (P26: `TOKEN_WORD`
# is a fixed 32-token machine word, no second value survives to compare against)
# is deleted; see `workflows/perf/p26_endgame.md` DELETIONS.
@pytest.mark.parametrize("widths,BC,p_read,p_write,label", [
    ((16,), 16, 0.4, 0.3, "D1 j=0"),
    ((8, 4), 16, 0.4, 0.3, "D2 j=0 sparse"),
    ((8, 4), 16, 1.0, 1.0, "D2 j=0 dense"),
    ((8, 8), 16, 0.4, 0.3, "D2 j=0, 4 owners"),
    ((8, 8), 64, 0.4, 0.3, "D2 BC == N, one owner"),
    ((4, 4, 4), 16, 0.5, 0.4, "D3 j=1, one fixed level"),
    ((4, 4, 4, 4), 16, 0.5, 0.4, "D4 j=2, two fixed levels (j+1 AND chain)"),
    ((2, 3, 4, 4), 16, 0.5, 0.4, "D4 j=2, non-power-of-two widths"),
    ((8, 4, 2), 16, 0.5, 0.4, "D3 mixed widths"),
    ((2, 2, 2, 2), 16, 0.6, 0.5, "D4 j=0, D at its cap"),
])
def test_algorithm_matches_oracle_structural_topologies(widths, BC, p_read, p_write, label):
    """§3/§5 control structure equals the canonical oracle across owner-rectangle
    shapes (the real axis this row set sweeps -- `normalization` is always
    `global` now, so there is nothing left to sweep it against)."""
    _check(widths, BC, 8, 2, 37, 3, p_read, p_write, nonzero_m0=False, seed=1234)


@pytest.mark.parametrize("widths,BC,p_read,p_write,nonzero_m0,label", [
    ((8, 4), 16, 0.4, 0.3, False, "sparse"),
    ((8, 4), 16, 1.0, 1.0, False, "dense"),
    ((4, 4, 4), 16, 0.4, 0.3, True, "fixed level, nonzero M_0"),
])
def test_algorithm_matches_oracle_global(widths, BC, p_read, p_write, nonzero_m0, label):
    """`global`: `num` and `den` accumulate separately and divide in an epilogue
    (D4.6) -- `den` is only complete after every owner has landed."""
    _check(widths, BC, 8, 2, 37, 3, p_read, p_write,
           nonzero_m0=nonzero_m0, seed=99)


def test_algorithm_carries_nonzero_initial_state():
    """P5.20.5 finding 3: with a nonzero incoming `M_0` a read-dead leaf still
    contributes `R * M_0`. Zero-route annihilation is a WRITE-side statement, so
    the carry path must be exercised from a nonzero state, where the distinction
    is visible."""
    _check((8, 4), 16, 8, 2, 37, 3, 0.4, 0.3, nonzero_m0=True, seed=7)


@pytest.mark.parametrize("T", [1, 8, 16, 17, 31, 32, 33, 64, 65])
def test_algorithm_token_tile_boundaries(T):
    """`TOKEN_WORD` edges: below one word, exactly one, one over, and across a
    word boundary. The amplitude stream is zero-padded past `T`, so a tail token
    must contribute exactly nothing -- checked from a nonzero `M_0` so that a
    spurious tail read would show up rather than multiplying zero."""
    _check((8, 4), 16, 8, 1, T, 2, 0.4, 0.3, nonzero_m0=True, seed=5)


# --- LeafMassDecay (increment 2) --------------------------------------------
# The dead `BT=16` row is deleted here too (same reason as the structural-topology
# matrix above); the `norm` parametrization collapses out (only `global` exists).


@pytest.mark.parametrize("widths,BC,label", [
    ((16,), 16, "D1 j=0"),
    ((8, 4), 16, "D2 j=0"),
    ((8, 8), 64, "BC == N, one owner"),
    ((4, 4, 4), 16, "D3 j=1, one fixed level"),
    ((4, 4, 4, 4), 16, "D4 j=2, two fixed levels"),
    ((2, 3, 4, 4), 16, "D4 non-power-of-two"),
])
@pytest.mark.parametrize("p_read,p_write", [(1.0, 1.0), (0.4, 0.3)])
def test_decay_matches_oracle(widths, BC, label, p_read, p_write):
    """The survival scan is *inclusive at* ``j = t``: the deposit at ``t`` is read
    at ``t`` with survival exactly 1. Off-by-one in either direction (decaying
    after the deposit, or reading before it) shows up here as an O(1) error, not a
    rounding one."""
    H = 3
    generator = torch.Generator().manual_seed(4242)
    _check(widths, BC, 8, 2, 37, H, p_read, p_write, nonzero_m0=True, seed=1234,
           decay_dials=_dials(widths, H, generator, hot=False))


@pytest.mark.parametrize("T", [1, 8, 16, 17, 31, 32, 33, 64, 65])
def test_decay_token_tile_boundaries(T):
    """Decay across word edges: survival must compose across the word boundary,
    which the token-sequential walk gets only because the owner's state is
    resident across words rather than reloaded per word."""
    H = 2
    generator = torch.Generator().manual_seed(77)
    _check((8, 4), 16, 8, 1, T, H, 0.4, 0.3, nonzero_m0=True, seed=5,
           decay_dials=_dials((8, 4), H, generator, hot=False))


def test_decay_near_unit_rate():
    """Dials near 1 drive the leaf rate ``prod_l delta_l`` toward 1, where
    ``1 - rate`` cancels catastrophically. fp64 here proves the *algorithm* is
    right in that regime; the fp32 accumulation requirement it implies for the
    kernel (§7.4 audit, rule 2's V3-specific note) is a separate claim that this
    test does not and cannot discharge."""
    H = 2
    generator = torch.Generator().manual_seed(909)
    _check((8, 4), 16, 8, 1, 37, H, 0.5, 0.5, nonzero_m0=True, seed=13,
           decay_dials=_dials((8, 4), H, generator, hot=True))


def test_decay_stored_clock_diverges_from_live_deposit():
    """The clock and the deposit come from genuinely different tensors.

    ``write_levels_stored`` is a bf16 round-trip of the live ``write_levels``, so
    the clock ``c[t,s]`` and the deposit ``W[t,s]`` differ in VALUE at essentially
    every entry -- which is what makes an emulator that confused the two visible.

    It does **not** reach P5.20.5 finding 2's condition (``stored == 0`` with
    ``live != 0``): measured on this exact fixture, 0 of 824 live-nonzero entries
    flush, because bf16's flush-to-zero threshold is ~9.2e-41 and simplex entries
    are ``O(1/width)`` (review finding F1). That condition is reached by direct
    injection in
    ``test_rola_v3_schedule.py::test_build_clock_stream_accepts_stored_zero_with_live_nonzero``."""
    H = 2
    generator = torch.Generator().manual_seed(31337)
    _check((8, 4), 16, 8, 2, 37, H, 0.5, 0.4, nonzero_m0=True, seed=21,
           decay_dials=_dials((8, 4), H, generator, hot=False), quantize_stored=True)
