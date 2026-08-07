"""TIER 2 — THE ENDGAME EXECUTOR gate (P26).

The executor's walk: membership = streamed fine scan (fixed-level token rows
AND the token-granular slice-union bit), stable ascending ballot into groups
of ``MMA_K_QUANTUM`` gathered tokens (ALWAYS FULL except each segment's final
flush), per-visit atom masks OR'd from the per-token atom-union bytes, and the
compact-into-registers atom walk on every leaf-axis consumer. The validation
target is the canonical fp64 oracle ``naive_rola`` — never a fresh reference
(spec §2.1).

Correctness criteria, and the DERIVED tolerance each cell is held to:

1. **Oracle agreement everywhere.** bf16 is the only storage format, so the
   derived budget is the bf16 family bound ``BF16_RTOL = 1e-2`` (u_bf16 = 2^-8
   = 3.9e-3 per requantization; the P14 remedy derivation measured H in
   [0.7, 1.4] on this kernel family, so ~0.5x of budget is the expected
   scale). There is no tf32x3 arm to compare against: the fp64 oracle IS the
   exactness reference (ruling 2026-08-06).
2. **Degeneracy = identity grouping.** Dense (full-support) inputs make every
   membership word all-ones and every atom mask FULL, so groups are
   consecutive 16-token blocks and the statically-unrolled dense arms run.
   Observable teeth: single-owner runs are bit-DETERMINISTIC across
   invocations (one CTA, one emission order), and the mutant recipe below
   breaks dense cells exactly as it breaks sparse ones.
3. **Causality/stability.** ``t_idx`` ascending IS the causality requirement
   (the intra Gram's mask is order-relative). The gate's teeth are the mutant
   pair (dropped-token, permuted-group) plus the boundary-straddle mutant
   (a member word split across two groups mis-carried) — each must fail this
   suite when injected into the kernel (build-cycle mutants, recorded in the
   workflow evidence, not shipped gates).
4. **Adversarial corners:** cold read (never-written owners), ragged tail
   (T % 32, T % 16), group boundary (memberships straddling word splits),
   decay (fp16 application; ticks-ride-writes exactness of the fold skip),
   union-visit semantics (read-only and write-only tokens interleaved),
   P in {2, 4} word-range segments.
"""

from __future__ import annotations

import pytest
import torch

from rola.ops.consumer import _consumer_forward
from rola.ops.naive import naive_rola
from rola.planner.schedule import Plan
from rola.routing.types import (
    EntmaxActivation,
    IndependentRouting,
    LeafMassDecay,
    LevelConfig,
    Topology,
)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="V3 consumer is a CUDA kernel"),
]

#: The DERIVED bf16 family budget (see the module docstring). Recorded here, not
#: imported, so the gate states its own bound.
BF16_RTOL = 1e-2

_ENTMAX_BOTH = IndependentRouting(
    tied=True, read_activation=EntmaxActivation(1.5), write_activation=EntmaxActivation(1.5))


def _relative(got: torch.Tensor, ref: torch.Tensor) -> float:
    ref = ref.to(torch.float64)
    return float((got.to(torch.float64) - ref).norm() / ref.norm().clamp_min(1e-30))


def _sparse_amps(B, T, H, w, p, gen):
    a = torch.rand(B, T, H, w, generator=gen, device="cuda", dtype=torch.float64)
    keep = torch.rand(B, T, H, w, generator=gen, device="cuda") < p
    keep[..., 0] |= ~keep.any(-1)  # a simplex row needs one live digit
    a = a * keep
    return (a / a.sum(-1, keepdim=True).clamp_min(1e-30)).to(torch.float32)


class Cell:
    def __init__(self, widths, BC, d_v, B, T, H, p_read, p_write, seed, *,
                 decay_on=False, hot_dials=False, nonzero_m0=False):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        self.topology = Topology(
            levels=tuple(LevelConfig(width=w, routing=_ENTMAX_BOTH) for w in widths), d_v=d_v)
        self.read = tuple(_sparse_amps(B, T, H, w, p_read, gen) for w in widths)
        self.write = tuple(_sparse_amps(B, T, H, w, p_write, gen) for w in widths)
        self.g_write = torch.rand(B, T, H, generator=gen, device="cuda") + 0.5
        self.v = torch.randn(B, T, H, d_v, generator=gen, device="cuda")
        self.decay = None
        if decay_on:
            hi = 0.9 if hot_dials else 0.15
            self.decay = LeafMassDecay(dials=tuple(
                (hi * torch.rand(H, w, generator=gen, device="cuda")).to(torch.float64)
                for w in widths))
        self.initial_state = None
        if nonzero_m0:
            cols = d_v + 1
            self.initial_state = torch.randn(
                B, H, self.topology.N, cols, generator=gen, device="cuda").abs()
        self.plan = Plan(topology=self.topology, normalization="global", BC=BC,
                         warp_tile=(1, max(1, d_v // 32)), decay=decay_on)

    def oracle(self):
        y, s = naive_rola(self.v, self.read, self.write, None, self.g_write,
                          self.topology, self.decay, "global",
                          initial_state=self.initial_state, output_final_state=True)
        return y, s

    def run(self, **kw):
        return _consumer_forward(self.v, self.read, self.write, None, self.g_write,
                                 self.plan, decay=self.decay,
                                 initial_state=self.initial_state, **kw)


def _check(cell: Cell, **kw):
    y_ref, s_ref = cell.oracle()
    y, s, _ = cell.run(**kw)
    assert _relative(y, y_ref) < BF16_RTOL, f"y off: {_relative(y, y_ref):.3e}"
    assert _relative(s.view_as(s_ref), s_ref) < BF16_RTOL, \
        f"state off: {_relative(s.view_as(s_ref), s_ref):.3e}"


# ---------------------------------------------------------------------------
# 1. Degeneracy: dense inputs = identity grouping (full masks, dense arms)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("widths,BC", [((16,), 16), ((8, 8), 64)])
@pytest.mark.parametrize("decay_on", [False, True])
@pytest.mark.parametrize("d_v", [32, 64])
def test_dense_identity_grouping(widths, BC, decay_on, d_v):
    """Full support: every membership word is all-ones, every atom mask FULL,
    groups are consecutive 16-blocks. Oracle-graded, and bit-DETERMINISTIC
    across invocations at B=H=1 (single owner per bh => one CTA, one order)."""
    cell = Cell(widths, BC, d_v, 1, 64, 1, 1.0, 1.0, 11, decay_on=decay_on,
                nonzero_m0=True)
    _check(cell)
    y1, s1, _ = cell.run()
    y2, s2, _ = cell.run()
    assert torch.equal(y1, y2) and torch.equal(s1, s2), \
        "dense identity grouping must be deterministic bit for bit"


# ---------------------------------------------------------------------------
# 2. Sparse, against the canonical oracle (membership scan + atom walk teeth)
# ---------------------------------------------------------------------------

_GRID = [
    ((16,), 16),         # D=1, one owner: membership = the union word alone
    ((8, 8), 16),        # j=0, c=2, 4 slices: multi-slice atom bytes
    ((8, 8), 64),        # BC == N, one owner, QPO=4 atom walk
    ((4, 4, 4), 16),     # j=1: ONE fixed token row in the AND
    ((2, 3, 4, 4), 16),  # j=2: TWO fixed rows, non-power-of-two widths
]


@pytest.mark.parametrize("widths,BC", _GRID)
@pytest.mark.parametrize("p_read,p_write", [(0.4, 0.3), (0.12, 0.1)])
def test_sparse_matches_the_canonical_oracle(widths, BC, p_read, p_write):
    # T=96 at density ~0.1: an owner's members straddle all source words, groups
    # genuinely mix words, and word-split straddles occur (the assembler carry).
    _check(Cell(widths, BC, 64, 1, 96, 2, p_read, p_write, 23, nonzero_m0=True))


# ---------------------------------------------------------------------------
# 3. Ragged: global tail, final flush, near-empty membership, cold read
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("widths,BC", [((4, 4, 4), 16), ((8, 8), 64)])
@pytest.mark.parametrize("T", [17, 33, 65])
@pytest.mark.parametrize("p_read,p_write", [(0.4, 0.3), (0.06, 0.05)])
def test_ragged_tail_and_final_flush(widths, BC, T, p_read, p_write):
    _check(Cell(widths, BC, 64, 1, T, 2, p_read, p_write, 31))


def test_cold_read_never_written_owner():
    """Write support confined to a token prefix; reads over the whole range hit
    owners whose state was never written (zero state; empty write masks)."""
    cell = Cell((8, 8), 16, 64, 1, 96, 2, 0.4, 0.3, 37)
    for lvl in cell.write:
        lvl[:, 48:] = 0.0
        lvl[:, 48:, :, 0] = 1.0  # keep simplex rows valid but concentrated
    _check(cell)


# ---------------------------------------------------------------------------
# 4. The decay arm: fp16 application + ticks-ride-writes atom-skip exactness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("widths,BC", [((8, 8), 16), ((4, 4, 4), 16), ((8, 8), 64)])
@pytest.mark.parametrize("hot", [False, True])
def test_decay_matches_the_oracle(widths, BC, hot):
    _check(Cell(widths, BC, 64, 1, 65, 2, 0.4, 0.3, 43, decay_on=True,
                hot_dials=hot, nonzero_m0=True))


# ---------------------------------------------------------------------------
# 5. Union-visit semantics: disjoint read-only / write-only token stretches
# ---------------------------------------------------------------------------

def test_union_visit_interleaving():
    """Alternating stretches where only one side is live: groups mix pure
    readers and pure writers; per-side masks and flags must sort them out."""
    cell = Cell((8, 8), 16, 64, 1, 96, 2, 0.5, 0.5, 47, nonzero_m0=True)
    for lvl in cell.read:
        lvl[:, 0::4] = 0.0
        lvl[:, 0::4, :, 0] = 1.0
    for lvl in cell.write:
        lvl[:, 1::4] = 0.0
        lvl[:, 1::4, :, 1] = 1.0
    _check(cell)


# ---------------------------------------------------------------------------
# 6. Composition with P (word-range segments) and dv=32
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("P", [2, 4])
@pytest.mark.parametrize("decay_on", [False, True])
def test_composes_with_P(P, decay_on):
    """P > 1 with WORD-RANGE segments: every pass rides the endgame walk (the
    summary pass ballots the write side alone), every segment flushes ragged."""
    from rola.routing.sequence_parallel import (
        SequenceParallelPlan,
        SequenceParallelPolicy,
        owner_slot_counts,
        select_sequence_parallelism,
    )
    cell = Cell((8, 8), 16, 64, 1, 96, 2, 0.4, 0.3, 71, decay_on=decay_on,
                nonzero_m0=True)
    y_ref, s_ref = cell.oracle()
    _, _, schedule = cell.run()
    base = select_sequence_parallelism(
        cell.plan, schedule, policy=SequenceParallelPolicy(workspace_cap=1 << 30))
    n_owners = int(schedule.owners.shape[0])
    Ps = tuple([P] * n_owners)
    sp = SequenceParallelPlan(
        P=Ps, limiting_gate=(None,) * n_owners, binding_gate=None,
        sms=base.sms, blocks_per_sm=base.blocks_per_sm,
        waves_before=base.waves_before, waves_after=0.0,
        base_traffic_bytes=0, roundtrip_bytes=0, roundtrip_seconds=0.0,
        base_traffic_seconds=0.0, ledger=base.ledger,
        n_slots=sum(owner_slot_counts(Ps)), BC=base.BC, cols=base.cols)
    y, s, _ = cell.run(schedule=schedule, sequence_parallel=sp)
    assert _relative(y, y_ref) < BF16_RTOL
    assert _relative(s.view_as(s_ref), s_ref) < BF16_RTOL


@pytest.mark.parametrize("p_read,p_write", [(0.4, 0.3), (0.12, 0.1)])
def test_dv32(p_read, p_write):
    _check(Cell((8, 8), 16, 32, 1, 96, 2, p_read, p_write, 53))
