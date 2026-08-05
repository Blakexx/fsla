"""Liveness + starvation gate for the consumer's COUNTER-GATED phase ordering (K3b).

THUNDER_RUN_SPEC.md §5 K3 moves the consumer's intra-tile phase ordering off
`__syncthreads()` and onto SMEM counters: the readout's consumption of the state
`S` is published by `kCtlReadDone` and the Gram's publication of `Gm` by
`kCtlGramDone`, both retired through `retire_item` and observed through
`eligible` (`consumer_kernel.cuh`), and the boundary fold's warp tiles become a
FLUID ITEM POOL claimed from `kCtlFoldClaim` instead of a strided warp loop.

WHY THIS FILE EXISTS AND WHY IT IS NOT THE ORACLE SUITE. A software barrier has a
failure mode a hardware barrier does not: it can HANG. The oracle suites
(`test_consumer_vs_oracle.py`, `test_consumer_smem_precision_gate.py`) gate the
VALUES, and they gate them thoroughly -- but a deadlocked kernel does not produce
a wrong value, it produces no value at all, and inside a pytest process that is an
un-attributable hang rather than a failure. So every cell below runs in a CHILD
PROCESS UNDER A WALL-CLOCK TIMEOUT, and a timeout is reported as a FAILED
LIVENESS CLAIM naming the cell.

Symbols: `BC` leafs per owner (one CTA = one owner); `BT` token tile; `DV` = `d_v`
the value width; `D` level count; `NM = BT/16` the tile's `m16` row blocks;
`NT_DV = DV/8` its `n8` column blocks; `kWarps = 8`; `P` the sequence-parallel
segment count.

THE STARVATION AXIS, and every cell below is chosen ON it rather than for coverage:

  (a) **more warps than items.** `BC = 16, DV = 32` gives `n_mt = 1` fold m-tile
      and `NT_DV = 4` n-tiles, i.e. FOUR fold items for EIGHT warps -- half the
      CTA claims nothing and must still retire both counters and reach the
      end-of-tile barrier. This is the cell that catches a retirement made
      conditional on having done work.
  (b) **no readout tiles for half the CTA.** `BT = 16, DV = 32` gives
      `NM * NT_DV = 4` output tiles, so warps 4-7 hold no `y` fragment at all and
      pass through phase A and phase C empty.
  (c) **tiles that are read-only, write-only and mixed within ONE launch.**
      Independent read/write densities make `is_read`/`is_write` vary tile by
      tile, so the CTA takes a different subset of the phase graph on successive
      iterations of the SAME tile loop -- which is where a counter that is not
      reset, or is reset on the wrong side of a barrier, shows up.
  (d) **many tiles.** `T = 521` at `BT = 16` is 33 tiles per owner, so a
      once-per-few-thousand-tiles race gets thousands of chances per cell rather
      than one.
  (e) **the `P > 1` three-pass path**, whose summary pass forces `is_read = false`
      on every tile -- the most one-sided phase graph the kernel has.

Each cell ALSO asserts its values against the fp64 oracle in the same child
process, so "it finished" is never the whole claim: a schedule that completes but
drops an item is a wrong value, and the fold's items write DISJOINT `S` elements
precisely so that which warp runs one cannot change a result.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import textwrap

import pytest
import torch

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="the V3 consumer is a CUDA kernel"),
]

#: Wall clock per child. The heaviest cell below is ~33 tiles x 6 owners x a few
#: hundred CTAs and runs in well under a second on the dev box; the budget is two
#: orders of magnitude of slack, so a timeout is a HANG and not a slow machine.
TIMEOUT_S = 240

#: The oracle bound. This file is not a precision gate -- `test_consumer_smem_
#: precision_gate.py` owns that -- so the bound here is the tier-1 bf16 value
#: bound, present as a tripwire for a DROPPED or DOUBLE-EXECUTED item, which
#: moves a result by ~100 %, not by ~1 %.
RTOL = 2e-2

_CHILD = textwrap.dedent(
    '''
    import os, sys, torch
    sys.path.insert(0, {repo!r})
    torch.cuda.set_per_process_memory_fraction(0.65)

    import rola
    assert os.path.realpath(rola.__file__).startswith(
        os.path.realpath({repo!r}) + os.sep), (
        "the K3 gate is testing a FOREIGN tree: rola resolved to "
        f"{{rola.__file__}} instead of under {repo!r}")
    from rola.ops.consumer import _consumer_forward
    from rola.ops.naive import naive_rola
    from rola.planner.schedule import Plan
    from rola.routing.types import (IndependentRouting, LeafMassDecay, LevelConfig,
                                    SoftmaxActivation, Topology)

    R = IndependentRouting(tied=False, read_activation=SoftmaxActivation(),
                           write_activation=SoftmaxActivation())

    def simplex(shape, p, g, device):
        x = torch.rand(shape, device=device, dtype=torch.float64, generator=g)
        if p < 1.0:
            x = x * (torch.rand(shape, device=device, dtype=torch.float64,
                                generator=g) < p)
        x = torch.where(x.sum(-1, keepdim=True) == 0, torch.ones_like(x), x)
        return x / x.sum(-1, keepdim=True)

    widths, BC, BT, d_v = {widths!r}, {BC!r}, {BT!r}, {d_v!r}
    norm, decay, prec = {norm!r}, {decay!r}, {prec!r}
    p_read, p_write, T, P = {p_read!r}, {p_write!r}, {T!r}, {P!r}
    B, H, device = 2, 3, "cuda"

    g = torch.Generator(device=device).manual_seed({seed!r})
    topo = Topology(levels=tuple(LevelConfig(w, R) for w in widths), d_v=d_v)
    rl = tuple(simplex((B, T, H, w), p_read, g, device) for w in widths)
    wl = tuple(simplex((B, T, H, w), p_write, g, device) for w in widths)
    gw = torch.rand(B, T, H, device=device, dtype=torch.float64, generator=g) + 0.5
    gr = (None if norm == "global"
          else torch.rand(B, T, H, device=device, dtype=torch.float64, generator=g) + 0.5)
    v = torch.randn(B, T, H, d_v, device=device, dtype=torch.float64, generator=g)
    dec = None
    if decay:
        dec = LeafMassDecay(dials=tuple(
            0.05 + 0.85 * torch.rand(H, w, device=device, dtype=torch.float64, generator=g)
            for w in widths))

    y_ref, s_ref = naive_rola(v, rl, wl, gr, gw, topo, dec, norm,
                              initial_state=None, output_final_state=True)

    plan = Plan(topology=topo, normalization=norm, BT=BT, BC=BC,
                warp_tile=(1, max(1, d_v // 32)), decay=dec is not None,
                mma_precision=prec)
    args = (v.to(torch.float32),
            tuple(t.to(torch.float32) for t in rl),
            tuple(t.to(torch.float32) for t in wl),
            None if gr is None else gr.to(torch.float32),
            gw.to(torch.float32), plan)
    y, state, schedule = _consumer_forward(*args, initial_state=None, decay=dec)
    if P > 1:
        from rola.routing.sequence_parallel import (SequenceParallelPlan,
            SequenceParallelPolicy, owner_slot_counts, select_sequence_parallelism)
        base = select_sequence_parallelism(plan, schedule,
                                           policy=SequenceParallelPolicy(workspace_cap=1 << 30))
        n_owners = schedule.owners.shape[0]
        Ps = tuple([P] * n_owners)
        sp = SequenceParallelPlan(P=Ps, limiting_gate=(None,) * n_owners, binding_gate=None,
            sms=base.sms, blocks_per_sm=base.blocks_per_sm, waves_before=base.waves_before,
            waves_after=0.0, base_traffic_bytes=0, roundtrip_bytes=0, roundtrip_seconds=0.0,
            base_traffic_seconds=0.0, ledger=base.ledger,
            n_slots=sum(owner_slot_counts(Ps)), BC=base.BC, cols=base.cols)
        y, state, _ = _consumer_forward(*args, schedule=schedule, sequence_parallel=sp,
                                        initial_state=None, decay=dec)
    torch.cuda.synchronize()

    def rel(a, r):
        return float((a.double() - r).abs().max()) / max(1e-30, float(r.abs().max()))

    ey, es = rel(y, y_ref), rel(state, s_ref)
    print("OK", ey, es)
    assert ey < {rtol!r} and es < {rtol!r}, ("item stream moved a value: "
                                             f"y {{ey:.3e}} state {{es:.3e}}")
    '''
)


def _run_cell(**kw):
    """Run one stress cell in a child process under a wall-clock timeout.

    The timeout IS the liveness gate. A counter-gated kernel that deadlocks
    leaves the child in an uninterruptible CUDA wait, so the child is killed and
    the cell is reported as a liveness failure naming its parameters -- which is
    the attribution a bare hang inside the pytest process cannot give.
    """
    repo = str(pathlib.Path(__file__).resolve().parents[2])
    src = _CHILD.format(repo=repo, rtol=RTOL, **kw)
    try:
        proc = subprocess.run([sys.executable, "-c", src], capture_output=True,
                              text=True, timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"LIVENESS FAILURE: the consumer did not complete in {TIMEOUT_S}s on cell "
            f"{kw}. The counter-gated phase ordering (kCtlReadDone / kCtlGramDone) or "
            f"the fold's claim pool (kCtlFoldClaim) has a warp waiting on a retirement "
            f"that never arrives -- see the deadlock-freedom argument at the head of "
            f"the tile body in consumer_kernel.cuh")
    assert proc.returncode == 0, (
        f"cell {kw} failed:\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")
    assert proc.stdout.startswith("OK"), proc.stdout


# ---------------------------------------------------------------------------
# (a)+(b)+(c)+(d) -- the imbalance sweep. `BC = 16, DV = 32` is the four-items-
# for-eight-warps cell; `BC = 64, DV = 64` is the flagship shape carried as the
# in-run control (every warp holds exactly one output tile and eight fold items).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "widths,BC,d_v",
    [((16,), 16, 32),          # (a) 4 fold items, 8 warps; (b) 4 output tiles
     ((4, 4, 4, 4), 16, 32),   # the same starvation at D = 4
     ((8, 8), 64, 32),         # 16 fold items, 4 output tiles
     ((8, 8), 64, 64)],        # the flagship shape -- the balanced control
)
@pytest.mark.parametrize("BT", [16, 32])
@pytest.mark.parametrize(
    "p_read,p_write",
    [(0.25, 0.25),   # (c) read-only, write-only and mixed tiles in ONE launch
     (1.0, 0.15),    # the read-dense freeze-arm pattern: most tiles read-only
     (1.0, 1.0)],    # every tile mixed -- no phase-subset variation at all
)
@pytest.mark.parametrize("decay", [False, True])
def test_item_stream_is_live_and_exact_under_role_imbalance(
        widths, BC, d_v, BT, p_read, p_write, decay):
    _run_cell(widths=widths, BC=BC, BT=BT, d_v=d_v, norm="global", decay=decay,
              prec="bf16", p_read=p_read, p_write=p_write,
              T=521, P=1, seed=8100 + BC + BT + d_v + int(decay))


# ---------------------------------------------------------------------------
# (e) -- the `P > 1` three-pass path. Its SUMMARY pass forces `is_read = false`
# on every tile, so phase A is empty on every iteration and the counters carry a
# phase graph with one side permanently absent. `raw` is included because it also
# removes the mass column, i.e. the only per-thread work phase A has left.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("P", [2, 3])
@pytest.mark.parametrize("norm", ["raw", "global"])
@pytest.mark.parametrize("decay", [False, True])
def test_item_stream_is_live_on_the_segmented_path(P, norm, decay):
    _run_cell(widths=(8, 8), BC=64, BT=16, d_v=64, norm=norm, decay=decay,
              prec="bf16", p_read=1.0, p_write=0.3,
              T=521, P=P, seed=8700 + P + int(decay))


# ---------------------------------------------------------------------------
# The tf32x3 exactness arm, at the starvation shape. It is a different
# instantiation of the same control flow (`READ_PREC`/`FOLD_PREC` change the
# staged element type, not the phase graph), and it is here so that a counter
# edit that compiles only on the narrow arm cannot ship.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("BC,d_v", [(16, 32), (64, 64)])
def test_item_stream_is_live_on_the_tf32x3_arm(BC, d_v):
    _run_cell(widths=((16,) if BC == 16 else (8, 8)), BC=BC, BT=16, d_v=d_v,
              norm="global", decay=True, prec="tf32x3",
              p_read=0.4, p_write=0.4, T=257, P=1, seed=8900 + BC + d_v)
