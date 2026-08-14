"""Liveness + starvation gate for the consumer's VISIT BARRIER.

The consumer orders one visit's phases with ONE `__syncthreads()`, placed after
the Gram and before phase C and the fold. It carries both of the visit's
intra-CTA edges at once -- the readout's reads of the state `S` complete before
the fold's writes, and the Gram's stores to `Gm` are published to phase C -- and
the fold's items are claimed by a STATIC strided partition
(`for (c = warp; c < n_fold_items; c += kWarps)`) rather than a pool.

WHY THIS FILE EXISTS AND WHY IT IS NOT THE ORACLE SUITE. A CTA-wide barrier has a
failure mode a value gate cannot see: it HANGS. `__syncthreads()` is only safe
where every thread of the CTA reaches it, so the guarantee the whole visit body
rests on is that every predicate wrapping a barrier -- `is_read`, `is_write`,
`mixed`, `DECAY && is_write` -- is CTA-UNIFORM. A predicate that becomes
per-warp deadlocks the launch, and the oracle suites
(`test_consumer_vs_oracle.py`) gate the VALUES: a deadlocked kernel does not
produce a wrong value, it produces no value at all, and inside a pytest process
that is an un-attributable hang rather than a failure. So every cell below runs
in a CHILD PROCESS UNDER A WALL-CLOCK TIMEOUT, and a timeout is reported as a
FAILED LIVENESS CLAIM naming the cell.

Symbols: `BC` leafs per owner (one CTA = one owner); `DV` = `d_v` the value
width; `D` level count; `NT_DV = DV/8` the readout's `n8` column blocks;
`kWarps = 8`.

THE STARVATION AXIS, and every cell below is chosen ON it rather than for coverage:

  (a) **more warps than items.** `BC = 16, DV = 32` gives one fold m-tile and
      `NT_DV = 4` n-tiles, i.e. FOUR fold items for EIGHT warps -- half the CTA
      claims nothing under the static partition and must still reach the
      barrier. This is the cell that catches a barrier made conditional on
      having done work.
  (b) **no readout tiles for half the CTA.** `DV = 32` gives `NT_DV = 4` output
      tiles, so warps 4-7 hold no `y` fragment at all and pass through the
      readout and phase C empty.
  (c) **tiles that are read-only, write-only and mixed within ONE launch.**
      Independent read/write densities make `is_read`/`is_write` vary group by
      group, so the CTA takes a different subset of the phase graph on
      successive iterations of the SAME visit loop -- which is where a barrier
      placed inside a predicate that is not actually CTA-uniform shows up.
  (d) **many visits.** `T = 521` is 33 groups per owner, so a
      once-in-thousands race gets thousands of chances per cell rather than one.

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

#: The oracle bound. This file is not a precision gate -- `test_consumer_vs_
#: oracle.py` owns that -- so the bound here is the tier-1 bf16 value bound,
#: present as a tripwire for a DROPPED or DOUBLE-EXECUTED item, which moves a
#: result by ~100 %, not by ~1 %. (P26: `test_consumer_smem_precision_gate.py`,
#: the file this comment used to point at, is deleted -- its whole subject was
#: the dead `mma_precision`/`raw` axis; see `workflows/perf/p26_endgame.md`
#: DELETIONS.)
RTOL = 2e-2

_CHILD = textwrap.dedent(
    '''
    import os, sys, torch
    sys.path.insert(0, {repo!r})
    torch.cuda.set_per_process_memory_fraction(0.65)

    import rola
    assert os.path.realpath(rola.__file__).startswith(
        os.path.realpath({repo!r}) + os.sep), (
        "the barrier gate is testing a FOREIGN tree: rola resolved to "
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

    widths, BC, d_v = {widths!r}, {BC!r}, {d_v!r}
    decay = {decay!r}
    p_read, p_write, T = {p_read!r}, {p_write!r}, {T!r}
    B, H, device = 2, 3, "cuda"

    g = torch.Generator(device=device).manual_seed({seed!r})
    topo = Topology(levels=tuple(LevelConfig(w, R) for w in widths), d_v=d_v)
    rl = tuple(simplex((B, T, H, w), p_read, g, device) for w in widths)
    wl = tuple(simplex((B, T, H, w), p_write, g, device) for w in widths)
    gw = torch.rand(B, T, H, device=device, dtype=torch.float64, generator=g) + 0.5
    # `normalization` is always "global" now, so `g_read` is always None.
    gr = None
    v = torch.randn(B, T, H, d_v, device=device, dtype=torch.float64, generator=g)
    dec = None
    if decay:
        dec = LeafMassDecay(dials=tuple(
            0.05 + 0.85 * torch.rand(H, w, device=device, dtype=torch.float64, generator=g)
            for w in widths))

    y_ref, s_ref = naive_rola(v, rl, wl, gr, gw, topo, dec, "global",
                              initial_state=None, output_final_state=True)

    plan = Plan(topology=topo, normalization="global", BC=BC,
                warp_tile=(1, max(1, d_v // 32)), decay=dec is not None)
    args = (v.to(torch.float32),
            tuple(t.to(torch.float32) for t in rl),
            tuple(t.to(torch.float32) for t in wl),
            None if gr is None else gr.to(torch.float32),
            gw.to(torch.float32), plan)
    y, state, schedule = _consumer_forward(*args, initial_state=None, decay=dec)
    torch.cuda.synchronize()

    def rel(a, r):
        return float((a.double() - r).abs().max()) / max(1e-30, float(r.abs().max()))

    ey, es = rel(y, y_ref), rel(state, s_ref)
    print("OK", ey, es)
    assert ey < {rtol!r} and es < {rtol!r}, ("the visit's phase schedule moved a "
                                             f"value: y {{ey:.3e}} state {{es:.3e}}")
    '''
)


def _run_cell(**kw):
    """Run one stress cell in a child process under a wall-clock timeout.

    The timeout IS the liveness gate. A kernel whose barrier is reached by only
    part of the CTA deadlocks, leaving the child in an uninterruptible CUDA
    wait, so the child is killed and the cell is reported as a liveness failure
    naming its parameters -- which is the attribution a bare hang inside the
    pytest process cannot give.
    """
    repo = str(pathlib.Path(__file__).resolve().parents[2])
    src = _CHILD.format(repo=repo, rtol=RTOL, **kw)
    try:
        proc = subprocess.run([sys.executable, "-c", src], capture_output=True,
                              text=True, timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"LIVENESS FAILURE: the consumer did not complete in {TIMEOUT_S}s on cell "
            f"{kw}. Some thread of the CTA is not reaching the visit's "
            f"__syncthreads() -- a predicate wrapping a barrier (is_read, is_write, "
            f"mixed, DECAY && is_write) has stopped being CTA-uniform. See the "
            f"barrier argument in docs/internals/consumer_kernel.md#visit-barrier")
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
@pytest.mark.parametrize(
    "p_read,p_write",
    [(0.25, 0.25),   # (c) read-only, write-only and mixed tiles in ONE launch
     (1.0, 0.15),    # the read-dense freeze-arm pattern: most tiles read-only
     (1.0, 1.0)],    # every tile mixed -- no phase-subset variation at all
)
@pytest.mark.parametrize("decay", [False, True])
def test_the_visit_barrier_is_live_and_exact_under_role_imbalance(
        widths, BC, d_v, p_read, p_write, decay):
    if decay:
        pytest.skip("decay retired from the operator; this battery retires "
                    "with the old consumer (see docs/internals/chunk/chunk.md)")
    _run_cell(widths=widths, BC=BC, d_v=d_v, decay=decay,
              p_read=p_read, p_write=p_write,
              T=521, seed=8100 + BC + d_v + int(decay))
