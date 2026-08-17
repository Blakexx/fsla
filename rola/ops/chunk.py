"""The CHUNK consumer's LAUNCHES (operator adoption; docs/internals/chunk/) -- the kernel arm.

`rola_op`'s facade is untouched: this module is the private seam's kernel-arm
implementation, opinionated about what it supports and LOUD about what it
does not (the facade stays generic; capabilities return through the paging
and backward work, not
through silent degradation).

THE SPLIT: what crosses the EXTENSION BOUNDARY lives here -- the consumer launch, the
readout divide, the seeds, the two scans and the un-packing. What a DAG walk produces
or decides lives in the library: the fact primitives in :mod:`rola.engine.facts` (the
packing, the write-atom bitmap, the state plane, the arm censuses), the pure folds in
:mod:`rola.engine.rules` (the arm, the envelope, retention), and the order in
:mod:`rola.engine.dags` (docs/internals/engine/engine.md).

Launch arguments therefore arrive already decided, as a
:class:`~rola.engine.plan.ChunkPlan` that the chunk DAG built
(:mod:`rola.engine.dags.chunk_dag`), and the reverse pass's as the
:class:`~rola.engine.plan.BackwardPlan` that holds it by reference
(:mod:`rola.engine.dags.backward_dag`).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from rola.ops._ext import extension

if TYPE_CHECKING:
    from rola.engine.plan import BackwardPlan, ChunkPlan
    from rola.engine.types import Seeds


def backward_seeds(plan: BackwardPlan, dy: torch.Tensor, y: torch.Tensor):
    """`(dnum, dden, dm)` -- the readout divide's own reverse, over one retained plan.

    `dm = eps * dden` exists exactly when the bundle carries a read mass. The eps
    floor is ADDITIVE, so all three are exact and none of them branches on `den`.
    """
    fwd = plan.forward
    dyk = dy if dy.is_contiguous() else dy.contiguous()
    out = extension().chunk_backward_seeds(dyk, y, plan.den, fwd.read_mass, fwd.eps)
    return out[0], out[1], (out[2] if len(out) > 2 else None)


def _scan_operands(plan: BackwardPlan, seeds: Seeds) -> tuple:
    """THE SCANS' SHARED ABI, spelled ONCE and read off the plan.

    Both scans take the same ten arguments because they walk the same schedule from
    the same tables in opposite directions (docs/internals/chunk/backward.md). What
    made that a hazard was the tuple being hand-built at the call site, where the two
    could drift apart by an edit to one of them.
    """
    fwd = plan.forward
    return (fwd.read_plane, fwd.write_plane, fwd.g_write, fwd.v, seeds.dnum,
            seeds.dden, fwd.union_table, fwd.arm.c, fwd.arm.bc, list(fwd.widths))


def scan_a(plan: BackwardPlan, seeds: Seeds) -> torch.Tensor:
    """SCAN A: the read side's digit gradient, fp32 in the kernel's layout.

    It rebuilds the forward state from zero over the RETAINED union table -- no
    per-chunk snapshot is read, because none was kept -- and is independent of
    :func:`scan_b`, which the backward DAG declares.
    """
    (d_read,) = extension().chunk_scan_a_probe(*_scan_operands(plan, seeds))
    return d_read


def scan_b(plan: BackwardPlan, seeds: Seeds):
    """SCAN B: `(dv, d_gain, d_write)`, fp32 in the kernels' layouts.

    It walks the same tokens downward carrying the adjoint state. Its slot-sum
    output is a probe byproduct that the operand gradients do not read.

    The two scans associate their sums differently by construction -- each walks its
    own partition -- so nothing either returns is bit-reproducible, and neither is
    the fan-in that assembles the digit gradients (`docs/api.md`).
    """
    dv, _slot_sum, d_gain, d_write = extension().chunk_scan_b_probe(
        *_scan_operands(plan, seeds))
    return dv, d_gain, d_write


def unpack_grads(plan: BackwardPlan, dv, d_read, d_write, d_gain, d_mass,
                 *, v_dtype: torch.dtype) -> list[torch.Tensor]:
    """The scans' fp32 kernel layouts back to the OPERANDS', in `tensors()`' order.

    That order is load-bearing rather than incidental: the autograd node receives the
    operands flat and in exactly that order, so a gradient list ordered any other way
    lands every gradient on the wrong tensor SILENTLY -- the shapes agree
    (`tests/unit/test_interface.py`). Each gradient is narrowed to its operand's dtype
    and un-permuted to its public layout because autograd requires
    `grad.dtype == input.dtype`, and the packing that produced the kernel layout was
    the node's own.
    """
    routes = plan.forward.routes

    def levels(packed_side, side):
        out, off = [], 0
        for level in side:
            w = level.shape[-1]
            out.append(packed_side[..., off:off + w].permute(0, 2, 1, 3)
                       .to(level.dtype).contiguous())
            off += w
        return out

    grads = [dv.permute(0, 2, 1, 3).to(v_dtype).contiguous()]
    grads += levels(d_read, routes.read)
    grads += levels(d_write, routes.write)
    #: `g_read` does not exist on this surface, so the gains are the write gain
    #: alone -- the same one clause `tensors()` builds.
    grads.append(d_gain.permute(0, 2, 1).to(routes.g_write.dtype).contiguous())
    if routes.read_mass is not None:
        #: SURPRISE 2: the m-carry is forward-exact but NOT gradient-neutral.
        grads.append(d_mass.permute(0, 2, 1).to(routes.read_mass.dtype).contiguous())
    return grads


def execute(plan: ChunkPlan) -> tuple[torch.Tensor, torch.Tensor]:
    """`(y [B,T,H,d_v], den)` -- the consumer launch and the divide, over one plan.

    Every launch argument is a plan field: nothing is derived, chosen or re-packed
    here, which is the whole of what the facts engine bought.

    A block-grain liveness bitmap was reduced over this call's union table, and a
    block whose byte says it contributes nothing exits before it touches anything
    (docs/internals/chunk/chunk_kernel.md#liveness-exit).
    """
    m = extension()
    num, den = m.chunk_forward_raw(
        plan.read_plane, plan.write_plane, plan.g_write, plan.v, plan.union_table,
        plan.arm.c, plan.arm.bc, list(plan.widths), plan.state_in, plan.state_out,
        plan.page_table, plan.block_bits)
    #: THE READ MASS RIDES INTO THE FLOOR. `read_mass` is the per-token scale the read
    #: levels still carry, so numerator and denominator are both `m` times the
    #: normalized ones; passing `eps * m` makes the divide `m*num / (m*(den + eps))`,
    #: which is the defined readout exactly rather than to within a scaled floor.
    y = m.chunk_divide_pass(num, den, plan.read_mass, plan.eps, plan.y_dtype)
    return y, den
