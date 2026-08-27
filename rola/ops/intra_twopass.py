# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""P78 v2 STEP A — the intra term as two GEMM passes on materialized Grams.

REFERENCE MODULE, NOT A SHIPPED PATH. Nothing in `rola/ops/` dispatches to it;
`rola.ops.intra` and `rola.ops.prefill` are untouched. It exists to state the
P78 v2 design in library calls so it can be gated against the fp64 intra
reference and MEASURED against the one-pass kernel before either form is chosen.

Symbols at first use: ``BH`` batch x heads, ``L`` the sequence length, ``W`` the
window in tokens, ``D`` routing levels, ``B`` a level's digit width, ``d_v = 64``
value channels, ``G`` the window's materialized Gram ``[W, W]``, ``R_l``/``W_l``
the level's read/write factor rows ``[W, B]``, ``gw`` the write side's per-token
gain, ``n`` the flattened ``(bh, window)`` batch index.

THE TERM. With ``A[t, t'] = gw[t'] * prod_l <R_l[t], W_l[t']>`` on the causal half
of one window, the intra's two outputs are ``o[t] = sum_t' A[t, t'] v[t']`` and
``den[t] = sum_t' A[t, t']`` -- `tests/k31/intra_reference.py`'s statement, which
`test_intra_reference.py` pins to the oracle.

THE COMPOSITION.
  pass 1  ``G = tril(prod_l R_l W_l^T)``   -- D batched GEMMs, bf16 operands,
          fp32 accumulate, chained IN PLACE (level l multiplies the running G;
          there are no per-level private matrices), then the causal mask. The
          chain's dtype is a DECLARATION, not a default: carrying it in bf16
          breaks the oracle band at the D = 4 shared-support arm (measured,
          FINDINGS ## I2), so fp32 is the rule and bf16 the flat arm's option.
  pass 2  ``o = G Vg`` and ``den = G gw`` (fp32 C) -- two batched GEMMs sharing G, with
          the gain folded into ``Vg = gw * v`` so the mass falls out of the same
          matrix as the numerator (``den``'s B operand is ``gw`` itself, so the
          mass carries no rounding the numerator does not).

THE BATCH IS AFFINE. Window ``w`` of stream ``bh`` starts at digit offset
``bh*L*D*B + w*W*D*B = (bh*(L/W) + w) * W*D*B``, so the whole ``(bh, window)``
grid is ONE uniformly strided batch over ``n = bh*(L/W) + w`` and every operand
is a view -- no gather, no packing pass, at any BH.

VRAM IS BOUNDED BY THE GROUP, NEVER BY L. Windows are processed in groups sized
to L2 (:func:`group_windows`): pass 2 follows pass 1 within the group, so G is
allocated for the windows IN FLIGHT and reused group after group. Peak Gram bytes
are ``group * W * W * 2`` and are reported by :func:`gram_bytes`.

NO DECAY. This is the no-decay form; decay enters at pass 2's Hadamard (a
decay-weighted G), which is a later card.
"""

from __future__ import annotations

import torch

VALUE_WIDTH = 64

#: THE ONE DIAL. The Gram of every window in flight must fit L2 for pass 2 to
#: read what pass 1 just wrote without a DRAM round trip; 4 MiB is two thirds of
#: the RTX 3080 Ti's 6 MiB, leaving room for the factor rows the GEMMs stream.
GROUP_BYTES = 4 << 20


def group_windows(window: int, group_bytes: int = GROUP_BYTES,
                  chain_dtype=torch.float32) -> int:
    """Windows per L2 group: as many windows' worth of Gram as ``group_bytes`` holds.

    Sized on the PEAK (:func:`gram_bytes`), not on the final G alone: everything
    pass 1 writes competes for the same L2 that pass 2 is meant to read G out of.
    """
    return max(1, group_bytes // gram_bytes(window, 1, chain_dtype))


def gram_bytes(window: int, group: int, chain_dtype=torch.float32) -> int:
    """Peak Gram bytes in flight -- the allocation that must not scale with L.

    The running chain (``chain_dtype``), the level GEMM's own output (bf16) and the
    masked bf16 G pass 2 reads are live at once; the shipped form's epilogue writes
    the chain IN PLACE and so holds only the first and the last.
    """
    element = 4 if chain_dtype == torch.float32 else 2
    return group * window * window * (element + 2 + 2)


def causal_mask(window: int, device, dtype=torch.bfloat16) -> torch.Tensor:
    """The window's inclusive lower triangle, as the value the chain multiplies by.

    It is a 0/1 bf16 matrix, so the mask multiply is exact and may be applied at
    any point in the chain -- in the shipped form it would be the FIRST level's
    epilogue source (one matrix, batch stride zero), which is why it is stated
    here as a factor of the product rather than as an indexing rule.
    """
    return torch.tril(torch.ones((window, window), device=device, dtype=dtype))


def _views(pread, pwrite, gwrite, v_bh, window):
    """The ``(bh, window)`` grid as one uniformly strided batch. All views."""
    bh, length, width = pread.shape
    n = bh * (length // window)
    return (pread.view(n, window, width), pwrite.view(n, window, width),
            gwrite.view(n, window, 1), v_bh.view(n, window, v_bh.shape[-1]))


def intra_terms(pread, pwrite, gwrite, v_bh, levels, level_width, window, group=None,
                mask=None, chain_dtype=None):
    """``(o, den)`` for the whole sequence -- fp32, allocated fresh, differentiable.

    ``pread``/``pwrite`` are ``[BH, L, D*B]`` bf16, ``gwrite`` is ``[BH, L]`` bf16 and
    ``v_bh`` is ``[BH, L, 64]`` bf16 (the STREAM-major value; the shipped op's
    token-major ``[B, T, H, 64]`` is :func:`from_token_major`).

    Autograd differentiates the composition: the three GEMM cotangents of P78 v2's
    backward are exactly what the graph of these bmm's produces.
    """
    bh, length, width = pread.shape
    if width != levels * level_width:
        raise ValueError(f"the plane is {width} wide, not {levels} x {level_width}")
    if length % window:
        raise ValueError(f"L = {length} is not a whole number of {window}-token windows")
    chain_dtype = chain_dtype or torch.float32
    group = group or group_windows(window, chain_dtype=chain_dtype)
    r, w, g, v = _views(pread, pwrite, gwrite, v_bh, window)
    if mask is None:
        mask = causal_mask(window, pread.device)
    n = r.shape[0]
    out, mass = [], []
    for start in range(0, n, group):
        cut = slice(start, min(n, start + group))
        gram = None
        for level in range(levels):
            span = slice(level * level_width, (level + 1) * level_width)
            #: bf16 in, fp32 accumulate, bf16 out -- the arithmetic the shipped
            #: pass 1 declares, with the running product carried in the same
            #: buffer the next level's epilogue would multiply into.
            partial = torch.bmm(r[cut][:, :, span], w[cut][:, :, span].transpose(1, 2))
            gram = (partial.to(chain_dtype) if gram is None
                    else (gram.float() * partial.float()).to(chain_dtype))
        #: THE CAUSAL MASK IS A FACTOR OF THE CHAIN, and the chain's LAST rounding
        #: is the only one pass 2 sees: G reaches pass 2 in bf16 because that is
        #: the operand its tensor cores take.
        gram = (gram.float() * mask).to(v.dtype)
        gain = g[cut]
        #: THE GAIN RIDES ON THE VALUE. `A`'s gain is a per-COLUMN scalar, so it
        #: folds into pass 2's B operand; the mass's B operand IS the gain.
        vg = (v[cut].float() * gain.float()).to(v.dtype)
        #: fp32 C, the shipped pass 2's `cublasGemmStridedBatchedEx` accumulator.
        #: `torch.bmm` cannot express bf16 operands with an fp32 output, so the
        #: operands are widened first: every value here is exactly a bf16, and
        #: tf32 carries a bf16 mantissa exactly, so the PRODUCTS are the shipped
        #: form's bit for bit -- only the tensor-core RATE differs, which is the
        #: one place Step A's measurement is not the shipped form's.
        out.append(torch.bmm(gram.float(), vg.float()))
        mass.append(torch.bmm(gram.float(), gain.float()))
    return (torch.cat(out).view(bh, length, v.shape[-1]),
            torch.cat(mass).view(bh, length))


def from_token_major(v, heads):
    """The shipped op's token-major ``[B, T, H, 64]`` value stream as ``[BH, L, 64]``."""
    batch, tokens, head_count, channels = v.shape
    if head_count != heads:
        raise ValueError(f"v carries {head_count} heads, not {heads}")
    return v.permute(0, 2, 1, 3).reshape(batch * heads, tokens, channels)


@torch.no_grad()
def intra_forward(pread, pwrite, gwrite, v_bh, levels, level_width, window, *, o=None,
                  den=None, group=None, mask=None, chain_dtype=None):
    """RED the intra term into ``(o, den)`` -- the shipped op's reduce-into contract.

    The caller owns the buffers and their initial contents: the carry's inter term
    is already in them when the two kernels run as one operator.
    """
    term_o, term_den = intra_terms(pread, pwrite, gwrite, v_bh, levels, level_width,
                                   window, group=group, mask=mask, chain_dtype=chain_dtype)
    if o is None:
        o = torch.zeros_like(term_o)
    if den is None:
        den = torch.zeros_like(term_den)
    o += term_o
    den += term_den
    return o, den


#: the mode word's two bits, `rola.ops.intra`'s -- restated so the restriction can
#: be read without the shipped op's constants in scope.
READ_SPARSE_BIT = 1
WRITE_SPARSE_BIT = 2


def support_restriction(plane, levels, level_width, modes, bit):
    """The declared-sparse restriction: ``True`` where a digit gradient may be live.

    A level whose mode word sets ``bit`` (:data:`READ_SPARSE_BIT` for the read
    plane, :data:`WRITE_SPARSE_BIT` for the write plane) restricts that side's digit
    gradient to the side's own stored support -- the producer's VJP never loads an
    off-support cotangent. A level that declares nothing is unrestricted.
    """
    parts = []
    for level in range(levels):
        span = slice(level * level_width, (level + 1) * level_width)
        block = plane[:, :, span]
        parts.append(block != 0 if (modes[level] & bit) else torch.ones_like(block,
                                                                            dtype=torch.bool))
    return torch.cat(parts, dim=-1)


def intra_backward(pread, pwrite, gwrite, v_bh, dnum, dden, levels, level_width, window,
                   modes, group=None):
    """The four cotangents of the composition, by autograd over the same GEMMs.

    ``dnum`` is ``[BH, L, 64]`` and ``dden`` ``[BH, L]`` -- the seeds the op seam
    forms from the readout's division. The digit gradients are restricted to their
    declared support, which is the contract `test_intra_backward.py` states.
    """
    #: THE LEAVES ARE fp32 HOLDING THE bf16 VALUES. The operand is bf16 and the
    #: cast back to it is exact, so the graph runs on exactly the stored numbers
    #: while the cotangent that comes out is fp32 -- which is what the gradient
    #: planes are, and what a bf16 leaf would have thrown away.
    pr = pread.detach().float().requires_grad_(True)
    pw = pwrite.detach().float().requires_grad_(True)
    gw = gwrite.detach().float().requires_grad_(True)
    vv = v_bh.detach().float().requires_grad_(True)
    o, den = intra_terms(pr.to(pread.dtype), pw.to(pwrite.dtype), gw.to(gwrite.dtype),
                         vv.to(v_bh.dtype), levels, level_width, window, group=group)
    ((o * dnum.float()).sum() + (den * dden.float()).sum()).backward()
    d_pr = pr.grad * support_restriction(pread, levels, level_width, modes, READ_SPARSE_BIT)
    d_pw = pw.grad * support_restriction(pwrite, levels, level_width, modes, WRITE_SPARSE_BIT)
    return d_pr, d_pw, gw.grad, vv.grad


__all__ = ["GROUP_BYTES", "READ_SPARSE_BIT", "VALUE_WIDTH", "WRITE_SPARSE_BIT", "causal_mask", "from_token_major", "gram_bytes",
           "group_windows", "intra_backward", "intra_forward", "intra_terms",
           "support_restriction"]
