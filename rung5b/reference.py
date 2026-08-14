"""References for PER-SIDE STREAM COMPACTION.

Rung 5A's references unchanged -- the recurrence, the block plan, the per-block
split, the regimes, the union table and the schedule derived from it -- plus the
two things compaction adds:

``cook``            the super-chunk partition, written from the DESIGN rather
                    than from the kernel: walk the tokens in order, keep a read
                    list and a write list, and close when either would exceed its
                    buffer; a read entry records the running write count, which
                    IS its causal threshold. This is the same walk
                    `rung5/size_compaction.py` counts, made to yield its lists.
``split_compacted`` the kernel's own arithmetic ON THAT PARTITION: per block, the
                    recurrence over the gathered panels with the monotone
                    staircase in place of the triangle, bf16 state and bf16 Gram.
                    Compaction is EXACT in exact arithmetic -- it regroups a sum
                    -- so what this models is only WHERE the state is rounded,
                    which the partition moves.

THE DENSE ASSERTION LIVES HERE. At dense the cook must return the contiguous
chunking with thresholds `J[i] = i + 1`, which is exactly today's triangle;
``assert_dense_identity`` states that rather than leaving it to be observed.
"""
from __future__ import annotations

import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from rung1.reference import (  # noqa: E402,F401
    amplitudes,
    chunked,
    chunked_bf16_state,
    digits,
    kron,
    kron_bf16,
    quadratic,
    split_levels,
    strides,
)
from rung2.reference import block_plan, block_slots, split  # noqa: E402,F401
from rung5.reference import (  # noqa: E402,F401
    REGIMES,
    chunked_norm,
    ratio,
    regime_amplitudes,
    schedule,
    schedule_exact,
    schedule_from_table,
    schedule_sides,
    schedule_tokens,
    split_norm,
    support_amplitudes,
    union_bits,
    union_entries,
    union_entries_law,
    union_unpack,
    with_ones,
)


def cook(read_live, write_live, chunk: int):
    """The super-chunk partition of ONE (block, b, h), from two bool sequences.

    Returns ``[(r_idx, w_idx, thresholds)]``. `r_idx` and `w_idx` are ascending
    token indices, at most `chunk` of each; `thresholds[i]` is the number of
    entries of `w_idx` at positions <= `r_idx[i]`, which is the causal threshold
    of read entry `i` and is what the kernel carries as one byte.

    THE CLOSING RULE IS THE DESIGN'S: the super-chunk ends at the first token
    that would push EITHER list past its buffer, so the range is set by whichever
    side saturates first and a stretch dead on both sides is swallowed whole.
    """
    L = len(read_live)
    out, t = [], 0
    while t < L:
        r_idx, w_idx, thr = [], [], []
        while t < L:
            if (read_live[t] and len(r_idx) >= chunk) or \
               (write_live[t] and len(w_idx) >= chunk):
                break
            if write_live[t]:
                w_idx.append(t)
            if read_live[t]:
                r_idx.append(t)
                thr.append(len(w_idx))
            t += 1
        out.append((r_idx, w_idx, thr))
    return out


def side_live(pr, pw, widths, bc: int):
    """(read, write) each [B*H, blocks, L] bool -- the table's bits, per side.

    This is `schedule_sides` reshaped: the SAME per-(side, block, token) bits the
    union table carries and the group schedule was a reduction over. Compaction
    reads them per token; that is the one-structure-two-consumers claim, and the
    gate checks this against the device table rather than against the kernel.
    """
    r, w = schedule_sides(pr, pw, widths, bc)
    B, H, nb, L = r.shape
    return r.reshape(B * H, nb, L), w.reshape(B * H, nb, L)


def assert_dense_identity(L: int, chunk: int):
    """AT DENSE, COMPACTION IS THE IDENTITY -- asserted, not observed.

    Both gathers are `t0 + i`, the partition is the contiguous chunking, and the
    thresholds are `i + 1`, which makes the monotone staircase `tc <= tr` term
    for term. This is what makes "bit-identical to rung 5A at dense" available as
    a gate, so it is stated where it can fail.
    """
    live = [True] * L
    plan = cook(live, live, chunk)
    assert len(plan) == L // chunk, f"{len(plan)} super-chunks, expected {L // chunk}"
    for k, (r_idx, w_idx, thr) in enumerate(plan):
        want = list(range(k * chunk, (k + 1) * chunk))
        assert r_idx == want and w_idx == want, f"super-chunk {k} is not contiguous"
        assert thr == list(range(1, chunk + 1)), f"super-chunk {k} is not the triangle"
    return True


def compacted_bf16_state(rb, wb, v, plan):
    """One block's recurrence over a compacted partition. `rb`/`wb` are that
    block's slot columns; `plan` is `cook`'s output for that block.

    The staircase is built from the thresholds rather than from a comparison of
    positions, because the thresholds are what the kernel has: read entry `i`
    keeps write entries `0..thr[i]-1`.
    """
    rb, wb, v = (t.double() for t in (rb, wb, v))
    S = torch.zeros(*rb.shape[:-2], rb.shape[-1], v.shape[-1],
                    dtype=torch.float64, device=rb.device)
    out = torch.zeros(*rb.shape[:-1], v.shape[-1], dtype=torch.float64,
                      device=rb.device)
    for r_idx, w_idx, thr in plan:
        if not r_idx and not w_idx:
            continue
        wc = wb[..., w_idx, :]
        vc = v[..., w_idx, :]
        if r_idx:
            rc = rb[..., r_idx, :]
            o = rc @ S.bfloat16().double()
            if w_idx:
                col = torch.arange(len(w_idx), device=rb.device)
                keep = col.unsqueeze(0) < torch.tensor(thr, device=rb.device).unsqueeze(1)
                g = torch.where(keep, rc @ wc.transpose(-1, -2), 0.0).bfloat16().double()
                o = o + g @ vc
            out[..., r_idx, :] = o
        if w_idx:
            S = S + wc.transpose(-1, -2) @ vc
    return out


def split_compacted(rb, wb, v, chunk: int, bc: int, plans):
    """The kernel's arithmetic: per block, the compacted recurrence, summed.

    `plans[k]` is block `k`'s partition. The partials are added in the
    accumulator's precision (fp64 here; fp32 in the kernel, far inside the bf16
    floor this models).
    """
    n = rb.shape[-1]
    out = None
    for k, s0 in enumerate(range(0, n, bc)):
        part = compacted_bf16_state(rb[..., s0:s0 + bc], wb[..., s0:s0 + bc], v,
                                    plans[k])
        out = part if out is None else out + part
    return out


def plans_for(pr, pw, widths, bc: int, chunk: int, b: int = 0, h: int = 0):
    """Every block's partition for one (b, h), as `split_compacted` wants them."""
    r, w = schedule_sides(pr, pw, widths, bc)
    rl = r[b, h].cpu().tolist()
    wl = w[b, h].cpu().tolist()
    return [cook(rl[k], wl[k], chunk) for k in range(len(rl))]


def model_compacted(pr, pw, v, widths, chunk: int, bc: int, g_write=None):
    """THE KERNEL'S OWN ARITHMETIC over the whole cell, on `d_v + 1` columns.

    The partition is a property of each (b, h, block) separately, so this runs the
    recurrence per (b, h) rather than batched. It is the compaction-aware
    counterpart of `split_norm` and it is what a floor must be measured against
    once the partition moves, since where the state is ROUNDED moves with it.
    """
    rb, wb = kron_bf16(pr, widths), kron_bf16(pw, widths, gain=g_write)
    v1 = with_ones(v)
    r, w = schedule_sides(pr, pw, widths, bc)
    B, H = pr.shape[0], pr.shape[1]
    out = torch.empty(B, H, pr.shape[2], v.shape[-1] + 1, dtype=torch.float64,
                      device=pr.device)
    for b in range(B):
        for h in range(H):
            rl, wl = r[b, h].cpu().tolist(), w[b, h].cpu().tolist()
            plans = [cook(rl[k], wl[k], chunk) for k in range(len(rl))]
            out[b:b + 1, h:h + 1] = split_compacted(
                rb[b:b + 1, h:h + 1], wb[b:b + 1, h:h + 1], v1[b:b + 1, h:h + 1],
                chunk, bc, plans)
    return out
