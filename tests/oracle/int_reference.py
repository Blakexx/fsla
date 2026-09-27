# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The fp64 reference of the int8 carry's DECLARED arithmetic, and the quantization-error report.

THE OP ORACLE MEASURES THAT THE MATH IS IMPLEMENTED; THE LAYER ORACLE MEASURES QUALITY (ruled
2026-09-26). This module is the first half's reference for the int8 form: the same windowed
recurrence as :func:`tests.oracle.reference.inter_reference`, run in fp64 on the OPERANDS THE
KERNEL DECLARES -- per-token unit-max s8 values (one digit, or an s8 high and u8 low digit), u8
amplitude products for the dense levels, the sparse levels' amplitude times the token's scale and
gain as a per-(token, write digit) scalar folded into the values at the (digit, window) scale, an
s32-exact state, and a readout operand that is the state quantized against its own bound. Every
product of two quantized operands is exact in fp64, so what this reference does NOT model is the
s32 panel's final rounding (half a step of ``envelope_t / 2^30``), which the op gate states as a
step term. ``exact=True`` skips every rounding and must reproduce ``inter_reference`` exactly:
that is the test that the recurrence's STRUCTURE is right before any rounding is judged.

The REPORT (:func:`quantization_report`) is comparison B: the declared arithmetic against the
unquantized op on the oracle cells, per slot against each slot's envelope and per token in norm,
reported and never gated (the doctrine's truth-oracle half). The Hadamard rotation is here so the
report can show the same error in the rotated and the real basis.

Symbols, each at first use: ``D`` levels, ``B_l`` widths, ``N = prod B_l`` leaves, ``dv`` value
channels, ``W`` the window, ``s_t`` a token's value scale (its max |channel|), ``a`` an amplitude,
``g`` the write gain, ``c`` a leaf coefficient (the product of its digits' amplitudes).
"""
from __future__ import annotations

import math

import torch

from tests.oracle.reference import inter_reference, leaf_product, radix_strides

TINY = 1e-300


def hadamard(n: int, device, dtype=torch.float64) -> torch.Tensor:
    """The normalized Sylvester Hadamard matrix of order ``n`` (a power of two): orthogonal, entries ±1/sqrt(n)."""
    assert n & (n - 1) == 0 and n >= 1
    h = torch.ones(1, 1, dtype=dtype)
    while h.shape[0] < n:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return (h / math.sqrt(n)).to(device)


def rotate(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Channels are the last axis; the layer folds ``h`` into W_V and ``h^T`` into W_O."""
    return x @ h


def unrotate(y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    return y @ h.T


def quantize_unit(x: torch.Tensor, digits: int) -> torch.Tensor:
    """Round ``x`` in [-1, 1] to the declared digit count: 127 levels (s8) or 32767 (s8 high + u8 low)."""
    levels = 127.0 if digits == 1 else 32767.0
    return torch.round(x * levels) / levels


def quantize_u8_unit(p: torch.Tensor) -> torch.Tensor:
    """An amplitude in [0, 1] as u8: ``round(255 p) / 255``."""
    return torch.round(p * 255.0) / 255.0


def unit_max(v: torch.Tensor):
    """``(v_hat, s_t)``: the per-token unit-max form, ``|v_hat| <= 1`` with the largest channel at ±1."""
    s = v.abs().amax(-1, keepdim=True).clamp_min(TINY)
    return v / s, s


def group_index(widths, levels) -> torch.Tensor:
    """For every canonical leaf, its mixed-radix index over the digits of ``levels`` (in level order); zeros when
    ``levels`` is empty."""
    N = math.prod(widths)
    strides = radix_strides(widths)
    idx = torch.zeros(N, dtype=torch.long)
    ell = torch.arange(N)
    for lvl in levels:
        idx = idx * widths[lvl] + (ell // strides[lvl]) % widths[lvl]
    return idx


def group_product(levels_all, widths, levels, sl, u8=None):
    """``(product, row_scale)``: the chosen levels' amplitudes multiplied over their digit tuples, ``[B, t, H, prod
    widths[levels]]``, and the per-token scalar factored out of it, ``[B, t, H, 1]``. With ``u8`` each level's row is
    first put in UNIT-MAX form (its max digit at 1) and quantized by ``u8``; the row maxima's product is the scalar --
    a per-token factor, which the fold migrates onto the values and the readout onto the output row. Without ``u8`` the
    scalar is one. Ones when ``levels`` is empty."""
    x, scale = None, None
    for lvl in levels:
        a = levels_all[lvl][:, sl].double()
        if u8 is not None:
            m = a.amax(-1, keepdim=True).clamp_min(TINY)
            a = u8(a / m)
            scale = m if scale is None else scale * m
        x = a if x is None else (x.unsqueeze(-1) * a.unsqueeze(-2)).flatten(-2)
    b, t, h = levels_all[0][:, sl].shape[:3]
    ones = torch.ones(b, t, h, 1, device=levels_all[0].device, dtype=torch.float64)
    return (ones if x is None else x), (ones if scale is None else scale)


def int_reference(read_levels, write_levels, g_write, v, widths, sparsity, window, *,
                  fold_digits: int = 1, readout_digits: int = 1, exact: bool = False, state_in=None,
                  scale_tokens: int | None = None):
    """``(num, den, state)`` in fp64 under the DECLARED int arithmetic; ``sparsity[l] = (read sparse, write sparse)``.

    Fold: ``state[leaf] += sum_t A_u8[t, dense(leaf)] * V'[t, sparse(leaf)]`` with ``V' = q(v_hat * scal / zeta) *
    zeta``, ``scal = a_sparse * s_t * g`` and ``zeta`` the max of ``scal`` over the window's tokens at that digit; the
    bound ``B[leaf] += sum_t A_u8 * scal`` beside it (the pre-pass's recurrence). Readout: ``R = A_u8[t, dense] *
    a_sparse[t, sparse]`` against ``S_op = q(state / B) * B``; ``den`` from the exact amplitudes and the mass, as the
    pre-pass computes it. ``exact`` turns every ``q`` and ``u8`` into the identity. ``scale_tokens`` = 32 takes
    ``zeta`` per (digit, tile of 32 LIVE tokens of that digit's list) instead of per (digit, window): the K range of one
    MMA, whose product a kernel rescales before it joins the state.
    """
    Bn, T, H, dv = v.shape
    D = len(widths)
    N = math.prod(widths)
    dev = v.device
    r_sp = [lvl for lvl in range(D) if sparsity[lvl][0]]
    r_dn = [lvl for lvl in range(D) if not sparsity[lvl][0]]
    w_sp = [lvl for lvl in range(D) if sparsity[lvl][1]]
    w_dn = [lvl for lvl in range(D) if not sparsity[lvl][1]]
    idx_rs, idx_rd = group_index(widths, r_sp).to(dev), group_index(widths, r_dn).to(dev)
    idx_ws, idx_wd = group_index(widths, w_sp).to(dev), group_index(widths, w_dn).to(dev)
    n_ws = int(idx_ws.max().item()) + 1
    u8 = None if exact else quantize_u8_unit
    q_fold = (lambda x: x) if exact else (lambda x: quantize_unit(x, fold_digits))
    q_read = (lambda x: x) if exact else (lambda x: quantize_unit(x, readout_digits))

    v_hat, s_t = unit_max(v.double())                    # [B,T,H,dv], [B,T,H,1]
    v_hat = q_fold(v_hat)                                 # the producer's s8 (or s8+u8) V-hat
    state = (torch.zeros(Bn, H, N, dv, device=dev, dtype=torch.float64) if state_in is None
             else state_in.double()[..., :dv].clone())
    mass = (torch.zeros(Bn, H, N, device=dev, dtype=torch.float64) if state_in is None
            else state_in.double()[..., dv].clone())
    bound = state.abs().amax(-1) if state_in is not None else torch.zeros(Bn, H, N, device=dev, dtype=torch.float64)
    num = torch.zeros(Bn, T, H, dv, device=dev, dtype=torch.float64)
    den = torch.zeros(Bn, T, H, device=dev, dtype=torch.float64)
    for t0 in range(0, T, window):
        sl = slice(t0, min(t0 + window, T))
        # --- readout against the state at the window start -----------------------------------------------------
        a_rd, m_rd = group_product(read_levels, widths, r_dn, sl, u8)     # [B,t,H,n_rd] unit-max u8, its row scale
        a_rs, _ = group_product(read_levels, widths, r_sp, sl)             # [B,t,H,n_rs] the sparse amps, fp
        r_q = a_rd[..., idx_rd] * (a_rs * m_rd)[..., idx_rs]               # [B,t,H,N]; m_rd is the output-row scale
        sigma = bound.clamp_min(TINY)                                      # [B,H,N]
        s_op = q_read(state / sigma.unsqueeze(-1)) * sigma.unsqueeze(-1)  # the readout's B operand, dequantized
        num[:, sl] = torch.einsum("bthn,bhnc->bthc", r_q, s_op)
        r_fp = leaf_product(read_levels, widths, sl)
        den[:, sl] = torch.einsum("bthn,bhn->bth", r_fp, mass)
        del r_q, r_fp, s_op
        # --- fold ------------------------------------------------------------------------------------------------
        a_wd, m_wd = group_product(write_levels, widths, w_dn, sl, u8)    # [B,t,H,n_wd] unit-max u8, its row scale
        a_ws, _ = group_product(write_levels, widths, w_sp, sl)            # [B,t,H,n_ws]
        scal = a_ws * (g_write[:, sl].double() * s_t[:, sl, :, 0]).unsqueeze(-1) * m_wd   # [B,t,H,n_ws]
        zeta = scal.amax(dim=1, keepdim=True).clamp_min(TINY)                       # [B,1,H,n_ws] per (digit, window)
        if scale_tokens is not None:                                                # per (digit, tile of its live list)
            zeta = torch.zeros_like(scal)
            for b in range(Bn):
                for hh in range(H):
                    for s_ in range(n_ws):
                        col = scal[b, :, hh, s_]
                        live = torch.nonzero(col > 0, as_tuple=False).flatten()
                        for i in range(0, live.numel(), scale_tokens):
                            tile = live[i:i + scale_tokens]
                            zeta[b, tile, hh, s_] = col[tile].max()
            zeta = zeta.clamp_min(TINY)
        vq = q_fold(v_hat[:, sl].unsqueeze(3) * (scal / zeta).unsqueeze(-1)) * zeta.unsqueeze(-1)  # [B,t,H,n_ws,dv]
        for s in range(n_ws):
            leaves = torch.nonzero(idx_ws == s, as_tuple=False).flatten()
            a_d = a_wd[..., idx_wd[leaves]]                                # [B,t,H,n_wd]
            state[:, :, leaves] += torch.einsum("bthd,bthc->bhdc", a_d, vq[:, :, :, s])
            bound[:, :, leaves] += torch.einsum("bthd,bth->bhd", a_d, scal[..., s])
        w_fp = g_write[:, sl].double()[..., None] * leaf_product(write_levels, widths, sl)
        mass += w_fp.sum(1)
        del a_wd, a_ws, scal, zeta, vq, w_fp
    full = torch.cat((state, mass.unsqueeze(-1)), -1)
    return num, den, full


def per_token_norm_ratio(err: torch.Tensor, envelope: torch.Tensor) -> torch.Tensor:
    """``||err[t]|| / ||envelope[t]||`` over the channel axis: the basis-free fidelity number."""
    return err.norm(dim=-1) / envelope.norm(dim=-1).clamp_min(TINY)


def quantization_report(drawn, sparsity, window, digit_pairs=((1, 1), (1, 2), (2, 2)), rotate_values=False,
                        scale_tokens=None):
    """Comparison B on one drawn cell: the declared arithmetic against the unquantized op. Returns one row per digit
    pair: the per-slot error over the slot's envelope (max and the fraction above 2^-8), the per-token norm ratio (max,
    mean) for ``num``, and the state's per-slot max over its envelope. With ``rotate_values`` the values are rotated by
    the Hadamard before quantization and the outputs unrotated, so the errors are read in the real basis."""
    read, write, gain, v = drawn.doubles()
    widths = drawn.cell.widths
    h = hadamard(v.shape[-1], v.device) if rotate_values else None
    v_in = rotate(v, h) if rotate_values else v
    true_num, true_den, true_state = inter_reference(read, write, gain, v, widths, window)
    env_num, _, env_state = inter_reference(read, write, gain, v.abs(), widths, window)
    rows = []
    for fd, rd in digit_pairs:
        num, den, state = int_reference(read, write, gain, v_in, widths, sparsity, window,
                                        fold_digits=fd, readout_digits=rd, scale_tokens=scale_tokens)
        if rotate_values:
            num, state = unrotate(num, h), torch.cat((unrotate(state[..., :-1], h), state[..., -1:]), -1)
        e_num = (num - true_num).abs()
        slot = e_num / (env_num * 2.0 ** -8).clamp_min(TINY)
        tok = per_token_norm_ratio(num - true_num, env_num)
        e_state = (state[..., :-1] - true_state[..., :-1]).abs() / (env_state[..., :-1] * 2.0 ** -8).clamp_min(TINY)
        rows.append(dict(fold_digits=fd, readout_digits=rd,
                         slot_max=float(slot.max()), slot_frac_over=float((slot > 1).double().mean()),
                         token_max=float(tok.max()), token_mean=float(tok.mean()),
                         state_slot_max=float(e_state.max()), state_slot_frac_over=float((e_state > 1).double().mean()),
                         den_max_rel=float(((den - true_den).abs() / true_den.abs().clamp_min(TINY)).max())))
    return rows
