"""Canonical fp64 reference implementation for the V3 routing-native RoLA oracle.

Implements ``ROLA_V3_SPEC.md`` §2 exactly. There is no Q/K, no feature map, and
no privileged routing level here -- every level is an ordinary mixed-radix
routing level, and the oracle consumes already-materialized per-level, per-side
routing allocations (``p_read``, ``p_write``) plus one positive side gain per
side (``g_read``, ``g_write``; spec §1.1). Computing those allocations from a
projected hidden state is producer scope (``ROLA_V3_SPEC.md`` §10 stage 2) and
is out of scope here.

This module intentionally does not import or reuse anything from the Q/K-era
oracle it replaces (``ResolvedRouting``, ``routing_factor_levels``,
``packed_router_logits_reference``, ``rola.internal.decay`` --
all tied to the packed-router / matrix-state ABI). The old oracle survives only
in git history at ``d6f73e97`` and is used there, read-only, by the migration
equivalence test.
"""

from __future__ import annotations

import torch

from rola.ops._oracle_contract import initial_cache, validate_inputs, validate_simplex
from rola.ops.constants import READOUT_EPS
from rola.routing.types import DecayConfig, Topology, validate_normalization


def _fold_bth(tensor: torch.Tensor) -> torch.Tensor:
    """Fold a public ``[B, T, H, *]`` tensor into ``[B*H, T, *]``."""
    return tensor.permute(0, 2, 1, 3).reshape(
        tensor.shape[0] * tensor.shape[2], tensor.shape[1], *tensor.shape[3:]
    )


def _leaf_capacity(widths: tuple[int, ...]) -> int:
    """``N = prod_l width_l`` -- the number of leaves the address space has.

    Derived here, from the widths alone. See :func:`_radix_strides` for why the
    oracle re-derives its own address map instead of importing production's.
    """
    capacity = 1
    for width in widths:
        capacity *= width
    return capacity


def _radix_strides(widths: tuple[int, ...]) -> tuple[int, ...]:
    """MSB-first mixed-radix strides: ``stride_l = prod(width_{l+1..D-1})``.

    **THE ORACLE DERIVES ITS OWN LEAF ADDRESSES, AND THAT IS THE POINT.** This is
    three lines that `rola.routing.types.Topology.radix_strides` also computes, and
    the duplication is deliberate: it is the difference between a reference and a
    second opinion. Every Tier 1 gate compares the kernel against this file, and the
    kernel's addressing comes from that production property; importing it here would
    make a wrong stride convention agree with itself in every gate the repository
    has. Two independent derivations disagree when one is wrong, which is the only
    reason to have a reference at all.

    The agreement is then a TESTED CLAIM rather than an unexamined shared import:
    `tests/unit/test_oracle_independence.py` asserts these strides equal the
    production property over the whole supported width space, and is allowed to fail.

    The convention, stated so the reader can check it without leaving the file: leaf
    ``s`` decomposes MSB-first, level 0 being the most significant digit, so level
    ``l``'s digit is ``(s // stride_l) % width_l`` with ``stride_l`` the product of
    every width BELOW it (`ROLA_V3_SPEC.md` section 2). Level ``D-1`` therefore has
    stride 1 and varies fastest.
    """
    strides = []
    for index in range(len(widths)):
        stride = 1
        for width in widths[index + 1:]:
            stride *= width
        strides.append(stride)
    return tuple(strides)


def _leaf_product(
    levels: tuple[torch.Tensor, ...], radix_strides: tuple[int, ...], N: int
) -> torch.Tensor:
    """Materialize the complete per-leaf product from per-level per-digit factors.

    ``levels[l]`` has shape ``[..., width_l]``. Returns ``[..., N]`` where leaf
    ``s``'s entry is ``prod_l levels[l][..., d_l(s)]`` under the MSB-first
    mixed-radix digit decomposition ``d_l(s) = (s // radix_strides[l]) % width_l``
    (``ROLA_V3_SPEC.md`` §2). This is the oracle's one leaf-materializing step;
    it is deliberately simple and auditable rather than fast.
    """
    device = levels[0].device
    state_ids = torch.arange(N, device=device, dtype=torch.long)
    leaves = None
    for level_factors, stride in zip(levels, radix_strides):
        width = level_factors.shape[-1]
        digit = (state_ids // stride) % width
        selected = level_factors.index_select(-1, digit)
        leaves = selected if leaves is None else leaves * selected
    return leaves


def naive_rola(
    v: torch.Tensor,
    read_levels: tuple[torch.Tensor, ...],
    write_levels: tuple[torch.Tensor, ...],
    g_read: torch.Tensor | None,
    g_write: torch.Tensor,
    topology: Topology,
    decay: DecayConfig,
    normalization: str,
    *,
    write_levels_stored: tuple[torch.Tensor, ...] | None = None,
    eps: float = READOUT_EPS,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the canonical routing-native RoLA reference in fp64 (``ROLA_V3_SPEC.md`` §2).

    There is no ``q``, no ``k``, and no feature map anywhere in this function.
    Inputs are per-level per-side routing allocations (already normalized
    simplices, one tensor per level, shape ``[B,T,H,width_l]``), one positive
    side gain per side (``g_read``/``g_write``, shape ``[B,T,H]``; ``g_read``
    must be ``None`` under ``normalization="global"`` -- spec §4.2 fixes
    ``g_read := 1`` there and production has no read-gain projection to supply
    one), a ``Topology``, and a ``DecayConfig`` (``None`` or ``LeafMassDecay``).

    ``write_levels`` feeds the deposit ``W`` and trains normally. ``write_levels_stored``
    is the (optionally separately quantized) representation the ``LeafMassDecay``
    clock reads (spec §2: ``c[t,s] = stop_gradient(prod_l p_write_stored[...])``);
    it defaults to ``write_levels`` itself, i.e. no separate storage boundary,
    since the bf16 storage/quantization numerics audit is explicit `TODO` (spec §7
    item 4) and out of scope for this oracle. Passing a distinct, separately
    quantized tensor here is how a caller exercises that boundary once it exists.

    Returns ``(y, final_state)`` with ``y: [B,T,H,d_v]`` and, if
    ``output_final_state``, ``final_state: [B,H,N,d_v(+1)]`` (the ``+1`` column
    carries the global-normalization mass state; absent for ``raw``).

    AN INPUT THE OUTPUT DOES NOT DEPEND ON GETS ``None`` FOR ITS GRADIENT, which
    is autograd's ordinary answer and this function states nothing else.
    ``write_levels_stored`` is the standing case: the clock is
    ``stop_gradient``-ed by the spec, so it never receives one. A test that wants
    a zero there asserts ``grad is None`` and supplies the zero itself.
    """
    validate_normalization(normalization)
    write_levels = tuple(write_levels)
    write_levels_stored = write_levels if write_levels_stored is None else tuple(write_levels_stored)
    B, T, H, d_v = validate_inputs(
        v, read_levels, write_levels, write_levels_stored, g_read, g_write, topology, decay, normalization
    )
    #: DERIVED HERE, not read off `topology`. The oracle's address map is its own
    #: (see `_radix_strides`); `topology` supplies the SHAPE (`widths`), which is
    #: configuration the caller chose, never the interpretation of it.
    widths = topology.widths
    N = _leaf_capacity(widths)
    strides = _radix_strides(widths)

    v64 = v.to(torch.float64)
    read64 = tuple(level.to(torch.float64) for level in read_levels)
    write64 = tuple(level.to(torch.float64) for level in write_levels)
    stored64 = tuple(level.to(torch.float64) for level in write_levels_stored)
    validate_simplex("read_levels", read64)
    validate_simplex("write_levels", write64)

    g_write64 = g_write.to(torch.float64)
    if normalization == "global":
        g_read64 = torch.ones((B, T, H), device=v.device, dtype=torch.float64)
    else:
        g_read64 = g_read.to(torch.float64)

    vf = _fold_bth(v64)                       # [BH,T,d_v]
    read_f = tuple(_fold_bth(level) for level in read64)     # each [BH,T,w_l]
    write_f = tuple(_fold_bth(level) for level in write64)
    stored_f = tuple(_fold_bth(level) for level in stored64)
    g_read_f = _fold_bth(g_read64.unsqueeze(-1)).squeeze(-1)   # [BH,T]
    g_write_f = _fold_bth(g_write64.unsqueeze(-1)).squeeze(-1)

    BH = B * H
    R = g_read_f[..., None] * _leaf_product(read_f, strides, N)    # [BH,T,N]
    W = g_write_f[..., None] * _leaf_product(write_f, strides, N)  # [BH,T,N]

    if decay is None:
        keep = torch.ones((BH, T, N), device=v.device, dtype=torch.float64)
    else:
        dial_f = tuple(
            dial.to(device=v.device, dtype=torch.float64).unsqueeze(0).expand(B, -1, -1).reshape(BH, -1)
            for dial in decay.dials
        )
        rate = _leaf_product(dial_f, strides, N)                          # [BH,N], trains through delta
        c = _leaf_product(stored_f, strides, N).detach()                  # [BH,T,N], stop_gradient (spec §2)
        keep = torch.pow(1.0 - rate[:, None, :], c)                       # [BH,T,N]

    cache = initial_cache(
        initial_state, B=B, H=H, N=N, d_v=d_v, norm=normalization, device=v.device
    ).reshape(BH, N, d_v + int(normalization == "global"))
    value_state = cache[..., :d_v]
    mass_state = cache[..., d_v] if normalization == "global" else None

    outputs = []
    for t in range(T):
        keep_t = keep[:, t, :]           # [BH,N]
        W_t = W[:, t, :]                 # [BH,N]
        # M[t,s,:] = keep[t,s] * M[t-1,s,:] + W[t,s] * v[t,:]  (spec §2)
        value_state = keep_t[..., None] * value_state + W_t[..., None] * vf[:, t, :][:, None, :]
        if mass_state is not None:
            mass_state = keep_t * mass_state + W_t
        # num[t,:] = sum_s R[t,s] * M[t,s,:], inclusive read-after-write (spec §2)
        R_t = R[:, t, :]
        numerator = torch.einsum("bs,bsv->bv", R_t, value_state)
        if mass_state is None:
            outputs.append(numerator)
        else:
            denominator = torch.einsum("bs,bs->b", R_t, mass_state)
            outputs.append(numerator / (denominator[:, None] + eps))

    out_bh = torch.stack(outputs, dim=1) if outputs else vf.new_empty(BH, 0, d_v)
    output_public = out_bh.view(B, H, T, d_v).permute(0, 2, 1, 3).contiguous().to(v.dtype)
    if not output_final_state:
        return output_public, None
    final_cache = value_state if mass_state is None else torch.cat((value_state, mass_state[..., None]), dim=-1)
    return output_public, final_cache.view(
        B, H, N, d_v + int(normalization == "global")).contiguous()


__all__ = ["naive_rola"]
