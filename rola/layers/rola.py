# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The RoLA layer: a value projection, the feature-map slot, and the op.

    RoLA(hidden_size, num_heads, routes, d_v=..., decay=..., layer_idx=...)

``routes`` is a :class:`~rola.routing.factors.RouteProducer` -- the feature map, which
carries the topology, the normalization arm and the producers that fill each level.
``d_v`` is the per-head value width, which is the LAYER's: it sizes ``v_proj`` and
``o_proj``, and the routing carries no value geometry at all. ``decay`` is a decay
SOURCE (:mod:`rola.routing.decay`), the slot that decides where the recurrence's
forgetting rates come from. Everything else the layer used to take was either one of
those facts spelled out flat, or a launch decision the op makes.

**Execution backends: two kernels, and the choice is STRUCTURAL.** ``'cuda'`` is the
chunk consumer through :func:`rola.rola_op`; ``'decode'`` is the ``T = 1`` sibling
kernel, which bypasses selection architecturally (at one token there is no tile to
choose, no split, a statistics population of one) and carries its own envelope --
BC-blind and general in the topology, so it is never gated on the prefill matrix.

**THE BUILT ENVELOPE IS THE WHOLE SURFACE** (docs/api.md §1). Everything outside it
raises: training mode and any grad-bearing call (the backward arrives with the native
backward), a CPU device, decay, a jagged topology, ``d_v != 64``, an unbuilt
``(D, B)``, ``T`` off the chunk. The layer's own clause is ``self.training``, which
has no meaning for a free function -- reentrant checkpointing runs its first pass under
``torch.no_grad()`` with ``module.training`` still ``True``, and that pass must refuse
too rather than produce a forward no backward can follow.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from einops import rearrange

from rola.interface import (
    BACKWARD_ARRIVES_WITH,
    BackwardNotImplemented,
    operands_of,
    requires_backward,
    rola_op,
)
from rola.ops.decode import decode_forward
from rola.routing.decay import validate_decay_source
from rola.routing.factors import RouteProducer

if TYPE_CHECKING:
    from rola._state import RoLAState

__all__ = ["RoLA"]


def _unpack_continuation(state):
    """``(op state, conv rings)`` from the pair a forward returns, or ``(None, None)``.

    A bare :class:`~rola.RoLAState` is refused rather than adopted: the pair is the
    contract in both directions, and accepting the half that happens to be the op's
    would make the rings silently optional to carry.
    """
    if state is None:
        return None, None
    if type(state) is not tuple or len(state) != 2:
        raise TypeError(
            "RoLA.forward's `state` is the (state, conv_rings) pair a forward returns, "
            f"got {type(state).__name__}. A bare rola.state() is the OP's half: pass "
            "(state, None) for a fresh sequence, and hand back what forward returned "
            "for a continuation.")
    return state


class RoLA(nn.Module):
    """Routed linear attention.

    Args:
        hidden_size: model dim. Must match the routing gateway's.
        num_heads: head count ``H``. Must match the routing gateway's.
        routes: the feature-map slot -- a :class:`~rola.routing.factors.RouteProducer`.
            It owns the topology (levels, widths), the normalization arm and the side
            gains, because those are facts about the map rather than about the layer
            wrapping it.
        d_v: per-head value width -- the width of the state each leaf holds. It sizes
            ``v_proj`` and ``o_proj``, which is why it is the layer's; the default is
            the width the chunk matrix is built at.
        decay: a decay source (:mod:`rola.routing.decay`), or ``None`` for a
            non-decaying recurrence. Validated against the topology at construction.
        value_conv: an optional causal short convolution over the value stream, e.g.
            ``fla.modules.ShortConvolution``. A module, not three scalars: the layer
            does not need to know how it is configured, only where it goes.
        route_conv: the same, over the routing source stream.
        layer_idx: index into the recurrent cache.
        expert: :class:`rola.expert.PlanOverrides`. Everything it pins is an axis the
            op decides; it is quarantined because a caller pinning one is not
            describing a model.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        routes: RouteProducer,
        *,
        d_v: int = 64,
        decay=None,
        value_conv: nn.Module | None = None,
        route_conv: nn.Module | None = None,
        layer_idx: int | None = None,
        expert=None,
    ) -> None:
        super().__init__()
        if type(routes) is not RouteProducer:
            raise TypeError(
                "RoLA's `routes` argument is the feature-map slot and must be a "
                f"rola.RouteProducer, got {type(routes).__name__}. It is what carries the "
                "topology and the producers that fill each level.")
        if routes.hidden_size != hidden_size or routes.num_heads != num_heads:
            raise ValueError(
                f"this layer is {hidden_size} wide with {num_heads} heads, but its RouteProducer "
                f"was built for {routes.hidden_size}/{routes.num_heads}")
        if type(d_v) is not int or d_v < 1:
            raise ValueError(f"d_v must be a positive exact int, got {d_v!r}")
        if decay is not None:
            validate_decay_source(decay, routes.topology, num_heads)

        from rola.expert import PlanOverrides

        overrides = PlanOverrides() if expert is None else PlanOverrides.of(expert)

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.d_v = d_v
        self.routes = routes
        self.decay = decay
        self.value_conv = value_conv
        self.route_conv = route_conv
        self.layer_idx = layer_idx
        self.expert = overrides

        self.value_dim = num_heads * d_v
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        # Execution telemetry. Plain attributes, not buffers: diagnostic state must not
        # enter a state_dict and must not be cast by `_apply`. Which kernel ran is
        # asserted from these rather than from wall-clock, which cannot tell two arms
        # apart.
        self._execution_backend_counts = {"cuda": 0, "decode": 0}
        self.last_execution_backend: str | None = None

    # --- accessors ----------------------------------------------------------

    @property
    def topology(self):
        """The routing topology, which is the feature map's, not the layer's."""
        return self.routes.topology

    @property
    def head_v_dim(self) -> int:
        return self.d_v

    @property
    def normalization(self) -> str:
        return self.routes.normalization

    @property
    def execution_backend_counts(self) -> dict[str, int]:
        """How many forwards took each backend, since construction. Read-only view."""
        return dict(self._execution_backend_counts)

    def validate_execution_envelope(self, *operation_inputs) -> None:
        """THE MODULE'S CLAUSES: training mode, and a gradient reaching the operation.

        Both refuse, because neither has a backward to be followed by. The
        configuration clauses are the op's (:func:`rola.interface.require_envelope`),
        asked separately so decode -- whose envelope is its own -- is never gated on
        the prefill matrix.
        """
        if self.training:
            raise BackwardNotImplemented(
                f"RoLA is in TRAINING mode and there is no training path: "
                f"{BACKWARD_ARRIVES_WITH}. Call `.eval()` for inference.")
        if requires_backward(*[t for t in operation_inputs if t is not None]):
            raise BackwardNotImplemented(
                f"a RoLA forward carrying a gradient: {BACKWARD_ARRIVES_WITH}. Run "
                "under torch.no_grad(), or freeze the operands.")

    def _bf16_projection_active(self, x: torch.Tensor) -> bool:
        """Whether this call's v/o projections run as bf16 GEMMs (P09 family).

        The GEMMs run with bf16 operands and cuBLAS's fp32 compute type (the
        accumulate-in-fp32 ruling; the global reduced-precision-reduction knob is
        deliberately not touched -- this is the same environment the FA comparison arm
        runs in).
        """
        return x.is_cuda and self.expert.mma_precision == "bf16"

    # --- forward -------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        state: tuple[RoLAState | None, tuple | None] | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, tuple[RoLAState | None, tuple | None]]:
        """``(y, (state, conv_rings))`` -- ALWAYS the pair. CONTINUATION IS LAYERED.

        ``state`` is ``None`` (a stateless call) or the pair a previous forward
        returned. The op's paged tensor is the op's; the short-convolution rings are
        the LAYER's, because the layer is what introduced that recurrence, and they
        are ``None`` when no convolution is configured. Each level of the stack holds
        exactly the recurrences its own level introduced (docs/api.md §1.2).
        """
        if attention_mask is not None and attention_mask.dim() != 2:
            raise ValueError(
                "attention_mask must be a [batch, seq_len] 0-1 padding matrix; "
                "[batch, seq_len, seq_len] masks are not supported.")
        for key in kwargs:
            # READ WHAT YOU SUPPORT, RAISE ON THE REST. A swallowed keyword is a knob
            # that silently does nothing, which is how a caller comes to believe a model
            # was configured a way it never was.
            raise TypeError(
                f"RoLA.forward got an unsupported keyword {key!r}. Variable-length packing "
                "(`cu_seqlens`) and attention-probability outputs are not implemented; RoLA "
                "has no attention matrix to return.")

        state, conv_rings = _unpack_continuation(state)
        x = hidden_states
        B, L, _ = x.shape
        H = self.num_heads

        # P09 matched-dtype family: on the kernel-bound bf16 arm both projections
        # run as bf16 GEMMs (weights cast per call), and `v` FLOWS bf16 straight
        # into the op -- round-tripping it through the module dtype was MEASURED
        # to cost two elementwise casts per call (a v downcast + a y upcast,
        # ~12-13% of a small cell's op time; workflows/perf/p09_bf16_family.md).
        # A configured value_conv runs in the module dtype, so its presence is
        # the one (structural) case where `v` returns to `x.dtype` here and the
        # op re-narrows at its boundary -- exactly, the values being
        # bf16-rounded already.
        bf16_proj = self._bf16_projection_active(x)
        if bf16_proj:
            v = nn.functional.linear(x.to(torch.bfloat16), self.v_proj.weight.to(torch.bfloat16))
            if self.value_conv is not None:
                v = v.to(x.dtype)
        else:
            v = self.v_proj(x)
        #: The rings ARE part of the recurrence: a continuation that dropped them would
        #: restart the causal conv from zeros and compute a different function for the
        #: first `conv_size - 1` tokens of every step.
        stateful = state is not None
        conv_states = [None, None] if conv_rings is None else list(conv_rings)
        if self.value_conv is not None:
            v, conv_states[0] = self.value_conv(
                v, cache=conv_states[0], output_final_state=stateful)
        source = x
        if self.route_conv is not None:
            source, conv_states[1] = self.route_conv(
                x, cache=conv_states[1], output_final_state=stateful)

        v = rearrange(v, "b l (h d) -> b l h d", d=self.head_v_dim)
        routes = self.routes(source)
        decay = None if self.decay is None else self.decay()
        self.validate_execution_envelope(*operands_of(routes, v, decay))
        #: a state that carries CONTENTS, which is what makes a step a continuation.
        #: A bound-but-empty state is a fresh sequence, and takes the prefill arm.
        carrying = stateful and state.carries

        if L == 1 and carrying:
            #: decode's envelope is its own -- BC-blind, general in the topology -- so
            #: it is never asked the PREFILL matrix's questions.
            out = self._decode(routes, v, decay, state)
            backend = "decode"
        else:
            out, state = rola_op(routes, v, decay, state=state, expert=self.expert)
            backend = "cuda"

        self._execution_backend_counts[backend] += 1
        self.last_execution_backend = backend
        conv_rings = (tuple(conv_states)
                      if self.value_conv is not None or self.route_conv is not None
                      else None)
        out = out.reshape(B, L, H * self.head_v_dim)
        if bf16_proj:
            #: `out`'s VALUES are already bf16-rounded (y is cast post-divide in the
            #: kernel), so this operand cast is exact.
            projected = nn.functional.linear(
                out.to(torch.bfloat16), self.o_proj.weight.to(torch.bfloat16)).to(x.dtype)
        else:
            projected = self.o_proj(out)
        return projected, (state, conv_rings)

    # --- the arm that needs layer-owned state --------------------------------

    def _decode(self, routes, v, decay, state):
        """The ``T = 1`` sibling kernel, which owns its own state I/O.

        The layer's only contribution is the plan overrides; the carrier, the scratch,
        the backing and its growth all live behind ``decode_forward``
        (``docs/internals/decode.md``), exactly as the chunk arm's tables live behind
        ``rola_op``.
        """
        out, state = decode_forward(routes, v, state, decay=decay, expert=self.expert)
        return out

    # --- state accounting (zoology's Hybrid.state_size reads these) ----------

    def get_stats(self):
        cols = self.head_v_dim + int(self.normalization != "raw")
        return {
            "d_v": self.head_v_dim,
            "n_heads": self.num_heads,
            "num_chunks": self.topology.N,
            "state_floats": self.num_heads * self.topology.N * cols,
        }

    def state_size(self, sequence_length: int = None, **kwargs) -> int:
        return self.get_stats()["state_floats"]
