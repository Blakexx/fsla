# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The RoLA layer: the declared routing, its weights, the producer and the op.

    RoLA(Routing([...]), hidden_size=..., num_heads=..., d_v=..., decay=..., layer_idx=...)

The routing is DECLARED (:class:`~rola.routing.producer.Routing`): data the layer compiles once into a producer
plan. The layer owns every weight the declaration implies -- the packed routing projection, its bias, the write
gain's bias, ``v_proj`` and ``o_proj`` -- and one producer call (:func:`rola.ops.produce.produce`) turns a stretch
of tokens into the bundle the op takes. ``d_v`` is the per-head value width, which is the LAYER's: it sizes
``v_proj`` and ``o_proj``, and the routing carries no value geometry at all. ``decay`` is a decay
SOURCE (:mod:`rola.routing.decay`), the slot that decides where the recurrence's
forgetting rates come from. Everything else the layer used to take was either one of
those facts spelled out flat, or a launch decision the op makes.

**Continuation is a container the layer owns** (:class:`LayerContinuation`). It drops
``value_conv``/``route_conv`` and the conv-ring half of the old ``(state, conv_rings)``
pair (docs/internals/DELETIONS.md) -- they were never reviewed as a mechanism and are
not paper-1 scope. What is left of the pair is one field, ``state``, and the container
exists anyway rather than collapsing back to a bare :class:`~rola._state.RoLAState`:
future layer-level recurrences become fields on this container, never on the op's, and
the membership rule (docs/internals/state.md §7) needs a place to land them.

**Execution backends: two arms, and the choice is STRUCTURAL.** ``'cuda'`` is the
prefill arm through :func:`rola.rola_op` — DELETED pending the rebuild (the op raises,
naming the ruling; baseline = tag ``baseline/pre-k31``); ``'decode'`` is the
``T = 1`` sibling kernel, which bypasses selection architecturally (at one token
there is no tile to choose, no split, a statistics population of one) and carries
its own envelope -- BC-blind and general in the topology, so it is never gated on
the prefill matrix.

**THE BUILT ENVELOPE IS THE WHOLE SURFACE** (docs/api.md §1). Everything outside it
raises: training mode and any grad-bearing call (the backward arrives with the native
backward), a CPU device, decay, a jagged topology, ``d_v != 64``, an unbuilt
``(D, B)``. The layer's own clause is ``self.training``, which
has no meaning for a free function -- reentrant checkpointing runs its first pass under
``torch.no_grad()`` with ``module.training`` still ``True``, and that pass must refuse
too rather than produce a forward no backward can follow.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from rola._inference_copy import InferenceCopy, reads_copy
from rola._state import state as rola_state
from rola.engine import PlanOverrides
from rola.engine.dags.decode_dag import decode_forward
from rola.interface import (
    BackwardNotImplemented,
    operands_of,
    requires_backward,
    rola_op,
)
from rola.ops.produce import ProducerPlan, ProducerWeights, hidden_contiguous, produce, projection_dtype
from rola.routing.decay import validate_decay_source
from rola.routing.producer import Routing, _init_packed_router
from rola.routing.side_gain import init_side_gain_bias

if TYPE_CHECKING:
    from rola._state import RoLAState

__all__ = ["LayerContinuation", "RoLA"]


@dataclass(frozen=True)
class LayerContinuation:
    """The LAYER's own recurrence bundle -- everything ``RoLA.forward`` needs handed
    back to continue a sequence, and nothing the op already owns.

    ``state`` is the op's paged tensor (:class:`~rola._state.RoLAState`) or ``None``
    for a fresh sequence. It is the only field today; a future layer-level recurrence
    is a field on THIS container, never on the op's state, per the membership rule
    (docs/internals/state.md §7): an object holds exactly the recurrences its own
    level introduced.
    """

    state: RoLAState | None = None


def _unpack_continuation(continuation):
    """The op's state out of the pair a forward returns, or ``None``.

    A bare :class:`~rola.RoLAState` and the retired ``(state, conv_rings)`` tuple are
    both refused by name rather than adopted: :class:`LayerContinuation` is the
    contract in both directions, so accepting a lookalike would make it silently
    optional to carry.
    """
    if continuation is None:
        return None
    if not isinstance(continuation, LayerContinuation):
        raise TypeError(
            "RoLA.forward's `continuation` is a LayerContinuation -- the container a "
            f"previous forward returned -- got {type(continuation).__name__}. A bare "
            "rola.state() is the OP's half: pass LayerContinuation(state=rola.state()) "
            "for a fresh sequence, and hand back what forward returned for a "
            "continuation. The old (state, conv_rings) tuple is retired "
            "(docs/internals/DELETIONS.md).")
    return continuation.state


class RoLA(nn.Module):
    """Routed linear attention.

    Args:
        routing: the declared routing (:class:`~rola.routing.producer.Routing`): the
            levels, the write gain and the projection bias. Data the layer compiles
            into its producer; the weights it implies are the layer's.
        hidden_size / num_heads: the source stream's width and the head count.
        d_v: per-head value width -- the width of the state each leaf holds. It sizes
            ``v_proj`` and ``o_proj``; the default is the width the chunk matrix is
            built at.
        decay: a decay source (:mod:`rola.routing.decay`), or ``None`` for a
            non-decaying recurrence. Validated against the topology at construction.
        layer_idx: index into the recurrent cache.
        expert: :class:`rola.expert.PlanOverrides`. Everything it pins is an axis the
            op decides; it is quarantined because a caller pinning one is not
            describing a model.
    """

    def __init__(
        self,
        routing: Routing,
        *,
        hidden_size: int,
        num_heads: int,
        d_v: int = 64,
        decay=None,
        layer_idx: int | None = None,
        expert=None,
        memory_budget: int | None = None,  #: reserved: the caller's declaration (docs/api.md section 4.2)
    ) -> None:
        super().__init__()
        if not isinstance(routing, Routing):
            raise TypeError(
                "RoLA's first argument is the declared routing, a rola.Routing -- got "
                f"{type(routing).__name__}. The routing is data the layer compiles into its producer; a "
                "callable feature map is not taken (docs/api.md section 2).")
        if type(d_v) is not int or d_v < 1:
            raise ValueError(f"d_v must be a positive exact int, got {d_v!r}")
        if memory_budget is not None and (type(memory_budget) is not int or memory_budget < 1):
            raise ValueError(f"memory_budget is a positive exact int of bytes, or None; got {memory_budget!r}")
        #: the plan validates `hidden_size` and `num_heads`
        self.plan = ProducerPlan(routing=routing, num_heads=num_heads, hidden_size=hidden_size, d_v=d_v)
        if decay is not None:
            validate_decay_source(decay, routing.topology, num_heads)

        overrides = PlanOverrides() if expert is None else PlanOverrides.of(expert)

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.d_v = d_v
        self.routing = routing
        self.decay = decay
        self.layer_idx = layer_idx
        self.expert = overrides
        self.memory_budget = memory_budget

        #: ONE ROUTING PARAMETER, and the write gain's columns are its trailing span: the gain is a function of the
        #: same stream as the routing logits, so its column is one more column of the same GEMM.
        self.route_W = nn.Parameter(torch.empty(num_heads, hidden_size, routing.columns))
        _init_packed_router(self.route_W)
        if routing.bias:
            self.route_bias = nn.Parameter(torch.zeros(num_heads, routing.columns))
        else:
            self.register_parameter("route_bias", None)
        gain_bias = 0.0 if routing.gain_bias_init is None else init_side_gain_bias(routing.gain_bias_init)
        self.gain_bias = nn.Parameter(torch.full((num_heads, routing.gain_columns), gain_bias))
        self.value_dim = num_heads * d_v
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
        #: the weights' copies for a step no gradient flows through (rola/_inference_copy.py).
        self._route_copy = InferenceCopy(lay=hidden_contiguous)
        self._v_copy = InferenceCopy(lay=None)
        self._o_copy = InferenceCopy(lay=None)

        #: Which kernel the last forward took. A plain attribute, not a buffer:
        #: diagnostic state must not enter a state_dict and must not be cast by
        #: `_apply`. It is PUBLISHED for the measurement guards -- a benchmark process
        #: cannot spy on the call it is timing, and wall-clock cannot tell two arms
        #: apart.
        self.last_execution_backend: str | None = None

    # --- accessors ----------------------------------------------------------

    @property
    def topology(self):
        """The routing topology, the declaration's."""
        return self.routing.topology

    @property
    def head_v_dim(self) -> int:
        return self.d_v

    # --- segmentation (docs/api.md section 4.2) -------------------------------

    def segment_bytes_per_token(self) -> int:
        """AN UPPER BOUND on a prefill segment's working set per batch-token, in bytes: every per-token tensor
        a segment allocates, summed as if all were alive at once. Gated against the measured peak
        (tests/integration/test_layer_segments.py); a kernel port that drops a tensor drops its term here.

        Per head, the routing's ``W`` digits of both sides: the router's fp32 logits (4), the solve's bf16
        levels (2), the stored rows (1), the seam's fp32 levels (4), the kernels' bf16 planes (2) and the
        activity pass's bf16 re-pack (2); the ``d_v`` value channels: the bf16 projection (2), the two stored
        digit planes (2), the seam's fp32 values (4), the kernels' two bf16 layouts (4), ``num`` (4) and ``y``
        (4); and the per-token scalars (gains, scales, masses, den: 64). Per token, the hidden stream's two bf16
        casts (the value and the router projections' inputs, 4).
        """
        W = 2 * sum(self.topology.widths)
        per_head = W * (4 + 2 + 1 + 4 + 2 + 2) + self.d_v * (2 + 2 + 4 + 4 + 4 + 4) + 64
        return self.num_heads * per_head + 4 * self.hidden_size

    def state_bytes(self, batch: int) -> int:
        """The carried state at full residency, in bytes: the fixed part of a segmented call's footprint."""
        return batch * self.num_heads * self.topology.N * (self.d_v + 1) * 4

    def segment_tokens(self, batch: int, length: int) -> int:
        """THE SEGMENT LENGTH, derived from the declared ``memory_budget``: the largest whole number of
        windows whose working set fits beside the state, capped at ``length``. No budget: one segment. A
        budget below one window beside the state is refused."""
        from rola.ops.carry import WINDOW

        if self.memory_budget is None:
            return length
        per_window = WINDOW * batch * self.segment_bytes_per_token()
        windows = (self.memory_budget - self.state_bytes(batch)) // per_window
        if windows < 1:
            raise ValueError(
                f"memory_budget={self.memory_budget} bytes cannot hold one {WINDOW}-token window "
                f"({per_window} bytes at batch {batch}) beside the state ({self.state_bytes(batch)} bytes)")
        return min(length, int(windows) * WINDOW)

    def validate_decode_envelope(self, *operation_inputs) -> None:
        """THE DECODE ARM'S OWN CLAUSE: it is forward-only and has no node.

        `decode_forward` is a bare extension call with no autograd node on it, so a
        grad-bearing continuation step would return a tensor whose graph is severed
        and say nothing -- the layer-dispatch autograd gap, on the one path that
        still has it. Training mode alone is enough: reentrant checkpointing runs its
        first pass under `no_grad` with `training` still True, and a decode step
        taken there must refuse too.
        """
        if self.training:
            raise BackwardNotImplemented(
                "the RoLA DECODE arm is forward-only and this layer is in TRAINING "
                "mode. Single-token continuation has no reverse pass -- the two-pass "
                "backward is the chunk arm's -- so call `.eval()` for inference, or "
                "take the prefill arm.")
        if requires_backward(*[t for t in operation_inputs if t is not None]):
            raise BackwardNotImplemented(
                "a RoLA DECODE step carrying a gradient: the single-token arm is "
                "forward-only and has no autograd node, so this would sever the "
                "graph silently. Run under torch.no_grad(), or freeze the operands.")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        continuation: LayerContinuation | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, LayerContinuation]:
        """``(y, LayerContinuation)`` -- ALWAYS the container. CONTINUATION IS LAYERED.

        ``continuation`` is ``None`` (a stateless call) or the container a previous
        forward returned. The op's paged tensor lives on it as ``.state``; a bare
        :class:`~rola.RoLAState` or the retired ``(state, conv_rings)`` tuple is
        refused by name (docs/api.md §1.2).
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

        state = _unpack_continuation(continuation)
        x = hidden_states
        B, L, _ = x.shape
        H = self.num_heads
        decay = None if self.decay is None else self.decay()
        #: a state that carries CONTENTS, which is what makes a step a continuation.
        #: A bound-but-empty state is a fresh sequence, and takes the prefill arm.
        carrying = state is not None and state.carries

        if L == 1 and carrying:
            routes, values = self._produce(x)
            #: decode's envelope is its own -- BC-blind, general in the topology -- so
            #: it is never asked the PREFILL matrix's questions.
            self.validate_decode_envelope(*operands_of(routes, values, decay))
            out, state = decode_forward(routes, values, state, decay=decay, expert=self.expert)
            self.last_execution_backend = "decode"
        else:
            out, state = self._prefill(x, state, decay)
            self.last_execution_backend = "cuda"

        out = out.reshape(B, L, H * self.head_v_dim)
        projected = nn.functional.linear(out.to(torch.bfloat16), self._bf16(self.o_proj.weight, self._o_copy)).to(x.dtype)
        return projected, LayerContinuation(state=state)

    def _produce(self, x: torch.Tensor):
        """One stretch of tokens into the format: ``(routes, values)``, from one producer call."""
        return produce(x, self.plan, self._weights(x))

    def _weights(self, x: torch.Tensor) -> ProducerWeights:
        """The layer's weights as the producer reads them: the reused copies when no gradient flows."""
        dtype = projection_dtype(x)
        route_W = self._route_copy(self.route_W, dtype) if reads_copy(self.route_W) else self.route_W.to(dtype)
        return ProducerWeights(route_W=route_W, route_bias=self.route_bias, gain_bias=self.gain_bias,
                               v_W=self._bf16(self.v_proj.weight, self._v_copy))

    @staticmethod
    def _bf16(W: torch.Tensor, copy: InferenceCopy) -> torch.Tensor:
        """A projection weight in bf16: the reused copy when no gradient flows, else the cast in the graph."""
        return copy(W, torch.bfloat16) if reads_copy(W) else W.to(torch.bfloat16)

    def _prefill(self, x: torch.Tensor, state, decay):
        """THE SEGMENT LOOP (docs/api.md section 4.2): each segment is projected, run and its state carried to
        the next, so the projections are bounded with the op. A stateless call that spans segments carries an
        internal state and hands none back."""
        B, L, _ = x.shape
        T = self.segment_tokens(B, L)
        carried = state if state is not None or T >= L else rola_state()
        outs = []
        for start in range(0, L, T):
            routes, values = self._produce(x[:, start:start + T])
            y, carried = rola_op(routes, values, decay, state=carried, expert=self.expert)
            outs.append(y)
        return torch.cat(outs, dim=1) if len(outs) > 1 else outs[0], state if state is None else carried

    # --- state accounting (zoology's Hybrid.state_size reads these) ----------

    def get_stats(self):
        cols = self.head_v_dim + 1  # the mass column is unconditional
        return {
            "d_v": self.head_v_dim,
            "n_heads": self.num_heads,
            "num_chunks": self.topology.N,
            "state_floats": self.num_heads * self.topology.N * cols,
        }

    def state_size(self, sequence_length: int = None, **kwargs) -> int:
        return self.get_stats()["state_floats"]
