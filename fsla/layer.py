# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The FSLA layer: the declared routing, its weights, the producer and the op.

    FSLA(Routing([...]), hidden_size=..., num_heads=..., d_v=..., decay=..., layer_idx=...)

The routing is DECLARED (:class:`~fsla.routing.producer.Routing`): data the producer reads. The layer owns every weight the declaration implies -- the packed routing projection, its bias, the write
gain's bias, ``v_proj`` and ``o_proj`` -- and one producer call (:func:`fsla.ops.produce.produce_prefill`) turns a stretch
of tokens into the bundle the op takes. ``d_v`` is the per-head value width, which is the LAYER's: it sizes
``v_proj`` and ``o_proj``, and the routing carries no value geometry at all. ``decay`` is a decay
SOURCE (:mod:`fsla.routing.decay`), the slot that decides where the recurrence's
forgetting rates come from. Everything else the layer used to take was either one of
those facts spelled out flat, or a launch decision the op makes.

**Continuation is a container the layer owns** (:class:`LayerContinuation`). It drops
``value_conv``/``route_conv`` and the conv-ring half of the old ``(state, conv_rings)``
pair (docs/internals/DELETIONS.md) -- they were never reviewed as a mechanism and are
not paper-1 scope. What is left of the pair is one field, ``state``, and the container
exists anyway rather than collapsing back to a bare :class:`~fsla._state.FSLAState`:
future layer-level recurrences become fields on this container, never on the op's, and
the membership rule (docs/internals/state.md §7) needs a place to land them.

**Execution backends: two arms, and the choice is STRUCTURAL.** ``'prefill'`` is the
prefill arm through :func:`fsla.prefill`, whose kernels are not built: the op validates
its inputs and refuses by name (:class:`~fsla.ops._ext.KernelNotBuilt`); ``'decode'`` is the
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

from fsla._inference_copy import InferenceCopy, reads_copy
from fsla.engine.decode_step import decode_step_gated
from fsla.engine.facts.call import refuse_backward
from fsla.interface import (
    BackwardNotImplemented,
    operands_of,
    prefill,
    requires_backward,
)
from fsla.ops._ext import not_built
from fsla.ops.produce import (
    ProducerWeights,
    hidden_contiguous,
    produce_decode,
    produce_prefill,
    projection_dtype,
)
from fsla.routing.decay import validate_decay_source
from fsla.routing.producer import Routing, _init_packed_router
from fsla.routing.side_gain import init_side_gain_bias

if TYPE_CHECKING:
    from fsla._state import FSLAState

__all__ = ["FSLA", "LayerContinuation"]


@dataclass(frozen=True)
class LayerContinuation:
    """The LAYER's own recurrence bundle -- everything ``FSLA.forward`` needs handed
    back to continue a sequence, and nothing the op already owns.

    ``state`` is the op's state (:class:`~fsla._state.FSLAState`) or ``None``
    for a fresh sequence. It is the only field today; a future layer-level recurrence
    is a field on THIS container, never on the op's state, per the membership rule
    (docs/internals/state.md §7): an object holds exactly the recurrences its own
    level introduced.
    """

    state: FSLAState | None = None


def _unpack_continuation(continuation):
    """The op's state out of the pair a forward returns, or ``None``.

    A bare :class:`~fsla.FSLAState` and the retired ``(state, conv_rings)`` tuple are
    both refused by name rather than adopted: :class:`LayerContinuation` is the
    contract in both directions, so accepting a lookalike would make it silently
    optional to carry.
    """
    if continuation is None:
        return None
    if not isinstance(continuation, LayerContinuation):
        raise TypeError(
            "FSLA.forward's `continuation` is a LayerContinuation -- the container a "
            f"previous forward returned -- got {type(continuation).__name__}. A bare "
            "fsla.dense_state() is the OP's half: pass LayerContinuation(state=fsla.dense_state()) "
            "for a fresh sequence, and hand back what forward returned for a "
            "continuation. The old (state, conv_rings) tuple is retired "
            "(docs/internals/DELETIONS.md).")
    return continuation.state


class FSLA(nn.Module):
    """Factored sparse linear attention.

    Args:
        routing: the declared routing (:class:`~fsla.routing.producer.Routing`): the
            levels, the write gain and the projection bias. Data the layer compiles
            into its producer; the weights it implies are the layer's.
        hidden_size / num_heads: the source stream's width and the head count.
        d_v: per-head value width -- the width of the state each leaf holds. It sizes
            ``v_proj`` and ``o_proj``; the default is the width the chunk matrix is
            built at.
        decay: a decay source (:mod:`fsla.routing.decay`), or ``None`` for a
            non-decaying recurrence. Validated against the topology at construction.
        layer_idx: index into the recurrent cache.
        memory_budget: the caller's declared working-set budget in bytes, or ``None`` for none (one prefill is one op call). The
            layer's, not the op's: with a budget the layer asks the planner how many tokens of the call fit it beside the state (a
            function of the prefill launch function, docs/open-work.md row A-f) and loops over segments carrying the state. That
            question cannot be answered yet, so a budgeted prefill is refused by name (:func:`fsla.ops._ext.not_built`, kernel ``prefill``).
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
        memory_budget: int | None = None,  #: reserved: the segment loop, once the prefill launch function (open-work A-f) answers how many tokens fit
    ) -> None:
        super().__init__()
        if not isinstance(routing, Routing):
            raise TypeError(
                "FSLA's first argument is the declared routing, a fsla.Routing -- got "
                f"{type(routing).__name__}. The routing is data the layer compiles into its producer; a "
                "callable feature map is not taken (docs/api.md section 2).")
        if type(d_v) is not int or d_v < 1:
            raise ValueError(f"d_v must be a positive exact int, got {d_v!r}")
        if memory_budget is not None and (type(memory_budget) is not int or memory_budget < 1):
            raise ValueError(f"memory_budget is a positive exact int of bytes, or None; got {memory_budget!r}")
        for name, value in (("hidden_size", hidden_size), ("num_heads", num_heads)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive exact int, got {value!r}")
        if decay is not None:
            validate_decay_source(decay, routing.topology, num_heads)

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.d_v = d_v
        self.routing = routing
        self.decay = decay
        self.layer_idx = layer_idx
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
        #: the weights' copies for a step no gradient flows through (fsla/_inference_copy.py).
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

    def validate_decode_envelope(self, *operation_inputs) -> None:
        """THE DECODE ARM'S OWN CLAUSE: it is forward-only and has no node.

        The gated decode step is a bare extension call with no autograd node on it, so a
        grad-bearing continuation step would return a tensor whose graph is severed
        and say nothing -- the layer-dispatch autograd gap, on the one path that
        still has it. Training mode alone is enough: reentrant checkpointing runs its
        first pass under `no_grad` with `training` still True, and a decode step
        taken there must refuse too.
        """
        if self.training:
            raise BackwardNotImplemented(
                "the FSLA DECODE arm is forward-only and this layer is in TRAINING "
                "mode. Single-token continuation has no reverse pass -- the two-pass "
                "backward is the chunk arm's -- so call `.eval()` for inference, or "
                "take the prefill arm.")
        if requires_backward(*[t for t in operation_inputs if t is not None]):
            raise BackwardNotImplemented(
                "a FSLA DECODE step carrying a gradient: the single-token arm is "
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
        forward returned. The op's state lives on it as ``.state``; a bare
        :class:`~fsla.FSLAState` or the retired ``(state, conv_rings)`` tuple is
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
                f"FSLA.forward got an unsupported keyword {key!r}. Variable-length packing "
                "(`cu_seqlens`) and attention-probability outputs are not implemented; FSLA "
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
            projected = self._step(x, state)
            self.last_execution_backend = "decode"
            return projected, LayerContinuation(state=state)

        out, state = self._prefill(x, state, decay)
        self.last_execution_backend = "prefill"
        return self._project(out.reshape(B, L, H * self.head_v_dim), x.dtype), LayerContinuation(state=state)

    def _step(self, x: torch.Tensor, state) -> torch.Tensor:
        """ONE DECODED TOKEN WITH NO HOST READ INSIDE: the producer's launch, the gated decode step and the output
        projection, ``x [B, 1, hidden]`` to the projected ``[B, 1, hidden]`` -- CUDA-graph capturable: the step builds its
        descriptor on every call (memoized process-wide: no device read after the first).
        """
        decay = None if self.decay is None else self.decay()
        routes, values, masks = produce_decode(x, self.routing, self._weights(x))
        #: decode's envelope is its own -- BC-blind, general in the topology -- so it is never asked the PREFILL
        #: matrix's questions.
        self.validate_decode_envelope(*operands_of(routes, values, decay))
        out, _state = decode_step_gated(routes, values, masks, state, decay=decay)
        return self._project(out.reshape(x.shape[0], 1, self.num_heads * self.head_v_dim), x.dtype)

    def _project(self, out: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """The output projection, bf16 operands, in the caller's dtype."""
        return nn.functional.linear(out.to(torch.bfloat16), self._bf16(self.o_proj.weight, self._o_copy)).to(dtype)

    def _produce(self, x: torch.Tensor):
        """One stretch of tokens into the format: ``(routes, values, masks, density)``, from one producer call, the masks in the
        prefill's layout and the density of those masks."""
        return produce_prefill(x, self.routing, self._weights(x))

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
        """THE PREFILL (docs/api.md section 4.2). With no ``memory_budget`` it is one op call: the producer's one call and :func:`fsla.prefill`, which builds its own descriptor and workspace. WITH a budget the layer must ask the planner how many tokens fit it beside the state -- the answer is the prefill
        launch function's (docs/open-work.md row A-f), which is not written -- and loop over segments carrying the state; until then a budgeted
        prefill is refused by name here, the one place the layer would have segmented."""
        if self.memory_budget is not None:
            not_built("prefill")
        routes, values, masks, density = self._produce(x)
        refuse_backward(routes, values, decay)  #: the gradient clause speaks before the descriptor's, so a training call is told why
        return prefill(routes, values, masks, density, state=state, decay=decay)

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
