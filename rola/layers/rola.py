# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The RoLA layer: a value projection, the feature-map slot, and the op.

    RoLA(hidden_size, num_heads, routes, decay=..., layer_idx=...)

``routes`` is a :class:`~rola.routing.factors.RouteProducer` -- the feature map, which
carries the topology, the value width, the normalization arm and the producers that
fill each level. ``decay`` is a decay SOURCE (:mod:`rola.routing.decay`), the slot that
decides where the recurrence's forgetting rates come from. Everything else the layer
used to take was either one of those two facts spelled out flat, or a launch decision
the planner makes.

**Execution backends, and the rule that picks between them.** The layer has three, and
the choice is STRUCTURAL -- decided before any of them runs:

* ``'oracle'`` -- ``naive_rola``, the canonical reference run in the layer's working
  dtype. It is fully differentiable and it is the ONLY implemented training path, which
  is not the same thing as a fallback: nothing degrades into it after a failure, and no
  configuration reaches it because something else declined.
* ``'cuda'`` -- the chunk consumer, through :func:`rola.rola_op`.
* ``'decode'`` -- the ``T = 1`` sibling kernel. At one token every selection decision is
  degenerate (no tile to choose, no split, a statistics population of one), so decode
  bypasses selection ARCHITECTURALLY rather than running the planner on degenerate
  inputs. It is a SIBLING of the tiled path, not a specialization: shared production
  contract, own consumer, frozen config. Its envelope is its own: decode is BC-blind
  and general in the topology, so it is never gated on the prefill matrix.

**The autograd trap this dispatch exists to close** (`[[layer-dispatch-autograd-gap]]`):
a forward-only kernel reachable from a gradient-requiring context does not raise. It
returns a tensor with no ``grad_fn`` and the graph is severed here. ``rola_op`` owns the
predicate -- it is evaluated on the tensors the OPERATION consumes, not on the module's
parameters -- and the layer adds ONE condition of its own, which has no meaning for a
free function: ``self.training``. Reentrant gradient checkpointing runs its first pass
under ``torch.no_grad()`` with ``module.training`` still ``True``, so a predicate keyed
only on gradient reachability would take the CUDA path there while backward recomputes
the same step on the oracle -- the value flowing downstream and the gradient taken would
come from two implementations agreeing only to a tolerance. Gating on training mode
closes that structurally.

**THE KERNEL ENVELOPE IS THE BUILT CHUNK MATRIX** (the unsupported-topology
retirement, 2026-08-14). Uniform widths, ``d_v = 64``, ``(D, B)`` in the manifest's
combinations, ``T`` a multiple of the chunk. A configuration outside it -- a jagged
topology, ``d_v = 32``, ``raw`` or per-state normalization, an unbuilt ``(D, B)`` --
DISPATCHES to the oracle, exactly as decay and an unratified activation already do,
and :attr:`~rola.interface.Dispatch.no_kernel_arm` names which clause refused it. That
is a dispatch and not a degradation: nothing failed, the kernel was never asked, and
the configuration is a labelled research surface rather than a broken one. It is also
a REAL capability narrowing -- those shapes had a kernel and now have the oracle's
throughput -- and ``docs/open-work.md`` carries the card that widens the matrix.
A caller who overrules the dispatch (``rola_op(_arm="kernel")``) still gets a loud
refusal: dispatch is not a lie, override is.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from einops import rearrange

from rola.interface import dispatch_for, rola_op, select_backend
from rola.ops.decode import decode_forward
from rola.routing.decay import validate_decay_source
from rola.routing.factors import RouteProducer

if TYPE_CHECKING:
    from rola._state import RoLAState

__all__ = ["RoLA"]


class RoLA(nn.Module):
    """Routed linear attention.

    Args:
        hidden_size: model dim. Must match the routing gateway's.
        num_heads: head count ``H``. Must match the routing gateway's.
        routes: the feature-map slot -- a :class:`~rola.routing.factors.RouteProducer`.
            It owns the topology (levels, widths), the per-head value width ``d_v``, the
            normalization arm and the side gains, because all of those are facts about
            the map rather than about the layer wrapping it.
        decay: a decay source (:mod:`rola.routing.decay`), or ``None`` for a
            non-decaying recurrence. Validated against the topology at construction.
        value_conv: an optional causal short convolution over the value stream, e.g.
            ``fla.modules.ShortConvolution``. A module, not three scalars: the layer
            does not need to know how it is configured, only where it goes.
        route_conv: the same, over the routing source stream.
        layer_idx: index into the recurrent cache.
        expert: :class:`rola.expert.PlanOverrides`. Everything it pins is an axis the
            planner derives; it is quarantined because a caller pinning one is not
            describing a model.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        routes: RouteProducer,
        *,
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
                "topology, the value width and the producers that fill each level.")
        if routes.hidden_size != hidden_size or routes.num_heads != num_heads:
            raise ValueError(
                f"this layer is {hidden_size} wide with {num_heads} heads, but its RouteProducer "
                f"was built for {routes.hidden_size}/{routes.num_heads}")
        if decay is not None:
            validate_decay_source(decay, routes.topology, num_heads)

        from rola.expert import PlanOverrides
        from rola.planner.derive import feasible_bc

        overrides = PlanOverrides() if expert is None else PlanOverrides.of(expert)
        # CONSTRUCTION TIME, NOT PLAN TIME. `feasible_bc` is the alignment-admissible set
        # intersected with the instantiated ladder, and it raises with the topology's own
        # diagnosis when the ladder cannot serve it at all. Resolving it here means a
        # topology nothing implements, and an inadmissible expert pin, both fail on the
        # line that chose them rather than inside a forward several frames away.
        #: The `BC` ladder this topology can run on. An expert pin SHRINKS it to itself.
        self.feasible_BC = feasible_bc(
            routes.topology.widths,
            **({} if overrides.state_block is None else {"pinned": overrides.state_block}))

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.routes = routes
        self.decay = decay
        self.value_conv = value_conv
        self.route_conv = route_conv
        self.layer_idx = layer_idx
        self.expert = overrides

        self.value_dim = num_heads * routes.d_v
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        # Execution telemetry. Plain attributes, not buffers: diagnostic state must not
        # enter a state_dict and must not be cast by `_apply`. The dispatch gate asserts
        # on these rather than on wall-clock, because timing cannot distinguish "took
        # the CUDA path" from "took a fast oracle path" and that is exactly the
        # distinction the autograd trap turns on.
        self._execution_backend_counts = {"oracle": 0, "cuda": 0, "decode": 0}
        self.last_execution_backend: str | None = None
        self.last_selected_plan = None

    # --- accessors ----------------------------------------------------------

    @property
    def topology(self):
        """The routing topology, which is the feature map's, not the layer's."""
        return self.routes.topology

    @property
    def head_v_dim(self) -> int:
        return self.routes.d_v

    @property
    def normalization(self) -> str:
        return self.routes.normalization

    @property
    def state_block(self) -> int:
        """``BC``, the leafs one owner block owns -- pinned, or the largest feasible.

        It is a PROPERTY rather than stored state because it is not a decision this
        layer makes: the planner chooses it per launch from the feasible set, and an
        expert pin is a set of one.
        """
        return self.feasible_BC[-1]

    @property
    def execution_backend_counts(self) -> dict[str, int]:
        """How many forwards took each backend, since construction. Read-only view."""
        return dict(self._execution_backend_counts)

    def select_execution_backend(self, *operation_inputs) -> str:
        """``'oracle'`` or ``'cuda'`` from the module's mode and the OPERATION's inputs.

        Training mode alone selects the oracle, independent of
        ``torch.is_grad_enabled()`` -- see this module's docstring for the
        reentrant-checkpointing case that condition exists for. The other two conditions
        (gradient reachability, device) are the op's own, shared verbatim so the layer
        and the free function cannot drift into two slightly different answers.

        THE MODULE'S CONDITIONS ONLY. Whether a topology has a built kernel arm at all
        is the op's question, answered by :func:`~rola.interface.dispatch_for` at the
        call site -- an ``'cuda'`` here means "nothing about this MODULE forces the
        oracle", and a configuration outside the built chunk matrix still dispatches
        there (the unsupported-topology retirement; this module's docstring states the
        envelope). Keeping the two separate is what lets decode, whose envelope is its
        own, be chosen from this answer without being gated on the prefill matrix.
        """
        if self.training:
            return "oracle"
        #: the decay retirement: decay is not implemented in the kernel arm; a decay-
        #: configured layer routes to the oracle like an unratified
        #: activation would -- dispatch, not silent degradation (the direct
        #: consumer entries still refuse loudly).
        if self.decay is not None:
            return "oracle"
        tensors = [t for t in operation_inputs if t is not None]
        return "oracle" if select_backend(*tensors) == "reference" else "cuda"

    def _bf16_projection_active(self, x: torch.Tensor) -> bool:
        """Whether this call's v/o projections run as bf16 GEMMs (P09 family).

        Keyed to the SAME conditions that reach the kernel arm -- eval mode, no
        grad anywhere (``torch.is_grad_enabled()`` off implies ``select_backend``
        cannot pick the reference for gradient reasons), CUDA -- plus the
        ``mma_precision`` format axis. Any grad-bearing call (training, the
        reentrant-checkpointing eval case, a dial-only finetune) keeps the
        module-dtype projections that feed the fp64 oracle. The GEMMs run with
        bf16 operands and cuBLAS's fp32 compute type (the accumulate-in-fp32
        ruling; the global reduced-precision-reduction knob is deliberately not
        touched -- this is the same environment the FA comparison arm runs in).
        """
        return (not self.training and not torch.is_grad_enabled() and x.is_cuda
                and self.expert.mma_precision == "bf16"
                and self.normalization == "global")

    # --- forward -------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        state: RoLAState | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, RoLAState | None]:
        """``(y, state)``, mirroring the op. THE LAYER IS A PASS-THROUGH FOR STATE.

        ``state`` is a :class:`~rola.RoLAState` or ``None``, and the predicate is
        the op's: ``state is not None`` engages the carried state, a stateless call
        returns ``(y, None)``. Nothing is wrapped, stamped or checked here -- state
        semantics are the state's (docs/internals/state.md), and a layer that
        re-validated them would be a second contract to keep in step with the first.
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
        #: The short-convolution rings ride ALONGSIDE the routed state, on the same
        #: object, because they are also part of the recurrence: a continuation that
        #: dropped them would restart the causal conv from zeros and compute a
        #: different function for the first `conv_size - 1` tokens of every step.
        stateful = state is not None
        conv_states = list(state.conv_state) if (
            stateful and state.conv_state is not None) else [None, None]
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
        operands = (v, *routes.tensors(), *(() if decay is None else tuple(decay.dials)))
        backend = self.select_execution_backend(*operands)
        #: a state that carries CONTENTS, which is what makes a step a continuation.
        #: A bound-but-empty state is a fresh sequence, and takes the prefill arms.
        carrying = stateful and state.carries
        if carrying and torch.is_grad_enabled():
            raise RuntimeError("RoLA carried-state execution is inference-only")

        if backend == "cuda" and L == 1 and carrying:
            #: decode is its own kernel family with its own envelope -- BC-blind, and
            #: general in the topology -- so it is chosen from the module's answer and
            #: never from the PREFILL matrix below.
            out = self._decode(routes, v, decay, state)
            backend = "decode"
        else:
            #: ONE CALL, TWO ARMS, AND THE OP OWNS THE CHOICE BETWEEN THEM. The layer's
            #: only contribution is `_arm="reference"`, its training-mode condition,
            #: which is a property of a module and has no meaning for a free function.
            #: Everything else -- gradient reachability, the device, an unratified
            #: activation, and the BUILT CHUNK MATRIX -- is `dispatch_for`'s, asked
            #: once so the counter below reports the arm that actually ran rather than
            #: the one the module would have preferred.
            arm = "reference" if backend == "oracle" else dispatch_for(
                routes, *operands).backend
            out, state = rola_op(routes, v, decay, state=state, expert=self.expert,
                                 _arm="reference" if arm == "reference" else "kernel")
            self.last_selected_plan = None
            backend = "oracle" if arm == "reference" else "cuda"
            # The reference arm is the fp64 oracle and returns fp64 whatever it was
            # handed; casting back is the caller's business because the caller is the
            # one who knows why. Here that reason is `o_proj`, which holds the module's
            # working dtype. The carried state deliberately stays fp64 -- the state
            # object is dtype-agnostic by design. The kernel arm already hands back the
            # operand dtype, where this is an exact no-op.
            out = out.to(v.dtype)

        self._execution_backend_counts[backend] += 1
        self.last_execution_backend = backend
        if stateful and (self.value_conv is not None or self.route_conv is not None):
            state.conv_state = tuple(conv_states)
        out = out.reshape(B, L, H * self.head_v_dim)
        if bf16_proj:
            # Off the kernel arm `out`'s VALUES are already bf16-rounded (y is
            # cast post-divide in the kernel and returned in the module dtype),
            # so this operand cast is exact; the result returns to the module
            # dtype at the layer boundary.
            projected = nn.functional.linear(
                out.to(torch.bfloat16), self.o_proj.weight.to(torch.bfloat16)).to(x.dtype)
        else:
            projected = self.o_proj(out)
        return projected, state

    # --- the arm that needs layer-owned state --------------------------------

    def _decode(self, routes, v, decay, state):
        """The ``T = 1`` sibling kernel, which owns its own state I/O.

        The layer's only contribution is the plan overrides; the carrier, the scratch,
        the backing and its growth all live behind ``decode_forward``
        (``docs/internals/decode.md``), exactly as the chunk arm's tables live behind
        ``rola_op``.
        """
        out, state = decode_forward(routes, v, state, decay=decay, expert=self.expert)
        self.last_selected_plan = None
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
