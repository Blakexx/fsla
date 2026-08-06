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
* ``'cuda'`` -- the plan-selected consumer, through :func:`rola.rola_op`.
* ``'decode'`` -- the ``T = 1`` sibling kernel. At one token every selection decision is
  degenerate (no tile to choose, no split, a statistics population of one), so decode
  bypasses selection ARCHITECTURALLY rather than running the planner on degenerate
  inputs. It is a SIBLING of the tiled path, not a specialization: shared production
  contract, own consumer, frozen config.

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

**There is no silent degradation in the other direction either.** Once the CUDA backend
is chosen, a configuration the consumer cannot run -- an un-instantiated tiling, an
unrealized warp tile -- raises out of the planner, the op, or the kernel's own checks.
It never falls back to the oracle.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from einops import rearrange

from rola.interface import rola_op, select_backend
from rola.ops.decode import decode_forward, freeze_decode_config
from rola.planner.select import freeze_for_decode
from rola.routing.decay import validate_decay_source
from rola.routing.factors import RouteProducer
from rola.utils.cache import get_layer_cache, update_layer_cache

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from rola.state import RoLAState

__all__ = ["RoLA"]


@dataclass(frozen=True)
class _DecodeConfigCache:
    """The frozen decode carrier plus the key it is valid for.

    A named type rather than a bare tuple because the validity KEY is the whole point:
    a cache whose key a reader has to reconstruct from an unpacking site is a cache that
    will eventually be reused across a shape it was not built for.
    """

    key: tuple
    value: object


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
            planner selects; it is quarantined because a caller pinning one is not
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
        from rola.planner.select import feasible_bc

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
        #: The paging arena, built on the first paged forward (it needs ``BH``, a
        #: forward-time quantity), and the frozen decode carrier with its scratch. Both
        #: are runtime state.
        self._page_arena = None
        self._decode_config = None
        self._decode_workspace = None

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
        """
        if self.training:
            return "oracle"
        tensors = [t for t in operation_inputs if t is not None]
        return "oracle" if select_backend(*tensors) == "reference" else "cuda"

    # --- forward -------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: RoLAState | None = None,
        use_cache: bool | None = False,
        budget: torch.Tensor | None = None,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, None, RoLAState | None]:
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
        cache_enabled = bool(use_cache)
        last_state = get_layer_cache(self, past_key_values)

        v = self.v_proj(x)
        conv_states = list(last_state.conv_state) if (
            last_state is not None and last_state.conv_state is not None) else [None, None]
        if self.value_conv is not None:
            if not cache_enabled and conv_states[0] is not None:
                conv_states[0] = conv_states[0].clone()
            v, conv_states[0] = self.value_conv(
                v, cache=conv_states[0], output_final_state=cache_enabled)
        source = x
        if self.route_conv is not None:
            if not cache_enabled and conv_states[1] is not None:
                conv_states[1] = conv_states[1].clone()
            source, conv_states[1] = self.route_conv(
                x, cache=conv_states[1], output_final_state=cache_enabled)

        v = rearrange(v, "b l (h d) -> b l h d", d=self.head_v_dim)
        recurrent_state = last_state.recurrent_state if last_state is not None else None
        if recurrent_state is not None and torch.is_grad_enabled():
            raise RuntimeError("RoLA carried-state execution is inference-only")

        routes = self.routes(source, budget=budget)
        decay = None if self.decay is None else self.decay()
        backend = self.select_execution_backend(
            v, *routes.tensors(), *(() if decay is None else tuple(decay.dials)))

        if backend == "cuda" and L == 1 and recurrent_state is not None and not self.paging:
            out, final_state = self._decode(routes, v, decay, recurrent_state, B, H, cache_enabled)
            backend = "decode"
        elif backend == "cuda" and self.paging:
            out, final_state = self._paged(routes, v, decay, recurrent_state, B, H, cache_enabled)
        else:
            # ONE op for both remaining arms. `rola_op` decides between the kernel and
            # the differentiable reference from the operands themselves; `_arm` is how
            # the layer contributes its training-mode condition, which is a property of
            # a module and has no meaning for a free function.
            out, final_state = rola_op(
                routes, v, decay, initial_state=recurrent_state,
                output_final_state=cache_enabled, expert=self.expert,
                _arm="reference" if backend == "oracle" else None)
            self.last_selected_plan = None

        self._execution_backend_counts[backend] += 1
        self.last_execution_backend = backend
        if cache_enabled:
            # The carried state has the SAME SHAPE and meaning on every backend; only the
            # accumulation dtype differs (fp64 oracle, fp32 kernel) and every entry point
            # accepts either, so a continuation may cross backends between steps.
            update_layer_cache(
                self, past_key_values, recurrent_state=final_state,
                conv_state=tuple(conv_states) if (
                    self.value_conv is not None or self.route_conv is not None) else None,
                offset=L)
        return self.o_proj(out.reshape(B, L, H * self.head_v_dim)), None, past_key_values

    # --- the two arms that need layer-owned state ----------------------------

    @property
    def paging(self) -> bool:
        """Whether the state is committed page by page (an allocator policy, not math).

        Paging is BITWISE INVISIBLE: one page is one MMA atom, pages are admitted by the
        write-side plan, and an atom without a page reads as zeros. It lives on the
        expert surface because it changes what the layer allocates and nothing about
        what it computes.

        A CAP IMPLIES IT: asking for `max_physical_pages` without asking for paging would
        be asking for a ceiling on an allocator that is not running, and refusing that
        combination would be a validation rule with no purpose. Paging without a cap is
        the uncapped arena, provisioned to the dense limit, where the ceiling can never
        bind.
        """
        return self.expert.paging or self.expert.max_physical_pages is not None

    def _paged(self, routes, v, decay, recurrent_state, B, H, cache_enabled):
        """The tiled arm with a page arena carried across steps."""
        from rola.expert import derive_plan, execute

        plan = derive_plan(routes, decay=decay, expert=self.expert, device=v.device)
        self.last_selected_plan = plan
        arena = self._resolve_page_arena(plan, B * H, v.device)
        continue_paging = recurrent_state is not None
        if continue_paging:
            # An explicit bridge, not an implicit one: `adopt_dense` provisions every
            # owner, so it must be a visible act. `reset` clears the page table before
            # loading the dense state; `continue_paging` is what then stops the
            # consumer's own default reset from wiping what was just committed.
            arena.reset()
            arena.adopt_dense(recurrent_state.reshape(arena.BH, arena.N, arena.cols)
                              .to(torch.float32).contiguous())
        out, _arena, _schedule = execute(
            plan, routes, v, decay=decay, paging=arena, continue_paging=continue_paging)
        final_state = (arena.materialize().view(B, H, arena.N, arena.cols)
                       if cache_enabled else None)
        return out, final_state

    def _decode(self, routes, v, decay, recurrent_state, B, H, cache_enabled):
        """The ``T = 1`` sibling kernel, over the frozen carrier."""
        config = self._resolve_decode_config(B * H, v.device)
        # `decode_forward` updates its state argument IN PLACE, and the cached state is
        # fp32-contiguous in steady decode, so both casts below are IDENTITY and would
        # alias the caller's cache. Under `use_cache` that aliasing IS the intended
        # carry; without it the caller's cache must be left untouched, so the alias is
        # severed. Only the identity case clones -- a real cast already copied.
        state = recurrent_state.to(torch.float32).contiguous()
        if not cache_enabled and state.data_ptr() == recurrent_state.data_ptr():
            state = state.clone()
        out, final_state, self._decode_workspace = decode_forward(
            v, routes.read, routes.write, routes.g_read, routes.g_write, config, state,
            decay=decay, workspace=self._decode_workspace)
        self.last_selected_plan = None
        return out, (final_state if cache_enabled else None)

    def _resolve_page_arena(self, plan, BH: int, device):
        """The page arena, built once and reused across steps.

        Pooled across every ``(batch, head)`` pair of this layer, which is what makes the
        savings math work: a per-pair arena would round each pair's tiny live fraction up
        to its own growth chunk. Rebuilt when ``BH``, the device or the column count
        moves -- and NOT when ``BC`` does, because a page is one MMA atom on both sides,
        so every stored object and every address the kernel forms is ``BC``-invariant and
        two launches at different ``BC`` address the same bytes of the same slots.
        """
        from rola.ops.paging import PageArena

        cols = self.head_v_dim + (1 if self.normalization == "global" else 0)
        arena = self._page_arena
        if (arena is not None and arena.BH == BH and arena.device == device
                and arena.cols == cols):
            return arena
        arena = PageArena(
            widths=self.topology.widths, BC=plan.BC, cols=cols, BH=BH, device=device,
            max_physical_pages=self.expert.max_physical_pages)
        self._page_arena = arena
        return arena

    def _resolve_decode_config(self, BH: int, device):
        """The frozen decode config, built once at the prefill/decode boundary.

        Rebuilt only when its inputs move (``BH`` and the device; everything else is a
        construction-time fact). It is frozen in the sense that matters: no per-step
        statistic enters it, so a decode step never runs a selection.
        """
        from rola.interface import _static_tier0

        cached = self._decode_config
        key = (BH, torch.device(device), self.decay is not None)
        if cached is not None and cached.key == key:
            return cached.value
        static = _static_tier0(
            self.topology, self.normalization, self.routes.density,
            decay=self.decay is not None, BH=BH, overrides=self.expert)
        value = freeze_decode_config(
            self.topology, self.normalization, decay=self.decay is not None, BH=BH, device=device,
            frozen=freeze_for_decode(static), mma_precision=self.expert.mma_precision)
        self._decode_config = _DecodeConfigCache(key=key, value=value)
        return value

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
