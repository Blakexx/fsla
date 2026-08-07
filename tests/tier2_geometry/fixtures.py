"""Shared tier-2 fixture infrastructure: one routing realization, one launch.

``_Fixture``, ``_masked_simplex``, ``_simplex``, ``_topology``,
``_SPARSE_WRITE_ROUTING`` and ``_relative`` are general-purpose tier-2 building
blocks; ``test_matrix_rows.py`` builds every geometry-pair row's fixtures through
``_Fixture``. This module carries no test of its own -- it is collected for its
helpers only.
"""

from __future__ import annotations

import torch

from rola.ops.consumer import _consumer_forward
from rola.ops.naive import naive_rola
from rola.planner.schedule import Plan, build_schedule
from rola.routing.types import (
    EntmaxActivation,
    IndependentRouting,
    LeafMassDecay,
    LevelConfig,
    SoftmaxActivation,
    Topology,
)

_ROUTING = IndependentRouting(
    tied=False, read_activation=SoftmaxActivation(), write_activation=SoftmaxActivation())

#: §7.1's mark phase reads the WRITE activation structurally (`can_zero`): a softmax
#: level cannot emit an exact zero, so the arena takes its dense short-circuit and
#: commits every page. A partially-resident arena requires an entmax-declared write
#: router, and the fixture must declare what it emits ([[sparsity-comes-from-entmax]]).
_SPARSE_WRITE_ROUTING = IndependentRouting(
    tied=False, read_activation=SoftmaxActivation(), write_activation=EntmaxActivation(1.5))


def _topology(widths, d_v, routing=_ROUTING):
    return Topology(levels=tuple(LevelConfig(w, routing) for w in widths), d_v=d_v)


def _simplex(shape, p_nonzero, generator, device):
    x = torch.rand(shape, device=device, dtype=torch.float64, generator=generator)
    if p_nonzero < 1.0:
        keep = torch.rand(shape, device=device, dtype=torch.float64, generator=generator) < p_nonzero
        x = x * keep
    x = torch.where(x.sum(-1, keepdim=True) == 0, torch.ones_like(x), x)
    result = x / x.sum(-1, keepdim=True)
    if p_nonzero < 1.0:
        assert bool((result == 0).any()), "the sparse fixture is vacuously dense"
    return result


def _masked_simplex(shape, mask, generator, device):
    """A simplex whose support is EXACTLY ``mask``. Structured supports are what the
    carve-out is about; a Bernoulli mask cannot express "this digit is dead over
    exactly this token range"."""
    mask = mask.to(device=device, dtype=torch.float64).expand(shape)
    assert bool((mask.sum(-1) > 0).all()), "every row must have a nonempty support"
    x = torch.rand(shape, device=device, dtype=torch.float64, generator=generator) * mask
    x = torch.where(x.sum(-1, keepdim=True) == 0, mask, x)
    return x / x.sum(-1, keepdim=True)


class _Fixture:
    """One routing realization plus the plan and schedule built from it."""

    def __init__(self, widths, BC, d_v, B, T, H, p_read, p_write, seed, *,
                 decay_on=False, nonzero_m0=False,
                 read_levels=None, write_levels=None, routing=_ROUTING):
        device = "cuda"
        gen = torch.Generator(device=device).manual_seed(seed)
        self.topology = _topology(widths, d_v, routing)
        self.d_v = d_v
        # `normalization="global"` is the only value `Plan` accepts, so the mass
        # column is unconditional: `cols = d_v + 1` always.
        self.cols = d_v + 1
        self.read_levels = read_levels or tuple(
            _simplex((B, T, H, w), p_read, gen, device) for w in widths)
        self.write_levels = write_levels or tuple(
            _simplex((B, T, H, w), p_write, gen, device) for w in widths)
        self.g_write = torch.rand(B, T, H, device=device, dtype=torch.float64, generator=gen) + 0.5
        self.g_read = None
        self.v = torch.randn(B, T, H, d_v, device=device, dtype=torch.float64, generator=gen)
        self.decay = None
        if decay_on:
            self.decay = LeafMassDecay(dials=tuple(
                0.05 + 0.85 * torch.rand(H, w, device=device, dtype=torch.float64, generator=gen)
                for w in widths))
        self.initial_state = None
        if nonzero_m0:
            m0 = 0.1 * torch.randn(B, H, self.topology.N, self.cols, device=device,
                                   dtype=torch.float64, generator=gen)
            # The mass column is a nonnegative accumulated write mass; a negative
            # incoming mass is not a state the recurrence can produce.
            m0[..., d_v] = m0[..., d_v].abs()
            self.initial_state = m0
        self.plan = Plan(topology=self.topology, normalization="global", BC=BC,
                           warp_tile=(1, max(1, d_v // 32)), decay=decay_on)
        self._f32 = dict(
            v=self.v.to(torch.float32),
            read_levels=tuple(t.to(torch.float32) for t in self.read_levels),
            write_levels=tuple(t.to(torch.float32) for t in self.write_levels),
            g_read=None if self.g_read is None else self.g_read.to(torch.float32),
            g_write=self.g_write.to(torch.float32))
        self.schedule = build_schedule(
            self._f32["read_levels"], self._f32["write_levels"],
            self._f32["g_read"], self._f32["g_write"], self.plan)
        self.n_owners = int(self.schedule.owners.shape[0])

    def oracle(self):
        return naive_rola(
            self.v, self.read_levels, self.write_levels, self.g_read, self.g_write,
            self.topology, self.decay, "global",
            initial_state=self.initial_state, output_final_state=True)

    def run(self):
        """Launch the consumer -- always the single ``P = 1`` path."""
        return _consumer_forward(
            self._f32["v"], self._f32["read_levels"], self._f32["write_levels"],
            self._f32["g_read"], self._f32["g_write"], self.plan,
            schedule=self.schedule, decay=self.decay,
            initial_state=(None if self.initial_state is None
                           else self.initial_state.to(torch.float32)))


def _relative(actual: torch.Tensor, reference: torch.Tensor) -> float:
    scale = max(1e-30, float(reference.abs().max()))
    return float((actual.double() - reference).abs().max()) / scale
