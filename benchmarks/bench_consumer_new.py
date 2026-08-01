"""The NEW V3 consumer's oracle gate and interleaved A/B bench (`v3_consumer_kernel.cuh`).

**WHY THIS IS A COMMITTED ARTIFACT AND NOT A SCRATCHPAD SCRIPT.** It already
happened once, in the same repository, and `benchmark_rola_v3_consumer.py`'s header
records it as review finding F6: a findings document quoted numbers that no
committed script reproduced. It happened AGAIN to this batch --
`workflows/20260801-rewrite/rewrite.md` §1.6-§2.8 quotes an at-scale cell whose
harness was lost, and the cell was described only by its PLAN STATISTICS
(`mean_K = 0.1908`, `mean_K_candidate = 0.9665`, `candidate_frac = 0.1975`). A
statistic is not a fixture. Sweeping the construction parameter that was supposed
to produce it traces a curve that does not pass through that triple at all, so
those numbers are permanently non-reproducible.

**THE RULE THIS FILE EXISTS TO ENFORCE: A CELL IS ITS CONSTRUCTOR AND ITS SEED.**
Every gate and bench cell below is a frozen :class:`Cell` -- widths, `BC`, shapes,
support construction, decay dials, seed -- and `Cell.build()` is deterministic from
it. Documents reference cells BY NAME (``atscale-L2048``, ``wk1-D2-w64x64-BC64``),
never by a description of their density. Two numbers carrying the same cell name
are commensurable; two numbers carrying different names are not, and the name makes
that visible instead of leaving it to be discovered later.

`--manifest` prints each cell's REALIZED statistics and `--check-manifest`
diffs them against the committed `benchmark_rola_v3_new_consumer_manifest.json`,
so a change in torch's RNG, in the entmax solver or in the planner announces itself
as a manifest diff rather than as a silently different benchmark.

Modes:

  --mode oracle   correctness against the canonical fp64 oracle `naive_rola_v3`,
                  at the declared per-arm tolerances. NEVER against a second
                  reference: validating against a freshly written one proves only
                  self-consistency (spec §2.1's prohibition; the shared-router bug
                  passed exactly that way).

  --mode atscale  DELETED by item 6b (2026-08-01): it was the OLD-consumer-vs-NEW
                  interleaved paired-ratio A/B, and there is one consumer now. Its
                  landed number (-24.7 % paired at `atscale-L2048` COMPACT, IQR
                  0.094) is recorded in `rewrite.md` §2.20. What it used to say
                  about the two arms
                  ALTERNATE REP BY REP -- never arm-A-block then arm-B-block, which
                  lets a drift step land between the arms and masquerade as an
                  effect. Measured on this box, that bias was real: sequential
                  blocks read x3.27-3.30 where interleaving read x3.33-3.35, and
                  interleaving cut the run-to-run spread by ~7x.

  --mode btab     `BT = 16` against `BT = 32`, BOTH ARMS THE NEW CONSUMER, the
                  schedule rebuilt at each `BT` by restricting the planner's
                  candidate set. This is the planner `BT` axis's calibration
                  instrument (§2.32); the ratio is `t32 / t16`, so > 1 means the
                  larger tile is slower.

  --mode skipshare  the read-only-`(owner, tile)` share per cell -- the fraction of
                  visits on which the fold GEMM and the decay panels are skipped.

  --mode armtime  ONE arm's absolute times on one cell. The only non-paired timing
                  mode, and it exists for the only comparison the paired doctrine
                  cannot express: two builds that differ by a COMPILE-TIME kernel
                  constant cannot share a process. The caller alternates PROCESSES
                  and pairs across them; see `arm_time`.

Reportable statistic under contention is the median of the PER-PAIR ratios (each
pair two measurements adjacent in time). Absolute times and IQRs are reported
alongside, never instead. **A cell whose IQR cannot discriminate the effect being
claimed is reported as a RANGE or dropped, and which was done is stated.**

Usage (GPU work belongs under `flock /tmp/rola_gpu.lock`):

    PYTHONPATH=<worktree> python benchmarks/ops/benchmark_rola_v3_new_consumer.py \
        --mode oracle
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from rola.ops.naive import naive_rola_v3
from rola.routing.entmax.reference import entmax
from rola.planner.schedule import (
    V3Plan, build_v3_schedule, owner_geometry, tile_union_stats, unpack_bits)
from rola.routing.types import (
    EntmaxActivation, IndependentRouting, LeafMassDecay, LevelConfig,
    SoftmaxActivation, Topology)
from rola.ops.consumer import v3_consumer_forward

MANIFEST = Path(__file__).with_name("benchmark_rola_v3_new_consumer_manifest.json")

# ---------------------------------------------------------------------------
# TOLERANCES -- the consumer suite's own, quoted, never widened
# ---------------------------------------------------------------------------

#: fp32 accumulation against an fp64 oracle. The exactness arm (`tf32x3`) meets it.
RTOL = 2e-5

#: The bf16 arm's `y` tolerance. bf16 A/B operands carry 8 mantissa bits (~4e-3
#: relative per product) and the readout contracts over `BC` of them: a FORMAT
#: tolerance, i.e. what the chosen numeric format costs, not slack.
BF16_RTOL = 1e-2

#: The bf16 arm's STATE tolerance, DERIVED rather than observed. Each fold deposit
#: `S += W~ * V` rounds TWO bf16 operands, each to within `2^-8` relative
#: (8-bit significand, round-to-nearest), so to first order the product carries
#: `2^-8 + 2^-8 = 2^-7`. **The accumulation is fp32 and contributes nothing at this
#: scale, so the bound DOES NOT GROW WITH THE NUMBER OF DEPOSITS** -- which is why
#: it is a per-deposit constant rather than an `O(T)` or `O(sqrt T)` expression. The
#: flat-in-L measurement (3.447e-03 at L=128 vs 3.441e-03 at L=256) is its
#: CONFIRMATION, not its source.
BF16_FOLD_STATE_RTOL = 2.0 ** -7


# ---------------------------------------------------------------------------
# SUPPORT CONSTRUCTORS -- committed, because a density is not a fixture
# ---------------------------------------------------------------------------

def _dense_simplex(shape, g, device):
    x = torch.rand(shape, device=device, dtype=torch.float64, generator=g)
    return x / x.sum(-1, keepdim=True)


def _bernoulli_simplex(shape, p, g, device):
    """Support drawn from a Bernoulli mask, with empty rows REPAIRED to dense.

    **The repair is why this constructor cannot express a sparse support**, and
    that is recorded here rather than in a comment somewhere downstream because it
    silently produced a VACUOUS gate once: at `p = 0.004` on width 256 most rows
    come up empty, every one of them is repaired to fully dense, and the resulting
    write support covered every owner on every tile -- 0.0 % read-only tiles, the
    exact opposite of the intent. Use `_topk_simplex` for a genuinely sparse
    support; this one is for MODERATE density, where the repair rarely fires.
    """
    x = torch.rand(shape, device=device, dtype=torch.float64, generator=g)
    x = x * (torch.rand(shape, device=device, dtype=torch.float64, generator=g) < p)
    x = torch.where(x.sum(-1, keepdim=True) == 0, torch.ones_like(x), x)
    return x / x.sum(-1, keepdim=True)


def _topk_simplex(shape, k, g, device):
    """A simplex whose support is EXACTLY ``min(k, width)`` entries per row.

    This is the constructor that actually drives owners write-dead, hence the one
    that reaches the decay lowering's read-only-tile exemption and the fold's
    tile-level write skip. See `_bernoulli_simplex` for why the obvious
    alternative does not.
    """
    x = torch.rand(shape, device=device, dtype=torch.float64, generator=g)
    kth = x.topk(min(k, shape[-1]), dim=-1).values[..., -1:]
    x = torch.where(x >= kth, x, torch.zeros_like(x))
    return x / x.sum(-1, keepdim=True)


def _entmax_simplex(shape, logit_scale, g, device):
    """`alpha = 1.5` entmax of `logit_scale * N(0,1)` logits -- the PRODUCER's own
    sparsity, not a synthetic dial. `logit_scale` is a committed construction
    parameter because it is the knob the whole at-scale density hangs on."""
    z = logit_scale * torch.randn(shape, device=device, dtype=torch.float64, generator=g)
    return entmax(z, alpha=1.5)


def _support(spec, shape, g, device):
    kind = spec[0]
    if kind == "dense":
        return _dense_simplex(shape, g, device)
    if kind == "bernoulli":
        return _bernoulli_simplex(shape, spec[1], g, device)
    if kind == "topk":
        return _topk_simplex(shape, spec[1], g, device)
    if kind == "entmax":
        return _entmax_simplex(shape, spec[1], g, device)
    raise ValueError(f"unknown support spec {spec!r}")


#: `delta in [0, 1)`, STRICT (spec §1). `hot` drives the leaf rate product toward
#: 1, where `1 - rate` cancels -- the regime the kernel's fp32 dial accumulation and
#: §12 rule 2's `log1pf` composition exist for.
_DIAL_RANGE = {"warm": (0.05, 0.85), "hot": (0.90, 0.099)}


def _with_f32(fx):
    """Materialize the consumer's fp32 inputs ONCE per fixture.

    **This is load-bearing, not tidiness.** `V3DecayStream`'s validity key is
    tensor IDENTITY plus `_version` (it has to be -- a `data_ptr` key is unsound
    across calls). A harness that re-ran `t.float()` per call handed the consumer a
    NEW tensor object every time on any fp64 fixture, so the memo missed every call
    and the decay preamble was paid every call -- while PRODUCER cells, whose
    routing is already fp32 and whose `.float()` therefore returns `self`, hit it.
    That split is exactly what the first post-fix measurement showed, and it was the
    harness, not the memo.
    """
    fx["read_f32"] = tuple(t.float() for t in fx["read_levels"])
    fx["write_f32"] = tuple(t.float() for t in fx["write_levels"])
    fx["v_f32"] = fx["v"].float()
    fx["g_write_f32"] = fx["g_write"].float()
    fx["g_read_f32"] = None if fx["g_read"] is None else fx["g_read"].float()
    return fx


@dataclass(frozen=True)
class Cell:
    """One benchmark/gate fixture, RECONSTRUCTIBLE EXACTLY from these fields.

    Two construction MODES, and the distinction is load-bearing rather than
    cosmetic -- it is the one that made the lost at-scale cell unreproducible:

    * **SYNTHETIC** (`producer_route_w_mul is None`). Routing tensors are drawn
      directly by the support constructors above. Full control over the density
      axis, and the only way to build a fixture the decay/`raw`/write-`k` arms need,
      because the trained layer this repo ships is `decay="none"` /
      `state_norm="global"` only.
    * **PRODUCER** (`producer_route_w_mul` set). A real
      :class:`~rola.layers.RoLA` is constructed and `produce_v3_routing` is run
      on it, so the supports are the ones the MODEL emits. The plan is then chosen
      by `plan_and_schedule_v3` -- the PLANNER, not a hand-built `V3Plan` -- which
      is what makes "plan-selected" claims about this cell true.

    **THE SYNTHETIC MODE CANNOT IMITATE THE PRODUCER MODE'S DENSITY, and that is a
    recorded finding, not a caveat.** `producer_route_w_mul` scales the routing
    PROJECTION WEIGHTS, so the level logits are `x @ route_W` -- correlated across
    tokens and across levels through one shared projection, with a scale that also
    depends on `hidden_size` and the init. A synthetic sweep over an iid Gaussian
    logit scale is a different knob on a different distribution and traces a
    different curve; see the `atscale-*` registry entries.

    `build()` draws every tensor from one seeded generator in a FIXED ORDER --
    including BOTH dial sets, always, whether or not the arm uses them -- so the
    routing tensors of a cell are identical across the decay axis. An arm selects;
    it never perturbs the draw.
    """

    name: str
    widths: tuple[int, ...]
    BC: int
    #: Support constructors, as `(kind, *params)`. Committed values, not densities.
    read: tuple = ("dense",)
    write: tuple = ("dense",)
    d_v: int = 64
    B: int = 1
    T: int = 128
    H: int = 2
    BT: int = 16
    seed: int = 0
    #: Entmax gain declared on the Topology's activations. It describes what the
    #: routing IS; a synthetic cell's own sparsity comes from the support
    #: constructors. Forced true on producer cells (they route `sparse`/`sparse`).
    sparse_activation: bool = False
    #: PRODUCER MODE. The `layer.route_W.mul_(...)` factor. `None` = synthetic.
    producer_route_w_mul: float | None = None
    note: str = ""

    @property
    def D(self) -> int:
        return len(self.widths)

    @property
    def is_producer(self) -> bool:
        return self.producer_route_w_mul is not None

    def topology(self) -> Topology:
        if self.is_producer:
            return self._layer().topology
        act = (EntmaxActivation(alpha=1.5) if self.sparse_activation
               else SoftmaxActivation())
        routing = IndependentRouting(
            tied=self.sparse_activation, read_activation=act, write_activation=act)
        return Topology(levels=tuple(LevelConfig(w, routing) for w in self.widths),
                        d_v=self.d_v)

    def _layer(self):
        """The producer cell's layer, built EXACTLY as the original harness did.

        `torch.manual_seed(seed)` before construction is part of the fixture: the
        layer's parameter init consumes the global RNG, so the seed determines the
        routing projection and therefore the supports.
        """
        from rola.layers import RoLA
        from rola.layers.rola import LevelRoutingConfig
        torch.manual_seed(self.seed)
        layer = RoLA(hidden_size=self.H * self.d_v, num_heads=self.H,
                     head_v_dim=self.d_v, levels_per_head=len(self.widths),
                     states_per_level=self.widths, decay="none", state_norm="global",
                     level_routing_configs=LevelRoutingConfig(
                         read="sparse", write="sparse", tied=False, alpha=1.5),
                     paging=False, state_block=self.BC,
                     layer_idx=0).to("cuda", torch.float32).eval()
        with torch.no_grad():
            layer.route_W.mul_(self.producer_route_w_mul)
        for p in layer.parameters():
            p.requires_grad_(False)
        return layer

    def _build_producer(self):
        from rola.routing.producer import produce_v3_routing
        layer = self._layer()
        hidden = self.H * self.d_v
        x = torch.randn(self.B, self.T, hidden, device="cuda", dtype=torch.float32)
        with torch.no_grad():
            v = layer.v_proj(x).view(self.B, self.T, self.H, self.d_v)
            ro = produce_v3_routing(
                x, None, v, layer.route_W, layer.route_bias_param, layer.gain_W,
                layer.gain_bias, layer.topology, "global", budget=layer.budget, r=None,
                budget_embed_W=layer.budget_embed_W, temp_W=layer.temp_W,
                temp_b=layer.temp_b, usage_bias_W=layer.usage_bias_W,
                usage_bias_b=layer.usage_bias_b)
        g = torch.Generator(device="cuda").manual_seed(self.seed + 977)
        dials = {t: tuple(lo + span * torch.rand(self.H, w, device="cuda",
                                                 dtype=torch.float64, generator=g)
                          for w in self.widths)
                 for t, (lo, span) in _DIAL_RANGE.items()}
        #: `produce_v3_routing(..., "global")` returns `g_read = None`, because spec
        #: §4.2 FIXES `g_read := 1` under `global` and production has no read-gain
        #: projection to supply one. A `raw` arm on a producer cell therefore has no
        #: producer-supplied read gain to use, and one is synthesised here from the
        #: cell's own seed -- stated rather than silently substituted, because it
        #: means a producer cell's `raw` arm is PART synthetic and its `raw` numbers
        #: are not "what the model emits" the way its `global` numbers are.
        g_read = ro.g_read
        if g_read is None:
            g_read = torch.rand(self.B, self.T, self.H, device="cuda",
                                dtype=torch.float32, generator=g) + 0.5
        return _with_f32(dict(read_levels=ro.read_levels, write_levels=ro.write_levels,
                              g_read=g_read, g_write=ro.g_write, v=v, dials=dials,
                              topology=layer.topology))

    def build(self, device="cuda"):
        if self.is_producer:
            return self._build_producer()
        g = torch.Generator(device=device).manual_seed(self.seed)
        shp = lambda w: (self.B, self.T, self.H, w)
        read_levels = tuple(_support(self.read, shp(w), g, device) for w in self.widths)
        write_levels = tuple(_support(self.write, shp(w), g, device) for w in self.widths)
        g_write = torch.rand(self.B, self.T, self.H, device=device,
                             dtype=torch.float64, generator=g) + 0.5
        g_read = torch.rand(self.B, self.T, self.H, device=device,
                            dtype=torch.float64, generator=g) + 0.5
        v = torch.randn(self.B, self.T, self.H, self.d_v, device=device,
                        dtype=torch.float64, generator=g)
        # BOTH dial sets, ALWAYS drawn, so the decay axis does not move the routing.
        dials = {}
        for temp, (lo, span) in _DIAL_RANGE.items():
            dials[temp] = tuple(
                lo + span * torch.rand(self.H, w, device=device, dtype=torch.float64,
                                       generator=g) for w in self.widths)
        return _with_f32(dict(read_levels=read_levels, write_levels=write_levels,
                              g_read=g_read, g_write=g_write, v=v, dials=dials,
                              topology=self.topology()))

    def plan(self, *, norm, decay_on, prec, fx=None, compact=False, bt=None):
        """A producer cell's plan is PLANNER-SELECTED; a synthetic cell's is stated.

        The distinction matters for what a measurement on the cell can claim: only
        the planner-selected form supports a sentence beginning "the plan selects".
        The selected axes are then overridden ONLY on the one arm this bench
        crosses -- the precision point -- so that everything else (`BT`, `BC`,
        `compact`, `ring_slots`) remains the planner's answer.

        **THE `lowering` PARAMETER IS GONE** (item 6b, 2026-08-01). The consumer
        includes `intra_panel.cuh` unconditionally and does ONE intra lowering, so
        there is nothing to force and nothing to inherit.

        ``bt`` RESTRICTS THE PLANNER'S ``BT`` CANDIDATE SET rather than stamping a
        `BT` onto a finished plan, and the difference is not cosmetic: the schedule
        is a `BT`-keyed object (the §4 tile bitmaps, the candidate lists and the
        clock stream are all built at a tile size), so a `dataclasses.replace` of
        `plan.BT` would leave the kernel reading a schedule built for the other tile
        size. Restricting the candidates re-runs selection AND the build at that
        `BT`, which is what makes the two arms of a `BT` A/B each a plan the planner
        could actually have emitted.
        """
        if self.is_producer and fx is not None:
            from rola.planner.schedule import plan_and_schedule_v3
            import dataclasses as _dc
            #: NO ORACLE IS INJECTED, and that is item 6b's point: the planner's
            #: DEFAULT occupancy oracle is now the one kernel's own. This file used
            #: to inject `consumer_occupancy()` because the default queried the OLD
            #: consumer -- a different footprint (31680 B against 44672 at
            #: `BT=16, BC=64, d_v=64`), reporting 2 CTAs/SM where the launched kernel
            #: runs 3, the residency contradiction `rewrite.md` §2.33 recorded and
            #: §2.34 resolved. With one kernel the injection would be a no-op that
            #: hid whether the default was right.
            kw = {}
            if bt is not None:
                kw["bt_candidates"] = (bt,)
            plan, _axes, sched = plan_and_schedule_v3(
                fx["read_levels"], fx["write_levels"], fx["g_read"], fx["g_write"],
                fx["topology"], self.BC, normalization=norm, decay=decay_on, **kw)
            #: `compact` is OVERRIDDEN rather than left to the planner, and that is
            #: RULING 2's whole point: the planner's cost model prices compaction
            #: against the OLD kernel's economics (a scalar walk's trip count, a
            #: dense-panel-to-compacted copy, a staging-ring probe) and selects
            #: `False` at every gating cell. Re-deriving the axis for the new shape
            #: requires driving it DIRECTLY; the planner's own choice is what the
            #: measurement is supposed to inform, so inheriting it here would beg the
            #: question. `compact=False` (the default) leaves the planner's answer.
            if compact:
                plan = _dc.replace(plan, compact=True)
            return _dc.replace(plan, mma_precision=prec), sched
        return V3Plan(topology=fx["topology"] if fx else self.topology(),
                      normalization=norm, BT=self.BT if bt is None else bt, BC=self.BC,
                      warp_tile=(1, max(1, self.d_v // 32)), decay=decay_on,
                      mma_precision=prec, compact=compact), None


# ---------------------------------------------------------------------------
# THE CELL REGISTRY. Documents cite these names.
# ---------------------------------------------------------------------------

def _cells() -> dict[str, Cell]:
    cs: list[Cell] = []

    #: TOPOLOGY/BC COVERAGE. Eight cells spanning `D = 1..4` at both legal `BC`,
    #: each innermost level sized to admit the kernel's `BC`. Dense on both sides.
    for widths, BC, seed in (((256,), 64, 1241), ((64, 64), 64, 1248),
                             ((16, 16, 64), 64, 1255), ((8, 8, 8, 64), 64, 1262),
                             ((64, 16), 16, 1248), ((8, 8, 16), 16, 1255),
                             ((256,), 16, 1241), ((8, 8, 8, 16), 16, 1262)):
        w = "x".join(str(x) for x in widths)
        cs.append(Cell(name=f"topo-D{len(widths)}-w{w}-BC{BC}", widths=widths, BC=BC,
                       seed=seed, note="dense read + dense write"))

    #: DENSE-READ + WRITE-k=1. These exist because the topology block above CANNOT
    #: REACH the decay lowering's read-only-tile exemption or the fold's tile-level
    #: write skip: dense-both makes `is_write` true on every candidate tile. They
    #: are also the shape of §10i.23's FA-comparable headline arm (dense read, an
    #: enforced small write budget). Realized read-only shares are in the manifest.
    for widths, BC, seed in (((64, 64), 64, 113), ((256, 4), 64, 113),
                             ((16, 16, 64), 64, 120), ((8, 8, 8, 64), 64, 127)):
        w = "x".join(str(x) for x in widths)
        cs.append(Cell(name=f"wk1-D{len(widths)}-w{w}-BC{BC}", widths=widths, BC=BC,
                       write=("topk", 1), seed=seed,
                       note="dense read + exactly-1 write per token per level"))

    #: AT-SCALE, PRODUCER MODE -- **THE CANONICAL CELL, AND THE HISTORICAL ONE.**
    #: `[256,256]` `N=65536`, `B=1 H=8 d_v=64`, sparse/sparse alpha=1.5 routing from
    #: a real `RoLA` layer with `route_W` scaled by 8.0, seed 0, plan chosen by
    #: `plan_and_schedule_v3`. This is byte-for-byte the constructor the pre-loss
    #: `rewrite.md` tables (§1.2, §1.6, §2.1-§2.8) were measured on, so numbers
    #: quoted against these names ARE commensurable with them.
    for L in (512, 2048):
        cs.append(Cell(name=f"atscale-L{L}", widths=(256, 256), BC=64, H=8, T=L,
                       sparse_activation=True, seed=0, producer_route_w_mul=8.0,
                       note="THE canonical at-scale cell: real producer routing, "
                            "planner-selected plan; historical continuity"))

    #: AT-SCALE AT `BC = 16` -- the SAME producer constructor, the OTHER legal `BC`.
    #:
    #: Added 2026-08-01 for the `BT` calibration matrix (§2.32). §2.31's `BT` A/B
    #: covered `BC = 64` only, and superseding a `(BC, decay) -> BT` table on
    #: `BC = 64` evidence alone would be exactly the extrapolation §2.31 refused to
    #: make. `BC` is not a routing parameter -- it partitions the SAME `N = 65536`
    #: state into 4x more owners -- so this cell's supports are drawn identically to
    #: `atscale-L2048`'s (same seed, same `route_W` scale, same layer init) and the
    #: only thing that moves is the owner geometry and therefore the plan. That is
    #: what makes the two cells' `BT` orderings commensurable as a `BC` sweep.
    cs.append(Cell(name="atscale-L2048-BC16", widths=(256, 256), BC=16, H=8, T=2048,
                   sparse_activation=True, seed=0, producer_route_w_mul=8.0,
                   note="atscale-L2048's routing at the other legal BC; the "
                        "BC=16 half of the BT calibration matrix"))

    #: AT-SCALE, SYNTHETIC -- the DENSER arm, kept because it is already published.
    #:
    #: **WHY IT EXISTS AND WHY IT IS NOT THE CANONICAL CELL.** While the producer
    #: harness was lost, the at-scale cell was reconstructed synthetically by
    #: sweeping an iid-Gaussian entmax logit scale to hit `rewrite.md`'s recorded
    #: plan stats. The sweep could not reach them, because it is a DIFFERENT KNOB:
    #: `route_W.mul_(8.0)` scales a shared routing PROJECTION, so the logits are
    #: correlated across tokens and levels and their scale also depends on
    #: `hidden_size` and the init -- nothing about it makes "8" a logit sigma.
    #: Scale 4 was taken as the closest match on `mean_K_candidate`. These cells are
    #: retained under their own names so the numbers already recorded against them
    #: stay attributable, and NOT deleted, because deleting a fixture that has
    #: published numbers is how this whole problem started.
    for L in (512, 2048):
        cs.append(Cell(name=f"atscale-syn4-L{L}", widths=(256, 256), BC=64, H=8, T=L,
                       read=("entmax", 4.0), write=("entmax", 4.0),
                       sparse_activation=True, seed=0,
                       note="synthetic reconstruction, DENSER than atscale-L*; "
                            "not commensurable with the pre-loss tables"))

    return {c.name: c for c in cs}


CELLS = _cells()

#: Named GROUPS, so a gate run is also cited by name rather than by a slice.
GROUPS = {
    "topo": [n for n in CELLS if n.startswith("topo-")],
    "wk1": [n for n in CELLS if n.startswith("wk1-")],
    "atscale": [n for n in CELLS if n.startswith("atscale-")],
    #: The `raw` arm is gated on a SUBSET of the topology cells crossed with both
    #: precision points and both decay points -- the normalization axis crossed with
    #: the two axes that share the STATE with it, rather than tested in a corner.
    "raw": ["topo-D1-w256-BC64", "topo-D2-w64x64-BC64",
            "topo-D3-w16x16x64-BC64", "topo-D4-w8x8x8x16-BC16"],
}


# ---------------------------------------------------------------------------
# ORACLE GATE
# ---------------------------------------------------------------------------

def run_cell(cell: Cell, *, norm="global", decay_temp=None, prec="bf16",
             fx=None, compact=False):
    """Run one arm of one cell and return `(y, state, y_ref, state_ref)`."""
    fx = cell.build() if fx is None else fx
    decay = None if decay_temp is None else LeafMassDecay(dials=fx["dials"][decay_temp])
    g_read = None if norm == "global" else fx["g_read"]
    topology = fx["topology"]

    y_ref, state_ref = naive_rola_v3(
        fx["v"], fx["read_levels"], fx["write_levels"], g_read, fx["g_write"],
        topology, decay, norm, initial_state=None, output_final_state=True)

    #: THE NEW CONSUMER MUST BE STAMPED `panel` UNDER DECAY, and it is a gating
    #: requirement rather than bookkeeping. Its tile algorithm IS §7.4.1's panel
    #: lowering -- two separately exponentiated survival factors -- so it inherits
    #: that lowering's DOMAIN, and `v3_consumer_forward` evaluates
    #: `panel_exponent_bound` against `PANEL_MAX_EXPONENT` only when the plan says
    #: `panel`. Stamping anything else would let the new kernel run outside the
    #: range in which its two panels are an exact rewrite of the recurrence, with
    #: the admission check silently skipped. Off the decay arm the axis is
    #: unreachable by this kernel and the planner's own choice stands.
    plan, sched = cell.plan(norm=norm, decay_on=decay is not None, prec=prec,
                            fx=fx, compact=compact)
    y, state, _ = v3_consumer_forward(
        fx["v_f32"], fx["read_f32"], fx["write_f32"],
        None if g_read is None else fx["g_read_f32"], fx["g_write_f32"],
        plan, decay=decay, schedule=sched)
    return y, state, y_ref, state_ref


def gate_cell(cell: Cell, *, norm, decay_temp, prec, verbose=True, compact=False):
    arm = f"{norm} {prec}" + (f" decay:{decay_temp}" if decay_temp else "")
    if compact:
        arm += " COMPACT"
    try:
        y, state, y_ref, state_ref = run_cell(
            cell, norm=norm, decay_temp=decay_temp, prec=prec, compact=compact)
    except ValueError as exc:
        #: A REFUSAL IS A PASS, and it is the only outcome that can be, because the
        #: alternative is a wrong answer rather than an inaccurate one. `panel`'s
        #: admission bound refuses a routing whose per-leaf reference exponent
        #: exceeds `PANEL_MAX_EXPONENT`; past it the write factor saturates at the
        #: kernel's clamp while the read factor underflows and their product --
        #: `O(1)` at `t ~ j` -- comes out as `~0`. The HOT-dial arm crossed into
        #: that regime on the write-`k=1` cells, which is exactly the condition the
        #: bound names (a near-unit leaf rate met by a near-one-hot write clock), so
        #: this path is EXERCISED rather than hypothetical. It is reported
        #: distinctly, never silently counted as a numeric pass.
        if "not admissible" not in str(exc):
            raise
        if verbose:
            print(f"  RFSD  {cell.name:26s} {arm:22s} "
                  f"admission bound refused (expected: this is the §12 rule-3 "
                  f"dynamic-range limit)")
        return True
    err_y = float((y.double() - y_ref).abs().max()) / max(1e-30, float(y_ref.abs().max()))
    err_s = float((state.double() - state_ref).abs().max()) / max(
        1e-30, float(state_ref.abs().max()))
    tol_y = RTOL if prec == "tf32x3" else BF16_RTOL
    tol_s = RTOL if prec == "tf32x3" else BF16_FOLD_STATE_RTOL
    ok = (err_y < tol_y) and (err_s < tol_s)
    if verbose:
        print(f"  {'PASS' if ok else 'FAIL':4s}  {cell.name:26s} {arm:22s} "
              f"y {err_y:.3e} (<{tol_y:.0e})  state {err_s:.3e} (<{tol_s:.2e})")
    return ok


def run_oracle(*, decay=True, hot=False, compact=False):
    """The gate, by named group. Every cell states its arm; nothing is implicit."""
    temps = [None] + ([("hot" if hot else "warm")] if decay else [])
    total = passed = 0
    for group, arms in (
            ("topo", [(n, "global", t, p) for n in GROUPS["topo"]
                      for p in ("bf16", "tf32x3") for t in temps]),
            #: `wk1` is decay-only: its whole purpose is the read-only-tile
            #: exemption, which only exists on the decay arm.
            ("wk1", [(n, "global", ("hot" if hot else "warm"), p)
                     for n in GROUPS["wk1"] for p in ("bf16", "tf32x3")]),
            ("raw", [(n, "raw", t, p) for n in GROUPS["raw"]
                     for p in ("bf16", "tf32x3") for t in temps]),
    ):
        print(f"-- group {group} ({len(arms)} arms)")
        for name, norm, temp, prec in arms:
            total += 1
            passed += int(gate_cell(CELLS[name], norm=norm, decay_temp=temp, prec=prec,
                                    compact=compact))
    print(f"\n{passed}/{total} oracle arms PASS" + (" [COMPACT]" if compact else ""))
    return passed == total


# ---------------------------------------------------------------------------
# §6.4 SEQUENCE PARALLELISM (`P > 1`)
# ---------------------------------------------------------------------------

def _forced_split(plan, schedule, P):
    """Split EVERY owner `P` ways, BYPASSING the three admission gates.

    The gates are a PERFORMANCE POLICY (occupancy waves, a relative bandwidth
    budget, a workspace cap) and they are selected at plan time from schedule
    statistics -- device code branches on nothing. The kernel's EXACTNESS must not
    depend on which `P` the policy happened to admit, so the numeric gate drives
    `P` directly. The policy's own behaviour is gated separately, in
    `tests/ops/test_rola_v3_sequence_parallel_policy.py`.
    """
    from rola.routing.sequence_parallel import (
        SequenceParallelPlan, SequenceParallelPolicy, owner_slot_counts,
        select_sequence_parallelism)
    base = select_sequence_parallelism(
        plan, schedule, policy=SequenceParallelPolicy(workspace_cap=1 << 30))
    n_owners = int(schedule.owners.shape[0])
    assert P <= schedule.n_tiles, (
        f"cannot split {P} ways over {schedule.n_tiles} tiles; raise T or lower P "
        "rather than letting the split degenerate into empty segments")
    Ps = tuple([P] * n_owners)
    return SequenceParallelPlan(
        P=Ps, limiting_gate=(None,) * n_owners, binding_gate=None, sms=base.sms,
        blocks_per_sm=base.blocks_per_sm, waves_before=base.waves_before,
        waves_after=0.0, base_traffic_bytes=0, roundtrip_bytes=0,
        roundtrip_seconds=0.0, base_traffic_seconds=0.0, ledger=base.ledger,
        n_slots=sum(owner_slot_counts(Ps)), BC=base.BC, cols=base.cols)


def run_split(cell: Cell, *, P, norm="global", decay_temp=None, prec="bf16",
              fx=None, verbose=True, compact=False):
    """`P`-way split against the canonical fp64 oracle, plus the `P = 1` control.

    **TWO DIFFERENT GATES, BECAUSE THE TWO QUANTITIES HAVE DIFFERENT STATUS.**

    * `P = 1` must be **BIT-IDENTICAL** to passing no plan at all. That is the
      structural carve-out and it is checkable exactly: the `nullptr` work arrays
      make `seg = 0`, `nseg = 1`, `last_seg` true and the range the whole axis
      WITHOUT READING MEMORY, so nothing about the launch changes.
    * `P > 1` is **NOT** bitwise and must not be gated as such. The split
      re-associates the `P - 1` boundary folds in fp32 -- pass 3's last segment
      still runs serially from the accumulated boundary, which is what confines the
      re-association to the boundaries -- so it is gated against the ORACLE at the
      `P = 1` tolerance, which is the honest statement: the split must not be worse
      than the unsplit kernel, not that it must be identical to it.
    """
    fx = cell.build() if fx is None else fx
    decay = None if decay_temp is None else LeafMassDecay(dials=fx["dials"][decay_temp])
    g_read = None if norm == "global" else fx["g_read"]
    plan, sched = cell.plan(norm=norm, decay_on=decay is not None, prec=prec,
                            fx=fx,
                            compact=compact)
    if sched is None:
        sched = build_v3_schedule(fx["read_f32"], fx["write_f32"],
                                  None if g_read is None else fx["g_read_f32"],
                                  fx["g_write_f32"], plan)
    y_ref, state_ref = naive_rola_v3(
        fx["v"], fx["read_levels"], fx["write_levels"], g_read, fx["g_write"],
        fx["topology"], decay, norm, initial_state=None, output_final_state=True)

    def launch(sp):
        return v3_consumer_forward(
            fx["v_f32"], fx["read_f32"], fx["write_f32"],
            None if g_read is None else fx["g_read_f32"], fx["g_write_f32"], plan,
            decay=decay, schedule=sched, sequence_parallel=sp)

    y1, st1, _ = launch(None)
    sp = _forced_split(plan, sched, P)
    yp, stp, _ = launch(sp)

    err_y = float((yp.double() - y_ref).abs().max()) / max(1e-30, float(y_ref.abs().max()))
    err_s = float((stp.double() - state_ref).abs().max()) / max(
        1e-30, float(state_ref.abs().max()))
    tol_y = RTOL if prec == "tf32x3" else BF16_RTOL
    tol_s = RTOL if prec == "tf32x3" else BF16_FOLD_STATE_RTOL
    ok = (err_y < tol_y) and (err_s < tol_s)
    #: THE `P = 1` CONTROL IS RUN IN THE SAME PROCESS ON THE SAME FIXTURE, so a
    #: failure at `P > 1` can be attributed to the SPLIT rather than to the cell.
    #: Without it a bad fixture and a bad split look identical -- the discipline
    #: that resolved the paging gate, where the control proved the gate wrong
    #: rather than the kernel.
    ctl = float((y1.double() - y_ref).abs().max()) / max(1e-30, float(y_ref.abs().max()))
    if verbose:
        arm = (f"P={P} {norm} {prec}" + (f" decay:{decay_temp}" if decay_temp else "")
               + (" COMPACT" if compact else ""))
        print(f"  {'PASS' if ok else 'FAIL':4s}  {cell.name:26s} {arm:24s} "
              f"y {err_y:.3e}  state {err_s:.3e}  (P=1 control y {ctl:.3e})")
    return ok


def run_split_gate(*, decay=True, compact=False):
    """`P` crossed with decay and normalization -- the two axes §2.13 names.

    `A` is decay's object (it is identically `1` on the `decay = None` arm and the
    accumulated `s0^2` under decay), and the summary ROW WIDTH is normalization's
    (`cols` carries the mass column under `global` and not under `raw`). Testing
    `P` at one point on either axis would leave the interaction untested.
    """
    total = passed = 0
    temps = [None] + (["warm"] if decay else [])
    for name in ("topo-D2-w64x64-BC64", "topo-D3-w16x16x64-BC64",
                 "wk1-D2-w64x64-BC64", "topo-D2-w64x16-BC16"):
        for P in (2, 3):
            for norm in ("global", "raw"):
                for temp in temps:
                    total += 1
                    passed += int(run_split(CELLS[name], P=P, norm=norm,
                                            decay_temp=temp, prec="tf32x3",
                                            compact=compact))
    print(f"\n{passed}/{total} P>1 arms PASS" + (" [COMPACT]" if compact else ""))
    return passed == total


def run_p1_identity():
    """`P = 1` with a plan vs `P = 1` with no plan -- **ON STATE, and `y` EXCLUDED.**

    `P = 1` is the primary benchmarked configuration (§6.4), so it must reach the
    kernel through exactly the arguments it reached it through before `P` existed:
    a perturbed baseline is not a baseline. An unsplit plan carries no work list and
    no workspace, so every sequence-parallel pointer is null and the kernel takes
    the pre-`P` path without reading memory.

    **`y` IS DELIBERATELY NOT ASSERTED, and the NULL CONTROL below is what
    establishes that rather than a claim.** The §5 readout lands through `atomicAdd`
    into `num`/`den` from concurrently scheduled owners, so its summation order is a
    RACE BY CONSTRUCTION -- two byte-identical launches already disagree in the last
    bits. Asserting `torch.equal` on `y` would be asserting a property the kernel has
    never had. STATE is the output `P` actually moves from deterministic to
    re-associated: it is per-owner and published exactly once, so it is the output
    the bitwise discipline can and does apply to.

    This is the same discipline the paging gate needed (§2.8), and it is run the
    same way round: there the control proved the GATE wrong rather than the kernel,
    and a first version of THIS gate asserted `y` bitwise and failed for exactly
    that reason.
    """
    from rola.routing.sequence_parallel import (
        SequenceParallelPlan, SequenceParallelPolicy, owner_slot_counts,
        select_sequence_parallelism)
    ok = True
    for name in ("topo-D2-w64x64-BC64", "wk1-D2-w64x64-BC64"):
        for temp in (None, "warm"):
            cell = CELLS[name]
            fx = cell.build()
            d = None if temp is None else LeafMassDecay(dials=fx["dials"][temp])
            plan, sched = cell.plan(norm="global", decay_on=d is not None,
                                    prec="bf16",
                                    fx=fx)
            if sched is None:
                sched = build_v3_schedule(fx["read_f32"], fx["write_f32"], None,
                                          fx["g_write_f32"], plan)
            base = select_sequence_parallelism(
                plan, sched, policy=SequenceParallelPolicy(workspace_cap=1 << 30))
            n = int(sched.owners.shape[0])
            Ps = tuple([1] * n)
            unsplit = SequenceParallelPlan(
                P=Ps, limiting_gate=(None,) * n, binding_gate=None, sms=base.sms,
                blocks_per_sm=base.blocks_per_sm, waves_before=base.waves_before,
                waves_after=0.0, base_traffic_bytes=0, roundtrip_bytes=0,
                roundtrip_seconds=0.0, base_traffic_seconds=0.0, ledger=base.ledger,
                n_slots=sum(owner_slot_counts(Ps)), BC=base.BC, cols=base.cols)

            def launch(sp):
                return v3_consumer_forward(
                    fx["v_f32"], fx["read_f32"], fx["write_f32"], None,
                    fx["g_write_f32"], plan, decay=d, schedule=sched,
                    sequence_parallel=sp)

            ya, sa, _ = launch(None)
            yb, sb, _ = launch(unsplit)
            # THE NULL CONTROL: two identical NO-PLAN launches. If `y` differs here
            # too, a `y` difference above is the atomic race and not the plan.
            yc, sc, _ = launch(None)
            state_bit = bool(torch.equal(sa, sb))
            null_state_bit = bool(torch.equal(sa, sc))
            y_plan = bool(torch.equal(ya, yb))
            y_null = bool(torch.equal(ya, yc))
            # The gate is STATE bitwise, on both the plan comparison and the null
            # control. `y` is reported, never asserted.
            bit = state_bit and null_state_bit
            ok = ok and bit
            print(f"  {'PASS' if bit else 'FAIL':4s}  {name:26s} "
                  f"{'decay' if temp else 'none ':5s}  STATE bitwise: plan {state_bit} "
                  f"null-control {null_state_bit}   "
                  f"(y bitwise: plan {y_plan}, NULL CONTROL {y_null} "
                  f"-- the atomicAdd race, not the plan)")
    return ok


def run_compact_identity():
    """`COMPACT = true` vs `COMPACT = false` at the DENSE DEGENERACY -- **ON STATE**.

    **This is the strongest available check on the compaction, and it is free.** The
    ballot emits the gathered leaf list ASCENDING, so on a tile where every leaf is
    active AND causal the list is `c_idx[k] == k`, `K == BC`, `KI == BC`, and every
    compacted extent equals its dense one. The compacted kernel is then not merely
    EQUAL to the dense one, it is BIT-IDENTICAL: same k-steps, same m-tiles, same
    accumulation order. Any indexing defect -- a wrong gather, a mis-ordered list, a
    dropped pad, a scatter aimed at the wrong leaf -- breaks that exactly.

    **WHICH CELLS IT IS REACHABLE ON, MEASURED RATHER THAN ASSUMED.** The degeneracy
    needs every leaf both READ and WRITTEN in the tile: a leaf with no in-tile write
    is not causal (`W_s == 0` gives an empty causal mask), so the two-pass ballot
    puts it AFTER the causal leaves and the list is a permutation rather than the
    identity. The committed manifest says `mean_K_candidate == BC` exactly on all
    eight `topo-*` cells (dense read AND dense write), so the degeneracy is realized
    there; on the `wk1-*` cells the write is `k = 1` per token, so `KI < K` and the
    order legitimately permutes -- those cells are gated against the ORACLE instead,
    which is where they belong.

    `y` IS NOT ASSERTED, for the reason the `P = 1` identity gate records: the §5
    readout lands through `atomicAdd` from concurrently scheduled owners, so two
    BYTE-IDENTICAL launches already disagree in the last bits. The null control below
    is run so that is a measurement rather than a claim.
    """
    ok = True
    for name in GROUPS["topo"]:
        for temp in (None, "warm"):
            cell = CELLS[name]
            fx = cell.build()
            def go(compact):
                return run_cell(cell, norm="global", decay_temp=temp, prec="bf16",
                                fx=fx, compact=compact)
            ya, sa, _, _ = go(False)
            yb, sb, _, _ = go(True)
            yc, sc, _, _ = go(False)   # THE NULL CONTROL: two identical dense launches
            state_bit = bool(torch.equal(sa, sb))
            null_state_bit = bool(torch.equal(sa, sc))
            y_bit = bool(torch.equal(ya, yb))
            y_null = bool(torch.equal(ya, yc))
            bit = state_bit and null_state_bit
            ok = ok and bit
            print(f"  {'PASS' if bit else 'FAIL':4s}  {name:26s} "
                  f"{'decay' if temp else 'none ':5s}  STATE bitwise: compact {state_bit} "
                  f"null-control {null_state_bit}   "
                  f"(y bitwise: compact {y_bit}, NULL CONTROL {y_null})")
    return ok


# ---------------------------------------------------------------------------
# INTERLEAVED PAIRED-RATIO A/B
# ---------------------------------------------------------------------------

def _iqr(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[int(0.75 * n)] - xs[int(0.25 * n)]


def _timed(cell, fx, plan, decay, g_read, schedule):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _, _, sch = v3_consumer_forward(
        fx["v_f32"], fx["read_f32"], fx["write_f32"],
        None if g_read is None else fx["g_read_f32"], fx["g_write_f32"],
        plan, decay=decay, schedule=schedule)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3, sch


def arm_time(name, *, reps=31, norm="global", decay_temp=None, compact=False,
             bt=None):
    """ONE arm of the new consumer, absolute times -- the CROSS-BUILD instrument.

    **Every other timing mode in this file is a same-process paired A/B, and that is
    the standing doctrine. This mode exists for the one comparison the doctrine
    cannot express: two arms that differ by a COMPILE-TIME constant of the kernel
    itself** (the launch bound's register budget, `consumer_reg_budget`), which cannot
    coexist in one binary and therefore cannot alternate rep by rep in one process.

    The pairing is recovered one level up: the caller alternates PROCESSES (build A,
    build B, build A, ...), each process reporting the median of its own reps, and
    the per-pair ratio is taken across adjacent process pairs. That is weaker than
    rep-level interleaving -- a drift step can land between two processes rather
    than between two reps -- so this mode prints median, IQR and min, the caller
    reports the IQR alongside, and the verdict is corroborated with ncu counters,
    which are contention-independent.
    """
    cell = CELLS[name]
    fx = cell.build()
    decay = None if decay_temp is None else LeafMassDecay(dials=fx["dials"][decay_temp])
    gr = None if norm == "global" else fx["g_read"]
    plan, sch = cell.plan(norm=norm, decay_on=decay is not None, prec="bf16",
                          fx=fx,
                          compact=compact, bt=bt)
    _, sch = _timed(cell, fx, plan, decay, gr, sch)      # warm + schedule
    ts = []
    for _ in range(reps):
        t, _ = _timed(cell, fx, plan, decay, gr, sch)
        ts.append(t)
    med = statistics.median(ts)
    print(f"ARMTIME {cell.name} norm={norm} decay={decay_temp} compact={compact} "
          f"reps={reps}")
    print(f"  median {med:.4f} ms   IQR {_iqr(ts):.4f}   min {min(ts):.4f}")
    return dict(median_ms=med, iqr=_iqr(ts), min_ms=min(ts), times=ts)


def run_btab(name, *, reps=21, norm="global", decay_temp=None, compact=False):
    """`BT = 16` against `BT = 32` ON THE NEW CONSUMER, interleaved rep by rep.

    **This is the calibration instrument for the planner's `BT` axis** (§2.31,
    §2.32). Both arms are the ONE consumer, so the only
    difference is the tile size, and the schedule is REBUILT at each `BT` by
    restricting the planner's candidate set (see `Cell.plan`), never stamped.

    The reported statistic is the median of the per-pair ratios `t32 / t16`, so a
    value above 1 means `BT = 32` is SLOWER. Absolute medians and IQRs are printed
    alongside; a cell whose IQR cannot resolve the effect is reported as
    unresolvable here and decided on `ncu` counters instead, which is stated in the
    findings rather than hidden by quoting the ratio anyway.
    """
    cell = CELLS[name]
    fx = cell.build()
    decay = None if decay_temp is None else LeafMassDecay(dials=fx["dials"][decay_temp])
    gr = None if norm == "global" else fx["g_read"]

    def mk(bt):
        #: The new consumer must be stamped `panel` on the decay arm so its
        #: admission bound is checked -- the same divergence `run_atscale` makes,
        #: and it is applied to BOTH arms here, so it cannot bias the ratio.
        pl, sch = cell.plan(norm=norm, decay_on=decay is not None, prec="bf16",
                            fx=fx, compact=compact, bt=bt)
        _, sch = _timed(cell, fx, pl, decay, gr, sch)
        return pl, sch

    p16, s16 = mk(16)
    p32, s32 = mk(32)
    if (p16.BT, p32.BT) != (16, 32):
        raise SystemExit(f"the planner returned BT={p16.BT}/{p32.BT}, not 16/32")
    t16, t32, pair = [], [], []
    for _ in range(reps):
        x, _ = _timed(cell, fx, p16, decay, gr, s16)
        y, _ = _timed(cell, fx, p32, decay, gr, s32)
        t16.append(x)
        t32.append(y)
        pair.append(y / x)
    m16, m32, mr = statistics.median(t16), statistics.median(t32), statistics.median(pair)
    print(f"{name} norm={norm} decay={decay_temp} compact={compact} "
          f"BC={cell.BC}")
    print(f"  BT=16 {m16:.3f} ms (IQR {_iqr(t16):.3f})   "
          f"BT=32 {m32:.3f} ms (IQR {_iqr(t32):.3f})")
    print(f"  PAIRED 32/16 {mr:.4f} (IQR {_iqr(pair):.4f})   = {100 * (mr - 1):+.1f} %")
    return dict(paired=mr, paired_iqr=_iqr(pair), ms16=m16, ms32=m32)


# ---------------------------------------------------------------------------
# PER-FEATURE PRICE -- the new kernel against ITSELF
#
# A production A/B cannot price a FEATURE, only a kernel: production implements
# decay with a different algorithm (the token-sequential walk) and `global` with a
# different state layout, so its decay-vs-none delta moves two things at once.
# Holding the kernel fixed and flipping one axis is what isolates the feature, and
# residency (3 CTAs/SM on every arm) is held fixed with it.
# ---------------------------------------------------------------------------

def _paired_one_kernel(cell, fx, arm_a, arm_b, reps, label, extra=""):
    """Interleaved paired ratio between two ARMS OF THE NEW KERNEL on one fixture.

    `arm_*` are `(norm, decay_temp)`. Interleaved rep by rep, per the standing
    rule's interleaving clause; the reportable statistic is the median of the
    per-pair ratios.
    """
    def mk(arm):
        norm, temp = arm
        d = None if temp is None else LeafMassDecay(dials=fx["dials"][temp])
        gr = None if norm == "global" else fx["g_read"]
        pl, sch = cell.plan(norm=norm, decay_on=d is not None, prec="bf16",
                            fx=fx)
        _, sch = _timed(cell, fx, pl, d, gr, sch)
        return pl, d, gr, sch

    pa, da, ga, sa = mk(arm_a)
    pb, db, gb, sb = mk(arm_b)
    ta, tb, pair = [], [], []
    for _ in range(reps):
        x, _ = _timed(cell, fx, pa, da, ga, sa)
        y, _ = _timed(cell, fx, pb, db, gb, sb)
        ta.append(x)
        tb.append(y)
        pair.append(y / x)
    ma, mb, mr = statistics.median(ta), statistics.median(tb), statistics.median(pair)
    print(f"{cell.name} {label}{extra}")
    print(f"  a={arm_a} {ma:.3f} ms (IQR {_iqr(ta):.3f})   "
          f"b={arm_b} {mb:.3f} ms (IQR {_iqr(tb):.3f})")
    print(f"  PAIRED b/a {mr:.4f} (IQR {_iqr(pair):.4f})   = {100 * (mr - 1):+.1f} %")
    return dict(paired=mr, paired_iqr=_iqr(pair), a_ms=ma, b_ms=mb)


def decay_price(name, *, reps=11, decay_temp="warm"):
    """DECAY'S OWN PRICE: the new kernel against itself, decay vs none.

    The structural expectation this replaces with a measurement: on a DENSE-WRITE
    cell decay adds two `expf` panels (`2 * BT * BC`, on the XU pipe, which §2.5's
    instruction mix measured at 0.8 M -- essentially idle), the clock prefix, the
    two coefficient folds and the `s0` entry/exit sweeps. On a SPARSE-WRITE cell
    §2's write-conditioned quiescence exempts read-only tiles STRUCTURALLY
    (`c = 0` => `keep == 1` exactly), so the cost should scale with the WRITE-LIVE
    share and approach free as that share falls.
    """
    cell = CELLS[name]
    fx = cell.build()
    share = skip_share(cell, fx)["read_only_frac"]
    r = _paired_one_kernel(
        cell, fx, ("global", None), ("global", decay_temp), reps,
        "DECAY PRICE (new kernel both arms)",
        f", read-only share {100 * share:.1f} %")
    #: **THE RATIO ABOVE IS NOW THE KERNEL'S PRICE, and it was not always.** The
    #: decay arm's preamble (`build_clock_stream` + `leaf_rate_dials` +
    #: `panel_exponent_bound`) used to be rebuilt on EVERY call, including calls
    #: handed a prebuilt `schedule=`, and landed entirely in this ratio because the
    #: `decay=None` arm pays none of it. It is now memoized on the schedule
    #: (`V3DecayStream`), so it is paid once on the warm call and is OUTSIDE the
    #: timed region -- which is why it is REPORTED here but NOT subtracted.
    #: Subtracting it now would double-count.
    #:
    #: It never contaminated the production A/B: there both arms are decay arms and
    #: paid it identically, so it cancelled.
    pre = _decay_preamble_ms(cell, fx, decay_temp)
    print(f"  (one-time preamble, now memoized OUT of the timed region: {pre:.3f} ms)")
    r.update(preamble_ms=pre)
    return r


def _decay_preamble_ms(cell, fx, decay_temp, reps=9):
    """Median cost of the decay arm's per-call preamble, on this fixture."""
    from rola.planner.schedule import (
        build_clock_stream, leaf_rate_dials, panel_exponent_bound)
    decay = LeafMassDecay(dials=fx["dials"][decay_temp])
    plan, _ = cell.plan(norm="global", decay_on=True, prec="bf16",
                        fx=fx)
    wl = fx["write_f32"]
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ca = build_clock_stream(wl, wl, plan)
        leaf_rate_dials(decay, plan.topology).to("cuda")
        panel_exponent_bound(ca, decay, plan.BT)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def compact_price(name, *, reps=11, norm="global", decay_temp=None, stats=True):
    """RULING 2's MEASUREMENT: compaction's own price, the new kernel against itself.

    Both arms are the NEW consumer on ONE fixture in ONE process, interleaved rep by
    rep; the only difference is the `COMPACT` template axis. That is what isolates
    the AXIS -- a production A/B could not, because production's compaction is a
    different mechanism (a dense-panel-to-compacted COPY, a staging-ring probe and a
    third barrier, none of which exists here) priced by a cost model calibrated on
    that mechanism.

    Residency is 3 CTAs/SM on BOTH arms (read off the driver), so it is not a
    confound: what the ratio measures is the gather's cost against the three MMA
    extents it shrinks, and nothing else.
    """
    cell = CELLS[name]
    fx = cell.build()

    def mk(compact):
        d = None if decay_temp is None else LeafMassDecay(dials=fx["dials"][decay_temp])
        gr = None if norm == "global" else fx["g_read"]
        pl, sch = cell.plan(norm=norm, decay_on=d is not None, prec="bf16",
                            fx=fx,
                            compact=compact)
        _, sch = _timed(cell, fx, pl, d, gr, sch)
        return pl, d, gr, sch

    pa, da, ga, sa = mk(False)
    pb, db, gb, sb = mk(True)
    ta, tb, pair = [], [], []
    for _ in range(reps):
        x, _ = _timed(cell, fx, pa, da, ga, sa)
        y, _ = _timed(cell, fx, pb, db, gb, sb)
        ta.append(x)
        tb.append(y)
        pair.append(y / x)
    ma, mb, mr = statistics.median(ta), statistics.median(tb), statistics.median(pair)
    extra = ""
    if stats:
        st = gather_stats(cell, fx)
        extra = (f"  [mean K {st['mean_K']:.2f}/{cell.BC}, mean KI {st['mean_KI']:.2f}, "
                 f"k-steps {st['mean_ksteps']:.2f}/{cell.BC // 16}, "
                 f"fold m-tiles {st['mean_mtiles']:.2f}/{cell.BC // 16}, "
                 f"{st['owners_sampled']} owners sampled]")
    print(f"{cell.name} COMPACT PRICE norm={norm} decay={decay_temp}{extra}")
    print(f"  dense  {ma:.3f} ms (IQR {_iqr(ta):.3f})   "
          f"COMPACT {mb:.3f} ms (IQR {_iqr(tb):.3f})")
    print(f"  PAIRED compact/dense {mr:.4f} (IQR {_iqr(pair):.4f})   "
          f"= {100 * (mr - 1):+.1f} %")
    return dict(paired=mr, paired_iqr=_iqr(pair), dense_ms=ma, compact_ms=mb)


def gather_stats(cell: Cell, fx=None, max_owners=48):
    """The gather's REALIZED extents, computed independently of the kernel.

    Re-derives §3 step 2's per-leaf support words and the two-pass ballot's counts in
    Python from the schedule's TOKEN bitmaps -- the same object the kernel reads --
    and reports what compaction actually buys per candidate `(owner, tile)`: `K`
    (the readout's k-extent and the fold's m-extent), `KI` (the intra Gram's), and
    the two in the QUANTA the kernel actually moves in, MMA k-steps and warp m-tiles.

    **The quanta are the point.** A cell can have `mean_K = 0.97` of `BC = 64` and
    still cost a full k-step, because `ceil(K/16)` cannot go below 1 on a candidate
    tile. `mean_K` alone overstates the saving; `mean_ksteps` is what the kernel
    pays.

    Owners are SAMPLED (evenly spaced, up to `max_owners`) because the at-scale cell
    has 8180 of them; the sample size is reported with the number rather than
    hidden.
    """
    fx = cell.build() if fx is None else fx
    plan, sch = cell.plan(norm="global", decay_on=False, prec="bf16", fx=fx)
    if sch is None:
        sch = build_v3_schedule(fx["read_f32"], fx["write_f32"], None,
                                fx["g_write_f32"], plan)
    geo = owner_geometry(tuple(cell.widths), cell.BC)
    j, nsl, nw = geo.first_spanned_level, geo.n_slices, sch.n_words
    #: `plan.BT`, NOT `cell.BT`. FIXED 2026-08-01 (item 6b), and it was a real defect
    #: rather than a tidy-up: the bitmaps below belong to `sch`, which is built at
    #: `plan.BT`, while the tile count was taken from the cell's NOMINAL `BT`.
    #: Whenever the planner selected the other tile size the two disagreed and
    #: `unpack_bits` read a tile axis the words do not have. It was reachable and it
    #: was reached: the committed `atscale-L2048-BC16` entry (749436 candidate pairs)
    #: was produced by a `BT = 32` schedule counted at `nt = 2048/16 = 128`. At
    #: `plan.BT == cell.BT` the two forms are identical, which is why every other
    #: cell's entry is unchanged.
    nt = (cell.T + plan.BT - 1) // plan.BT
    widths = tuple(cell.widths)
    m32 = (1 << 32) - 1

    def leaf_digits(level, slice_index):
        """`base + (i / stride) % modulus` for `i in [0, BC)` -- the kernel's map."""
        i = torch.arange(cell.BC, dtype=torch.int64)
        if level < j:
            return torch.zeros(cell.BC, dtype=torch.int64)   # base only; added below
        inner = 1
        for w in widths[level + 1:]:
            inner *= w
        modulus = geo.slice_digits if level == j else widths[level]
        base = slice_index * geo.slice_digits if level == j else 0
        return base + (i // inner) % modulus

    tok_r = sch.read_token_bits.to(torch.int64) & m32     # [BH, n_tiles, total_rows]
    tok_w = sch.write_token_bits.to(torch.int64) & m32
    ru = sch.read_slice_union.view(-1, nsl, nw)
    wu = sch.write_slice_union.view(-1, nsl, nw)
    owners = sch.owners.cpu()
    n_own = int(owners.shape[0])
    step = max(1, n_own // max_owners)
    picks = list(range(0, n_own, step))[:max_owners]
    valid = torch.tensor([min(cell.BT, max(0, cell.T - t * cell.BT)) for t in range(nt)],
                         dtype=torch.int64)
    valid = (1 << valid) - 1                                     # [n_tiles]
    tot_K = tot_KI = tot_ks = tot_mt = 0
    n_cand = 0
    for k in picks:
        bh, ow = int(owners[k, 0]), int(owners[k, 1])
        digits, slice_index = geo.owner_digits(ow)
        r, w = ru[bh, ow % nsl].clone(), wu[bh, ow % nsl].clone()
        for l in range(j):
            row = sch.level_offsets[l] + digits[l]
            r &= sch.read_tile_bits[bh, row]
            w &= sch.write_tile_bits[bh, row]
        cand = (unpack_bits(r, nt).bool() | unpack_bits(w, nt).bool()).cpu()
        # (a') the FIXED levels' AND, as token validity; (b) the per-leaf AND.
        fr = valid.clone()
        fw = valid.clone()
        for l in range(j):
            row = sch.level_offsets[l] + digits[l]
            fr = fr & tok_r[bh, :, row].cpu()
            fw = fw & tok_w[bh, :, row].cpu()
        rw = fr[:, None].repeat(1, cell.BC)
        ww = fw[:, None].repeat(1, cell.BC)
        for l in range(j, len(widths)):
            rows = sch.level_offsets[l] + leaf_digits(l, slice_index)
            rw &= tok_r[bh][:, rows].cpu()
            ww &= tok_w[bh][:, rows].cpu()
        active = (rw | ww) != 0
        first_write = ww & (-ww)
        in_intra = (rw & (-first_write)) != 0
        K = active.sum(1)[cand]
        KI = in_intra.sum(1)[cand]
        n_cand += int(K.numel())
        tot_K += int(K.sum())
        tot_KI += int(KI.sum())
        unit = 16          # bf16 `m16n8k16`
        tot_ks += int(((K + unit - 1) // unit).sum())
        tot_mt += int(((K + 15) // 16).sum())
    d = max(1, n_cand)
    return dict(owners_sampled=len(picks), candidate_pairs=n_cand,
                mean_K=tot_K / d, mean_KI=tot_KI / d,
                mean_ksteps=tot_ks / d, mean_mtiles=tot_mt / d)


def norm_price(name, *, reps=11, decay_temp=None):
    """NORMALIZATION'S OWN PRICE: `global` vs `raw`, new kernel both arms.

    `global` costs, structurally: the mass column in the fold (`v := 1`, one extra
    column of `DV + 1`), the scalar denominator matvec (a `[BT x BC] x [BC x 1]`
    reduction deliberately NOT lowered to an MMA -- an `n8` tile would waste 7/8 of
    a fragment), one extra element in each `s0` sweep under decay, the denominator
    atomics, and the `v3_divide` epilogue.
    """
    cell = CELLS[name]
    fx = cell.build()
    return _paired_one_kernel(
        cell, fx, ("raw", decay_temp), ("global", decay_temp), reps,
        "NORMALIZATION PRICE (new kernel both arms), global vs raw")


# ---------------------------------------------------------------------------
# REALIZED STATISTICS -- the manifest, and the read-only share
# ---------------------------------------------------------------------------

def skip_share(cell: Cell, fx=None):
    """Fraction of candidate `(owner, tile)` pairs that are READ-ONLY.

    On those the fold GEMM is skipped entirely, and with it (on the decay arm) both
    survival panels and the clock prefix -- all of them are under the same
    block-uniform `is_write` schedule bit. This is the measurement that decides
    whether a fixture actually EXERCISES the read-only exemption, and it is run
    before the fixture is trusted rather than after: a first attempt scored 0.0 %
    and would have made those gate cells vacuous.
    """
    fx = cell.build() if fx is None else fx
    plan, sch = cell.plan(norm="global", decay_on=True, prec="bf16",
                          fx=fx)
    if sch is None:
        sch = build_v3_schedule(fx["read_f32"], fx["write_f32"],
                                None, fx["g_write_f32"], plan)
    geo = owner_geometry(tuple(cell.widths), cell.BC)
    j, nsl, nw = geo.first_spanned_level, geo.n_slices, sch.n_words
    #: `plan.BT`, NOT `cell.BT`. FIXED 2026-08-01 (item 6b), and it was a real defect
    #: rather than a tidy-up: the bitmaps below belong to `sch`, which is built at
    #: `plan.BT`, while the tile count was taken from the cell's NOMINAL `BT`.
    #: Whenever the planner selected the other tile size the two disagreed and
    #: `unpack_bits` read a tile axis the words do not have. It was reachable and it
    #: was reached: the committed `atscale-L2048-BC16` entry (749436 candidate pairs)
    #: was produced by a `BT = 32` schedule counted at `nt = 2048/16 = 128`. At
    #: `plan.BT == cell.BT` the two forms are identical, which is why every other
    #: cell's entry is unchanged.
    nt = (cell.T + plan.BT - 1) // plan.BT
    ru = sch.read_slice_union.view(-1, nsl, nw)
    wu = sch.write_slice_union.view(-1, nsl, nw)
    owners = sch.owners.cpu()
    cand = ronly = 0
    for k in range(owners.shape[0]):
        bh, ow = int(owners[k, 0]), int(owners[k, 1])
        digits, _ = geo.owner_digits(ow)
        r, w = ru[bh, ow % nsl].clone(), wu[bh, ow % nsl].clone()
        for l in range(j):
            row = sch.level_offsets[l] + digits[l]
            r &= sch.read_tile_bits[bh, row]
            w &= sch.write_tile_bits[bh, row]
        rb, wb = unpack_bits(r, nt).bool(), unpack_bits(w, nt).bool()
        cand += int((rb | wb).sum())
        ronly += int((rb & ~wb).sum())
    return dict(n_owners=int(owners.shape[0]), candidate_pairs=cand,
                read_only_pairs=ronly, read_only_frac=ronly / max(cand, 1))


def cell_manifest(cell: Cell):
    fx = cell.build()
    st = tile_union_stats(fx["read_f32"], fx["write_f32"],
                          fx["topology"], cell.BC, cell.BT)
    out = dict(name=cell.name, widths=list(cell.widths), BC=cell.BC, D=cell.D,
               T=cell.T, H=cell.H, seed=cell.seed,
               mode="producer" if cell.is_producer else "synthetic",
               route_w_mul=cell.producer_route_w_mul,
               read=list(cell.read), write=list(cell.write),
               mean_K=round(st.mean_K, 6),
               mean_K_candidate=round(st.mean_K_candidate, 6),
               candidate_frac=round(st.candidate_frac, 6))
    out.update({k: (round(v, 6) if isinstance(v, float) else v)
                for k, v in skip_share(cell, fx).items()})
    return out


def build_manifest():
    return {c.name: cell_manifest(c) for c in CELLS.values()}


def check_manifest():
    """A cell that has DRIFTED is a cell whose numbers are no longer comparable to
    the ones already published under its name. That is exactly the failure this
    file exists to prevent, so it is a hard check rather than a report."""
    if not MANIFEST.exists():
        raise SystemExit(f"no committed manifest at {MANIFEST}; run --manifest --write")
    want = json.loads(MANIFEST.read_text())
    got = build_manifest()
    bad = []
    for name in sorted(set(want) | set(got)):
        if want.get(name) != got.get(name):
            bad.append(name)
            print(f"  DRIFT {name}\n    committed {want.get(name)}\n    realized  {got.get(name)}")
    print(f"{len(got) - len(bad)}/{len(got)} cells match the committed manifest")
    return not bad


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="oracle",
                    choices=["oracle", "skipshare", "manifest",
                             "decayprice", "normprice", "split", "p1identity",
                             "compactgate", "compactprice", "gatherstats",
                             "armtime", "btab"])
    ap.add_argument("--cell", default="atscale-L2048")
    ap.add_argument("--norm", default="global", choices=["global", "raw"])
    ap.add_argument("--decay", default=None, choices=[None, "warm", "hot"])
    ap.add_argument("--no-decay-axis", action="store_true",
                    help="oracle mode: skip the decay arms")
    ap.add_argument("--hot", action="store_true")
    ap.add_argument("--reps", type=int, default=11)
    ap.add_argument("--write", action="store_true", help="manifest mode: write it")
    ap.add_argument("--compact", action="store_true",
                    help="oracle/split modes: drive the COMPACT template axis true")
    ap.add_argument("--bt", type=int, default=None, choices=[16, 32],
                    help="armtime mode: RESTRICT the planner's BT candidates to this "
                         "one value, so the arm is a plan the planner could have "
                         "emitted at that tile size (the schedule is rebuilt with it)")
    a = ap.parse_args()

    if a.mode == "oracle":
        raise SystemExit(0 if run_oracle(decay=not a.no_decay_axis, hot=a.hot,
                                         compact=a.compact) else 1)
    if a.mode == "btab":
        run_btab(a.cell, reps=a.reps, norm=a.norm, decay_temp=a.decay,
                 compact=a.compact)
        return
    if a.mode == "armtime":
        arm_time(a.cell, reps=a.reps, norm=a.norm, decay_temp=a.decay,
                 compact=a.compact, bt=a.bt)
        return
    if a.mode == "split":
        raise SystemExit(0 if run_split_gate(compact=a.compact) else 1)
    if a.mode == "compactgate":
        raise SystemExit(0 if run_compact_identity() else 1)
    if a.mode == "compactprice":
        #: RULING 2's cells: the TRAINED-SPARSE at-scale cells (where the density is
        #: `mean_K_candidate ~ 1` of 64) AND the DENSE cells (where compaction can
        #: only cost), at both legal `BC`. The disposition is whatever these say.
        for name in ("atscale-L512", "atscale-L2048", "atscale-syn4-L2048",
                     "topo-D2-w64x64-BC64", "topo-D2-w64x16-BC16",
                     "wk1-D2-w64x64-BC64"):
            compact_price(name, reps=a.reps, norm=a.norm, decay_temp=a.decay)
        return
    if a.mode == "gatherstats":
        for name in (GROUPS["topo"] + GROUPS["wk1"] + GROUPS["atscale"]):
            st = gather_stats(CELLS[name])
            print(f"  {name:26s} BC={CELLS[name].BC:3d}  mean K {st['mean_K']:7.3f}"
                  f"  mean KI {st['mean_KI']:7.3f}  k-steps {st['mean_ksteps']:.3f}"
                  f"  m-tiles {st['mean_mtiles']:.3f}"
                  f"  ({st['owners_sampled']} owners, {st['candidate_pairs']} pairs)")
        return
    if a.mode == "p1identity":
        raise SystemExit(0 if run_p1_identity() else 1)
    if a.mode == "decayprice":
        #: Three cells BRACKETING the write-live share, because the structural
        #: prediction is that decay's price scales with it: a dense-write cell, the
        #: sparse-routing at-scale cell, and a write-`k=1` cell whose read-only
        #: share is 77.9 %.
        for name in ("topo-D2-w64x64-BC64", "atscale-L2048", "wk1-D2-w64x64-BC64"):
            decay_price(name, reps=a.reps)
        return
    if a.mode == "normprice":
        for name in ("topo-D2-w64x64-BC64", "atscale-L2048", "wk1-D2-w64x64-BC64"):
            norm_price(name, reps=a.reps, decay_temp=a.decay)
        return
    if a.mode == "skipshare":
        for name in (GROUPS["topo"] + GROUPS["wk1"]):
            s = skip_share(CELLS[name])
            print(f"  {name:26s} owners {s['n_owners']:5d}  candidate {s['candidate_pairs']:7d}"
                  f"  read-only (FOLD SKIPPED) {s['read_only_pairs']:7d}"
                  f" = {100 * s['read_only_frac']:5.1f} %")
        return
    if a.write:
        MANIFEST.write_text(json.dumps(build_manifest(), indent=2) + "\n")
        print(f"wrote {MANIFEST}")
        return
    raise SystemExit(0 if check_manifest() else 1)


if __name__ == "__main__":
    main()
