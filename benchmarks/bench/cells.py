"""WHAT is measured: the frozen cell registry and its committed manifest.

**A CELL IS ITS CONSTRUCTOR AND ITS SEED.** Split out of
``benchmarks/bench_consumer_new.py`` unchanged, because the rule that file exists
to enforce is a property of THIS module: every gate and bench cell is a frozen
:class:`Cell` -- widths, ``BC``, shapes, support construction, decay dials, seed --
and ``Cell.build()`` is deterministic from it. Documents reference cells BY NAME
(``atscale-L2048``, ``wk1-D2-w64x64-BC64``), never by a description of their
density; two numbers carrying the same cell name are commensurable and two carrying
different names are not.

It already went wrong twice in this repository's history -- ``benchmark_rola_v3_consumer.py``'s
review finding F6, and the at-scale cell of ``rewrite.md`` §1.6-§2.8 that was
described only by its plan statistics (``mean_K = 0.1908``, ``mean_K_candidate =
0.9665``, ``candidate_frac = 0.1975``) and is now permanently non-reproducible,
because sweeping the construction parameter that was supposed to produce it traces
a curve that does not pass through that triple at all. **A statistic is not a
fixture.**

``check_manifest`` is the standing gate: a change in torch's RNG, in the entmax
solver, or in the planner announces itself as a manifest DIFF rather than as a
silently different benchmark.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from rola.planner.schedule import Plan, build_schedule, owner_geometry, tile_union_stats, unpack_bits
from rola.routing.entmax.reference import entmax
from rola.routing.types import EntmaxActivation, IndependentRouting, LevelConfig, SoftmaxActivation, Topology

#: The committed statistics of every cell. Sibling of this package, not of the
#: driver, because the manifest describes the CELLS and outlives any one driver.
MANIFEST = Path(__file__).resolve().parents[1] / "cells" / "topology_manifest.json"



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

    **This is load-bearing, not tidiness.** `DecayStream`'s validity key is
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
      :class:`~rola.layers.RoLA` is constructed and `produce_routing` is run
      on it, so the supports are the ones the MODEL emits. The plan is then chosen
      by `plan_and_schedule` -- the PLANNER, not a hand-built `Plan` -- which
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
        from rola.routing.producer import produce_routing
        layer = self._layer()
        hidden = self.H * self.d_v
        x = torch.randn(self.B, self.T, hidden, device="cuda", dtype=torch.float32)
        with torch.no_grad():
            v = layer.v_proj(x).view(self.B, self.T, self.H, self.d_v)
            ro = produce_routing(
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
        #: `produce_routing(..., "global")` returns `g_read = None`, because spec
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
        def shp(w):
            return (self.B, self.T, self.H, w)

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
            import dataclasses as _dc

            from rola.planner.schedule import plan_and_schedule
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
            plan, _axes, sched = plan_and_schedule(
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
        return Plan(topology=fx["topology"] if fx else self.topology(),
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
    #: `plan_and_schedule`. This is byte-for-byte the constructor the pre-loss
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
        sch = build_schedule(fx["read_f32"], fx["write_f32"],
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

