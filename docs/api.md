# The API

RoLA is linear attention under a structured feature map. That is not a slogan about the
paper; it is the specification of this API. A head's recurrent state is expanded into
many sub-states that share the head's projections, and a learned routing decides which
sub-states each token reads and writes — so the public contract is *linear attention's*
contract, with exactly **one** new concept: the feature map.

A user who knows linear attention should find one unfamiliar thing on this page.

```python
import rola

producer = rola.SimplexProducer((64, 64), rola.sparse_routing(1.5),
                                hidden_size=1024, num_heads=8)
routes   = rola.RouteProducer([producer], d_v=64, num_heads=8, hidden_size=1024)
decay    = rola.LearnedDecay(2 ** -8, widths=routes.widths, num_heads=8)

layer = rola.RoLA(1024, 8, routes, decay=decay, layer_idx=0).cuda().eval()
y, _, cache = layer(hidden_states)
```

or, without the layer:

```python
factors = routes(hidden_states)          # the (q, k) pair, after the feature map
y, final_state = rola.rola_op(factors, v, decay())
```

---

## 1. The op

```python
rola_op(routes: RouteFactors, v: Tensor, decay: LeafMassDecay | None = None, *,
        initial_state=None, output_final_state=False, expert=None) -> (y, final_state)
```

`routes` is the `(q, k)` pair after the feature map. `v` is `[B, T, H, d_v]`. `decay` is
the recurrence dial; a decay-configured call runs on the fp64 reference arm — decay is
not a feature of the CUDA consumer (the measured mass-clock price fired the stop-gate;
docs/internals/chunk/chunk.md records the retirement and its reopen conditions). `y` is `[B, T, H, d_v]`; `final_state` is
`[B, H, N, d_v + (norm == 'global')]` when asked for, where `N = prod(widths)` is the
leaf capacity.

There is no plan argument, no configuration argument, and no statistics argument. Those
are not model facts — they are how the op RUNS, and the op decides how it runs, per
call, from the routing it was actually given. They live behind
[`rola.expert`](#6-the-expert-surface).

`rola_op` is the **single autograd boundary**. A call that wants a gradient runs the
whole operation on the differentiable reference, forward and backward from one graph; a
call that does not reaches the CUDA consumer. It is one decision in one place because
the failure it prevents — a forward-only kernel reached from a gradient context returns
a tensor with no `grad_fn` and severs the graph silently — is structural, not a bug in
any one call site.

## 2. The feature map

### 2.1 The gateway

```python
RouteProducer(producers, *, d_v, num_heads, hidden_size, normalization='global',
              gain=True, gain_bias_init=None, gain_config=None) -> RouteFactors
```

`RouteProducer` is the **only** way to obtain a `RouteFactors`, and that is the point.
Every field of the bundle — the stored simplices, the side gains, the support
statistics, the declared density — is computed by the gateway's own derivation code from
the tensors the producers returned. A caller therefore cannot hand the op a bundle whose
statistics disagree with its tensors, because a caller never authors the bundle: they
choose producers, and producers return tensors.

That is what makes this surface trustless rather than merely documented. Correctness is
derived, never declared. Construction validates the whole assembly — widths in range, a
buildable topology, a gain layout consistent with the normalization arm — so an
unbuildable model fails on the line that built it, and the op type-gates on the result,
so no well-formed bundle means no call.

The gateway also owns the topology (`routes.topology`, `routes.widths`), the value width
`d_v`, and the normalization arm, because all three are facts about the map rather than
about the layer wrapping it.

### 2.2 The producer interface

A producer fills one or more routing levels. The interface is duck-typed — the library
never asks what a producer is, only that its output is well formed:

| member | required | meaning |
|---|---|---|
| `widths` | yes | one branch width per level it covers, in its own order |
| `forward(source) -> ProducedRoutes` | yes | ONE call producing every level it covers |
| `dense_sides` | no | per-level `(read, write)` structural declaration |
| `level_routing` | no | per-level routing tag, for the pricing columns |

`source` is a `RouteSource` carrying the `[B, T, hidden_size]` read and write streams
(the same object when the model does not split them). `ProducedRoutes` carries one
`[B, T, H, width_l]` tensor per level per side, and optionally any statistic the
producer already knew.

Levels need not be unit-sum: the gateway factors each into `(mass, simplex)` and folds
the mass into the side gain, which is where a per-level magnitude belongs.

**Batching is a producer's business.** The gateway calls each producer once, with every
level that producer covers. A producer covering several levels is therefore how a model
gets one launch across them — `SimplexProducer` takes a width tuple and runs the whole
group through one packed projection and one batched solve, which is what every shipped
configuration does. Batching across two *different* producers is not built: the prize is
launch overhead at `D <= 4` levels, nobody has measured it to matter, and the machinery
(a merge protocol, a scatter, a bit-identity gate) is real. It is a revisit-if-measured
decision, not an oversight.

**The gateway owns packing.** A producer returns the plain documented layout and knows
nothing about how the kernel wants its tensors arranged. The solve emits `[BH, T, width]`,
which is already what the kernel's view wants, and the public layout is a permute of it —
so exactly one of the two can be the materialized tensor, and it is the kernel's. The
public tensors are views. Materializing the public layout instead and folding it back is
an inverse copy pair. It was priced at 2.82 % of the whole step at `L = 4096`
(`workflows/refactor/p12_lane3.md`), and the same harness measures it directly against
the production call it rides: 3.085 ms vs 2.732 ms at `L = 4096` and 0.942 ms vs
0.835 ms at `L = 1024` — 12.9 % and 12.7 % of production-plus-fold, at both lengths.

### 2.3 The shipped producer

```python
SimplexProducer(widths, routing=dense_routing(), *, hidden_size, num_heads,
                bias=False, expose_stats=False)
```

A learned packed per-head router projection, solved with the exact production
entmax/softmax closed form. `routing` is the tag vocabulary directly:
`dense_routing()` (softmax on both duties), `sparse_routing(alpha)` (one exact entmax
solve determining a shared support that the two duties split), or any
`IndependentRouting` / `UnionRouting` you build yourself. One tag broadcasts across the
covered levels; a sequence gives per-level control.

Activation *properties* are never supplied by a caller — they come from the closed
registry in `rola/routing/activations.py`, keyed on the tag. Metadata a caller could
supply is metadata a caller could supply wrong, and a validator reading it would be
checking a claim instead of a fact.

### 2.4 Foreign producers

Any module satisfying the interface works, with no registration and no tier. This is
sound because no producer-specific execution path exists: the consumer eats tensors and
cannot tell what produced them — no producer identity reaches the kernel's parameters at
all. Support bits and summaries are computed FROM the tensors, so a foreign producer
changes nothing about correctness. Its only cost is inside its own module.

A level filled by a producer that does not name one of our activations is *opaque*: it
reads the `("opaque", None)` registry row, whose columns are argued in
`rola/routing/activations.py`. In practice that means its density is MEASURED rather
than substituted from a closed form — bit for bit the same result, one summary slice
more work — unless the producer declares `dense_sides`.

### 2.5 The one modelling consequence to know

The decay clock is defined on the STORED (normalized) level. A producer whose write-side
mass differs from 1 has therefore moved magnitude out of the clock and into `g_write`.
That is the defined semantics rather than a defect, but it is a choice a producer author
is making, so it is stated here rather than buried.

## 3. The decay-source slot

Where the recurrence's forgetting rates come from is a CHOICE, and the recurrence cannot
tell the answers apart:

```python
rola.ConstantDecay(rate, widths=..., num_heads=...)                    # user-fixed
rola.LearnedDecay(rate, widths=..., num_heads=..., scope='state')      # 'state' | 'global'
```

Both emit a `LeafMassDecay` whose `dials[l]` is `[H, width_l]` in `[0, 1)`. A source is
validated against the topology at layer construction. Duck-typed like a producer:
declare `widths` and `num_heads`, return a `LeafMassDecay`.

`state` and `global` are the two IDENTIFIABLE resolutions: under `keep = (1 - prod_l
delta_l)**c` (one write-conditioned clock per leaf), a rate only ever enters the
recurrence through that per-leaf product, so per-digit and per-head are the two degrees
of freedom the product structure lets the recurrence actually distinguish.

`ConstantDecay` set high is a legitimate performance lever: the state forgets fast, the
planner OBSERVES that in the realized routing, and prices accordingly — through exactly
the same door a learned source uses. Nothing about it is a special case.

The decay is **write-conditioned** (mass decay), never time decay. A leaf whose write
amplitude is exactly zero ticks its clock by zero, so `keep = (1 - rate) ** 0 = 1`
exactly and its state is left bit-unchanged. Several decisions elsewhere in the system
rest on that property.

## 4. The layer

```python
RoLA(hidden_size, num_heads, routes, *, decay=None,
     value_conv=None, route_conv=None, layer_idx=None, expert=None)
```

Four public arguments and two module slots. `value_conv` / `route_conv` take a MODULE
(e.g. `fla.modules.ShortConvolution`) rather than three scalars describing one: the layer
does not need to know how a convolution is configured, only where it goes.

The layer adds exactly one thing the op cannot know: `self.training` gates the CUDA path
independently of gradient reachability, because reentrant gradient checkpointing runs its
first pass under `no_grad()` with the module still in training mode. Without that
condition the value flowing downstream and the gradient taken would come from two
implementations agreeing only to a tolerance.

`layer.execution_backend_counts` reports which of `oracle` / `cuda` / `decode` ran, which
is a fact about the branch rather than an inference from a timing.

## 5. Statistics

The auxiliary statistics — exact per-token support counts, end-of-sequence touched
bits — are derived ON DEMAND:

```python
factors.support_counts()   # {(level, side): [B, T, H]}
factors.touched_bits()     # {level: [B, H, width_l]}
```

Nothing in the library reads them; they exist for diagnostics only. Producing them
eagerly spent launches per level per step on numbers no consumer had asked for —
measured at 3.0 % of the step at `L = 4096`.

A producer that already knows a statistic as a byproduct of its own solve may supply it
(`expose_stats=True` on the shipped producer), and the gateway then measures only what is
left. **Supplied statistics feed pricing only.** Membership and support bits are always
recomputed from the tensors, so a wrong declaration costs its author performance and can
never cost anyone correctness.

Which arm is faster is a measurement. `benchmarks/bench_api_stats.py` is the harness;
median per-call duration over 21 rounds, CUDA events, on `B=2, hidden=1024, H=8,
widths=(64,64)`, union entmax at α=1.5, raw samples in the performance ledger (rows
recorded at `1338854` on a then-dirty tree — immaterial to the claims, which are
relative comparisons among five arms sharing one `run_id` on one tree, as the rows
themselves record):

| arm | `L = 4096` | `L = 1024` |
|---|---|---|
| `stats_unused` — production alone | 2.599 ms | 0.794 ms |
| `stats_gateway` — derived from the stored tensors | 3.221 ms (+23.9 %) | 1.008 ms (+27.0 %) |
| `stats_fused` — supplied by the producer | 3.277 ms (+26.1 %) | 1.041 ms (+31.2 %) |

**The fused arm loses, at both lengths, and the reason is instructive.** "Fused" at this
seam means the producer computes the counts on its own solve output — still a separate
reduction, and one that then has to be laid out in the public statistic layout, which is
a transpose the gateway's arm gets for free from tensors it is already reading. A fused
arm that WON would have to fall out of the solve kernel itself, where the support is
already known; the entmax kernel emits a support bitmap, so that door is open, but it is
a kernel change and not an API one. What the measurement actually settles is the third
row: the statistics cost roughly a quarter of the production call, and nothing in the
library reads them, which is why they are derived on demand and not before.

## 6. The expert surface

`rola.expert` holds everything BELOW the contract line — plans, schedules, the
derivation that fixes their geometry, the seam the op is split at, the allocator cap,
the numeric arm:

```python
from rola.expert import PlanOverrides
layer = rola.RoLA(1024, 8, routes, expert=PlanOverrides())
```

The contract for everything there is **stable construction, advisory contents**: build
these objects and pass them; do not assert on their fields. A plan's geometry moves with
every recalibration by design — it is the answer to "how should this launch run", and
that answer is allowed to change. The model facts a plan carries (`topology`,
`normalization`) are the exception, because they are not decisions.

The op is split there too: `_derive_plan(routes) -> Plan` chooses the launch geometry
from resident facts, and `_execute(plan, routes, v)` builds the schedule from the
realized masks and runs the kernel. Exported as `derive_plan` / `execute`, they are what
lets a conformance test drive execution with a SYNTHETIC plan — a stale one, one
straddling a boundary — which a pipeline that derived and immediately used its plan
could not be adversarially tested against.

Selection is per step and never cached per shape: the statistics it prices are exact over
the REALIZED routing, so a cached plan prices a support distribution that no longer
exists the moment training moves it.

## 7. What is deliberately not here

- **No dict or JSON configuration surface.** Raw configuration belongs to consumers. The
  library type-checks structured objects and parses nothing.
- **No pluggable recurrence.** RoLA *is* the feature map, not a framework for
  recurrences. The state update is not an extension point and must never be presented as
  one.
- **No silent second path.** Every arm is chosen from a property of the configuration or
  of the operands, before the call, and named in the exception when a configuration
  cannot run. Nothing degrades into anything after a failure.
