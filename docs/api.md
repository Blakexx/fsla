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
y, _ = layer(hidden_states)                     # stateless: (y, None)
```

or, without the layer:

```python
factors = routes(hidden_states)          # the (q, k) pair, after the feature map
y, _ = rola.rola_op(factors, v, decay())
```

---

## 1. The op

```python
rola_op(routes: RouteFactors, v: Tensor, decay: LeafMassDecay | None = None, *,
        state=None, expert=None) -> (y, state)
```

`routes` is the `(q, k)` pair after the feature map. `v` is `[B, T, H, d_v]`. `decay` is
the recurrence dial; a decay-configured call runs on the fp64 reference arm — decay is
not a feature of the CUDA consumer (the measured mass-clock price fired the stop-gate;
docs/internals/chunk/chunk.md records the retirement and its reopen conditions). `y` is `[B, T, H, d_v]`.

`state` is `None` (a stateless call, returning `(y, None)`) or a `rola.state()` object,
which is mutated in place and handed back. THE PREDICATE IS `state is not None` — there
is no second flag, because a stateless call runs a kernel variant with no state I/O
compiled into it at all. Read the carried state with
`state.materialize() -> [B, H, N, d_v + (norm == 'global')]`, where `N = prod(widths)` is
the leaf capacity; that is a bridge for the oracle and for a test, never a measured
path. See [`docs/internals/state.md`](internals/state.md).

There is no plan argument, no configuration argument, and no statistics argument. Those
are not model facts — they are how the op RUNS, and the op decides how it runs, per
call, from the routing it was actually given. They live behind
[`rola.expert`](#6-the-expert-surface).

### 1.1 The kernel envelope

The kernel arm is the chunk consumer, and the chunk consumer is a BUILT MATRIX. A
configuration outside it dispatches to the fp64 reference arm — a labelled research
surface, not a fallback: nothing failed, the kernel was never asked, and
`rola.expert.dispatch_for(routes, *operands).no_kernel_arm` names the clause that
refused, before the call and in the refusal's own words.

| axis | the built matrix | outside it |
|---|---|---|
| `d_v` | `64` | reference arm |
| `normalization` | `"global"` | reference arm (`raw`, per-state) |
| widths | uniform (`len(set(widths)) == 1`) | reference arm (a jagged topology) |
| `(D, B)` | the manifest's combinations (`rola.ops.chunk`'s `chunk_arms()`) | reference arm |
| `T` | a multiple of the chunk `C = 32` | reference arm |
| `decay` | not implemented at all | reference arm |
| an unratified activation | — | reference arm |

**This is a real capability narrowing, and it is stated rather than softened.** Shapes
outside the matrix — including the flagship `widths = (8, 16)`, which is refused for
being non-uniform — had a kernel before the tiled consumer's retirement and now have
the reference arm's throughput, which is the textbook recurrence in fp64 and orders of
magnitude slower. The card that widens the matrix is in
[`docs/open-work.md`](open-work.md).

A caller who overrules the dispatch gets a refusal rather than the oracle:
`rola_op(..., _arm="kernel")` on a configuration with no built arm raises by name.
Dispatch is not a lie; an override is.

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

`RouteProducer` is how a MODEL obtains a `RouteFactors`: it runs each producer once,
places its levels in the slots the list assigned, folds each level into
`(mass, simplex)`, and hands the mass to the side gain. Construction validates the
whole assembly — widths in range, a buildable topology, a gain layout consistent with
the normalization arm — so an unbuildable model fails on the line that built it.

**The bundle is a record of tensors, and it is constructible.** Its fields are the
stored simplices, the side gains and the topology they are shaped by — nothing else,
and in particular no declaration a caller could pair with tensors that do not describe
it. `RouteFactors.__post_init__` therefore checks the STRUCTURE rather than the
provenance: one level per topology level at that level's width, one shared `[B, T, H]`
prefix, one device, one dtype, `g_read` present iff the arm is `raw`, `g_write` shaped
`[B, T, H]`. A test that plants a specific routing pattern builds one directly and is
held to exactly those invariants. Unit-sum-ness is deliberately not among them: it is a
device reduction per level per call, it is the gateway's own output guarantee, and
conformance fixtures plant non-canonical amplitudes on purpose. The op still type-gates
on the class, so handing it bare tensors is still a clean refusal.

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
| `level_routing` | no | per-level routing tag, so the topology can answer the activation questions the dispatch asks |

`source` is a `RouteSource` carrying the `[B, T, hidden_size]` read and write streams
(the same object when the model does not split them). `ProducedRoutes` carries one
`[B, T, H, width_l]` tensor per level per side.

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
                bias=False)
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
all. Support bits are computed FROM the tensors, so a foreign producer changes nothing
about correctness. Its only cost is inside its own module.

A level filled by a producer that does not name one of our activations is *opaque*: it
reads the `("opaque", None)` registry row, whose columns are argued in
`rola/routing/activations.py`. What that costs it is the activation questions the
topology can answer about it — ratification, in particular — and nothing else.

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

`ConstantDecay` set high is a legitimate performance lever: the state forgets fast and
the realized routing shows it, through exactly the same door a learned source uses.
Nothing about it is a special case. (Today every decay-configured call runs on the
reference arm — §1.1 — so the lever is a modelling one until the kernel-side decay
retirement is reopened.)

The decay is **write-conditioned** (mass decay), never time decay. A leaf whose write
amplitude is exactly zero ticks its clock by zero, so `keep = (1 - rate) ** 0 = 1`
exactly and its state is left bit-unchanged. Several decisions elsewhere in the system
rest on that property.

## 4. The layer

```python
RoLA(hidden_size, num_heads, routes, *, decay=None,
     value_conv=None, route_conv=None, layer_idx=None, expert=None)

y, state = layer(hidden_states, state=rola.state())    # or state=None -> (y, None)
```

`forward(x, attention_mask=None, state=None) -> (y, state)` MIRRORS the op, and the
layer is a PASS-THROUGH for the state: it unwraps nothing and checks nothing, because
state semantics are the state's.

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

## 5. The expert surface

`rola.expert` holds everything BELOW the contract line — the dispatch's own answer,
the pins over axes the op otherwise decides, and the fine-grained gain layout behind
the layer's one public flag:

```python
from rola.expert import PlanOverrides, dispatch_for
layer = rola.RoLA(1024, 8, routes, expert=PlanOverrides())
```

| name | what it is |
|---|---|
| `PlanOverrides` | the pins: `state_block`, `mma_precision`, `paging` |
| `Dispatch` / `dispatch_for` | how a call WILL run, asked before making it — including `no_kernel_arm`, the refusing clause |
| `select_backend`, `requires_backward` | the two predicates the dispatch is built from |
| `GainConfig` | the two-boolean gain layout behind the public `gain` flag |

The contract for everything there is **stable construction, advisory contents**:
build these objects and pass them; do not assert on their fields. What a `Dispatch`
reports is the answer to "how will this call run", and that answer moves with the
machine and with the built matrix. The model facts it carries are the exception,
because they are not decisions.

**Two of the three pins are declared restrictions rather than dials, and each
refuses by name rather than being ignored.** `state_block` (`BC`) is not
host-selectable: the chunk arm's `BC` belongs to the built arm the manifest carries
for a topology, so any value but `None` is refused at the call — a knob that
silently does nothing is how a caller comes to believe a launch was shaped a way it
never was. `mma_precision` has one legal value, `"bf16"`; the precision escape is
the fp64 reference path. `paging` overrides the MAPPING only: `None` is the paged
default, `False` pins the dense plane the bit-identity gate compares against, and
`True` is refused because paging is on by default and bitwise invisible.

**There is no plan object and no derive/execute seam.** `Plan`, `Schedule`,
`derive_plan` and `execute` were the tiled consumer's, and went with it in P67 D2
([`internals/DELETIONS.md`](internals/DELETIONS.md)). The geometry they carried is
now a property of the built arm rather than of a per-launch derivation, so there is
no launch-shaping object for a caller to hold, synthesize or staleness-test. What
replaces the adversarial use they enabled: a conformance family drives
`rola.ops.chunk` directly and declares the refusal clause it expects, and
`dispatch_for` answers the same question a synthetic plan used to be built to probe.

## 6. What is deliberately not here

- **No dict or JSON configuration surface.** Raw configuration belongs to consumers. The
  library type-checks structured objects and parses nothing.
- **No pluggable recurrence.** RoLA *is* the feature map, not a framework for
  recurrences. The state update is not an extension point and must never be presented as
  one.
- **No silent second path.** Every arm is chosen from a property of the configuration or
  of the operands, before the call, and named in the exception when a configuration
  cannot run. Nothing degrades into anything after a failure.
