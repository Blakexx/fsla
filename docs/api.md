# The API

RoLA is linear attention under a structured feature map. That is not a slogan about the
paper; it is the specification of this API. A head's recurrent state is expanded into
many sub-states that share the head's projections, and a learned routing decides which
sub-states each token reads and writes — so the public contract is *linear attention's*
contract, with exactly **one** new concept: the feature map.

A user who knows linear attention should find one unfamiliar thing on this page.

```python
import rola

routes   = rola.RouteProducer(rola.uniform(2, rola.union_routing(64, alpha=1.5)),
                              hidden_size=1024, num_heads=8)
decay    = rola.LearnedDecay(2 ** -8, widths=routes.widths, num_heads=8)

layer = rola.RoLA(routes, d_v=64, decay=decay, layer_idx=0).cuda().eval()
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
the recurrence dial. `y` is `[B, T, H, d_v]`.

`state` is `None` (a stateless call, returning `(y, None)`) or a `rola.state()` object,
which is mutated in place and handed back. THE PREDICATE IS `state is not None` — there
is no second flag, because a stateless call runs a kernel variant with no state I/O
compiled into it at all. Read the carried state with
`state.materialize() -> [B, H, N, d_v + 1]`, where `N = prod(widths)` is the leaf
capacity; that is a bridge for the oracle and for a test, never a measured path. See
[`docs/internals/state.md`](internals/state.md).

There is no plan argument, no configuration argument, and no statistics argument. Those
are not model facts — they are how the op RUNS, and the op decides how it runs, per
call, from the routing it was actually given. They live behind
[`rola.expert`](#6-the-expert-surface).

### 1.1 The built envelope IS the surface

The arm is the chunk consumer, and the chunk consumer is a BUILT MATRIX. **Everything
outside it raises.** There is no second arm behind this surface: the fp64 oracle
(`rola.ops.naive.naive_rola`) is the executable spec — what tests and ratification
call directly — and it is not reachable by dispatch, because a dispatch to it at
inference is a silent ~1000x-slower route shipping unnoticed, which is the exact thing
the no-fallback rule forbids.

| axis | the built matrix | outside it |
|---|---|---|
| a gradient anywhere in the operands | none | `BackwardNotImplemented` |
| the module's mode (layer only) | `eval()` | `BackwardNotImplemented` |
| device | CUDA | refused |
| `d_v` | `64` | refused |
| `normalization` | `"global"` | refused |
| widths | uniform (`len(set(widths)) == 1`) | refused (a jagged topology) |
| `(D, B)` | the manifest's combinations (`rola.ops.chunk`'s `chunk_arms()`) | refused |
| `T` | a multiple of the chunk `C = 32` | refused |
| `decay` | not implemented in the chunk arm (decode has it) | refused |
| an unratified activation | — | refused |

Ask before calling: `rola.expert.envelope_refusal(routes, v, decay)` returns the
sentence the refusal will carry, or `None`. `rola.expert.require_envelope(...)` is the
raising form the op itself uses.

**This is a real capability narrowing, and it is stated rather than softened.** Shapes
outside the matrix — including the flagship `widths = (8, 16)`, which is refused for
being non-uniform — had a kernel before the tiled consumer's retirement, and there is
NO TRAINING PATH through this surface at all until the native backward lands. The cards
that widen the matrix and bring the backward are in
[`docs/open-work.md`](open-work.md).

`rola_op` is the **single autograd boundary**, and it is a refusal rather than a
branch: the failure it prevents — a forward-only kernel reached from a gradient
context returns a tensor with no `grad_fn` and severs the graph silently — is
structural, not a bug in any one call site. `ChunkConsumer`'s backward is the last
line, loud if the first is ever bypassed.

### 1.2 Continuation is LAYERED

The op's state is the paged tensor and nothing else
([`docs/internals/state.md`](internals/state.md)). Each level of the stack returns its
OWN thin bundle, and the rule is a membership test:

> **An object holds exactly the recurrences ITS level introduced.**
> Op: the routed state. Layer: the short-convolution rings. Model: the per-layer
> bundles.

So `RoLA.forward` returns `(y, (state, conv_rings))` — always the pair, `conv_rings`
`None` when no convolution is configured — and takes that same pair back. The pair is
the contract in both directions: a bare `RoLAState` is refused by name, because
accepting the op's half would make the layer's half silently optional to carry.

This is not the old pooled state object recreated. That object's defect was LAYER
LUGGAGE INSIDE THE OP's state; keeping each level's object pure is what leaves
accretion nowhere to land.

## 2. The feature map

### 2.1 The bundle IS the contract

```python
RouteFactors(topology, normalization, read, write, g_read, g_write, read_mass=None)
```

**The stored form is bf16.** The routing sides have one consumer, and it reads bf16, so
that is what the shipped producer's solve WRITES -- there is no fp32 plane between the
solve and the operator, and no narrowing pass. `read_simplex()` divides in at least
fp32 whatever the levels are stored in, because a bf16 quotient of a bf16 sum is three
roundings where one is forced. The bundle's own invariant is only that every level
shares ONE dtype, so a bundle in any dtype is legal; bf16 is what the shipped producer
emits and what every kernel-vs-oracle tolerance in the tree is derived against
(`tests/oracle/tolerances.py`).

`RouteFactors` is the op's first operand and the whole producer contract: **any callable
returning a well-formed one is a producer.** There is no producer base class, no
per-level tier and no assembly gateway — none of them was a boundary anything checked.
What is checked is the bundle's own invariants, and they are STRUCTURAL rather than
provenance-based.

**The bundle is a record of tensors, and it is constructible.** Its fields are the
stored levels, the side gains, the read mass and the topology they are shaped by —
nothing else, and in particular no declaration a caller could pair with tensors that do
not describe it. `RouteFactors.__post_init__` therefore checks the STRUCTURE rather than
the provenance: one level per topology level at that level's width, one shared
`[B, T, H]` prefix, one device, one dtype, `g_read` present iff the arm is `raw`,
`g_write` and (when present) `read_mass` shaped `[B, T, H]`. A test that plants a
specific routing pattern builds one directly and is held to exactly those invariants.
Unit-sum-ness is deliberately not among them: it is a device reduction per level per
call, it is the shipped producer's own output guarantee, and conformance fixtures plant
non-canonical amplitudes on purpose. The op still type-gates on the class, so handing it
bare tensors is still a clean refusal.

A producer also owns the topology (`routes.topology`, `routes.widths`) and the
normalization arm, because both are facts about the map rather than about the layer
wrapping it. It does NOT own `d_v`: the value width is the width of the state each leaf
holds, which is a fact about the value stream — the layer sizes `v_proj`/`o_proj` with
it and the op reads it off `v.shape[-1]` at call time.

**The side gains are routing's, and they ride the bundle.** Two things put them there
rather than on the layer. Their SHAPE is defined by the normalization arm — `g_read`
exists iff `normalization == 'raw'` — so a layer-owned gain producer would have to be
told the routing arm in order to know its own columns, which is the coupling running
backwards. And the gain is where the per-level mass product lands: the producer folds
each level's mass into it, so the gain and the routing mass are the same number by the
time any consumer sees either. They ride the bundle rather than arriving as a third op
argument because `RouteFactors.tensors()` is the autograd-reachability enumeration, and
an operand outside that flat list is invisible to autograd — the exact failure the
single autograd boundary exists to make impossible.

### 2.2 The mass fold, and what each duty does with it

Levels need not be unit-sum: each is factored into `(mass, simplex)`. The two sides then
take the mass differently, because the two duties use their levels differently. The
WRITE side APPLIES it — the stored level is the simplex and the magnitude folds into
`g_write`, which is where a per-level magnitude belongs, and the decay clock is defined
on the stored level. The READ side CARRIES it — the readout is a ratio, so a per-token
scale common to every leaf is homogeneous of degree zero through it, and
`RouteFactors.read` is stored with the mass still on it while `RouteFactors.read_mass`
holds the per-token product (`None` when there is none). `RouteFactors.read_simplex()`
divides it back out for every consumer defined on the simplex instead: the fp64 oracle
and its input contract, the decode arm, the reference recurrence.

**The read mass reaches the readout as a scaled floor.** The one term that does not
scale with a ratio is its floor, so it is handed over pre-scaled: the consumer computes
`m·num / (m·den + READOUT_EPS·m)`, which is `num / (den + READOUT_EPS)` exactly — the
same function, not a differently conditioned one. The alternative — dropping the mass
and leaving the floor alone — is an *effective* floor of `READOUT_EPS / m`, and the
union read mass runs as low as 7e-4 across the gain sweep, so it is a change to the
definition rather than to its conditioning. `READOUT_EPS` is a DEFINITION constant
(`rola/ops/constants.py`); the scaled floor is what keeps it one.

**The mass is only COMPUTED where it is not declared to be 1, and only there.** Each
`(level, duty)` cell answers `simplex_output` — "does this cell's own output already
land on the simplex?" — and where it answers yes, the mass is 1, so its `sum` is
omitted, it contributes nothing to the product, and the write side's `div` goes with it.
The answer is a declaration read off the activation registry
(`rola/routing/activations.py`) through the level's routing form, never a runtime test
of the tensor: a `sum(...) == 1` check is the very reduction the skip exists to delete,
and it would make the bundle's arithmetic depend on its data rather than on its
configuration. It is asked per *duty* and not per *activation*, because a routing form
may hand one duty a share of a solve rather than the whole of it — union routing's read
half is exactly that, so it answers `False` while the entmax it names answers `True`.
A FOREIGN producer folds its own mass, because it returns the finished bundle: the
opaque row's `simplex_output = False` is what keeps `normalized` from being read
circularly for a cell nobody vouched for, and it reaches the shipped fold only through
a topology a foreign producer declared. Measured across the shipped census (`D` in 1..4, fp32 and fp64), every cell answering
`True` has `max |sum − 1| ≤ 2.4e-7` in fp32 and `≤ 4.4e-16` in fp64.

**The producer owns packing.** The solve emits `[BH, T, width]`, which is already what
the kernel's view wants, and the public `[B, T, H, width_l]` layout is a permute of it —
so exactly one of the two can be the materialized tensor, and it is the kernel's. The
public tensors are views. Materializing the public layout instead and folding it back is
an inverse copy pair. It was priced at 2.82 % of the whole step at `L = 4096`
(`workflows/refactor/p12_lane3.md`), and the same harness measures it directly against
the production call it rides: 3.085 ms vs 2.732 ms at `L = 4096` and 0.942 ms vs
0.835 ms at `L = 1024` — 12.9 % and 12.7 % of production-plus-fold, at both lengths.

### 2.3 The shipped producer

```python
RouteProducer(levels, *, hidden_size, num_heads, normalization='global',
              bias=False, gain=True, gain_bias_init=None, gain_config=None)
```

A learned packed per-head router projection, solved with the exact production
entmax/softmax closed form, folded, gained, and returned as one bundle. Construction
validates the whole assembly — widths in range, a buildable topology, a gain layout
consistent with the normalization arm — so an unbuildable model fails on the line that
built it.

**`levels` is the configuration, and it is a LIST.** One entry per routing level,
outermost first, each carrying that level's WHOLE spec:

```python
IndependentRouting(width, read, write, tied=False)   # per-duty activations
UnionRouting(width, alpha=1.5)                       # one solve, support split
```

Depth, leaf count `N`, the packed router width, the per-level logit offsets and the
topology are all DERIVED from that list. None of them is a second input, so none of them
can disagree with it. The four spellings the shipped configurations use:

```python
dense_routing(width)                      # softmax on both duties
union_routing(width, alpha=1.5)           # one exact entmax solve, shared support
split_routing(width, alpha, sparse_duty)  # entmax one duty, softmax the other
uniform(depth, level)                     # `depth` levels spelled the same way
```

`uniform` is a list builder, not a broadcast: a producer takes a list and nothing else,
because a broadcast is a rule the reader has to know before they can read the widths off
the call.

**Every level is one launch group, not one launch each.** The levels share a packed
`route_W` and are dispatched per level inside the solve kernel with no privileged level,
so a model pays one GEMM and one solve group for the whole topology. Levels sharing an
operator, an alpha, a width class and an operand set share a single launch; levels that
do not are genuinely different `(operator, source, destination)` triples and issue
separately.

Activation *properties* are never supplied by a caller — they come from the closed
registry in `rola/routing/activations.py`, keyed on the tag. Metadata a caller could
supply is metadata a caller could supply wrong, and a validator reading it would be
checking a claim instead of a fact. Activations stay width-free: an activation is a
function on a row of any length, so a width on one would be a second place the same
number lives.

### 2.4 Foreign producers

Any callable returning a well-formed bundle works, with no registration and no tier.
This is sound because no producer-specific execution path exists: the consumer eats
tensors and cannot tell what produced them — no producer identity reaches the kernel's
parameters at all. Support bits are computed FROM the tensors, so a foreign producer
changes nothing about correctness. Its only cost is inside its own module. The layer's
`routes` slot checks STRUCTURE for the same reason: callable, plus the four facts the
layer reads (`hidden_size`, `num_heads`, `topology`, `normalization`).

A level filled by a producer that does not name one of our activations is *opaque*: it
declares `OpaqueRouting(width)` and reads the `("opaque", None)` registry row, whose
columns are argued in `rola/routing/activations.py`. What that costs it is the
activation questions the topology can answer about it — ratification, in particular —
and nothing else.

### 2.5 The one modelling consequence to know

The decay clock is defined on the STORED (normalized) level. A level whose write-side
mass differs from 1 has therefore moved magnitude out of the clock and into `g_write`.
That is the defined semantics rather than a defect, but it is a choice a level's author
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
Nothing about it is a special case. (Today the chunk arm has no decay and refuses it
— §1.1 — so the lever reaches production only through the decode arm, which does have
it, until the kernel-side decay retirement is reopened.)

The decay is **write-conditioned** (mass decay), never time decay. A leaf whose write
amplitude is exactly zero ticks its clock by zero, so `keep = (1 - rate) ** 0 = 1`
exactly and its state is left bit-unchanged. Several decisions elsewhere in the system
rest on that property.

## 4. The layer

```python
RoLA(routes, *, d_v=64, decay=None,
     value_conv=None, route_conv=None, layer_idx=None, expert=None)

y, (state, conv_rings) = layer(hidden_states, state=(rola.state(), None))
y, (state, conv_rings) = layer(next_tokens, state=(state, conv_rings))
y, _ = layer(hidden_states)                            # stateless -> (y, (None, None))
```

`forward(x, attention_mask=None, state=None) -> (y, (state, conv_rings))` — ALWAYS the
pair, `conv_rings` `None` when no convolution is configured. See §1.2; a bare
`RoLAState` is refused rather than adopted.

`routes` is the only positional argument, and it is the only declaration of
`hidden_size`/`num_heads`: the layer reads both off the producer rather than taking a
second copy to check against. `d_v` is the per-head value width — the layer's, because
it sizes `v_proj`/`o_proj`; the default is the width the chunk matrix is built at. `value_conv` / `route_conv`
take a MODULE
(e.g. `fla.modules.ShortConvolution`) rather than three scalars describing one: the layer
does not need to know how a convolution is configured, only where it goes.

The layer adds exactly one clause the op cannot know: `self.training` REFUSES,
independently of gradient reachability, because reentrant gradient checkpointing runs
its first pass under `no_grad()` with the module still in training mode — and a
forward that pass could take would be a forward no backward can follow.

`layer.execution_backend_counts` reports which of `oracle` / `cuda` / `decode` ran, which
is a fact about the branch rather than an inference from a timing.

### 4.1 The bf16 projection arm

On CUDA with the (only) `mma_precision="bf16"` pin, `v_proj` and `o_proj` run as bf16
GEMMs with weights cast per call, and `v` FLOWS bf16 straight into the op. Round-tripping
it through the module dtype was MEASURED to cost two elementwise casts per call — a `v`
downcast and a `y` upcast, ~12-13% of a small cell's op time — which is why the layer
narrows once, at the projection, rather than at the op boundary. The GEMMs use cuBLAS's
fp32 compute type (the accumulate-in-fp32 ruling); the global
reduced-precision-reduction knob is deliberately left alone, so this is the same
environment the FlashAttention comparison arm runs in.

A configured `value_conv` is the one structural exception: it runs in the module dtype,
so `v` returns to `x.dtype` before it and the op re-narrows at its own boundary —
exactly, the values being bf16-rounded already. `o_proj`'s operand cast is exact for the
same reason: `y` leaves the kernel already bf16-rounded.

## 5. The expert surface

`rola.expert` holds everything BELOW the contract line — the dispatch's own answer,
the pins over axes the op otherwise decides, and the fine-grained gain layout behind
the layer's one public flag:

```python
from rola.expert import PlanOverrides, envelope_refusal
layer = rola.RoLA(routes, expert=PlanOverrides())
```

| name | what it is |
|---|---|
| `PlanOverrides` | the pins: `state_block`, `mma_precision`, `paging`, `sparse_grid` |
| `envelope_refusal(routes, v, decay=None)` / `require_envelope(...)` | whether a call has an arm, asked before making it — the refusing clause in its own words, or `None`. `v` is named rather than assumed: the `d_v` clause is a fact about the value stream, so the answer is a property of the configuration AND of `v.shape[-1]` |
| `requires_backward` | the gradient-reachability predicate the envelope's first clause is built from |
| `GainConfig` | the two-boolean gain layout behind the public `gain` flag |

The contract for everything there is **stable construction, advisory contents**:
build these objects and pass them; do not assert on their fields. What a `Dispatch`
reports is the answer to "how will this call run", and that answer moves with the
machine and with the built matrix. The model facts it carries are the exception,
because they are not decisions.

**Three of the four pins are declared restrictions rather than dials, and each
refuses by name rather than being ignored.** `state_block` (`BC`) is not
host-selectable: the chunk arm's `BC` belongs to the built arm the manifest carries
for a topology, so any value but `None` is refused at the call — a knob that
silently does nothing is how a caller comes to believe a launch was shaped a way it
never was. `mma_precision` has one legal value, `"bf16"`; the precision escape is
the fp64 reference path. `paging` overrides the MAPPING only: `None` is the paged
default, `False` pins the dense plane the bit-identity gate compares against, and
`True` is refused because paging is on by default and bitwise invisible.

**`sparse_grid` is the one that is a real dial**, and it is here because nothing else
decides it yet. It runs the chunk launch over LIVE BLOCKS ONLY — a block-grain
liveness reduction over the call's union table, then a persistent grid whose work loop
skips the dead ones. The reduction costs a launch of its own, so it pays where the
dead-block count is large and loses where it is not, and the automatic rule (a liveness
popcount against a per-arm density knee) is not built. `None` (the default) is OFF,
`True` is the explicit expert choice to pay for it, `False` pins it off. It is
BITWISE INVISIBLE where it applies: the same launch over a smaller grid, gated in
`tests/integration/test_persistent_grid.py`.

**There is no plan object and no derive/execute seam.** `Plan`, `Schedule`,
`derive_plan` and `execute` were the tiled consumer's, and went with it in P67 D2
([`internals/DELETIONS.md`](internals/DELETIONS.md)). The geometry they carried is
now a property of the built arm rather than of a per-launch derivation, so there is
no launch-shaping object for a caller to hold, synthesize or staleness-test. What
replaces the adversarial use they enabled: a conformance family drives
`rola.ops.chunk` directly and declares the refusal clause it expects, and
`envelope_refusal` answers the same question a synthetic plan used to be built to probe.

## 6. What is deliberately not here

- **No dict or JSON configuration surface.** Raw configuration belongs to consumers. The
  library type-checks structured objects and parses nothing.
- **No pluggable recurrence.** RoLA *is* the feature map, not a framework for
  recurrences. The state update is not an extension point and must never be presented as
  one.
- **No silent second path.** Every arm is chosen from a property of the configuration or
  of the operands, before the call, and named in the exception when a configuration
  cannot run. Nothing degrades into anything after a failure.

## Gradients are not bitwise reproducible

`rola_op` trains, and its gradients are **not** run-to-run deterministic. Both scans
assemble their outputs through fp32 `atomicAdd` fan-ins whose order is the
scheduler's, and the two scans walk DIFFERENT partitions of the token axis by
construction — each closes its chunks by its own walk — so nothing about the reverse
pass is bit-reproducible even on identical inputs on identical hardware. The forward
already had this property through the same fan-ins; the backward inherits it in a
worse regime (more contributors, narrower destinations).

Two consequences, stated rather than discovered:

* a training run cannot be reproduced bit-for-bit by re-running it, only
  statistically;
* a gate on a gradient must be a tolerance, never an equality.

**Training is STATELESS.** A call carrying both a gradient and a `state=` plane is
refused by name: the state is caller-owned and mutated in place, and the reverse
pass has no `d initial_state` — it rebuilds from zero. **Decode has no reverse pass
at all**, so a grad-bearing single-token continuation is refused too. And the
reverse pass carries a NARROWER arm matrix than the forward: a grad-bearing call on
an arm it does not carry is refused at the boundary rather than three launches later.
