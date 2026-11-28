# The API

FSLA is linear attention under a structured feature map. That is not a slogan about the
paper; it is the specification of this API. A head's recurrent state is expanded into
many sub-states that share the head's projections, and a learned routing decides which
sub-states each token reads and writes — so the public contract is *linear attention's*
contract, with exactly **one** new concept: the feature map.

A user who knows linear attention should find one unfamiliar thing on this page.

```python
import fsla

routing = fsla.Routing(topology=(64, 64), write=(fsla.entmax(1.5), fsla.softmax()),
                       read=((fsla.softmax(), fsla.entmax(1.5)),))
layer = fsla.FSLA(routing, hidden_size=1024, num_heads=8, d_v=64, layer_idx=0).cuda().eval()
y, _ = layer(hidden_states)                     # stateless: (y, None)
```

or, without the layer, from levels you computed yourself -- the format's own utilities build the bundles:

```python
factors = fsla.RouteFactors.from_levels(fsla.Topology.opaque(widths), read_levels, write_levels)
masks, density = fsla.masks_and_density(factors)          # the masks the rows' nonzero digits make, and their density
y, _ = fsla.prefill(factors, v, masks, density)            # v: the bf16 values, [B, T, H_w, d_v]
```

---

## 1. The op

```python
prefill(routes: RouteFactors, values: Tensor, masks: Masks, density: Tensor, *,
        state=None, decay: LeafMassDecay | None = None) -> (y, state)
decode_step_gated(routes: RouteFactors, values: Tensor, masks: Masks, state, *,
                  decay: LeafMassDecay | None = None) -> (y, state)
```

`routes` is the `(q, k)` pair after the feature map. `values` is the value stream, the `[B, T, H_w, d_v]` projection in
bf16 as `v_proj` writes it: no scale and no normalization ([§2.2](#values)).
`masks` is the data beside the rows, like attention's mask (§2.6): decode's op computes with `r * m`, and the prefill's masks are the caller's
guarantee about the rows. `density` is the prefill's own input (§2.6): a `torch.uint8` `[1 + G, D]` tensor, how full each level is for
each read head and the write side (255 is every digit live), which chooses how the work
is laid out; decode takes none ("decode doesn't need it, once again it has its own design", the owner, 2026-11-24).
`decay` is the recurrence dial (§3): decode has its arm, the prefill refuses it; leave it `None` for a prefill. `y` is
`[B, T, H, d_v]` in bf16 ([§4.1](#output)).

The owner's ruling (2026-11-23): "is it better to just have a decode and prefill producer, have each internal call handle its descriptor stuff
and not be exposed to the user, and have the public contract carry all inputs, workpsace is something we build inside just to pass around?" --
agreed: yes. A call carries ALL ITS INPUTS and nothing else: the rows, the values, the masks, the prefill's density, the state and the decay dial, which is what changes the
mathematics. Each call builds its own DESCRIPTOR -- the validated run-time context, every device fact and every call fact, checked against the
contract before any launch -- from what it is handed: the bundles' extents and declaration, whether a decay and a state are handed, and the
operands' device (`fsla.ops.produce`'s memoized builders). The WORKSPACE is internal, sized by the call's descriptor and passed between the
call's steps: the prefill's holds the leaf groups' mask words and counts the
mask-preparation pass prepares ([`architecture/mask-preparation.md`](architecture/mask-preparation.md#the-mask-preparation-pass)), and its layout is [`mask_prepare_layout.h`](internals/common/plan/mask_prepare_layout.md); decode needs
none. The state is its own argument, in and out. None of these changes the mathematical output, at most rounding within the oracle's tolerance.

`decode_step_gated` has no host participation and is capturable in a CUDA graph: it reads and updates the state's plane in place and returns
`(y, state)`. The step's second call builds its descriptor by a memoized lookup: no device read and no allocation.

<a id="batch-heads"></a>A decode step serves B times H from 1 to 536870911 (`2^29 - 1`: within what the step's work queue of `3 B H + 2` int32
counters indexes); the step's descriptor states it as the `batch_heads` fact and a step outside it is refused when the descriptor is made.

`state` is `None` (a stateless call, returning `(y, None)`) or a state object,
which is mutated in place and handed back. THE PREDICATE IS `state is not None` — there
is no second flag, because a stateless call runs a kernel variant with no state I/O
compiled into it at all. Read the carried state with
`state.materialize() -> [B, H, N, d_v + 1]`, where `N = prod(widths)` is the leaf
capacity, its leaves in the declared level order whatever order the state stores them in; that is a bridge for the
oracle and for a test, never a measured path. See [`docs/internals/state.md`](internals/state.md).

<a id="the-state"></a>`fsla.FSLAState` is the state INTERFACE and `fsla.dense_state(...)` builds its one implementation, the dense state: one
plane of the stored page format, the whole leaf space resident. A STATE IS CONSTRUCTED FROM THE `fsla.Routing` AND THE
CALL'S SHAPE -- the batch, the write heads, the value width -- and its device: its format is fixed at construction and
never changes, and its plane is allocated and zeroed there. A layer or op whose routing or shape differs from the
state's refuses it, naming the field. The plane's contents advance in place each call; only the format is immutable.
`layer.new_state(batch, device)` is the layer's convenience, the state of its own routing, write heads and value width
(§4). A paged state is future work and will be built by its own constructor, taking the parameters its storage needs
(see [`internals/state.md`](internals/state.md#interface)). NOT BUILT: `fsla/_state.py` still builds an unbound state
that binds its format at its first call.

There is no plan object. There is no configuration argument and no statistics argument: those are not model facts -- they are
how the op RUNS. The descriptor and the workspace are internal to each call: it builds them from its operands through the producer's cached
builders (`fsla.ops.produce`); the tests have a helper for it (`tests/descriptors.py`). The prefill kernel is not built, so the descriptor of a prefill the
contract holds as OWED is refused when it is made, as `KernelNotBuilt` naming `prefill`
([`internals/hardware/registry.md`](internals/hardware/registry.md#the-owed-folder)).

**Any level width, any `d_v`.** A `Routing` level's width and a layer's `d_v` are
never required to be a power of two or a shipped value — host-side padding at the op's
own call surface serves the rest, with no kernel change
([`docs/internals/padding.md`](internals/padding.md)): the prefill's carry leg and the
decode step pad below the op, and the intra reads the rows at their own widths.

### 1.1 The built envelope IS the surface

The prefill is the carry and the intra over a BUILT MATRIX
(`fsla.engine.prefill_call`). **Everything outside it raises.** There is no second arm behind this surface: the
fp64 oracle
(`fsla.ops.naive.naive_fsla`) is the executable spec — what tests and measurements
call directly — and it is not reachable by dispatch, because a dispatch to it at
inference is a silent ~1000x-slower route shipping unnoticed, which is the exact thing
the no-fallback rule forbids.

| axis | the built matrix | outside it |
|---|---|---|
| a gradient anywhere in the operands | none | `BackwardNotImplemented` |
| the module's mode (layer only) | `eval()` | `BackwardNotImplemented` |
| device | CUDA | refused |
| `d_v` | the intra's value width (`fsla.ops.intra.VALUE_WIDTH`), which the binary's carry arms at the depth must reach | refused |
| widths | one or two levels, each a whole number of 16-digit granules up to 256, any mix (`fsla.ops.intra.reads`) | refused |
| `T` | any (a partial last window runs inside the kernels; the layer's segments start on window boundaries, §4.2) | — |
| `decay` | decode only (§3); outside the first paper's scope | refused |

No activation reaches the op: it computes on the factors and the masks the producer made and is agnostic to what
produced them, so there is no row for an activation here. What a producer supports is its own registry
(`fsla/routing/activations.py`, the layer's configuration selecting one).

A call outside the contract is refused by name when it is made, and the supported configurations are the generated table in the README. The call builds its descriptor after
validating its operands: a prefill is refused there as `KernelNotBuilt` naming
`prefill` while no carry is built, operands that are not on a CUDA device are refused there naming the device, and any other call outside the contract
is refused naming the facts. A call that carries a gradient is refused by name before any of them (`BackwardNotImplemented`).

**A walk is an action.** `write_listed_k` (decode's call) states whether the write side of level `k` is walked by its lists (1) or digit by digit (0). The producer's rule chooses it from the declaration (a side whose activation has exact zeros is walked by its lists); a walk costs time and never the result. The prefill's call states no walk and no density fact: each side's per-level densities are the call's own input (§2.6), and the prefill's kernels derive their groups and the CTA's carve from them ([`internals/common/plan/prefill_launch.md`](internals/common/plan/prefill_launch.md)).

**There is no plan object.** Everything a launch derives from the facts -- the carve, the window, the grid, the workspace's layout --
is computed by a family's ONE launch function in `common/plan/`, right before the launch, and never stored; the segment plan is the
planner's too, made once a call for the layer that supplies each segment to the op (§4.2); only `common/plan/` reads
hardware facts. Inside, one launch bundle per kernel family (`DecodeLaunch`, `PrefillLaunch`) carries three named parts -- the
`descriptor`, the `operands` (the data, masks included) and the `workspace` -- built by a call's launch step and never public. The `Descriptor`
and the workspace's builder (`fsla.ops.prefill`'s private builder) belong to the library's own code and tests, and are not on the top level. The
state's storage is the state's own, chosen by the constructor (`fsla.dense_state(...)`), never a fact of the call. A conformance family drives the envelope's own rule (`fsla.engine.rules.envelope`) directly and declares the refusal
clause it expects, and the descriptor of a call answers the same question before it.

**The envelope is narrow, and that is stated rather than softened.** Shapes outside the
matrix are refused, and there is NO TRAINING PATH through this surface until the native
backward lands; what widens it is in [`docs/open-work.md`](open-work.md).

`prefill` is the **single autograd boundary**, and it is a refusal rather than a
branch: the failure it prevents — a forward-only kernel reached from a gradient
context returns a tensor with no `grad_fn` and severs the graph silently — is
structural, not a bug in any one call site. The prefill's autograd node, whose backward
raises `BackwardNotImplemented`, is the last line, loud if the first is ever bypassed.

### 1.2 Continuation is LAYERED

The op's state is the dense state and nothing else
([`docs/internals/state.md`](internals/state.md)). Each level of the stack returns its
OWN thin bundle, and the rule is a membership test:

> **An object holds exactly the recurrences ITS level introduced.**
> Op: the factored sparse state. Layer: `LayerContinuation`. Model: the per-layer bundles.

So `FSLA.forward` returns `(y, LayerContinuation)` — always the container — and takes
one back. `LayerContinuation.state` is the op's state; it is the container's
only field today, and any future layer-level recurrence is a field ON IT, never on the
op's state. The container is the contract in both directions: a bare `FSLAState` or a
`(state, conv_rings)` tuple is refused by name, because accepting a lookalike would make
the layer's own fields silently optional to carry. Keeping each level's object pure is
what leaves layer luggage nowhere to land inside the op's state.

### 1.3 TWO CONTRACTS, ONE TRANSLATOR

There are two boundaries in this library and they accept different things. Confusing
them is what produces a kernel that quietly pads, masks or half-serves a shape, so they
are stated apart.

| | what it accepts | what it does with the rest |
|---|---|---|
| **the API** (this page: `prefill`, `FSLA`, `fsla.dense_state(...)`) | any level width `b_l >= 2`, any `d_v`, any length `L` | **pads** — level logits to the next power of two at or above 16 with `-inf`, `v` with zero columns to the state's stored width `DV`, `d_v` rounded up to the format's 32-channel group, one width for the prefill and decode |
| **the kernel** (`fsla.ops.carry`, `fsla.ops.decode`) | only the state descriptor's strict shape: `B_l` a power of two at or above 16, the stored width `DV` (a kernel whose tile is wider, as the prefill's 64 channels on the MMA's M axis, pads in registers or shared memory at load and stores back only the stored channels), `N` a multiple of 16, the internal format's rows and values (§2.1, §4.1) | **refuses**, by name — it never pads its operands in memory and never masks |

**The seam is the only translator.** A padded digit gets amplitude *exactly* zero
(`-inf` before the entmax or softmax solve), so it is never live, never read, never
written, never marks a box and never causes a page to be allocated: the kernel sees a
dead digit like any other and no kernel code knows padding happened. `d_v` padding is
the same mechanism on the value side — the projection writes the first `d_v` columns of
a `DV`-wide buffer and `y` comes back as a view of the first `d_v` — so a kernel never
sees a width other than its own.

**Both boundaries are tested.** API tests prove a padded run is bit-identical to the
same shape declared natively; kernel-boundary tests call the launch surface directly
with unpadded shapes and assert the refusal
(`tests/oracle/test_padding_seam.py`). A logical-width field in a kernel entry
signature is refused by nothing yet, though a kernel that could read one could pad: the lint that
would refuse it is owed ([`CONSTITUTION.md` R13](CONSTITUTION.md#r13)).

**What the descriptor fixes.** A state's format is fixed when it is constructed ([§1](#the-state)) — `D`, the
per-level `B_l`, the leaf order, the page rectangle, `DV`, and the stored form
(`fsla._state.STATE_DTYPE`, [`internals/common/state_page.md`](internals/common/state_page.md)) — and every call
is *checked* against it rather than assumed
to fit. `DV`, the stored value width, is `d_v` rounded up to the format's 32-channel group and no further: a kernel
whose tile is wider pads at load and stores back only the stored channels, so one state serves the prefill and decode
alike ([`internals/state.md`](internals/state.md#format)). A call that disagrees is refused at the seam, naming the field, instead of
addressing somebody else's bytes inside a kernel. Re-laying out a state is an explicit
op, never a side effect of a call. See
[`internals/state.md`](internals/state.md#format).

**Status.** `prefill` validates the call against the envelope and runs the prefill call.
No carry kernel is built (`fsla.ops.carry.carry` raises `KernelNotBuilt`), so a prefill does not run
([`docs/open-work.md`](open-work.md)).

## 2. The feature map

### 2.1 The bundle IS the contract, and it is THE INTERNAL FORMAT

```python
RouteFactors(topology, read, write)                      # one ScaledBlock a level-side, in the ROWS form
RouteFactors.from_levels(topology, read, write)          # the framework's epilogue: floating levels in, the blocks out
```

**The stored form is the internal quantized format** (ruled 2026-09-26). Each level-side is ONE scaled block
(`fsla.ScaledBlock`, §2.2) whose rows are
UNIT-MAX -- u8 with the row's largest digit at 255, at every level, dense or sparse (CONSTITUTION R17, R23); a zero
digit is an exact zero, with no floor under it; an all-zero row is a dead token at that level -- with the row's maximum
carried beside them as its fp32 scale (a block's `data` and `scale`; `read` and `write` one
block per level over its heads, `H` read and `H_w` write).
<a id="tile-major"></a>THE ROWS ARE PUBLIC IN THEIR TILE-MAJOR LAYOUT (the owner's ruling of 2026-11-28 on a level's public form): a level-side's `data` is u8
`[B, H, ceil(T / 64), ceil(width / 16), 64, 16]` (`H_w` on the write side) -- a (sequence, head) STREAM's 64-token
TILES one after another, each tile its 16-digit BLOCKS one after another, each block its 64 tokens' 16 bytes, a token a
line: digit `d` of token `t` of head `h` of sequence `b` at `((((b H + h) ceil(T / 64) + t / 64) ceil(width / 16) + d / 16)
64 + t % 64) 16 + d % 16`. PADDING: a stream's last tile is whole, its tokens past `T` zero rows; a row's digits past
`width` zero to its last block; both are exact zeros a reader multiplies (a padded token is a dead token, a padded digit
a dead digit). Why this order: a CTA reads a (sequence, head)'s level a tile at a time, so batch and head are outermost and
a tile is one contiguous run (the intra stages a tile's level as one copy so, against 16 strided ones token-major:
[`intra_window`](lab/intra_window.md)); a block is the 16 bytes a token's integer step reads; and the producer's warpgroup
writes one tile, a contiguous run of `ceil(width / 16)` KB, with no word or line shared with another CTA
([`produce.md`](internals/produce/produce.md#plan)). The scale stays token-major, fp32 `[B, T, H]`. Every level is stored as rows, once (ruled 2026-10-01: the
intra and decode read rows, and the carry reads rows through the words the masks are prepared into). The bundle carries NO liveness: which
digits are live is the masks' (§2.6), supplied beside the rows and never derived from them or from the activations.
NOT BUILT: the bundle's reference path (`fsla/routing/quant.py`'s `aligned_rows`) still holds the rows `[B, T, H, width]`
token-major; the tile-major form [above](#tile-major) is the producer's
([`ARCHITECTURE.md` §1](ARCHITECTURE.md#1-what-the-kernel-computes)), and the bundle takes it with the producer's launch.
<a id="grouped-queries"></a>
GROUPED QUERIES: the read side is the `H` query heads' (`heads`), the write side the `H_w` write heads' (`write_heads`):
its rows and scales (`[B, T, H_w]`), and with it the values, the decay dials and the state. `H_w` divides
`H`, and query head `h` reads write head `h // group` (`group = H / H_w`, consecutive: PyTorch SDPA's `enable_gqa`
order); `H_w = H` is the ungrouped layer. The oracle reads grouped bundles (`tests/oracle/test_oracle_grouped.py`: a
grouped run is the ungrouped run with each write head's side repeated for its query heads). Query head `w * G + g` is READ HEAD `g` of
write head `w`, produced by the routing's `g`-th read tuple (§2.3): the layer's producer makes grouped bundles, and its read
heads' activations and listed levels may differ. A carry reads them a CTA a write head, its query heads' rows with it, as does the intra, so the prefill reads them. The decode step and the carried state read one head count
today, and each refuses a grouped bundle by name (`RouteFactors.refuse_grouped`).
`RouteFactors.rows(level, side)` is a level's rows, and its scale the row's max: the level as its producer handed it
over, whatever its sum ([§2.2](#22-the-scaled-block-and-the-gain)). This is what
the consumers read, and it is written by the framework's producer
alone: the fused kernel's epilogue, and the reference's (`fsla/routing/quant.py`, `RouteFactors.from_levels`),
the first gated against the second where it is built. The op's oracle takes the
bundle itself as its input.

**Nothing floating is stored.** `RouteFactors.dequantized()` computes the levels, `rows x scale`
in fp32, at the call: a consumer that still computes on floating levels calls it at its seam, so
it computes on the quantized VALUES and nothing in the tree sees a number the kernel would not;
the call goes when that consumer reads the rows. A block's `source` is the producer's floating level, held
ONLY when it requires a gradient -- the anchor `dequantized()` steps the gradient straight
through the rounding to (`quant.straight_through`); an inference bundle holds the rows and nothing
else.

**The bundle is a record of blocks, and it is constructible.** Its fields are the
stored levels, their scales and the topology they are shaped by — nothing else, and in
particular no declaration a caller could pair with tensors that do not describe it.
`RouteFactors.__post_init__` therefore checks the STRUCTURE rather than the provenance: one
block per topology level at that level's width, each in the `ROWS` form, one shared
prefix a side, one device, each scale fp32 at its side's prefix (the block's own check). A test that plants
a specific routing pattern builds one directly and is held to exactly those invariants.
Unit-sum-ness is not among them: no level need sum to 1 ([§2.2](#22-the-scaled-block-and-the-gain)).
The op still type-gates on the class, so handing it bare tensors is still a clean refusal.

A producer also owns the topology (`routes.topology`, `routes.widths`), because that is
a fact about the map rather than about the layer
wrapping it. It does NOT own `d_v`: the value width is the width of the state each leaf
holds, which is a fact about the value stream — the layer sizes `v_proj`/`o_proj` with
it and the op reads it off `v.shape[-1]` at call time.

**The side gain is COLUMNS OF THE ROUTER GEMM.** The gain is a function of the read
stream and so are the routing logits, so `route_W` carries both: the routing slots
first, the gain's span last. One parameter, one cast of the source stream, one
projection. `gain=False` removes the column rather than pinning it, so a disabled gain
is a narrower parameter and not a parameter with an identically zero gradient.

**The write gain is the producer's, and it rides write level 0's scale.** There is no
read-side gain at all — the ratio readout cancels a per-token read factor but in the floor —
so the write side carries the only gain column there is, and a layer-owned gain producer would
have to be told that in order to know its own columns, which is the coupling running
backwards. The producer folds the gain into write level 0's scale, in the job that writes that
level, so the format has no gain field and no consumer reads one; it rides the bundle's own
tensors, which `RouteFactors.tensors()` enumerates for autograd.

### 2.2 The scaled block, and the gain

A level is numbers in a representation, not a distribution (ruled 2026-11-07,
[`CONSTITUTION.md` R23](CONSTITUTION.md#r23)). `q` (read) and `k` (write) are each a
SCALED BLOCK: unit-max rows in a FORM that states its storage type and its signedness -- u8
nonnegative at 255 (a routing level, dense or sparse, a zero a zero) -- with the row's max as its fp32 scale. <a id="values"></a>THE
VALUES `v` ARE RAW BF16, `v_proj`'s output as it is: no scale a token and no unit-max normalization, the precision
class FlashAttention's values have (the owner, 2026-11-28: "everyone uses bf16 right. lets just be simple and use raw
bf16"). ONE type holds every scaled block, `fsla.ScaledBlock(data, scale, form, source=None)`
(`fsla/routing/quant.py`): a routing level-side's rows (form `ROWS`), `[B, T, H, width]` rows and
`[B, T, H]` scales, checked for structure (dtype, shapes, device, the alignment law where the form states it) and never for values. The form
is `fsla.routing.quant`'s `Form`, quantizing into and dequantizing out of it written once on it
(`ScaledBlock.quantize(x, form)`, `ScaledBlock.dequantized()`), and the factory refuses an entry its form does not
hold -- a negative or non-finite routing level -- with one host read a call. NOT BUILT: the values' `VALUES` form
and `fsla.ScaledBlock.quantize(v, VALUES)` still stand in `fsla/routing/quant.py`. A side's shared scale is the product of its levels'
scales, STORED FACTORED: each level's scale is written by the producer job that quantizes that
level and multiplied by the reader as it loads them, so no job waits on another. The op every
consumer computes, and the fp64 oracle with them:

    W(t, s) = prod_l k_l(t, d_l(s))          R(t, s) = prod_l q_l(t, d_l(s))
    S_t(s)  = S_{t-1}(s) + W(t, s) v(t)      den_t(s) likewise with v = 1
    y(t)    = sum_s R(t, s) S_t(s) / (sum_s R(t, s) den_t(s) + eps)

A READ ROW IS A SIMPLEX'S: FSLA is normalized attention only, so a read row's true values sum to one and a readout is a
weighted average of state entries ([§2.3](#simplex)). The write side carries no such rule: no write level is required to
sum to 1. A producer's magnitude is its own: ours, the write gain `softplus(gain logit + bias)`, is folded
into write level 0's scale (§2.1); a caller's own levels carry whatever magnitude they have into
their scales, `RouteFactors.from_levels` quantizing each against its own max and nothing else.

**The read scales are kept, for the floor.** A per-token read factor common to every leaf
cancels in the ratio everywhere but the floor, so the reference reads the read scales' product `r`
only there: its readout is `num' / (den' + eps / r)` (`fsla/ops/prefill.py`'s `readout`),
which is the formula above exactly, and it is what makes `eps` mean the same thing on every
routing. A kernel applies each read row's fp32 scale where it expands the row instead
([`architecture/hopper-carry.md`](architecture/hopper-carry.md#the-scales-the-den-and-the-outputs-line)), so its
`num` and `den` carry `r` and its readout is the same formula. `READOUT_EPS` is a DEFINITION constant (`fsla/ops/constants.py`). Decode computes the
same `y`: neither normalizes a read level.

**The decay computes its own shares** ([§3](#3-the-decay-source-slot)): its clock is each leaf's
SHARE of the token's write, `c = prod_l k_l(d_l(s)) / sum_d k_l(d)`, a row sum a level and
scale-free, so no stored level need be a simplex and a gain folded into a scale is invisible to
it.

**The producer owns packing.** The solve emits `[BH, T, width]`, which is already what
the kernel's view wants, and the public `[B, T, H, width_l]` layout is a permute of it —
so exactly one of the two can be the materialized tensor, and it is the kernel's. The
public tensors are views. Materializing the public layout instead and folding it back is
an inverse copy pair. A harness measures it directly against the production call it
rides: 3.085 ms vs 2.732 ms at `L = 4096` and 0.942 ms vs
0.835 ms at `L = 1024` — 12.9 % and 12.7 % of production-plus-fold, at both lengths.

### 2.3 The declared routing, and the producer

```python
Routing(topology=(w0, w1, ...), write=(a0, a1, ...), read=((r0, r1, ...), ...), *,
        gain=True, gain_bias_init=None, gain_config=None, bias=False)
FSLA(routing, *, hidden_size, num_heads, group=1, d_v=64, decay=None, layer_idx=None, ...)
```

RULED 2026-09-29: the producer is a DECLARED CONFIGURATION. A `Routing` is data -- the topology, its activations, the
write gain and the projection bias -- and the layer owns every weight it implies: the
packed per-write-head routing projection `route_W`, its bias, the write gain's bias, `v_proj` and `o_proj`. One producer
call (`fsla.ops.produce.produce_prefill`, or `produce_decode` for the decode step) turns a stretch of tokens into the bundle -- the projection, the exact
entmax/softmax solves, the write gain, the unit-max rows and the bf16 values -- and on CUDA
that call is ONE LAUNCH of the fused producer kernel, the bundle's rows and scales views of the planes it writes --
a kernel not built on this line, so the no-grad call raises `fsla.ops._ext.KernelNotBuilt` there
([`open-work.md`](open-work.md)). The torch composition behind the same call is the reference: fp64 and the host.

<a id="training"></a>**ONE FORWARD SERVES INFERENCE AND TRAINING.** The fused producer kernel is the forward with or
without a gradient. With a gradient on, the same launch also writes a LO PLANE beside the u8 rows -- one byte a digit,
the rounding residual of the digit the readers take -- saved privately by the producer's autograd node, from which
the backward rebuilds each level's value, at least bf16's precision above a 255th of the row's max; the writing of it is a compile-time variant of
the kernel, on exactly when a gradient is needed. The public API and the op's u8 operands are the same either way:
nothing reads the lo plane but the producer's backward. The memory a training step holds is the caller's to bound with
torch's own checkpointing, so the forward is pure in its inputs and re-runs under `torch.utils.checkpoint` to the
same result. THE BACKWARD IS SEVERAL LAUNCHES, each with its own partition: the solve's backward, then the `dx` and
`dW` GEMMs -- `dx` sums over every head's columns and `dW` over every token, so the two want opposite partitions; the
one-launch rule below binds the forward. NOT BUILT: the fused forward, the lo plane and the backward kernels; until the
fused forward lands, the composition's CUDA entmax solve stands under the call, taking bf16 logits (the projection's
output) and writing the bf16 stored form -- fp64 logits take the torch lowering, and CUDA logits of any other dtype are
refused by name -- and it and the composition's CUDA training path are deleted with the fused forward's landing.

Construction validates the whole declaration --
widths in range, a buildable topology, a gain layout -- so a model the declaration cannot state fails on the line that built it (the one declaration it accepts and the solve refuses is named below).

<a id="the-routing"></a>**The routing is declared ONCE: one topology, flat activations, `G` read heads** (the owner, 2026-11-27: "its
not worth it. just have a flat activation for each pattern"). `topology` is the level widths, outermost first. `write` is one flat
activation a level -- `softmax()` or `entmax(alpha)`, a registry row; none is chained. `read` is one tuple a READ HEAD of the group --
`G` of them, `G` the layer's query heads a write head (`FSLA(group=G)`) -- or ONE tuple that every head reads by; each tuple is one entry
a level, an activation or `tied()`. The read side states no widths: they are the topology's.

```python
Routing(topology=(64, 64), write=(entmax(1.5), softmax()), read=((softmax(), entmax(1.5)),))   # the alternating split
Routing(topology=(64, 64), write=(entmax(1.5), softmax()),
        read=((tied(), entmax(1.5)), (softmax(), softmax())))                                # G = 2: each head its own tuple
```

`tied()` is a POSITIONAL MARKER, not an activation: the read head's level IS the write side's routing on that level -- the same rows,
one solve, no read columns of its own -- and any rule asking about the cell reads the write activation (a tied softmax level is legal and
dense on both sides). <a id="tied-copies"></a>IN THE FORMAT A TIED LEVEL IS A MATERIALIZED COPY: every read head has its own rows and
scale on every level, the write job's epilogue storing the same rows and scale to each tied head's planes, so a consumer reads each head's
own rows natively and nothing downstream knows a level was tied. Each read head's activations and listed levels may differ; every read head
has the write side's topology (a per-side or per-head structure is not declared).

**Every activation is nonnegative** (the registry's `nonnegative` column, `fsla/routing/activations.py`): the v1 routing is nonnegative, and a
row that is not is refused at the declaration. <a id="simplex"></a>**FSLA IS NORMALIZED ATTENTION ONLY** (the owner, 2026-11-28: "we are
locked into only normalized form"): every activation declares whether it is a SIMPLEX -- its row's true values sum to one -- in its registry
row beside whether it makes exact zeros, and the read side refuses a non-simplex activation by name, so a read row sums to one and a readout
is a weighted average of state entries. A write level may be non-simplex. Construction refuses, naming the reason: an activation with no
registry row, one whose row is not nonnegative, a read entry whose activation is not a simplex, one the producer does not solve, a read tuple
whose length is not the topology's depth, and a width below 2. NOT BUILT: the registry's simplex column and the read side's refusal. A level whose untied read entry and write side both make exact zeros is allowed: two solves, two supports, and nothing makes them overlap, where a tied entry is one solve and one support. One accepted declaration is refused later, at the solve: a level SEVERAL read heads tie has no gradient built, so a call on the kernels that requires one is refused by name ([`internals/entmax/factor.md`](internals/entmax/factor.md#read-heads)); its forward without a gradient runs.
The layer refuses a read count that is neither its `G` nor one, naming both (`Routing.layout`).

Depth, leaf count `N`, the packed router width and the per-level logit offsets are all DERIVED from the declaration
(`fsla.routing.projection.RouterLayout` at the layer's `G`). None of them is a second input, so none of them
can disagree with it. `route_W` is `[H_w, hidden, columns]`, a write head's slice holding its write levels' columns in level order, then each
of its read heads' untied levels' columns (head 0's, then head 1's, ...), then the gain's span: every column is consumed and the projection is
one GEMM over the whole parameter, a side's logits column spans of its output.

<a id="one-launch"></a>**Every level, every head and the values are one launch by design** -- a model pays one launch for the whole
topology, whatever its levels' operators, alphas and widths. The one launch is the FORWARD's, with or without a gradient; the backward is
several ([above](#training)). That kernel is not built on this line: the no-grad CUDA call raises `fsla.ops._ext.KernelNotBuilt`, and
the torch composition behind the same call serves fp64, the host and, until the fused forward lands, a gradient.

<a id="projection-weight"></a>**The projection reads a laid-out copy of `route_W` when no gradient flows.** The GEMM's
operands are bf16 and the kernel reads each column's hidden run contiguous (`hidden_contiguous`), so an inference
step would otherwise cast the whole fp32 parameter and re-lay it out on every call -- measured at
~55 us of GPU a decode step at `H = 16`, `(256, 256)`, more than the decode kernel. The copy
is made once and reused while the parameter's version counter, storage, device and the
projection dtype are unchanged; an in-place update (an optimizer step, `load_state_dict`)
moves the version and so the copy. A step that trains casts inside the graph, since its
gradient must reach the parameter and the parameter changes every step anyway. (Writing
through `route_W.data` bypasses the version counter, as it does for every autograd cache.)
The layer's value and output projections read their bf16 weights the same way (another ~6 us
a step and two launches at `H = 16`); the one mechanism is `fsla/_inference_copy.py`.

Activation *properties* are never supplied by a caller — they come from the closed
registry in `fsla/routing/activations.py`, keyed on the tag. Metadata a caller could
supply is metadata a caller could supply wrong, and a validator reading it would be
checking a claim instead of a fact. Activations stay width-free: an activation is a
function on a row of any length, so a width on one would be a second place the same
number lives.

### 2.4 Production lives in the framework

RULED 2026-09-26, and 2026-09-29: there is no duck-typed producer, and the layer takes no callable feature map. A
new kind of production is added to the framework -- its activation to the closed registry, its solve to the
producer, its reference and its oracle -- and what it returns is the internal format. A caller of the ops with its
own levels builds the bundles through the format's own utilities (`RouteFactors.from_levels`), which are the same epilogue the
producer applies, and hands its bf16 values as they are; nothing else constructs a bundle the op
admits: the format's fields are required, checked structurally, and a bundle assembled from bare floating levels is
refused by name. Correctness is still producer-blind -- liveness is the supplied masks, never derived from the rows, and nothing about a producer's identity reaches a kernel -- but the FORMAT is ours, because it is what every
kernel reads and what the oracle gates against.

A level filled by a production that names none of our activations is *opaque*: its topology is
`Topology.opaque(widths)`, every cell the `("opaque", None)` registry row. What that costs it is the
activation questions the topology can answer about it -- ratification, in particular -- and
nothing else.

### 2.5 The one modelling consequence to know

The decay clock is each leaf's SHARE of the token's write ([§2.2](#22-the-scaled-block-and-the-gain)),
scale-free: a magnitude a write level carries -- a producer's gain, a level summing past one --
scales the deposit and not the clock. That is the defined semantics rather than a defect, but it
is a choice a level's author is making, so it is stated here rather than buried.

### 2.6 The masks are data, in each consumer's native layout

A mask is a bit per digit per token, ONE MASK A (HEAD, LEVEL), THE WRITE SIDE INCLUDED: the write head's mask on each
level and each read head's own (the owner, 2026-11-28: "one mask per (head, level) including write"). No mask is
shared across read heads and the write side's is never duplicated for them. A mask is an operand BESIDE the rows, like
attention's mask (ruled 2026-11-22). The masks are validated for shape, dtype and device, never for
their values. What a bit says depends on the consumer.

**DECODE computes with `r * m`**, the product under the mask: a bit under a nonzero row drops that digit's term, a bit over a
zero row adds nothing, and the oracle (`fsla.naive_fsla(..., masks=)`) is the result the data fixes. The backward saves the masks and they
carry no gradient: a masked coefficient's gradient is zero.

**The PREFILL's published math is over the rows and the values**, and its mask is a guarantee from the caller (the owner, 2026-11-24:
"if it's zero you are guaranting to use this value is zero. If it's not zero, you will get a wrong answer. If it's 1, it can be anything"):
A mask bit of 0 is your guarantee that the entry's row is zero. If it is not, the result may be wrong. A bit of 1 guarantees nothing: the
row can be anything, zero included. Masks built by the library's helpers from the rows always keep the guarantee. How the kernels use the
bits (which groups they skip, how they lay the work out) is an implementation detail and is not exposed ("groups are not exposed at all"):
the masks are a suggestion the kernels may take. The oracle keeps computing rows times bits; for masks that keep the guarantee that is the rows alone.

```python
fsla.Masks(consumer, read, write)   # write: one int32 tensor a level; read: one such tuple a read head, G of them
                                    # (prefill: an entry may be None, dense); consumer = "prefill" | "decode"
```

The producer emits them in the layout of the consumer it is made for (`produce_prefill` the prefill's, `produce_decode` the decode step's), so no
conversion stands between the producer and the op, and each op documents the layout it reads:

| consumer | a level's mask | bit |
|---|---|---|
| `"prefill"` (`prefill`, `FSLA.forward`) | int32 `[B, H_w, ceil(T / 32), width_l]` a (head, level), TOKEN-packed | bit `t % 32` of word `t // 32` = token `t` live at that digit |
| `"decode"` (`decode_step_gated`) | int32 `[B, H, ceil(width_l / 32)]`, DIGIT-packed, one token | bit `d % 32` of word `d // 32` = digit `d` live |

Bits past `T` (prefill) or past `width_l` (decode) are zero. **THE DENSE FLAG IS THE DENSITY BYTE 255** (CONSTITUTION R9): a level-side whose byte is 255 is computed with every digit live and its mask is never
read, so in the prefill layout its entry in the bundle may be ABSENT (`None`). An absent entry's byte is 255: that is its supplier's duty, as a mask's
guarantee is, and the call does not check it (it validates no value of the masks or the density, and reads nothing back to the host). The library's
producer and helpers leave a level-side's mask absent, and state its byte 255, exactly where its declared activation cannot make exact zeros (softmax
today; on the read side, the head's own entry's, a tied entry reading the write side's; `fsla.routing.types.activation_walks`), from the declaration and never from the data (the owner, 2026-11-24: "like if it's declared no zeroes, we just don't make the masks").
Each read head's level is its own: one head's level may be absent and 255 while another's on the same level holds a mask, and each has its own
density byte; a (head, level) that
can make zeros (entmax, and an opaque level) always gets its mask, however dense its data turns out, and where its count rounds to 255 it is computed
densely too, which is exact (the owner: "It's fine if we mark something as dense when it's possible it's like 99.99 due to rounding or whatever ... the non zero flag is also data, and in this case there is no mask to disagree with"). The kernels are not built. The oracle multiplies nothing for an absent entry. DECODE is unchanged: its masks are
made for every level-side, an all-live one where the activation has no exact zeros, and a decode bundle with an absent entry is refused by name. The rows, scales and values need no conversion: the producer already emits them as the consumers read them
(u8 `[B, T, H, width_l]` rows and fp32 `[B, T, H]` scales, raw bf16 `[B, T, H_w, d_v]` values).

**The density is the prefill's own input, beside the masks** (the owner, 2026-11-24: "I almost think weights should be outside the masks? They
need not be coupled, and are independent variables really. Someone may want to give crazy weights as some experiment. No reason we need to stop
them" -- "It's also more clear what the contract is if the weights are their own variable vs hidden in the masks"; the name: "just density is fine
initially"). <a id="density"></a>It is one `torch.uint8` tensor `[1 + G, D]` on the rows' device, a row a head and an entry a level, as the
masks are one a (head, level): rows `0 .. G - 1` read head `g`, row `G` the write side -- at `G = 1`, row 0 the read side and row 1 the write
side. A byte `b` means the
share `b / 255` of the level's (token, digit) pairs that are live: 255 is every digit live on every token and 0 is none (the owner, 2026-11-24: "Why can't
s byte just express 255/255?" -- "Ok to switch to 255"). The call checks its type, shape (`[1 + G, D]` for the bundle's read heads and depth), uint8 dtype (a float
tensor is refused, by name) and device, and NOTHING about its values: every byte is a valid density, so there is no value to refuse. It is carried with the call's other inputs to where the prefill's kernels will read it. Today
the GPU prefill is refused by name as not built and the reference path does not use it: as a fact of the contract, the density changes no defined
result. It chooses how the work is laid out, and it changes no result while the masks keep their guarantee.

The op computes nothing for a caller ("the op call signature shouldn't change or compute anything for them. They need to use our library helps to
build what they want from their data that they have"); the library's helpers build what a caller has not got, on the operands' device with torch
operations and no host read:

```python
masks, density = fsla.masks_and_density(routes)     # from the rows: a digit is live where its row is not zero
density = fsla.density_of_masks(routes, masks)      # the density of PREFILL masks a caller already has
```

`masks_and_density` returns the PREFILL-layout `Masks` and the `[1 + G, D]` uint8 density; `density_of_masks` returns the same density of the set bits, over all
`(token, digit)` pairs of each (head, level) (over the batch and the write heads). It takes the bundle rather than a token count: the bundle carries the
token count and the shapes the masks are checked against, so a count that disagrees with the masks cannot be handed over. The helpers round
to the nearest byte, a half up, in integer arithmetic (the library's rule): with `live` the set bits and `total` the level's (token, digit) pairs over
the batch and the side's heads, `b = (2 * 255 * live + total) // (2 * total)`, no float entering. A level all live gives exactly 255 and a level all
dead exactly 0. The prefill's kernels, which are not built, will read a byte `b` as the integer `b + 1` in 256ths of the level (so every byte is a valid
density, and the kernels need no clamp, no rounding and no NaN case): the carve clamps an integer outside 1..256 into the range
([`internals/common/plan/cta_carve.md`](internals/common/plan/cta_carve.md)). The producer's `produce_prefill` returns the density beside its masks, from `density_of_masks` after the
solve, and the layer hands it to `prefill`. <a id="mask-bit"></a>THE PRODUCER'S MASK BIT SAYS THE U8 DIGIT IS NONZERO: it ballots the
quantized digit in the same epilogue that writes it, so a 0 is written exactly where the digit the op reads is 0 and the density counts the
work there really is; the solve's support, a looser superset, is not the mask. A 1 over a zero stays legal, as every mask's is (wasted work,
never a wrong answer), and the backward reads no bit. NOT BUILT: the per-(head, level) masks and the `[1 + G, D]` density -- the helpers
and `fsla.Masks` still hold one read mask a level over every query head, and a `[2, D]` density. A prefill level-side whose activation cannot make exact zeros has no mask (its entry is absent, dense) and the byte 255, which the
helpers write from the declaration without reading any data; `density_of_masks` gives 255 for an absent entry and counts the set bits of a present one. A level-side that can make zeros always has its mask, and the decode's masks are always made, for every activation.

## 3. The decay-source slot

**OUTSIDE THE FIRST PAPER'S SCOPE.** Decode builds a decay arm, checked against the oracle (`tests/oracle/test_decode_vs_oracle.py`'s
decay rows); the prefill has none and refuses a decay (§2). The source classes below are the declared surface.

Where the recurrence's forgetting rates would come from is a CHOICE, and the recurrence could
not tell the answers apart:

```python
fsla.ConstantDecay(rate, widths=..., num_heads=...)                    # user-fixed
fsla.LearnedDecay(rate, widths=..., num_heads=..., scope='state')      # 'state' | 'global'
```

Both emit a `LeafMassDecay` whose `dials[l]` is `[H, width_l]` in `[0, 1)`, `H` the write heads (the decay is the
state's: [grouped queries](#grouped-queries)). A source is
validated against the topology at layer construction. Duck-typed like a producer:
declare `widths` and `num_heads`, return a `LeafMassDecay`.

`state` and `global` are the two IDENTIFIABLE resolutions: under `keep = (1 - prod_l
delta_l)**c` (one write-conditioned clock per leaf, `c` the leaf's share of the token's write:
[§2.2](#22-the-scaled-block-and-the-gain)), a rate only ever enters the
recurrence through that per-leaf product, so per-digit and per-head are the two degrees
of freedom the product structure lets the recurrence actually distinguish.

Nothing about a constant source is a special case: it enters through exactly the same door a learned source would.
No call runs a decay today (§1.1).

The decay is **write-conditioned** (mass decay), never time decay. A leaf whose write
amplitude is exactly zero ticks its clock by zero, so `keep = (1 - rate) ** 0 = 1`
exactly and its state is left bit-unchanged. Several decisions elsewhere in the system
rest on that property.

## 4. The layer

```python
FSLA(routing, *, hidden_size, num_heads, group=1, d_v=64, decay=None, layer_idx=None, segments=SegmentPolicy(max_tokens=65_536))

y, continuation = layer(hidden_states, continuation=LayerContinuation(state=layer.new_state(B, device)))
y, continuation = layer(next_tokens, continuation=continuation)
y, _ = layer(hidden_states)                            # stateless -> (y, LayerContinuation(state=None))
```

`forward(x, attention_mask=None, continuation=None) -> (y, LayerContinuation)` — ALWAYS
the container (§1.2).

`routing` is the declared `Routing` (§2.3), and the layer owns every weight it implies.
`num_heads` are the WRITE heads `H_w` -- the state's, the values' and the decay's heads, `route_W`'s and `v_proj`'s -- and `group` is
`G`, the query heads reading each ([grouped queries](#grouped-queries)): `num_heads * group` query heads in all, query head `w * G + g`
read head `g` of write head `w`, and `o_proj` takes their `num_heads * group * d_v` outputs. The routing declares `G` read tuples or one for
every head; any other count is refused at construction. `group = 1` is the ungrouped layer.
`d_v` is the per-head value width — the layer's, because it sizes `v_proj`/`o_proj`; the
default, 64, is the intra's value width.

<a id="one-token-step"></a>**One token of a carried sequence is `_step`.** `FSLA._step(x, state)`
is the whole token with no host read inside it -- the producer's one launch, the gated decode
step, the output projection -- so a caller can capture it in a CUDA graph and replay it a
token (`docs/internals/decode/decode.md#capture`).

A convolution over the value or routing-source stream, if one is added, is a field on
`LayerContinuation`.

The layer adds exactly one clause the op cannot know: `self.training` REFUSES,
independently of gradient reachability, because reentrant gradient checkpointing runs
its first pass under `no_grad()` with the module still in training mode — and a
forward that pass could take would be a forward no backward can follow.

`layer.last_execution_backend` names the arm the last forward took — `'prefill'` or
`'decode'`, and `None` until one completes. It is a fact about the branch rather than an
inference from a timing, and it is published for exactly that: a benchmark process
cannot spy on the call it is timing, and wall-clock cannot tell two arms apart. It is a
SCALAR, not a tally — a caller that needs how many times an arm ran spies on the arm.

### 4.1 The projections and the value format

`v_proj` and `o_proj` run as bf16 GEMMs (their weights read through the inference copy when no
gradient flows, §2.3), and `v_proj`'s bf16 output IS THE VALUE OPERAND, as it is ([§2.2](#values)): no scale a token, no
unit-max normalization and no pass between the projection and the op. The values' precision is bf16's, the class
FlashAttention's values have. Round-tripping
them through the module dtype was MEASURED to cost two elementwise casts per call — a `v`
downcast and a `y` upcast, ~12-13% of a small cell's op time — which is why the layer
narrows once, at the projection, rather than at the op boundary. The GEMMs use cuBLAS's
fp32 compute type (the accumulate-in-fp32 ruling); the global
reduced-precision-reduction knob is deliberately left alone, so this is the same
environment the FlashAttention comparison arm runs in.

<a id="output"></a>THE OP RETURNS `y` IN BF16, as FlashAttention and torch's SDPA do: `num` and `den` are summed in fp32 (inside a CTA
always; across CTAs fp32 or the f16 line, [`architecture/hopper-carry.md`](architecture/hopper-carry.md#output-line)), and a
FINISH step divides and writes the bf16 `y` once -- likely a tiny kernel of its own. `o_proj`'s bf16 GEMM reads `y`
directly, with no cast pass: the output store's one bf16 rounding, about 0.16% a token, is the one rounding on the
output side and inside the family's accepted class ([`architecture/precision.md`](architecture/precision.md#quality)).
NOT BUILT: the finish; the op's reference and the layer return `y` in fp32 today (`fsla/layer.py`).

### 4.2 Segments: the layer bounds its own working set

<a id="decode-segments"></a>THE SEGMENT POLICY BOUNDS THE PREFILL. Decode runs every head of the call in one launch and takes no
segmentation: its working memory beyond the resident state is a row a query head and a queue, so splitting heads would only add launches.

The op runs one segment and knows nothing of segmentation: SEGMENTATION IS THE LAYER'S CONCERN (the owner, 2026-11-26, on whether
the constitution's R24 needed amending for it: "the prefill op is not aware segmentation, it still computes everything for its kernel in its one launch
function from its input. segmnetation is a layer concern"). R24 governs the op and its kernels and stands as written: the segment the op is handed is its
whole call, and its launch function derives everything for that call. The layer is handed a SEGMENT POLICY (`segments=`), the most tokens and the
most WRITE heads a step may hold, `None` on an axis meaning no limit there (the owner, 2026-11-26: "i think its reasonable to support a
segmentationpolicy primtive that has like max_tokens, max_heads, null meaning no limit"), or a SEGMENT HANDLE, a function of the call's context
returning one, or `None`, one prefill one op call. The plan of a call is made once, by the planner, which reads the card's window
([`internals/common/plan/segments.md`](internals/common/plan/segments.md); the owner, 2026-11-26: "i was thinking we could just call it once and it
yields the whole exceution plan"):

```python
fsla.SegmentPolicy(max_tokens=8192, max_heads=16)   # no step longer than 8,192 tokens or wider than 16 write heads
fsla.max_tokens(8192)                               # SegmentPolicy(max_tokens=8192)
fsla.max_memory(1 << 30)                            # a handle: no step whose workspace and inputs, at their worst case, pass 1 GiB
fsla.segment_plan(length, policy, batch=B, heads=H) # the plan: its head blocks, and the token steps in each
```

- **The default is `SegmentPolicy(max_tokens=65_536)`**: 64 windows a step of every head. It is revisited when the prefill is first timed.
- **A policy states maxima and the plan takes the largest whole number of windows under the token maximum**, so a caller never needs the
  card's window ("so the user doesnt need to be aware of our window sematncis"). The window grid is global, so a boundary on a window boundary
  moves no pair: a segmented call is the whole call (the op's readout to fp32 ULPs, `tests/integration/test_layer_segments.py`).
- **`max_heads` counts write heads, the state's owners, and never splits a query group** (the owner, 2026-11-26: "we are just saying heads is
  write heads, and all read heads must live for any write head? we wont split inside a write head"): a step takes every query head reading its
  write heads. The state is batch-major with write heads inside a sequence, and the op updates a contiguous plane in place, so a head block is
  WHOLE SEQUENCES (`max_heads // H` of them where the maximum holds one sequence's `H` write heads) or a RANGE OF ONE SEQUENCE'S write heads (where it
  does not). The plan is made over write heads, and a block of them carries their `G` query heads each: a write head's `route_W` slice holds
  its read heads' columns, and the block's output is the query heads `[first * G, last * G)`.
- **The layer's loop: for each head block, for each token range**, the step projects only that block's heads (each per-head weight's rows of
  them: `route_W`, `route_bias`, `gain_bias`, and `v_W`'s `d_v` rows a head) over that stretch of tokens, runs `prefill` on it and writes the
  output's block, so the projections are bounded along with the op, which an op-level split cannot do once the whole call's bundle exists. Blocks
  are independent; within a block the token steps run in order, carrying that block's state. STATELESS: a block that spans steps carries a fresh
  state of its own, and nothing is handed back. STATEFUL: where there is more than one block, the caller's state already covers the whole call
  (constructed for its batch and write heads, its plane zeroed there: [§1](#the-state)) and each block runs on a VIEW of its rows of the plane
  ([`internals/state.md#view`](internals/state.md#view)), so its update lands in the caller's plane in place.
- **A sequence of any length is served, and the caller pads nothing.** One no longer than the token maximum is ONE step, a partial window
  included (the kernels run a partial last window). The last step of a longer one holds what is left.
- **A token maximum below one window is refused where the sequence must be split.** Tokens of one window interact, so a split sits on a
  window boundary and the least step of a split sequence is one window: 10,000 tokens under a maximum of 500, or 800 under 500, has no
  plan, and the planner refuses it, naming the window (the owner, 2026-11-26: "sure we can not allow such a case").
- **`max_memory` bounds what a step holds beside the state**: the op's workspace at its WORST CASE and the step's projected inputs
  (the owner, 2026-11-26: "why wouldnt it? thats also part of the memory usage"). Both are functions of the call's shape alone (its
  levels' digits, which levels each side can hold a mask on, its batch-heads, its value width), so the bound holds whatever the data
  and no density is read: the workspace is the prepared masks of the most leaf groups the shape allows, a sixteenth of its leaves a
  side ([`internals/common/plan/leaf_groups.md`](internals/common/plan/leaf_groups.md#most)), and the inputs are each side's rows and
  scales, the masks and the values. One planner function sums them, each part from the function that sizes it
  ([`internals/common/plan/segments.md`](internals/common/plan/segments.md#step)). TOKENS FIRST: the handle's policy is the most whole windows
  whose step of every head fits; only where one window of every head does not fit, one window and the most write heads whose one-window step
  fits ([`internals/common/plan/segments.md`](internals/common/plan/segments.md#policy)). Where one window of one write head does not fit the
  limit, no segmentation serves it and the call is refused, naming the bytes ("will refuse if there is no form of segmentation that exists that
  can serve it"). Outside the limit: the state, the same however the call is cut, and the call's output, assembled whole.
- **A handle is any function of the context** (`fsla.segmentation.SegmentContext`: the levels' digits, each side's maskable levels, the
  batch, the write heads and query heads a sequence, the value width, the device) that returns a `SegmentPolicy`, or `None` for one op call. It
  is told no window and needs none.

A call of the layer needs the producer and the prefill, which are not built on this line (`KernelNotBuilt`); the plan, the policy, the
handle, their refusals, the state's view and the layer's loop (with a stand-in for the op) are built and tested on the host
(`tests/unit/test_segmentation.py`, `tests/unit/test_state_view.py`).

## 5. What is deliberately not here

- **No dict or JSON configuration surface.** Raw configuration belongs to consumers. The
  library type-checks structured objects and parses nothing.
- **No pluggable recurrence.** FSLA *is* the feature map, not a framework for
  recurrences. The state update is not an extension point and must never be presented as
  one.
- **No silent second path.** Every arm is chosen from a property of the configuration or
  of the operands, before the call, and named in the exception when a configuration
  cannot run. Nothing degrades into anything after a failure.
