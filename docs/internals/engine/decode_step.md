# The decode step — one plain function

`fsla/engine/decode_step.py`. The kernel-side page is
[`internals/decode/decode.md`](../decode/decode.md); this one is the HOST side — what
the step's values are, in what order they are found, and how the two properties a decode
step lives or dies by is held: it must be capturable.

Symbols used below: `B` batch, `H` heads, `BH = B*H`, `D` level count, `N` leaf count,
`d_v` value width, **atom** = 16 consecutive leaves = the page granule.

## <a id="the-state"></a>1. The step is ONE function over the dense state

The step reads and updates the state's plane in place and returns `(y, state)`; the state is the one dense implementation
of the interface ([`state.md#interface`](../state.md#interface)), and `backing` below reaches it only through the interface's seam,
`_decode_entry`. There is no host-visible verdict to read: nothing in the step is conditional on a fact the host holds, which is what
makes a CUDA graph captured over it valid for the sequence's life.

## <a id="the-order"></a>2. The step, read top to bottom

`decode_step_gated(routes, values, masks, state, *, decay=None)` takes its inputs only. It validates (`validate_context`, the fast-fail) and then calls, in this order:

| step | returns |
|---|---|
| `decode_call` | `BH`, `device` (refuses a grouped bundle) |
| `descriptor_of` | the call's descriptor, from the bundle's declaration, `d_v`, whether a decay is handed, `BH` and the device |
| `scratch` | the per-sequence buffers |
| `backing` | `carried_plane` |
| `state_plane` | the plane the kernel indexes |
| `build_launch` | the launch |
| `step` | `y` |

Each step runs inside a profiler range under its own name ([`engine.md`](engine.md#the-profiler-range)).

**THERE IS NO OPERAND STEP AND NO ADMISSION STEP.** The step's per-token operands are
folded inside the step kernel, from the producer's own views
([`decode/decode_fold.md`](../decode/decode_fold.md)). So a
decode step is ALWAYS one launch, and has no stage whose only output is a widened copy of an input or a residency answer for a
launch still to come. The detached clock is the write rows' own shares
([`decode.md#detached-clock`](../decode/decode.md#detached-clock)), so it is not a stream
of its own either.

`backing` is `FSLAState._decode_entry`: the plane the state holds, checked against the bound format. `build_launch` derives the decay arm's `dials`: a per-HEAD constant, and the one
operand the step kernel does not fold. Everything else on the launch's `operands` is a view the producer
already holds; its `descriptor` is the one `descriptor_of` built, and the state plane's leaf order is the lattice of the descriptor's level widths, solved by the launch function
([`decode_launch.md`](../common/plan/decode_launch.md); `fsla.ops.decode.lattice_of` reads the same solve for the oracle and test seam).

`state_plane` carries the per-step SHAPE agreements — `d_v` against the descriptor's arm,
the plane's extent against `[BH, N/16, page_bytes]`. They are
checked per step and not when the descriptor is made because that is where the plane the step actually
indexes is known.

## <a id="capturable"></a>3. The step is capturable: no host read of a device value, no allocation a step

Nothing the step calls reads a device value on the host or allocates per step, which is
precisely what a capture requires. The scratch is allocated once per
sequence ([section 4](#the-scratch)), so the first step builds it and every later step finds it.

The property is stated here and proven by `tests/integration/test_decode_graph_step.py`,
which captures the step and replays it against the eager run. A step added to the call that reads a device value, or allocates per step, fails
there at someone's first `graph.replay()`.

## <a id="the-scratch"></a>4. The descriptor and the scratch, and why the scratch is held the way it is

THERE IS NO DECODE PLAN. The decode op builds the validated `Descriptor` itself (`descriptor_of`, over `fsla.ops.produce._decode_descriptor`: every device fact and every call fact, immutable, hashable; the
key of the scratch), and what the call derives is the descriptor's or the launch function's: the widths, each side's walk of each level (`write_dense` derived from it) and `d_v` are call facts the descriptor states and
`plan::decode_launch` ([`decode_launch.md`](../common/plan/decode_launch.md)) reads; the per-level dial row offsets are the launch function's output (the widths' prefix sums); the `(k, m)`
lattice the state plane is written in ([`decode_lattice.md`](../decode/decode_lattice.md)) is the launch function's output, a function of the widths;
`eps` is data and `ctas` a private keyword of the internal `step`. None of that is a derived axis. `BC` is deliberately NOT a
fact: owner-blocked AMORTIZATION is a property of how a tiled consumer shares a
`[BC, cols]` block's cost across many tokens, and with one token there is nothing to
amortize, so the decode kernels are BC-blind by construction in that sense — the
lattice's OWN `BC` (the leaf count one owner holds) is a layout fact the `(k, m)` fields
carry instead, never a launch-shape one
([`decode.md#bc-absent`](../decode/decode.md#bc-absent)).

`scratch` holds its buffers in one module-level `WeakKeyDictionary` keyed on the sequence's state object, each
value a dict keyed `(descriptor, BH, device)`; the terminal values beside it — the folded amplitudes, the plane, the kernel's required buffers —
are per-call and never held.

The weakness is REQUIRED and not incidental: a dropped
sequence must drop its buffers, and two live sequences must never share one. Holding the scratch
is also what hoists the allocation, and the once-only initialization of `yacc`
and `queue`, above any captured region — the step's own finishers and
last CTA out reset each of them, so no `memset` ever stands between two decode steps
([`decode.md#capture`](../decode/decode.md#capture)). The state is the table's key and never an input of the
build, so two scratches built for one geometry are interchangeable by construction
(`tests/unit/test_decode_scratch.py`).

Because the descriptor is FIXED for the sequence, a state has exactly one live
scratch; `_scratch_of` reads it through that table and refuses a state holding more than
one, which is the observable form of the decay arm changing mid-sequence.
