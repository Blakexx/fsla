# The decode family's DAG — M3 and M4

`rola/engine/dags/decode_dag.py`. The kernel-side page is
[`internals/decode/decode.md`](../decode/decode.md); this one is the HOST side — what
the step's facts are, in what order they are found, and which of the framework's
declarations carry the two properties a decode step lives or dies by: it must be
capturable, and its growth must self-correct.

Symbols used below: `B` batch, `H` heads, `BH = B*H`, `D` level count, `N` leaf count,
`d_v` value width, **atom** = 16 consecutive leaves = the page granule, **gate** = the
`[1]` int32 the growth probe reduces into and the step reads on the device.

## <a id="the-two-masks"></a>1. M3 and M4 are the two BACKINGS, and a growth step is neither

`Mask.M3` is decode-dense and `Mask.M4` is decode-paged. That is the whole of the
specialization, because it is the whole of the difference in the node set: a dense step
addresses the leaf space directly and has nothing to admit, so `GrowthProbe` — and with
it the gate and the verdict — is the one band M3 masks off.

**A GROWTH STEP IS NOT A THIRD MASK.** It is M4 walked twice, around an admission the
host performs between the two walks. A mask selects a subgraph; growth selects the same
one, which is exactly why a CUDA graph captured over the step stays valid across an
admission and why a replay is self-correcting: the probe is INSIDE the captured region,
so on the second walk its verdict is 0 and the step it had skipped performs.

## <a id="the-order"></a>2. The step, read top to bottom

| node | gives | stream | mask |
|---|---|---|---|
| `decode_call` | `widths`, `d_v`, `has_decay`, `BH`, `device`, `paged` | host | both |
| `decode_mask` (rule) | `mask` | host | both |
| `geometry` | `geometry` | host, `pure_of` | both |
| `scratch` | `scratch` | host, `per_state` | both |
| `growth_probe` | `gate` | main | M4 |
| `backing` | `carried_plane`, `page_table` | main | both |
| `state_plane` | `state_plane` | main | both |
| `build_plan` | `plan` | main | both |
| `step` | `y` | main | both |

**THERE IS NO OPERAND NODE.** The step's per-token operands are folded inside the step
kernel, from the producer's own views, and the growth probe folds its own write side the
same way ([`decode/decode_fold.md`](../decode/decode_fold.md)). So a dense step is ONE
launch and a paged step is two, and neither of them has a stage whose only output is a
widened copy of an input. The detached clock is the write allocation itself
([`decode.md#detached-clock`](../decode/decode.md#detached-clock)), so it is not a stream
of its own either.

`build_plan` carries a stream because the decay arm's `dials` are derived there: a per-HEAD
constant, and the one operand the step kernel does not fold. Everything else on the plan is
a view the producer already holds.

`state_plane` carries the per-step SHAPE agreements — `d_v` against the frozen config,
the plane's extent against the backing, the page table's against `[BH, N/16]`. They are
checked per step and not at plan build because that is where the plane the step actually
indexes is known.

## <a id="the-verdict"></a>3. The verdict is TWO nodes, and that is the host order

`decode_forward` is this same tuple with the host-visible verdict spliced in:
`request_verdict` (a `copy_(non_blocking)` into the pinned mirror, then
`Event.record()`) ahead of `backing`, and `read_verdict` (`Event.synchronize()`, then the
one 4-byte read) after `execute`.

Two nodes and not one epilogue node, because the copy must be ENQUEUED AHEAD of the step
it gates. That is the whole of P67-a: the step is enqueued behind its own verdict, gated
so that a growth step is a device-side no-op, and the host's wait for the flag is spent on
the launches instead of in front of them
([`decode.md#the-gate`](../decode/decode.md#the-gate)). `read_verdict` is the family's
ONE `host_sync` node, and it is the last node in the tuple.

`decode_step_gated` runs the tuple WITHOUT either: the caller owns the verdict, at
whatever synchronization it already has, through `growth_pending` and `admit_growth`
([`decode.md#capture`](../decode/decode.md#capture)).

## <a id="capturable"></a>4. `capturable` is a declaration, not a hope

Every node of the step declares `capturable=True`, and
`rola.engine.runner.validate_capturable` refuses the tuple at import if one does not. A
node cannot declare both `capturable` and `host_sync`: reading a device value on the host
is precisely what a capture forbids, and the two attributes are one claim written from
two ends.

The declaration is what makes the property survive editing. A node added to the step that
reads a device value, or allocates per step, fails at import rather than at some caller's
first `graph.replay()` — and `tests/integration/test_decode_graph_step.py` is what proves
the declaration is TRUE of a real capture.

## <a id="host-carrier"></a>5. The carrier and the scratch, and why each is memoized the way it is

`DecodeGeometry` (`rola/engine/plan.py`) answers "what shape is the launch?" — widths,
mixed-radix strides, spans, row offsets, word and group counts, `d_v`, the normalization
and decay arms, `eps`, `n_split`. None of that is a derived axis. `BC` is deliberately NOT
a field: owner blocking is a property of how a tiled consumer amortizes a `[BC, cols]`
block across many tokens, and with one token there is nothing to amortize, so the decode
kernels are BC-blind by construction
([`decode.md#bc-absent`](../decode/decode.md#bc-absent)).

It is `DecodePlan`'s CONFIG HALF rather than a field of it, because the memo key is over
this alone: `geometry` is declared `pure_of` `(widths, d_v, has_decay, BH, device)` and
the terminal facts beside it — the folded amplitudes, the plane, the gate — are per-call
and never cached.

`scratch` is declared `per_state`, which makes its table a `WeakKeyDictionary` keyed on
the sequence's state object. The weakness is REQUIRED and not incidental: a dropped
sequence must drop its buffers, and two live sequences must never share one. Being
memoized is also what hoists the allocation, and the once-only zeroing of `ctr` and
`probe_ctr`, above any captured region — the last-arriving CTA of every step resets the
counter, so no `memset` ever stands between two steps
([`decode.md#capture`](../decode/decode.md#capture)).

Because the launch config is FROZEN for the sequence, a state has exactly one live
scratch; `growth_pending` and `admit_growth` read it through that table and refuse a
state holding more than one, which is the observable form of the decay arm changing
mid-sequence.

## <a id="reference-only"></a>6. `reference_only` — the enumerated gate reference, refused mechanically

`write_atom_bitmap` is declared as a node and belongs to no DAG. It is the gate's
INDEPENDENT second derivation of the write-atom set, written from the definition rather
than from the kernel's structure
([`decode.md#second-derivation`](../decode/decode.md#second-derivation)); the shipped path
derives that set on the device, inside the probe. Splicing it into a shipped DAG would
make the gate compare the probe against itself, and `validate_dag` refuses any DAG
containing a `reference_only` node — which is the mechanical form of the warning the
function's docstring used to carry alone.
