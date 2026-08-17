# The backward family's DAG

Mirror of `rola/facts/backward_dag.py`, `BackwardPlan` in `rola/facts/plan.py`, and
the reverse-pass bodies in `rola/ops/chunk.py`. Framework vocabulary — masks, seeds,
the fact bag, the Join — is [the facts engine's page](facts.md); what the two scans
COMPUTE is [the backward's own page](../chunk/backward.md).

Symbols: `T`/`L` tokens, `N` leaf count, `BC` owner block, `BH = B*H`, `den` the
readout denominator, `dy` the output cotangent.

## <a id="the-plan-is-the-input"></a>1. The plan is this DAG's INPUT

```
seeds  = (plan, dy, y, v, call_class)
backward_seeds  -> seeds                    @main
scan_a          -> d_read                   @main
scan_b          -> dv, d_gain, d_write      @main
unpack          -> grads                    @main
```

Every other family's DAG ENDS at a plan. This one starts at one, because the reverse
pass finds no facts: each fact it reads is one the forward already produced, so its
plan is the forward's `ChunkPlan` held by reference. What the walk produces is the
gradient list in the operands' own layouts.

`dy` arrives with the call; `y` and `v` ride `save_for_backward`, because they are
the autograd node's own output and operand and autograd owns their versioning. `v`
is read for its DTYPE alone — the scans consume the plan's packed bf16 copy.

## <a id="the-ctx"></a>2. `BackwardPlan` is two fields, and `den` is the second

`(forward: ChunkPlan, den: Tensor)`. `den` is not a `ChunkPlan` field because it is
the forward's OUTPUT rather than one of its launch arguments: `execute(plan)` returns
it, so a plan carrying it would have to be built after the launch it configures, or
be mutable. It is fed here instead, at retention time — which is also the moment the
retention decision is actually made, and the reason the decision reads as one.

RETENTION IS A DECISION, NOT A MASK. Whether a call carries a gradient is
`needs_input_grad`, evaluated in the autograd node's `forward`, and its only effect
is whether a `BackwardPlan` is constructed. It selects no subgraph, compiles no
variant and changes no launch, so `grad` is a `CallClass` bit that no mask reads.

`retained_bytes()` counts `replayed_from()` — the packed read and write sides, the
write gain, the read mass, the union table and `den` — which is the `O(L)` closed
form and is what a grad-bearing call costs over an inference one. The plan reference
additionally pins `block_bits` (`[BH, N / BC]` bytes, no `T`) and `v` (the op's own
operand, pinned before this too). Both are NAMED and gated in
`tests/integration/test_backward_ctx.py` rather than folded into the closed form.

## <a id="the-scans-are-independent"></a>3. The scans' independence is declared

`scan_a` and `scan_b` read the same two facts — the plan and the seeds — and neither
reads anything the other produces. That was true before and invisible: both were
handed the SAME hand-built ten-argument tuple, spelled twice at the call site, where
an edit to one could silently drift from the other. The tuple is now
`chunk._scan_operands(plan, seeds)`, read off the plan, and the independence is
`needs` edges that an import-time check asserts stay absent.

They are still issued in tuple order on one stream. Putting them on two is a
measurement and no stage has claimed one.

## <a id="what-the-seeds-are"></a>4. The seeds are the divide's own reverse

`chunk_backward_seeds` returns `dnum`, `dden` and — when the bundle carries a read
mass — `dm = eps * dden`. The eps floor is ADDITIVE, so all three are exact and none
of them branches on `den`. `Seeds` is a `NamedTuple` and nothing in it is retained:
it is built from `dy` inside the backward and dies with it, which is why it is a fact
of the walk and not a field of anything the forward keeps.

## <a id="the-order"></a>5. `unpack`'s order is load-bearing

The autograd node receives the operands FLAT, in `RouteFactors.tensors()`' order, so
the gradient list must come back in that order or every gradient lands on the wrong
tensor silently — the shapes agree. `tests/unit/test_interface.py` asserts the
correspondence. The narrowing to each operand's dtype and the un-permute to its
public layout happen here because autograd requires `grad.dtype == input.dtype` and
the packing that produced the kernel layout was the node's own.
