# `segments.h`: the segments of a call

How a long prefill is cut into op calls. The op runs one segment and knows nothing of segmentation; the layer loops over the steps of a
PLAN, head blocks and token ranges in each, carrying each block's state ([`../../../api.md`](../../../api.md) section 4.2). The plan is the
planner's: it needs the card's window, a hardware fact, and Python shapes no call from one. `csrc/fsla/src/common/plan/segments.h` is pure and
`constexpr`, with no device; its cases are proved at compile time (`segments_proof`).

## The plan

`segment_plan(length, max_tokens, max_heads, batch, heads, window)` returns `SegmentPlan{steps, tokens, sequences, heads}`: head blocks of
`sequences` whole sequences, or of `heads` write heads of one sequence (`sequences == 1`), and in each `steps` op calls of `tokens` tokens, the last
holding the rest. A maximum of `kNoMaximum`, 0, is no limit on its axis. The two axes are independent, and the head axis lifts no token refusal.

**Heads.** `max_heads` counts WRITE heads, the state's owners: a step takes every query head reading them and never splits a write head's query
group (the owner, 2026-11-26: "we are just saying heads is write heads, and all read heads must live for any write head? we wont split inside a write
head"). The state is batch-major with write heads inside a sequence, and the op updates a contiguous plane in place, so a block is a contiguous run of
the state's rows: either WHOLE SEQUENCES, `max_heads / heads` of them, where the maximum holds one sequence's write heads, or a RANGE OF ONE SEQUENCE'S
write heads, `max_heads` of them, where it does not; the last block of the batch, or of a sequence, holds the rest. A maximum of every write head of
the call is one block.

**Tokens.**

- A call that fits (`length <= max_tokens`, or no limit) is ONE step of `length` tokens, a partial window included: the kernels run a partial last
  window and the caller pads nothing.
- A longer call is split on window boundaries only: tokens of one window interact, and the window grid is global, so a boundary on a window boundary
  moves no pair. `tokens` is the most whole windows under the maximum, `max_tokens / window * window`, and `steps = ceil(length / tokens)`.
- Where a split is needed and the maximum is under one window there is no plan: `steps == 0`. The owner (2026-11-26): "sure we can not allow such
  a case".

| length | max_tokens | window | tokens |
|---|---|---|---|
| 300 | none | 1,024 | 1 step of 300 |
| 300 | 500 | 1,024 | 1 step of 300 |
| 10,000 | 4,000 | 1,024 | 4 steps of 3,072, the last 784 |
| 2,049 | 2,048 | 1,024 | 2 steps of 2,048, the last 1 |
| 10,000 | 500 | 1,024 | none |
| 800 | 500 | 1,024 | none |

| batch | heads | max_heads | blocks |
|---|---|---|---|
| 4 | 8 | none, or 32 | one |
| 4 | 8 | 20 | 2 sequences each |
| 4 | 8 | 8 | a sequence each |
| 4 | 8 | 3 | heads 0-2, 3-5, 6-7 of each sequence |

## <a id="step"></a>A step's bytes

The memory limit is a PREFILL question -- what does one step of this call hold? -- so one function answers it, summing a step's parts,
each from the function that sizes the real thing (the owner, 2026-11-26, on counting a step's inputs: "why wouldnt it? thats also part
of the memory usage"; and on the form: "why isnt it like a prefill function that calls the actual same underlying thigns so drift is
impossilbe"). A step's shape is `StepShape`: its levels' widths as log2, each side's masked levels (a bit a level) and batch-heads (the
read side, the query side, first), and the values' channels. `step_shape(widths, levels, masked, heads, value_width)` is the one place a shape is
made from a call's numbers and judged (`levels == 0` where a part is outside the op's: no level or more than four, a width that is no
power of two from 2 to 256, no batch-head on a side, read batch-heads that are no whole number of the write side's -- `RouteFactors`' own law -- or no
value channel); the library's entries and `plan_tool --step` and `--policy` all call it.

| part | function | what sizes the real thing |
|---|---|---|
| the op's workspace, at the most leaf groups the shape allows | `workspace_bytes(shape, windows, window)` | `prepared_layout` ([`mask_prepare_layout.md`](mask_prepare_layout.md)), the function `mask_prepare_bytes` allocates the workspace by: the same code |
| the projected operands | `operand_bytes(shape, windows, window)` | the op's operand formats: a route digit a byte and a scale an fp32 a level a side, a mask word an int32 of 32 tokens of one digit on each masked level-side, a value channel a bf16 and the values' scale an fp32 on the write side. PyTorch allocates these from the producer's formats, so this figure is a statement of them, tied to a real bundle's tensors by `tests/unit/test_segmentation.py` |

`step_bytes` is their sum. Every part is a window's bytes a window and a batch-head's bytes a batch-head of its side, so a step's bytes are
`windows` times its one window's, and its one window's are its write heads times ONE WRITE HEAD's: `write_head_of(shape)`, one write head with every
query head that reads it (`segments_proof` asserts both). A part the prefill gains later -- the carry's workspace -- is a term added here. OUTSIDE THE
SUM: the state, the same however the call is cut, and the call's output, assembled whole.

## <a id="policy"></a>A step's policy under a byte limit

`step_policy_within(shape, limit, window)` returns `StepPolicy{tokens, heads}`, TOKENS FIRST: the most whole windows of every head under `limit`
(`step_tokens_within`, `limit / step_bytes(shape, 1, window) * window`, with `heads` no limit); only where one window of every head does not fit, one
window and the most write heads whose one window fits, `limit / step_bytes(write_head_of(shape), 1, window)`; none (`tokens == 0`) where one window of
one write head does not fit. The owner (2026-11-26): "i think its reasonable to support a segmentationpolicy primtive that has like max_tokens,
max_heads, null meaning no limit".

## <a id="entries"></a>The entries

All three are the planner's (`environment.cu`, declared in `plan_api.cuh`), on the card `device`, whose window is `window_of_cc` of its
compute capability ([`levels.md`](levels.md)).

- `fsla_segment_plan(length, max_tokens, max_heads, batch, heads, device)` returns `[steps, tokens, sequences, heads]`. A call with no plan is
  refused, naming the window. `fsla.segment_plan(length, policy, *, batch, heads, device=None)` is its Python face (`fsla/segmentation.py`), and its
  `SegmentPlan` lists the blocks (`blocks()`) and the token ranges (`bounds()`); a call no maximum cuts is one op call and asks no card.
- `fsla_step_bytes(widths, masked, heads, value_width, tokens, device)` returns `[workspace, operands]` of a step of `tokens` tokens of every
  head, in whole windows (`fsla.segmentation.step_bytes`).
- `fsla_step_policy(widths, masked, heads, value_width, limit, device)` returns `[max_tokens, max_heads]`, 0 for no limit, the policy under `limit`
  bytes; where one window of one write head does not fit it is refused, naming that window's bytes and both parts. `fsla.max_memory(bytes)` is the
  handle that asks it, and returns its answer as a `fsla.SegmentPolicy`.
- `plan_tool --segments length,max_tokens,max_heads,batch,heads,window;...`, `plan_tool --step widths/masked/heads/value_width/windows/window` and
  `plan_tool --policy widths/masked/heads/value_width/limit/window` ([`plan_tool.md`](plan_tool.md)) print the plan, a step's bytes and its policy
  as JSON, with no device: `tests/unit/test_segmentation.py` runs the cases above and a sweep of both axes through the first, checks the operand
  bytes against the tensors a producer makes through the second, and the policy against the second's bytes through the third; on a card it checks the
  library's plan, the workspace against `mask_prepare_bytes` and `fsla.max_memory` against `step_bytes`.
