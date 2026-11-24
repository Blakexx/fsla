# The prefill call

`fsla/engine/prefill_call.py`: one SEGMENT of a prefill -- its routes and values -- into its readout,
against the state the call is handed. `fsla.prefill` validates the context in the caller's grad mode
(`validate_context`: what a kernel would fault or misread on -- the bundles' types, and the form of the masks and of the `density` -- and never a value of either: a mask is its supplier's guarantee, and a level-side's entry may be ABSENT (`None`: dense, every digit of every token live; the library makes no mask for a level-side whose activation cannot make exact zeros) and is then not checked -- its density byte is 255, its supplier's duty, the dense flag being the byte -- and a density's values are never refused, since every byte is a valid density and it chooses a layout and no result (its dtype is `torch.uint8`);
the density rides the context into `PrefillOperands` for the kernels, which are not built), builds the call's descriptor from its inputs (`descriptor_of`: the bundles' extents and declaration, whether a decay and a state are handed, the operands'
device, through `fsla.ops.produce`'s memoized builder; its device refusal and its owed-kernel refusal are made there), then runs `run_prefill` inside an autograd node whose
backward refuses. `run_prefill` builds the workspace the steps pass between them (`fsla.ops.prefill`'s private builder; its layout is owed, `docs/open-work.md` row A-g). The layer (`fsla.FSLA`) calls the op once a segment; the segment
length is an output of the prefill launch function (`docs/api.md` section 4.2, `docs/open-work.md` row A-f).

## The steps, in order

`run_prefill` is one plain function, stateless and stateful alike, calling its steps in this order:

| Step | Returns | What it does |
|---|---|---|
| `backing` | `page_table`, `state_out` | the planes the launch is handed; a continuation's are one |
| `launch` | `num`, `den` | the carry at its bytes (`num` and the v+1 channel `den`) and the intra into the same planes, each over the reader factor (`fsla.ops.prefill.prefill`, [`#the-seam`](#the-seam)) |
| `post` | `y` | THE POST PASS, `y = num / (den + eps / r)`, `[B, T, H, d_v]` fp32 (`fsla.ops.prefill.readout`; `r` the reader factor the kernels leave out) |

Each step runs inside a profiler range under its own name ([`engine.md`](engine.md#the-profiler-range)).

The call holds the steps that run over its descriptor and workspace. The state's entry (`FSLAState._kernel_entry`) needs the arena's
allocation extent, the carry's box leaves, which only the prefill launch function's carve gives, so the entry on the prefill path is that
function's (`docs/open-work.md` rows A-c and A-f); a prefill is refused before any of it could run.

## The seam

The call is handed the bundle itself. The carry and the intra (not built) read its rows and values where the
producer stores them, grouped or not (`docs/api.md#grouped-queries`). The `launch` step is one composition
(`fsla.ops.prefill.prefill`): the carry reduces its partials and reads each token's earlier-window denominator out as
its v+1 channel, and the intra reduces its in-window pairs into the same two planes, both sums the recurrence's over
the reader factor (`fsla/ops/prefill.py`'s header). No carry or intra kernel is built on this line: the envelope
refuses every segment for want of a carry arm, and the composition raises by name
([`docs/open-work.md`](../../open-work.md)).
